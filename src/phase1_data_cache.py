"""Canonical, pre-split cleaned-data cache for Phase 1.

The cache boundary is intentionally narrow:

raw Zeek rows -> split_label_column -> clean_flows -> stable row ID -> Parquet

No split membership, sampling result, fitted preprocessing state, transformed
feature, class weight, or model state is allowed in this cache.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
import shutil
import tempfile
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import pandas as pd

from src.data_io import split_label_column
from src.preprocess import (
    DROP_COLUMNS,
    FLOAT_COLUMNS,
    INT_COLUMNS,
    MISSING_TOKENS,
    STRING_COLUMNS,
    clean_flows,
)


logger = logging.getLogger(__name__)

CACHE_FORMAT = "parquet"
CACHE_SCHEMA_VERSION = 2
PARSER_CLEANING_VERSION = "phase1-canonical-pandas-v2"
ROW_ID_COLUMN = "_clean_row_id"
FORBIDDEN_CACHE_TOKENS = (
    "split",
    "mask",
    "scaler",
    "encoder",
    "vocab",
    "class_weight",
    "undersampl",
    "feature_vector",
    "edge_attr",
)


@dataclass(frozen=True)
class CacheLoadReport:
    scenario: str
    source_path: str
    cache_path: str | None
    fingerprint: str
    cache_status: str
    raw_open_count: int
    parsed_rows: int
    returned_rows: int
    file_size_bytes: int
    elapsed_seconds: float
    rows_per_second: float
    megabytes_per_second: float
    parse_error_rows: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "Phase 1 Parquet cache requires pyarrow. Install project "
            "requirements (or `pip install pyarrow>=15`)."
        ) from exc
    return pa, pq


def canonical_schema_contract() -> dict[str, Any]:
    """Return the stable schema rules that participate in cache identity."""

    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "row_id_column": ROW_ID_COLUMN,
        "float_columns": list(FLOAT_COLUMNS),
        "int_columns": list(INT_COLUMNS),
        "string_columns": list(STRING_COLUMNS),
        "drop_columns": list(DROP_COLUMNS),
        "missing_tokens": list(MISSING_TOKENS),
        "missing_flag_suffix": "_missing",
        "stable_row_id_version": "phase1-clean-v1",
    }


def parser_cleaning_digest() -> str:
    """Digest the explicit version and actual parsing/cleaning functions."""

    payload = {
        "version": PARSER_CLEANING_VERSION,
        "pandas_version": pd.__version__,
        "field_parser": inspect.getsource(_field_names_from_handle),
        "split_label_column": inspect.getsource(split_label_column),
        "clean_flows": inspect.getsource(clean_flows),
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cache_fingerprint(path: str | os.PathLike[str]) -> tuple[str, dict[str, Any]]:
    """Fingerprint raw identity plus parser and canonical schema contract."""

    source = Path(path)
    stat = source.stat()
    payload = {
        "raw_file_size": int(stat.st_size),
        "raw_file_mtime_ns": int(stat.st_mtime_ns),
        "parser_cleaning_digest": parser_cleaning_digest(),
        "canonical_schema": canonical_schema_contract(),
    }
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


def cache_path_for(
    cache_dir: str | os.PathLike[str],
    scenario: str,
    fingerprint: str,
) -> Path:
    return Path(cache_dir) / scenario / f"{fingerprint}.{CACHE_FORMAT}"


def _stable_row_ids(scenario: str, start: int, count: int) -> list[str]:
    return [
        hashlib.sha256(
            f"phase1-clean-v1:{scenario}:{position}".encode("utf-8")
        ).hexdigest()
        for position in range(start, start + count)
    ]


def _field_names_from_handle(handle: Any, path: Path) -> list[str]:
    canonical = {
        "det_label": "detailed-label",
        "detailed_label": "detailed-label",
        "label_val": "label",
    }
    for line in handle:
        if line.startswith("#fields"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                raise ValueError(f"Invalid #fields row in {path}.")
            return [canonical.get(name, name) for name in parts[1:]]
    raise ValueError(f"Missing #fields row in {path}.")


def _source_bytes_read(handle: Any, file_size: int) -> int:
    try:
        position = int(handle.buffer.tell())
    except (AttributeError, OSError, ValueError):
        try:
            position = int(handle.tell())
        except (AttributeError, OSError, ValueError):
            return 0
    return min(max(position, 0), file_size)


def _log_progress(
    *,
    scenario: str,
    status: str,
    rows: int,
    bytes_processed: int,
    total_bytes: int,
    started_at: float,
) -> None:
    elapsed = max(time.monotonic() - started_at, 1e-9)
    rows_rate = rows / elapsed
    mb_rate = bytes_processed / (1024**2) / elapsed
    logger.info(
        "phase1 cache %s | scenario=%s | bytes=%d/%d | rows=%d | "
        "elapsed=%.1fs | rate=%.0f rows/s (%.2f MiB/s)",
        status,
        scenario,
        bytes_processed,
        total_bytes,
        rows,
        elapsed,
        rows_rate,
        mb_rate,
    )


def _iter_clean_raw_chunks(
    path: Path,
    scenario: str,
    *,
    chunksize: int,
    progress_interval_seconds: float,
    counters: dict[str, int],
) -> Iterator[pd.DataFrame]:
    """Open raw input once and yield deterministic cleaned chunks."""

    file_size = int(path.stat().st_size)
    started_at = time.monotonic()
    last_progress = started_at
    row_offset = 0
    bad_lines = 0

    def on_bad_line(_: list[str]) -> None:
        nonlocal bad_lines
        bad_lines += 1
        return None

    counters["raw_open_count"] += 1
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        field_names = _field_names_from_handle(handle, path)
        handle.seek(0)
        reader = pd.read_csv(
            handle,
            sep="\t",
            comment="#",
            header=None,
            names=field_names,
            na_values=[],
            keep_default_na=False,
            skip_blank_lines=True,
            dtype=str,
            engine="python",
            on_bad_lines=on_bad_line,
            chunksize=chunksize,
        )
        for raw_chunk in reader:
            parsed = split_label_column(raw_chunk)
            cleaned = clean_flows(parsed).reset_index(drop=True)
            cleaned[ROW_ID_COLUMN] = _stable_row_ids(
                scenario, row_offset, len(cleaned)
            )
            row_offset += len(cleaned)
            counters["parsed_rows"] = row_offset
            counters["parse_error_rows"] = bad_lines
            now = time.monotonic()
            if now - last_progress >= progress_interval_seconds:
                _log_progress(
                    scenario=scenario,
                    status="MISS",
                    rows=row_offset,
                    bytes_processed=_source_bytes_read(handle, file_size),
                    total_bytes=file_size,
                    started_at=started_at,
                )
                last_progress = now
            yield cleaned

    _log_progress(
        scenario=scenario,
        status="MISS",
        rows=row_offset,
        bytes_processed=file_size,
        total_bytes=file_size,
        started_at=started_at,
    )


def _column_order(chunk_columns: Iterable[Iterable[str]]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for columns in chunk_columns:
        for column in columns:
            if column not in seen:
                seen.add(column)
                output.append(column)
    return output


def _normalize_chunk(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Normalize chunk schemas without introducing learned state."""

    output = frame.reindex(columns=columns).copy()
    for column in columns:
        if column in FLOAT_COLUMNS:
            output[column] = pd.to_numeric(
                output[column], errors="coerce"
            ).astype("float64")
        elif column in INT_COLUMNS:
            output[column] = (
                pd.to_numeric(output[column], errors="coerce")
                .fillna(0)
                .astype("int64")
            )
        elif column.endswith("_missing"):
            output[column] = (
                pd.to_numeric(output[column], errors="coerce")
                .fillna(0)
                .astype("int8")
            )
        else:
            # clean_flows deliberately exports categorical/identifier columns
            # as object; preserve that pandas contract through Parquet.
            output[column] = output[column].astype(object)
    return output


def validate_cache_columns(columns: Iterable[str]) -> None:
    lowered = [str(column).lower() for column in columns]
    forbidden = sorted(
        column
        for column in lowered
        if any(token in column for token in FORBIDDEN_CACHE_TOKENS)
    )
    if forbidden:
        raise ValueError(
            "Canonical Phase 1 cache contains forbidden learned/split fields: "
            + ", ".join(forbidden)
        )
    required = {"id.orig_h", "id.resp_h", "detailed-label", ROW_ID_COLUMN}
    missing = sorted(required - set(columns))
    if missing:
        raise ValueError(
            "Canonical Phase 1 cache is missing required fields: "
            + ", ".join(missing)
        )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


@contextmanager
def _cache_build_lock(path: Path) -> Iterator[None]:
    """Serialize builders so concurrent seed processes do not reparse raw."""

    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - Phase 1 targets Unix hosts
        raise RuntimeError(
            "Canonical cache locking requires fcntl on the training host."
        ) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _build_cache_file(
    path: Path,
    scenario: str,
    target: Path,
    *,
    fingerprint: str,
    fingerprint_payload: dict[str, Any],
    chunksize: int,
    progress_interval_seconds: float,
) -> dict[str, Any]:
    """Parse raw once, then atomically assemble one canonical Parquet file."""

    pa, pq = _require_pyarrow()
    target.parent.mkdir(parents=True, exist_ok=True)
    counters = {
        "raw_open_count": 0,
        "parsed_rows": 0,
        "parse_error_rows": 0,
    }
    temp_root = Path(
        tempfile.mkdtemp(prefix=f".{fingerprint}.", dir=target.parent)
    )
    chunk_paths: list[Path] = []
    chunk_columns: list[list[str]] = []
    chunk_rows: list[int] = []
    label_counts: Counter[str] = Counter()
    started_at = time.monotonic()
    try:
        for index, chunk in enumerate(
            _iter_clean_raw_chunks(
                path,
                scenario,
                chunksize=chunksize,
                progress_interval_seconds=progress_interval_seconds,
                counters=counters,
            )
        ):
            chunk_file = temp_root / f"chunk-{index:08d}.parquet"
            chunk.to_parquet(chunk_file, engine="pyarrow", index=False)
            chunk_paths.append(chunk_file)
            chunk_columns.append(list(chunk.columns))
            chunk_rows.append(len(chunk))
            label_counts.update(
                chunk["detailed-label"].astype(str).tolist()
            )
        if not chunk_paths:
            raise ValueError(f"Scenario {scenario!r} contains no parsed rows.")

        columns = _column_order(chunk_columns)
        validate_cache_columns(columns)
        assembled = temp_root / target.name
        writer = None
        try:
            for chunk_file in chunk_paths:
                chunk = pd.read_parquet(chunk_file, engine="pyarrow")
                normalized = _normalize_chunk(chunk, columns)
                table = pa.Table.from_pandas(
                    normalized, preserve_index=False
                )
                if writer is None:
                    writer = pq.ParquetWriter(
                        assembled,
                        table.schema,
                        compression="snappy",
                    )
                writer.write_table(table, row_group_size=len(normalized))
        finally:
            if writer is not None:
                writer.close()
        os.replace(assembled, target)

        elapsed = max(time.monotonic() - started_at, 1e-9)
        file_size = int(path.stat().st_size)
        manifest = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "cache_format": CACHE_FORMAT,
            "scenario": scenario,
            "source_path": str(path),
            "cache_path": str(target),
            "fingerprint": fingerprint,
            "fingerprint_payload": fingerprint_payload,
            "parser_cleaning_version": PARSER_CLEANING_VERSION,
            "columns": columns,
            "row_groups": [
                {
                    "index": index,
                    "rows": rows,
                    "source_columns": columns_for_chunk,
                }
                for index, (rows, columns_for_chunk) in enumerate(
                    zip(chunk_rows, chunk_columns)
                )
            ],
            "parsed_rows": int(counters["parsed_rows"]),
            "label_counts": dict(sorted(label_counts.items())),
            "parse_error_rows": int(counters["parse_error_rows"]),
            "raw_open_count": int(counters["raw_open_count"]),
            "elapsed_seconds": elapsed,
            "rows_per_second": counters["parsed_rows"] / elapsed,
            "megabytes_per_second": file_size / (1024**2) / elapsed,
        }
        _write_json_atomic(target.with_suffix(".json"), manifest)
        return manifest
    except Exception:
        raise
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _select_with_cap(
    chunks: Iterable[pd.DataFrame],
    cap_per_class: int | None,
) -> pd.DataFrame:
    buffers: list[pd.DataFrame] = []
    counts: dict[str, int] = {}
    all_columns: list[list[str]] = []
    for chunk in chunks:
        all_columns.append(list(chunk.columns))
        if cap_per_class is None:
            buffers.append(chunk)
            continue
        for label, group in chunk.groupby("detailed-label"):
            already = counts.get(str(label), 0)
            remaining = cap_per_class - already
            if remaining <= 0:
                continue
            if len(group) > remaining:
                group = group.sample(n=remaining, random_state=42)
            buffers.append(group)
            counts[str(label)] = already + len(group)
    columns = _column_order(all_columns)
    if not buffers:
        return pd.DataFrame(columns=columns)
    return pd.concat(buffers, axis=0, ignore_index=True).reindex(columns=columns)


def iter_cache_chunks(path: str | os.PathLike[str]) -> Iterator[pd.DataFrame]:
    """Yield canonical Parquet row groups without loading a full scenario."""

    _, pq = _require_pyarrow()
    parquet = pq.ParquetFile(Path(path))
    for index in range(parquet.num_row_groups):
        frame = parquet.read_row_group(index).to_pandas()
        yield _normalize_chunk(frame, list(frame.columns))


def _read_cache_with_cap(
    path: Path,
    cap_per_class: int | None,
    manifest: dict[str, Any],
) -> pd.DataFrame:
    """Read only row groups/rows needed by the deterministic class cap."""

    if cap_per_class is None:
        return _select_with_cap(iter_cache_chunks(path), None)

    _, pq = _require_pyarrow()
    parquet = pq.ParquetFile(path)
    columns = list(manifest.get("columns", parquet.schema.names))
    support = {
        str(label): int(count)
        for label, count in manifest.get("label_counts", {}).items()
    }
    targets = {
        label: min(cap_per_class, count)
        for label, count in support.items()
    }
    counts: dict[str, int] = {}
    buffers: list[pd.DataFrame] = []
    selected_column_groups: list[list[str]] = []
    row_groups = list(manifest.get("row_groups", ()))

    for row_group_index in range(parquet.num_row_groups):
        labels = parquet.read_row_group(
            row_group_index, columns=["detailed-label"]
        ).to_pandas()["detailed-label"]
        selections: list[pd.Index] = []
        for label, group in labels.to_frame().groupby("detailed-label"):
            label_string = str(label)
            already = counts.get(label_string, 0)
            remaining = cap_per_class - already
            if remaining <= 0:
                continue
            if len(group) > remaining:
                group = group.sample(n=remaining, random_state=42)
            selections.append(group.index)
            counts[label_string] = already + len(group)
        if selections:
            full = parquet.read_row_group(row_group_index).to_pandas()
            source_columns = (
                list(row_groups[row_group_index].get("source_columns", ()))
                if row_group_index < len(row_groups)
                else columns
            )
            full = full.reindex(columns=source_columns)
            full = _normalize_chunk(full, source_columns)
            selected_column_groups.append(source_columns)
            for selected_index in selections:
                buffers.append(full.loc[selected_index])
        if targets and all(
            counts.get(label, 0) >= target
            for label, target in targets.items()
        ):
            break

    if not buffers:
        return pd.DataFrame(columns=columns)
    selected_columns = _column_order(selected_column_groups)
    return pd.concat(buffers, axis=0, ignore_index=True).reindex(
        columns=selected_columns
    )


def ensure_canonical_cache(
    path: str | os.PathLike[str],
    scenario: str,
    *,
    cache_dir: str | os.PathLike[str] = "artifacts/data_cache",
    rebuild_cache: bool = False,
    cache_format: str = CACHE_FORMAT,
    chunksize: int = 200_000,
    progress_interval_seconds: float = 10.0,
) -> tuple[Path, dict[str, Any], str]:
    """Ensure one immutable fingerprinted cache exists without loading it."""

    if cache_format != CACHE_FORMAT:
        raise ValueError(
            f"Unsupported cache format {cache_format!r}; only parquet is supported."
        )
    source = Path(path)
    fingerprint, payload = cache_fingerprint(source)
    target = cache_path_for(cache_dir, scenario, fingerprint)
    manifest_path = target.with_suffix(".json")
    with _cache_build_lock(target.with_suffix(".lock")):
        if target.is_file() and manifest_path.is_file() and not rebuild_cache:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("fingerprint") == fingerprint
                and manifest.get("fingerprint_payload") == payload
            ):
                validate_cache_columns(manifest.get("columns", ()))
                return target, manifest, "HIT"
        manifest = _build_cache_file(
            source,
            scenario,
            target,
            fingerprint=fingerprint,
            fingerprint_payload=payload,
            chunksize=chunksize,
            progress_interval_seconds=progress_interval_seconds,
        )
        return target, manifest, "MISS"


def load_canonical_scenario(
    path: str | os.PathLike[str],
    scenario: str,
    *,
    cache_dir: str | os.PathLike[str] = "artifacts/data_cache",
    cache_enabled: bool = True,
    rebuild_cache: bool = False,
    cache_format: str = CACHE_FORMAT,
    cap_per_class: int | None = None,
    chunksize: int = 200_000,
    progress_interval_seconds: float = 10.0,
) -> tuple[pd.DataFrame, CacheLoadReport]:
    """Load one canonical frame and an observable raw/cache I/O report."""

    if cache_format != CACHE_FORMAT:
        raise ValueError(
            f"Unsupported cache format {cache_format!r}; only parquet is supported."
        )
    if chunksize < 1:
        raise ValueError("chunksize must be positive.")
    if cap_per_class is not None and cap_per_class < 1:
        raise ValueError("cap_per_class must be positive when supplied.")

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    fingerprint, payload = cache_fingerprint(source)
    target = cache_path_for(cache_dir, scenario, fingerprint)
    started_at = time.monotonic()
    raw_open_count = 0
    parse_error_rows = 0
    cache_status = "DISABLED"

    if cache_enabled:
        _require_pyarrow()
        target, manifest, cache_status = ensure_canonical_cache(
            source,
            scenario,
            cache_dir=cache_dir,
            rebuild_cache=rebuild_cache,
            cache_format=cache_format,
            chunksize=chunksize,
            progress_interval_seconds=progress_interval_seconds,
        )
        if cache_status == "MISS":
            raw_open_count = int(manifest["raw_open_count"])
        parsed_rows = int(manifest.get("parsed_rows", 0))
        parse_error_rows = int(manifest.get("parse_error_rows", 0))
        frame = _read_cache_with_cap(target, cap_per_class, manifest)
        if parsed_rows == 0:
            _, pq = _require_pyarrow()
            parsed_rows = int(pq.ParquetFile(target).metadata.num_rows)
    else:
        with tempfile.TemporaryDirectory(prefix="phase1-no-cache-") as tmp:
            temporary = Path(tmp) / f"{fingerprint}.parquet"
            manifest = _build_cache_file(
                source,
                scenario,
                temporary,
                fingerprint=fingerprint,
                fingerprint_payload=payload,
                chunksize=chunksize,
                progress_interval_seconds=progress_interval_seconds,
            )
            raw_open_count = int(manifest["raw_open_count"])
            parsed_rows = int(manifest["parsed_rows"])
            parse_error_rows = int(manifest["parse_error_rows"])
            frame = _read_cache_with_cap(
                temporary, cap_per_class, manifest
            )

    elapsed = max(time.monotonic() - started_at, 1e-9)
    file_size = int(source.stat().st_size)
    report = CacheLoadReport(
        scenario=scenario,
        source_path=str(source),
        cache_path=str(target) if cache_enabled else None,
        fingerprint=fingerprint,
        cache_status=cache_status,
        raw_open_count=raw_open_count,
        parsed_rows=parsed_rows,
        returned_rows=len(frame),
        file_size_bytes=file_size,
        elapsed_seconds=elapsed,
        rows_per_second=parsed_rows / elapsed,
        megabytes_per_second=file_size / (1024**2) / elapsed,
        parse_error_rows=parse_error_rows,
    )
    logger.info(
        "phase1 cache %s | scenario=%s | bytes=%d | parsed_rows=%d | "
        "returned_rows=%d | elapsed=%.2fs | raw_opens=%d | "
        "rate=%.0f rows/s (%.2f MiB/s)",
        report.cache_status,
        scenario,
        file_size,
        parsed_rows,
        len(frame),
        elapsed,
        raw_open_count,
        report.rows_per_second,
        report.megabytes_per_second,
    )
    return frame, report
