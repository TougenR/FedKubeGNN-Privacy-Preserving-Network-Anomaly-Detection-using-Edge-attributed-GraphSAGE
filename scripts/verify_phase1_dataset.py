#!/usr/bin/env python3
"""Verify the six raw IoT-23 scenarios required by clean Phase 1.

This command does not train or sample the dataset. It streams every file,
parses it with the same Zeek/IoT-23 conventions as ``src.data_io``, and writes
machine-readable manifests before returning a non-zero status on an incomplete
or seriously malformed dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data_io import split_label_column
from src.phase1_clean import FIXED_LABELS
from src.preprocess import clean_flows


REQUIRED_SCENARIOS = ("1-1", "3-1", "9-1", "34-1", "36-1", "39-1")
REQUIRED_PARSED_COLUMNS = (
    "id.orig_h",
    "id.resp_h",
    "label",
    "detailed-label",
)


def parse_scenario_arguments(items: Sequence[str]) -> dict[str, str]:
    """Parse the same ``name=PATH`` format accepted by ``src.phase1_clean``."""

    output: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Scenario must use name=PATH format, got {item!r}."
            )
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Invalid scenario mapping {item!r}.")
        if name in output:
            raise ValueError(f"Duplicate scenario name {name!r}.")
        output[name] = path
    return output


def scenarios_from_config(config_path: Path) -> dict[str, str]:
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return {
        str(item["name"]): str(item["path"])
        for item in config.get("experiments", {}).get("scenarios", ())
        if item.get("name") and item.get("path")
    }


def _field_names(path: Path) -> list[str]:
    canonical = {
        "det_label": "detailed-label",
        "detailed_label": "detailed-label",
        "label_val": "label",
    }
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("#fields"):
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    raise ValueError("Invalid #fields row.")
                return [canonical.get(name, name) for name in parts[1:]]
    raise ValueError("Missing #fields row; not a readable Zeek conn log.")


def _fingerprint_and_raw_rows(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    raw_rows = 0
    with path.open("rb") as handle:
        for binary_line in handle:
            digest.update(binary_line)
            stripped = binary_line.strip()
            if stripped and not stripped.startswith(b"#"):
                raw_rows += 1
    return digest.hexdigest(), raw_rows


def inspect_scenario(
    name: str,
    path_value: str | os.PathLike[str],
    *,
    chunksize: int = 200_000,
    expected_labels: Iterable[str] = FIXED_LABELS,
    max_parse_errors: int = 0,
) -> dict[str, Any]:
    """Return a complete verification record for one scenario."""

    path = Path(path_value)
    record: dict[str, Any] = {
        "scenario": name,
        "path": str(path),
        "exists": path.is_file(),
        "readable": False,
        "file_size_bytes": None,
        "sha256": None,
        "raw_data_rows": 0,
        "parsed_rows": 0,
        "clean_rows": 0,
        "parse_error_rows": 0,
        "label_counts": {},
        "unique_source_ips": 0,
        "unique_destination_ips": 0,
        "missing_expected_labels": list(expected_labels),
        "unexpected_labels": [],
        "warnings": [],
        "serious_parser_error": False,
        "status": "ERROR",
        "error": None,
    }
    if not path.is_file():
        record["error"] = "FILE_NOT_FOUND"
        record["warnings"].append("Required scenario file is missing.")
        return record
    if not os.access(path, os.R_OK):
        record["error"] = "FILE_NOT_READABLE"
        record["warnings"].append("Scenario file is not readable.")
        return record

    record["readable"] = True
    record["file_size_bytes"] = int(path.stat().st_size)
    try:
        sha256, raw_rows = _fingerprint_and_raw_rows(path)
        record["sha256"] = sha256
        record["raw_data_rows"] = raw_rows
        names = _field_names(path)
        parser_bad_lines = 0

        def bad_line(_: list[str]) -> None:
            nonlocal parser_bad_lines
            parser_bad_lines += 1
            return None

        reader = pd.read_csv(
            path,
            sep="\t",
            comment="#",
            header=None,
            names=names,
            na_values=[],
            keep_default_na=False,
            skip_blank_lines=True,
            dtype=str,
            engine="python",
            on_bad_lines=bad_line,
            chunksize=chunksize,
        )
        label_counts: Counter[str] = Counter()
        source_ips: set[str] = set()
        destination_ips: set[str] = set()
        missing_columns: set[str] = set()
        parsed_rows = 0
        clean_rows = 0
        for raw_chunk in reader:
            parsed_rows += len(raw_chunk)
            parsed = split_label_column(raw_chunk)
            missing_columns.update(
                column
                for column in REQUIRED_PARSED_COLUMNS
                if column not in parsed.columns
            )
            if missing_columns:
                continue
            clean = clean_flows(parsed)
            clean_rows += len(clean)
            label_counts.update(
                clean["detailed-label"].astype(str).tolist()
            )
            source_ips.update(
                clean["id.orig_h"].dropna().astype(str).tolist()
            )
            destination_ips.update(
                clean["id.resp_h"].dropna().astype(str).tolist()
            )

        inferred_skips = max(raw_rows - parsed_rows, 0)
        parse_errors = max(parser_bad_lines, inferred_skips)
        record.update(
            {
                "parsed_rows": int(parsed_rows),
                "clean_rows": int(clean_rows),
                "parse_error_rows": int(parse_errors),
                "label_counts": dict(sorted(label_counts.items())),
                "unique_source_ips": len(source_ips),
                "unique_destination_ips": len(destination_ips),
            }
        )
        observed = set(label_counts)
        missing_expected = sorted(set(expected_labels) - observed)
        unexpected_labels = sorted(observed - set(expected_labels))
        record["missing_expected_labels"] = missing_expected
        record["unexpected_labels"] = unexpected_labels
        if raw_rows == 0:
            record["warnings"].append("Scenario contains no data rows.")
        if missing_expected:
            record["warnings"].append(
                "Expected fixed-taxonomy labels absent in this scenario: "
                + ", ".join(missing_expected)
            )
        if unexpected_labels:
            record["warnings"].append(
                "Labels outside the fixed taxonomy: "
                + ", ".join(unexpected_labels)
            )
        if parse_errors:
            record["warnings"].append(
                f"Parser skipped or failed to retain {parse_errors} data rows."
            )
        if missing_columns:
            record["warnings"].append(
                "Missing required parsed columns: "
                + ", ".join(sorted(missing_columns))
            )

        serious = bool(
            missing_columns
            or unexpected_labels
            or (raw_rows > 0 and parsed_rows == 0)
            or parse_errors > max_parse_errors
            or parsed_rows > raw_rows
        )
        record["serious_parser_error"] = serious
        record["status"] = "ERROR" if serious else (
            "WARNING" if record["warnings"] else "OK"
        )
        if serious:
            record["error"] = "SERIOUS_PARSER_ERROR"
    except Exception as exc:  # preserve the manifest even on parser failure
        record["serious_parser_error"] = True
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["warnings"].append("Parser raised an exception.")
    return record


def build_manifest(
    scenario_paths: Mapping[str, str],
    *,
    chunksize: int = 200_000,
    max_parse_errors: int = 0,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for name in REQUIRED_SCENARIOS:
        path = scenario_paths.get(name)
        if path is None:
            records.append(
                {
                    "scenario": name,
                    "path": None,
                    "exists": False,
                    "readable": False,
                    "file_size_bytes": None,
                    "sha256": None,
                    "raw_data_rows": 0,
                    "parsed_rows": 0,
                    "clean_rows": 0,
                    "parse_error_rows": 0,
                    "label_counts": {},
                    "unique_source_ips": 0,
                    "unique_destination_ips": 0,
                    "missing_expected_labels": list(FIXED_LABELS),
                    "unexpected_labels": [],
                    "warnings": [
                        "Required scenario was not supplied on CLI/config."
                    ],
                    "serious_parser_error": False,
                    "status": "ERROR",
                    "error": "SCENARIO_NOT_SUPPLIED",
                }
            )
            continue
        records.append(
            inspect_scenario(
                name,
                path,
                chunksize=chunksize,
                max_parse_errors=max_parse_errors,
            )
        )
    supplied_extra = sorted(set(scenario_paths) - set(REQUIRED_SCENARIOS))
    complete = all(
        record["exists"]
        and record["readable"]
        and not record["serious_parser_error"]
        and int(record["parsed_rows"]) > 0
        for record in records
    )
    return {
        "manifest_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "required_scenarios": list(REQUIRED_SCENARIOS),
        "supplied_extra_scenarios": supplied_extra,
        "max_parse_errors": max_parse_errors,
        "complete": complete,
        "scenarios": records,
    }


def write_manifest(manifest: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    rows = []
    for record in manifest["scenarios"]:
        row = dict(record)
        row["label_counts"] = json.dumps(
            row["label_counts"], ensure_ascii=False, sort_keys=True
        )
        row["missing_expected_labels"] = "|".join(
            row["missing_expected_labels"]
        )
        row["unexpected_labels"] = "|".join(row["unexpected_labels"])
        row["warnings"] = " | ".join(row["warnings"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        output_dir / "dataset_manifest.csv", index=False
    )


def print_summary(manifest: Mapping[str, Any], output_dir: Path) -> None:
    print("Phase 1 dataset verification")
    print(f"Output: {output_dir.resolve()}")
    for record in manifest["scenarios"]:
        size = record["file_size_bytes"]
        size_text = "N/A" if size is None else f"{size / (1024**2):.2f} MiB"
        print(
            f"- {record['scenario']}: {record['status']} | "
            f"size={size_text} | raw={record['raw_data_rows']} | "
            f"parsed={record['parsed_rows']} | "
            f"parse_errors={record['parse_error_rows']} | "
            f"labels={len(record['label_counts'])}"
        )
        if record["error"]:
            print(f"  error: {record['error']}")
        for warning in record["warnings"]:
            print(f"  warning: {warning}")
    print(
        "Dataset readiness: "
        + ("READY" if manifest["complete"] else "NOT_READY")
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify all raw scenarios required by clean Phase 1."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        help="Scenario mappings in the same name=PATH format as src.phase1_clean.",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/phase1_clean",
    )
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument(
        "--max-parse-errors",
        type=int,
        default=0,
        help="Allowed skipped/bad rows before verification fails.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.chunksize < 1 or args.max_parse_errors < 0:
        raise SystemExit("--chunksize must be positive and --max-parse-errors nonnegative.")
    try:
        paths = (
            parse_scenario_arguments(args.scenarios)
            if args.scenarios
            else scenarios_from_config(Path(args.config))
        )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    manifest = build_manifest(
        paths,
        chunksize=args.chunksize,
        max_parse_errors=args.max_parse_errors,
    )
    output_dir = Path(args.out_dir)
    write_manifest(manifest, output_dir)
    print_summary(manifest, output_dir)
    return 0 if manifest["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
