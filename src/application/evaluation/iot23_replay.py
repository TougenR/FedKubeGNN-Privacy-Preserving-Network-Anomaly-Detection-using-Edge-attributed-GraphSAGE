"""Reconstruct exact timestamped IoT-23 held-out replay data transactionally."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.federated.data.partitioners.scenario import deterministic_edge_masks
from src.federated.data.sources.iot23 import read_clean_priority_sample


class ReplayPreparationError(RuntimeError):
    """Raised before publishing replay data with unverifiable provenance."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_source_contract(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ReplayPreparationError("Unsupported IoT-23 replay source contract.")
    sources = document.get("sources")
    if not isinstance(sources, list) or len(sources) != 6:
        raise ReplayPreparationError("Replay source contract must contain six clients.")
    ids = [str(source.get("client_id", "")) for source in sources]
    if len(set(ids)) != 6 or any(not client_id for client_id in ids):
        raise ReplayPreparationError("Replay client IDs are missing or duplicated.")
    for source in sources:
        if not str(source.get("url", "")).startswith("https://mcfp.felk.cvut.cz/"):
            raise ReplayPreparationError("Replay sources must use the official HTTPS host.")
        if int(source.get("size", 0)) < 1 or len(str(source.get("sha256", ""))) != 64:
            raise ReplayPreparationError("Replay source size/digest is invalid.")
    return document


def download_verified(
    source: Mapping[str, Any], destination: Path, *, attempts: int = 6
) -> None:
    """Resume an official source download and fail closed on size or SHA-256."""
    expected_size = int(source["size"])
    expected_digest = str(source["sha256"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == expected_size:
        if sha256_file(destination) == expected_digest:
            return
        destination.unlink()
    for attempt in range(1, attempts + 1):
        offset = destination.stat().st_size if destination.exists() else 0
        if offset > expected_size:
            destination.unlink()
            offset = 0
        request = urllib.request.Request(
            str(source["url"]),
            headers={"Range": f"bytes={offset}-", "User-Agent": "FedKube-Phase4/1"},
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                status = int(getattr(response, "status", 200))
                if offset and status != 206:
                    destination.unlink(missing_ok=True)
                    raise ReplayPreparationError(
                        "Official source did not honor a resumable range request."
                    )
                mode = "ab" if offset else "wb"
                with destination.open(mode) as output:
                    shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
        except (OSError, urllib.error.URLError, ReplayPreparationError) as exc:
            if attempt == attempts:
                raise ReplayPreparationError(
                    f"Download failed after {attempts} attempts: {source['client_id']}"
                ) from exc
            time.sleep(min(30, attempt * 5))
            continue
        if destination.stat().st_size == expected_size:
            break
    actual_size = destination.stat().st_size if destination.exists() else -1
    if actual_size != expected_size:
        raise ReplayPreparationError(
            f"Source size mismatch for {source['client_id']}: "
            f"expected={expected_size}, actual={actual_size}."
        )
    actual_digest = sha256_file(destination)
    if actual_digest != expected_digest:
        raise ReplayPreparationError(
            f"Source digest mismatch for {source['client_id']}: "
            f"expected={expected_digest}, actual={actual_digest}."
        )


def split_sampled_frame(
    frame: Any, *, contract: Mapping[str, Any]
) -> dict[str, Any]:
    labels = frame["detailed-label"].astype(str).to_numpy()
    ratios = contract["split"]
    masks = deterministic_edge_masks(
        labels,
        train_ratio=float(ratios["train"]),
        validation_ratio=float(ratios["validation"]),
        test_ratio=float(ratios["test"]),
        seed=int(contract["seed"]),
    )
    split_names = np.full(len(frame), "test", dtype=object)
    split_names[masks[0]] = "train"
    split_names[masks[1]] = "validation"
    retained = labels != str(contract["dropped_class"])
    result = frame.loc[retained].copy()
    result.insert(0, "source_edge_index", np.flatnonzero(retained))
    result.insert(1, "evaluation_split", split_names[retained])
    return {
        split: result.loc[result["evaluation_split"] == split].reset_index(drop=True)
        for split in ("validation", "test")
    }


def _class_counts(frame: Any, classes: list[str]) -> list[int]:
    counts = frame["detailed-label"].astype(str).value_counts()
    return [int(counts.get(label, 0)) for label in classes]


def prepare_replay(
    *, contract_path: str | Path, output_root: str | Path, work_root: str | Path
) -> Path:
    """Publish exact labeled held-out replay files; raw sources never persist."""
    contract = load_source_contract(contract_path)
    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(f"Replay destination already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    workspace = Path(work_root).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    aggregate = {
        split: np.zeros(len(contract["classes"]), dtype=np.int64)
        for split in ("validation", "test")
    }
    clients: dict[str, Any] = {}
    try:
        for index, source in enumerate(contract["sources"]):
            client_id = str(source["client_id"])
            raw = workspace / f"{client_id}.conn.log.labeled"
            try:
                download_verified(source, raw)
                frame = read_clean_priority_sample(
                    raw,
                    cap_per_class=int(contract["cap_per_class"]),
                    chunk_size=int(contract["chunk_size"]),
                    seed=int(contract["seed"]) + index,
                )
                splits = split_sampled_frame(frame, contract=contract)
                client_document: dict[str, Any] = {}
                client_root = temporary / "clients" / client_id
                client_root.mkdir(parents=True)
                for split, split_frame in splits.items():
                    relative = Path("clients") / client_id / f"{split}.jsonl.gz"
                    path = temporary / relative
                    split_frame.to_json(
                        path,
                        orient="records",
                        lines=True,
                        compression="gzip",
                        double_precision=15,
                    )
                    counts = _class_counts(split_frame, list(contract["classes"]))
                    aggregate[split] += np.asarray(counts, dtype=np.int64)
                    client_document[split] = {
                        "path": relative.as_posix(),
                        "rows": int(len(split_frame)),
                        "class_counts": counts,
                        "sha256": sha256_file(path),
                    }
                clients[client_id] = client_document
            finally:
                raw.unlink(missing_ok=True)
        observed = {split: values.astype(int).tolist() for split, values in aggregate.items()}
        if observed != contract["expected_split_class_counts"]:
            raise ReplayPreparationError(
                "Reconstructed split support differs from the immutable derived dataset: "
                f"expected={contract['expected_split_class_counts']}, actual={observed}."
            )
        manifest = {
            "schema_version": 1,
            "kind": "labeled-scientific-evaluation-only",
            "contains_raw_network_identifiers": True,
            "source_contract_digest": _canonical_digest(contract),
            "source_dataset_id": contract["source_dataset_id"],
            "source_dataset_digest": contract["source_dataset_digest"],
            "derived_dataset_id": contract["derived_dataset_id"],
            "derived_dataset_digest": contract["derived_dataset_digest"],
            "classes": contract["classes"],
            "split_class_counts": observed,
            "clients": clients,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output)
        return output
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_replay(
        contract_path=args.contract, output_root=args.output, work_root=args.work
    )
    print(result)


if __name__ == "__main__":
    main()
