"""Compare completed run summaries without re-evaluating test data."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable


def compare_runs(run_roots: Iterable[str | Path], output: str | Path) -> Path:
    rows = []
    expected_provenance: dict[str, str] | None = None
    for root_value in run_roots:
        root = Path(root_value)
        status = json.loads((root / "run.json").read_text(encoding="utf-8"))
        if status.get("status") != "completed":
            raise ValueError(f"Run is not completed: {root}")
        provenance = {
            key: str(status.get(key, ""))
            for key in ("config_digest", "dataset_digest", "model_digest")
        }
        missing = [key for key, value in provenance.items() if not value]
        if missing:
            raise ValueError(f"Run {root} is missing provenance fields: {missing}.")
        if expected_provenance is None:
            expected_provenance = provenance
        elif provenance != expected_provenance:
            mismatches = {
                key: (expected_provenance[key], provenance[key])
                for key in expected_provenance
                if provenance[key] != expected_provenance[key]
            }
            raise ValueError(
                f"Run {root} is not comparable with the first run: {mismatches}."
            )
        summary = json.loads(
            (root / "metrics/summary.json").read_text(encoding="utf-8")
        )
        test = summary.get("test_metrics", {})
        rows.append(
            {
                "run_id": status["run_id"],
                "strategy": summary.get("strategy", summary.get("kind", "unknown")),
                "best_round": summary.get("best_round", summary.get("epochs")),
                "validation_macro_f1": summary.get(
                    "validation_macro_f1",
                    summary.get("validation_metrics", {}).get("macro_f1"),
                ),
                "test_macro_f1": test.get("macro_f1"),
                "test_accuracy": test.get("accuracy"),
                "upload_bytes": summary.get("total_upload_bytes", 0),
                "download_bytes": summary.get("total_download_bytes", 0),
                **provenance,
            }
        )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]) if rows else ["run_id"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return destination
