from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.federated.experiments.comparison import compare_runs


def _completed_run(root: Path, run_id: str, *, dataset_digest: str) -> Path:
    run_root = root / run_id
    (run_root / "metrics").mkdir(parents=True)
    (run_root / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": "completed",
                "config_digest": "config",
                "dataset_digest": dataset_digest,
                "model_digest": "model",
            }
        ),
        encoding="utf-8",
    )
    (run_root / "metrics" / "summary.json").write_text(
        json.dumps(
            {
                "strategy": "fedavg",
                "best_round": 1,
                "validation_macro_f1": 0.8,
                "test_metrics": {"macro_f1": 0.7, "accuracy": 0.75},
            }
        ),
        encoding="utf-8",
    )
    return run_root


class ComparisonTests(unittest.TestCase):
    def test_compatible_provenance_is_written_to_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = _completed_run(root, "first", dataset_digest="dataset")
            second = _completed_run(root, "second", dataset_digest="dataset")
            output = compare_runs((first, second), root / "comparison.csv")
            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["dataset_digest"], "dataset")
            self.assertEqual(rows[0]["config_digest"], "config")

    def test_incompatible_provenance_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = _completed_run(root, "first", dataset_digest="dataset-a")
            second = _completed_run(root, "second", dataset_digest="dataset-b")
            with self.assertRaisesRegex(ValueError, "not comparable"):
                compare_runs((first, second), root / "comparison.csv")
            self.assertFalse((root / "comparison.csv").exists())


if __name__ == "__main__":
    unittest.main()
