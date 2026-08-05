from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.federated.experiments.visualization import (
    visualize_class_aware_summary,
    visualize_runs,
)


def _completed_run(
    root: Path,
    strategy: str,
    *,
    dataset_digest: str = "dataset",
    runtime: str = "flower",
) -> Path:
    run_root = root / strategy
    metrics = run_root / "metrics"
    metrics.mkdir(parents=True)
    (run_root / "run.json").write_text(
        json.dumps(
            {
                "run_id": f"{strategy}-run",
                "status": "completed",
                "strategy": strategy,
                "config_digest": "config",
                "dataset_digest": dataset_digest,
                "model_digest": "model",
            }
        ),
        encoding="utf-8",
    )
    round_root = metrics if runtime == "flower" else metrics / "rounds"
    round_root.mkdir(exist_ok=True)
    for round_number, macro_f1 in ((1, 0.5), (2, 0.7), (3, 0.6)):
        round_name = (
            f"validation-round-{round_number:04d}.json"
            if runtime == "flower"
            else f"round-{round_number:04d}.json"
        )
        round_metrics = {
            "round": round_number,
            "accuracy": macro_f1 + 0.1,
            "weighted_f1": macro_f1 + 0.05,
        }
        if runtime == "flower":
            round_metrics.update({"loss": 1.0 / round_number, "macro-f1": macro_f1})
        else:
            round_metrics.update(
                {
                    "validation_loss": 1.0 / round_number,
                    "macro_f1": macro_f1,
                }
            )
        (round_root / round_name).write_text(
            json.dumps(
                round_metrics
            ),
            encoding="utf-8",
        )
    if runtime == "flower":
        test_metrics = {
            "loss": 0.25,
            "accuracy": 0.8,
            "macro-f1": 0.75,
            "weighted-f1": 0.78,
            "num-classes": 2,
            "confusion-matrix": [8, 2, 1, 9],
            "f1-class-0": 0.8,
            "f1-class-1": 0.7,
        }
    else:
        test_metrics = {
            "loss": 0.25,
            "accuracy": 0.8,
            "macro_f1": 0.75,
            "weighted_f1": 0.78,
            "per_class": {
                "Benign": {"f1": 0.8},
                "Attack": {"f1": 0.7},
            },
        }
    (metrics / "summary.json").write_text(
        json.dumps(
            {
                "strategy": strategy,
                "class_names": ["Benign", "Attack"],
                "best_round": 2,
                "validation_macro_f1": 0.7,
                "test_metrics": test_metrics,
                "test_confusion_matrix": [[8, 2], [1, 9]],
            }
        ),
        encoding="utf-8",
    )
    return run_root


class VisualizationTests(unittest.TestCase):
    def test_class_aware_summary_renders_without_test_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = root / "summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "validation": {
                            "baseline": {"42": 0.5, "1337": 0.55},
                            "selected": {
                                "42": {"macro_f1": 0.7},
                                "1337": {"macro_f1": 0.75},
                            },
                        },
                        "test": {
                            "42": {
                                "accuracy": 0.8,
                                "weighted_f1": 0.78,
                                "macro_f1": 0.7,
                                "per_class_f1": {"Benign": 0.9, "Attack": 0.5},
                            },
                            "1337": {
                                "accuracy": 0.82,
                                "weighted_f1": 0.79,
                                "macro_f1": 0.72,
                                "per_class_f1": {"Benign": 0.91, "Attack": 0.53},
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            output = root / "visualizations"
            manifest_path = visualize_class_aware_summary(summary, output)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(manifest["test_evaluation_reused"])
            self.assertEqual(len(manifest["figures"]), 6)
            for name in manifest["figures"]:
                self.assertGreater((output / name).stat().st_size, 0)

    def test_completed_strategies_render_phase1_style_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fedavg = _completed_run(root, "fedavg")
            fedprox = _completed_run(root, "fedprox")
            output = root / "visualizations"

            manifest_path = visualize_runs((fedavg, fedprox), output)

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(manifest["test_evaluation_reused"])
            self.assertEqual(
                [run["strategy"] for run in manifest["runs"]],
                ["fedavg", "fedprox"],
            )
            self.assertEqual(len(manifest["figures"]), 8)
            for name in manifest["figures"]:
                self.assertGreater((output / name).stat().st_size, 0)
            with (output / "round_metrics.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                round_rows = list(csv.DictReader(handle))
            self.assertEqual(len(round_rows), 6)
            self.assertEqual(round_rows[1]["macro_f1"], "0.7")
            with (output / "final_metrics.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                final_rows = list(csv.DictReader(handle))
            self.assertEqual(final_rows[0]["test_macro_f1"], "0.75")

    def test_inprocess_metric_layout_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fedavg = _completed_run(root, "fedavg", runtime="inprocess")
            output = root / "visualizations"

            manifest_path = visualize_runs((fedavg,), output)

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["runs"][0]["rounds"], 3)
            self.assertGreater((output / "federated_per_class_f1.png").stat().st_size, 0)

    def test_incompatible_provenance_is_rejected_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fedavg = _completed_run(root, "fedavg", dataset_digest="dataset-a")
            fedprox = _completed_run(root, "fedprox", dataset_digest="dataset-b")
            output = root / "visualizations"

            with self.assertRaisesRegex(ValueError, "not comparable"):
                visualize_runs((fedavg, fedprox), output)

            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
