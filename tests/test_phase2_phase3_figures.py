from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from generate_phase2_phase3_figures import discover_phase2_runs, generate_all


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class ReportFigureTests(unittest.TestCase):
    def _phase2_run(self, root: Path, strategy: str, seed: int) -> Path:
        run = root / f"{strategy}-{seed}"
        _write_json(
            run / "run.json",
            {"run_id": run.name, "status": "completed", "strategy": strategy},
        )
        _write_json(run / "config.snapshot.json", {"training": {"seed": seed}})
        _write_json(
            run / "metrics" / "summary.json",
            {
                "run_id": run.name,
                "strategy": strategy,
                "best_round": 2,
                "validation_macro_f1": 0.75,
                "test_metrics": {
                    "accuracy": 0.8,
                    "macro_f1": 0.7,
                    "weighted_f1": 0.78,
                    "per_class": {
                        "Benign": {"f1": 0.9},
                        "Attack": {"f1": 0.5},
                    },
                },
                "test_confusion_matrix": [[8, 2], [1, 9]],
                "total_upload_bytes": 2**20,
                "total_download_bytes": 2**20,
            },
        )
        rounds = run / "metrics" / "rounds.csv"
        rounds.parent.mkdir(parents=True, exist_ok=True)
        with rounds.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["round", "macro_f1"])
            writer.writeheader()
            writer.writerows(
                [
                    {"round": 1, "macro_f1": 0.6},
                    {"round": 2, "macro_f1": 0.75},
                ]
            )
        return run

    def _phase3(self, root: Path) -> None:
        with (root / "latency.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "endpoint",
                    "batch_size",
                    "mean_ms",
                    "p95_ms",
                    "flows_per_second",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "endpoint": "local",
                    "batch_size": 10,
                    "mean_ms": 20,
                    "p95_ms": 25,
                    "flows_per_second": 500,
                }
            )
        _write_json(
            root / "equivalence_local.json",
            {
                "compared_predictions": 10,
                "flow_id_agreement": 1.0,
                "label_agreement": 1.0,
                "max_probability_delta": 0.0,
                "max_confidence_delta": 0.0,
                "max_entropy_delta": 0.0,
            },
        )
        _write_json(root / "contract_tests.json", {"status": "PASS"})
        _write_json(root / "minikube_smoke.json", {"status": "BLOCKED"})

    def test_discovers_completed_phase2_runs(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._phase2_run(root, "fedavg", 42)
            runs = discover_phase2_runs(root)
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0].num_classes, 2)
            self.assertEqual(runs[0].rounds[-1]["macro_f1"], 0.75)

    def test_generates_observed_figures_and_manifest(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            phase2 = root / "phase2"
            phase3 = root / "phase3"
            output = root / "figures"
            phase3.mkdir()
            self._phase2_run(phase2, "fedavg", 42)
            self._phase2_run(phase2, "fedprox", 42)
            self._phase3(phase3)

            manifest = generate_all(
                phase2_root=phase2,
                phase3_root=phase3,
                output_root=output,
                formats=("png",),
            )

            names = {figure["name"] for figure in manifest["figures"]}
            self.assertIn("fig_p2_validation_macro_f1_by_round_k2", names)
            self.assertIn("fig_p2_test_metrics_k2", names)
            self.assertIn("fig_p2_per_class_f1_k2", names)
            self.assertIn("fig_p2_communication_cost_k2", names)
            self.assertIn("fig_p3_latency_and_throughput", names)
            self.assertIn("fig_p3_offline_fastapi_equivalence", names)
            self.assertIn("fig_p3_validation_coverage", names)
            self.assertTrue((output / "figure_manifest.json").is_file())
            for figure in manifest["figures"]:
                for path in figure["paths"]:
                    self.assertGreater(Path(path).stat().st_size, 1_000)

    def test_missing_phase2_results_are_explicitly_skipped(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            phase3 = root / "phase3"
            phase3.mkdir()
            self._phase3(phase3)
            manifest = generate_all(
                phase2_root=root / "missing-phase2",
                phase3_root=phase3,
                output_root=root / "figures",
                formats=("png",),
            )
            self.assertTrue(any(item["phase"] == 2 for item in manifest["skipped"]))
            self.assertFalse(any(figure["phase"] == 2 for figure in manifest["figures"]))


if __name__ == "__main__":
    unittest.main()
