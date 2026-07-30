from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.analyze_phase1_clean import (
    NOT_AVAILABLE,
    PROBABILITY_COLUMNS,
    aggregate_seed_metrics,
    analyze_bundle,
    binary_metrics,
    entropy_analysis,
    fixed_class_macro_f1,
    seen_class_macro_f1,
    write_analysis,
)
from src.phase1_clean import FIXED_LABELS


class Phase1CleanAnalysisTests(unittest.TestCase):
    def test_fixed_class_macro_f1_always_averages_eight_classes(self):
        score = fixed_class_macro_f1(
            ["Benign", "Attack"],
            ["Benign", "Attack"],
        )
        self.assertAlmostEqual(score, 2.0 / 8.0)

    def test_seen_class_macro_f1_excludes_zero_train_support_classes(self):
        support = {label: 0 for label in FIXED_LABELS}
        support.update({"Benign": 10, "Attack": 5})
        score = seen_class_macro_f1(
            ["Benign", "Attack", "Attack"],
            ["Benign", "Benign", "Attack"],
            support,
        )
        self.assertAlmostEqual(float(score), 2.0 / 3.0)

    def test_binary_collapsing_treats_every_non_benign_label_as_malicious(self):
        metrics = binary_metrics(
            ["Benign", "Attack", "DDoS"],
            ["Attack", "Attack", "Benign"],
        )
        self.assertAlmostEqual(metrics["accuracy"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["malicious_precision"], 0.5)
        self.assertAlmostEqual(metrics["malicious_recall"], 0.5)
        self.assertAlmostEqual(metrics["malicious_f1"], 0.5)

    def test_entropy_groups_and_absent_class_auroc(self):
        rows = [
            {
                "true_label": "Benign",
                "predicted_label": "Benign",
                "probabilities": [0.01, 0.93, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
            },
            {
                "true_label": "Attack",
                "predicted_label": "Benign",
                "probabilities": [0.40, 0.54, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
            },
            {
                "true_label": "DDoS",
                "predicted_label": "Benign",
                "probabilities": [0.125] * 8,
            },
        ]
        frame_rows = []
        for row in rows:
            record = {
                "true_label": row["true_label"],
                "predicted_label": row["predicted_label"],
            }
            record.update(dict(zip(PROBABILITY_COLUMNS, row["probabilities"])))
            frame_rows.append(record)
        support = {label: 1 for label in FIXED_LABELS}
        support["DDoS"] = 0
        summary, auc, error = entropy_analysis(
            pd.DataFrame(frame_rows),
            support,
        )
        counts = dict(zip(summary["group"], summary["count"]))
        self.assertEqual(counts["known-correct"], 1)
        self.assertEqual(counts["known-incorrect"], 1)
        self.assertEqual(counts["class-absent-from-train"], 1)
        self.assertAlmostEqual(float(auc), 1.0)
        self.assertIsNone(error)
        absent_mean = float(
            summary.loc[
                summary["group"] == "class-absent-from-train", "mean"
            ].iloc[0]
        )
        self.assertAlmostEqual(absent_mean, math.log(8), places=6)

    def test_multi_seed_aggregation_reports_population_mean_and_std(self):
        frame = pd.DataFrame(
            [
                {
                    "protocol": "pooled",
                    "held_out": "ALL",
                    "seed": 42,
                    "accuracy": 0.6,
                },
                {
                    "protocol": "pooled",
                    "held_out": "ALL",
                    "seed": 1337,
                    "accuracy": 0.8,
                },
            ]
        )
        result = aggregate_seed_metrics(
            frame,
            group_columns=("protocol", "held_out"),
            metric_columns=("accuracy",),
        )
        self.assertEqual(int(result.iloc[0]["seed_count"]), 2)
        self.assertAlmostEqual(float(result.iloc[0]["accuracy_mean"]), 0.7)
        self.assertAlmostEqual(float(result.iloc[0]["accuracy_std"]), 0.1)
        self.assertEqual(
            result.iloc[0]["accuracy_mean_std"],
            "0.700000 ± 0.100000",
        )

    def test_missing_prediction_artifact_is_reported_without_inference(self):
        with tempfile.TemporaryDirectory(prefix="phase1-analysis-") as temp:
            root = Path(temp)
            bundle = root / "pooled-seed-42"
            bundle.mkdir()
            metadata = {
                "protocol": "pooled",
                "seed": 42,
                "held_out": None,
                "class_support": {
                    split: {label: 1 for label in FIXED_LABELS}
                    for split in ("train", "validation", "test")
                },
            }
            metrics = {
                "final": {
                    "accuracy": 0.5,
                    "weighted_f1": 0.4,
                    "macro_f1": 0.3,
                }
            }
            (bundle / "metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            (bundle / "metrics.json").write_text(
                json.dumps(metrics), encoding="utf-8"
            )
            run = analyze_bundle(bundle)
            self.assertEqual(run["prediction_status"], NOT_AVAILABLE)
            self.assertEqual(run["seen_class_macro_f1"], NOT_AVAILABLE)
            self.assertIsNone(run["binary"])
            self.assertTrue(
                any("predictions.csv" in note for note in run["notes"])
            )

            output = root / "analysis"
            result = write_analysis([run], output, [root])
            self.assertFalse(result["entropy_available"])
            self.assertFalse((output / "entropy_summary.csv").exists())
            report = (output / "report.md").read_text(encoding="utf-8")
            self.assertIn("NOT_AVAILABLE", report)
            self.assertIn("probability::<fixed-label>", report)


if __name__ == "__main__":
    unittest.main()
