from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analyze_phase1_clean import (
    NOT_AVAILABLE,
    PROBABILITY_COLUMNS,
    _entropy_products,
    aggregate_seed_metrics,
    analyze_bundle,
    binary_metrics,
    entropy_analysis,
    fixed_class_macro_f1,
    seen_class_macro_f1,
    write_analysis,
)
from src.phase1_contract import FIXED_LABELS


def _write_report_bundle(
    root: Path,
    *,
    protocol: str,
    seed: int,
    held_out: str | None,
) -> Path:
    name = (
        f"pooled-seed-{seed}"
        if protocol == "pooled"
        else f"loso-held-out-{held_out}-seed-{seed}"
    )
    bundle = root / name
    bundle.mkdir(parents=True)
    support = {
        split: {label: 4 for label in FIXED_LABELS}
        for split in ("train", "validation", "test")
    }
    if protocol == "loso":
        support["train"]["DDoS"] = 0
        support["validation"]["DDoS"] = 0
    metadata = {
        "protocol": protocol,
        "seed": seed,
        "held_out": held_out,
        "class_support": support,
        "best_epoch": 2,
        "validation_metric": 0.6,
    }
    rows = []
    for index, true_label in enumerate(FIXED_LABELS):
        predicted_label = (
            true_label
            if index % 3 != 0
            else FIXED_LABELS[(index + 1) % len(FIXED_LABELS)]
        )
        if true_label == "DDoS" and protocol == "loso":
            probabilities = np.full(len(FIXED_LABELS), 1 / len(FIXED_LABELS))
        else:
            probabilities = np.full(len(FIXED_LABELS), 0.2 / 7)
            probabilities[FIXED_LABELS.index(predicted_label)] = 0.8
        row = {
            "row_id": f"{protocol}-{seed}-{index}",
            "scenario": held_out or "1-1",
            "split": "test",
            "protocol": protocol,
            "seed": seed,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "true_class_train_support": support["train"][true_label],
            "true_class_absent_from_train": (
                support["train"][true_label] == 0
            ),
            "entropy": float(
                -np.sum(
                    probabilities
                    * np.log(np.clip(probabilities, 1e-12, 1.0))
                )
            ),
        }
        row.update(dict(zip(PROBABILITY_COLUMNS, probabilities)))
        rows.append(row)
    metrics = {
        "best_epoch": 2,
        "validation_macro_f1": 0.6,
        "final": {
            "accuracy": 0.5,
            "weighted_f1": 0.5,
            "macro_f1": 0.5,
        },
        "history": [
            {
                "epoch": epoch,
                "train_loss": 1.0 / epoch,
                "validation_macro_f1": 0.2 * epoch,
            }
            for epoch in (1, 2, 3)
        ],
    }
    (bundle / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    (bundle / "metrics.json").write_text(
        json.dumps(metrics), encoding="utf-8"
    )
    (bundle / "split_manifest.json").write_text(
        json.dumps({"seed": seed, "protocol": protocol}),
        encoding="utf-8",
    )
    (bundle / "model.pt").write_bytes(b"fixture")
    pd.DataFrame(rows).to_csv(bundle / "predictions.csv", index=False)
    return bundle


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

    def test_three_seed_aggregation_reports_min_max_and_seed_count(self):
        frame = pd.DataFrame(
            [
                {"held_out": "34-1", "seed": seed, "score": score}
                for seed, score in ((42, 0.1), (1337, 0.2), (2026, 0.3))
            ]
        )
        result = aggregate_seed_metrics(
            frame,
            group_columns=("held_out",),
            metric_columns=("score",),
        )
        self.assertEqual(int(result.iloc[0]["seed_count"]), 3)
        self.assertAlmostEqual(float(result.iloc[0]["score_mean"]), 0.2)
        self.assertAlmostEqual(float(result.iloc[0]["score_min"]), 0.1)
        self.assertAlmostEqual(float(result.iloc[0]["score_max"]), 0.3)

    def test_prediction_audit_entropy_groups_and_fixed_confusion_order(self):
        with tempfile.TemporaryDirectory(prefix="phase1-report-") as temp:
            bundle = _write_report_bundle(
                Path(temp),
                protocol="loso",
                seed=42,
                held_out="34-1",
            )
            run = analyze_bundle(bundle)
            self.assertEqual(run["prediction_status"], "AVAILABLE")
            self.assertEqual(
                run["prediction_audit"]["held_out"],
                "DERIVED_FROM_METADATA",
            )
            self.assertEqual(
                run["prediction_audit"]["class_present_in_train"],
                "DERIVED_FROM_EXPORTED_ABSENCE_FLAG",
            )
            self.assertEqual(run["confusion_matrix"].shape, (8, 8))
            attack_index = FIXED_LABELS.index("Attack")
            benign_index = FIXED_LABELS.index("Benign")
            self.assertEqual(
                int(run["confusion_matrix"][attack_index, benign_index]), 1
            )
            groups = set(run["entropy_summary"]["group"])
            self.assertEqual(
                groups,
                {
                    "known_correct",
                    "known_incorrect",
                    "absent_from_train",
                    "benign",
                    "malicious",
                },
            )
            absent = run["entropy_detection"][
                run["entropy_detection"]["comparison"]
                == "absent_from_train vs known_correct"
            ].iloc[0]
            self.assertEqual(absent["status"], "AVAILABLE")
            self.assertTrue(math.isfinite(float(absent["auroc"])))
            self.assertTrue(math.isfinite(float(absent["auprc"])))

    def test_report_ready_end_to_end_single_seed(self):
        with tempfile.TemporaryDirectory(prefix="phase1-report-") as temp:
            root = Path(temp)
            pooled = _write_report_bundle(
                root,
                protocol="pooled",
                seed=42,
                held_out=None,
            )
            loso = _write_report_bundle(
                root,
                protocol="loso",
                seed=42,
                held_out="34-1",
            )
            output = root / "report_analysis"
            result = write_analysis(
                [analyze_bundle(pooled), analyze_bundle(loso)],
                output,
                [root],
            )
            self.assertTrue(result["single_seed"])
            self.assertEqual(
                result["statistical_uncertainty"], NOT_AVAILABLE
            )
            self.assertGreaterEqual(result["figures_created"], 10)
            for filename in (
                "pooled_metrics_by_seed.csv",
                "loso_metrics_by_seed.csv",
                "loso_aggregate.csv",
                "class_support.csv",
                "per_class_metrics.csv",
                "entropy_summary.csv",
                "entropy_detection_metrics.csv",
                "report.md",
                "FIGURE_SELECTION.md",
            ):
                self.assertTrue((output / filename).is_file(), filename)
            self.assertFalse(
                (output / "figures" / "fig15_seed_stability.png").exists()
            )
            selection = (output / "FIGURE_SELECTION.md").read_text(
                encoding="utf-8"
            )
            self.assertLessEqual(selection.count("figures/fig"), 15)

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
            self.assertTrue((output / "entropy_summary.csv").exists())
            report = (output / "report.md").read_text(encoding="utf-8")
            self.assertIn("NOT_AVAILABLE", report)
            self.assertIn("probability::<fixed-label>", report)


if __name__ == "__main__":
    unittest.main()
