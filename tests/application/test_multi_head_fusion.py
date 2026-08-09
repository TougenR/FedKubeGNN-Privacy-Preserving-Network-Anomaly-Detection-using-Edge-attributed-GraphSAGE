from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from src.application.inference.fusion import (
    FusionPolicyError,
    MultiHeadFusionPolicy,
    load_fusion_policy,
    policy_digest,
)
from src.application.inference.runtime import RoutedPrediction


class MultiHeadFusionTests(unittest.TestCase):
    def test_logistic_stacking_uses_both_heads_and_returns_probabilities(self) -> None:
        def prediction(client_id: str, values: list[list[float]]) -> RoutedPrediction:
            probabilities = torch.tensor(values, dtype=torch.float32)
            confidence, indices = probabilities.max(dim=-1)
            return RoutedPrediction(
                client_id=client_id,
                probabilities=probabilities,
                predicted_indices=indices,
                confidence=confidence,
                entropy=torch.zeros(len(values)),
            )

        policy = MultiHeadFusionPolicy(
            policy_id="test",
            policy_digest="a" * 64,
            heads=("1", "2"),
            classes=("Benign", "Attack"),
            method="multinomial-logistic-stacking",
            class_head_weights=None,
            feature_transform="probability",
            feature_mean=torch.zeros(4, dtype=torch.float64),
            feature_scale=torch.ones(4, dtype=torch.float64),
            coefficients=torch.tensor(
                [[1.0, -1.0, 1.0, -1.0], [-1.0, 1.0, -1.0, 1.0]],
                dtype=torch.float64,
            ),
            intercept=torch.zeros(2, dtype=torch.float64),
            class_alert_thresholds={"Attack": 0.9},
            provenance={"validation_report_sha256": "b" * 64},
        )
        benign = policy.fuse(
            {
                "1": prediction("1", [[0.9, 0.1]]),
                "2": prediction("2", [[0.8, 0.2]]),
            }
        )
        attack = policy.fuse(
            {
                "1": prediction("1", [[0.2, 0.8]]),
                "2": prediction("2", [[0.1, 0.9]]),
            }
        )
        self.assertEqual(int(benign.predicted_indices[0]), 0)
        self.assertEqual(int(attack.predicted_indices[0]), 1)
        torch.testing.assert_close(
            benign.probabilities.sum(dim=-1), torch.ones(1)
        )

    def test_loader_rejects_policy_bound_to_another_model(self) -> None:
        manifest = {
            "bundle_id": "bundle",
            "model_digest": "a" * 64,
            "dataset_digest": "b" * 64,
            "graph_protocol": "rolling-window-v1",
            "head_digests": {"1": "c" * 64, "2": "d" * 64},
        }
        bundle = SimpleNamespace(
            manifest=manifest,
            heads={"1": object(), "2": object()},
            class_to_idx={"Benign": 0, "Attack": 1},
        )
        document = {
            "schema_version": 1,
            "kind": "validation-selected-multi-head-probability-fusion",
            "policy_id": "test",
            "selection_split": "validation",
            **manifest,
            "model_digest": "e" * 64,
            "heads": ["1", "2"],
            "classes": ["Benign", "Attack"],
            "method": "class-f1-weighted-probability",
            "class_head_weights": {
                "Benign": {"1": 0.5, "2": 0.5},
                "Attack": {"1": 0.5, "2": 0.5},
            },
            "class_alert_thresholds": {"Attack": 0.9},
            "provenance": {"validation_report_sha256": "f" * 64},
        }
        document["policy_digest"] = policy_digest(document)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(FusionPolicyError, "model_digest"):
                load_fusion_policy(path, bundle)


if __name__ == "__main__":
    unittest.main()
