from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.core.legacy_bundle import ModelContractError
from src.phase1_clean import (
    CLEAN_IMBALANCE_MODE,
    FIXED_LABELS,
    ROW_ID_COLUMN,
    CleanProtocolError,
    _canonical_digest,
    fixed_class_to_idx,
    make_toy_clean_frames,
    make_transductive_split_plans,
    prepare_loso_clean,
    prepare_transductive_clean,
    resolve_clean_imbalance_mode,
    train_prepared_clean,
    validate_clean_bundle,
    with_stable_row_ids,
    write_clean_bundle,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _load_test_config() -> dict:
    with (REPOSITORY_ROOT / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config = copy.deepcopy(config)
    config["training"]["epochs"] = 1
    config["training"]["early_stop_patience"] = 1
    return config


def _tagged_frames(frames):
    return {
        scenario: with_stable_row_ids(frame, scenario)
        for scenario, frame in frames.items()
    }


def _plans(frames, config, *, seed=42):
    return make_transductive_split_plans(
        _tagged_frames(frames),
        fixed_class_to_idx(config),
        seed=seed,
        protocol="pooled",
        train_ratio=float(config["training"]["train_ratio"]),
        val_ratio=float(config["training"]["val_ratio"]),
        test_ratio=float(config["training"]["test_ratio"]),
    )


def _positions_for_ids(frame, scenario, row_ids):
    tagged = with_stable_row_ids(frame, scenario)
    wanted = set(row_ids)
    return tagged.index[tagged[ROW_ID_COLUMN].isin(wanted)].tolist()


class Phase1CleanProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config = _load_test_config()
        self.frames = make_toy_clean_frames()

    def test_01_pooled_numeric_sentinels_do_not_change_fitted_scaler(self):
        plans = _plans(self.frames, self.config)
        baseline = prepare_transductive_clean(
            self.frames,
            self.config,
            seed=42,
            split_plans=plans,
        )
        modified = copy.deepcopy(self.frames)
        for scenario, plan in plans.items():
            held_ids = tuple(plan.val_ids) + tuple(plan.test_ids)
            positions = _positions_for_ids(
                modified[scenario], scenario, held_ids
            )
            modified[scenario].loc[positions, "duration"] = 1e15
            modified[scenario].loc[positions, "orig_bytes"] = 1e18
        changed = prepare_transductive_clean(
            modified,
            self.config,
            seed=42,
            split_plans=plans,
        )
        np.testing.assert_allclose(
            baseline.preprocessor.scaler.mean_,
            changed.preprocessor.scaler.mean_,
        )
        np.testing.assert_allclose(
            baseline.preprocessor.scaler.scale_,
            changed.preprocessor.scaler.scale_,
        )

    def test_02_unseen_validation_and_test_categories_do_not_enter_vocab(self):
        plans = _plans(self.frames, self.config)
        modified = copy.deepcopy(self.frames)
        for scenario, plan in plans.items():
            held_ids = tuple(plan.val_ids) + tuple(plan.test_ids)
            positions = _positions_for_ids(
                modified[scenario], scenario, held_ids
            )
            modified[scenario].loc[positions, "proto"] = "heldout-proto"
            modified[scenario].loc[positions, "service"] = "heldout-service"
            modified[scenario].loc[positions, "conn_state"] = "HELD"
        first = prepare_transductive_clean(
            modified,
            self.config,
            seed=42,
            split_plans=plans,
        )
        second = prepare_transductive_clean(
            modified,
            self.config,
            seed=42,
            split_plans=plans,
        )
        self.assertNotIn(
            "heldout-proto", first.preprocessor.proto_categories
        )
        self.assertNotIn(
            "heldout-service", first.preprocessor.service_categories
        )
        self.assertNotIn(
            "HELD", first.preprocessor.conn_state_categories
        )
        self.assertEqual(
            first.preprocessor.feature_columns,
            second.preprocessor.feature_columns,
        )
        for scenario in first.graphs:
            self.assertTrue(
                torch.equal(
                    first.graphs[scenario].edge_attr,
                    second.graphs[scenario].edge_attr,
                )
            )

    def test_03_class_weights_depend_only_on_training_membership(self):
        plans = _plans(self.frames, self.config)
        baseline = prepare_transductive_clean(
            self.frames,
            self.config,
            seed=42,
            split_plans=plans,
        )
        modified = copy.deepcopy(self.frames)
        for scenario, plan in plans.items():
            held_ids = tuple(plan.val_ids) + tuple(plan.test_ids)
            positions = _positions_for_ids(
                modified[scenario], scenario, held_ids
            )
            modified[scenario].loc[
                positions, "detailed-label"
            ] = "Okiru-Attack"
        changed = prepare_transductive_clean(
            modified,
            self.config,
            seed=42,
            split_plans=plans,
        )
        self.assertTrue(
            torch.equal(baseline.class_weights, changed.class_weights)
        )

    def test_04_undersampling_changes_only_training_membership(self):
        imbalanced = copy.deepcopy(self.frames)
        for scenario, frame in imbalanced.items():
            frame["detailed-label"] = "Benign"
            frame.loc[frame.index[-12:], "detailed-label"] = "Attack"
        plans = _plans(imbalanced, self.config)
        prepared = prepare_transductive_clean(
            imbalanced,
            self.config,
            seed=42,
            split_plans=plans,
            imbalance_mode="undersample",
        )
        dropped_train_rows = 0
        for scenario, original in plans.items():
            updated = prepared.split_plans[scenario]
            self.assertEqual(original.val_ids, updated.val_ids)
            self.assertEqual(original.test_ids, updated.test_ids)
            self.assertTrue(set(updated.train_ids).issubset(original.train_ids))
            dropped_train_rows += len(original.train_ids) - len(
                updated.train_ids
            )
        self.assertGreater(dropped_train_rows, 0)

    def test_05_loso_held_out_values_cannot_change_fit_weights_or_mapping(self):
        baseline = prepare_loso_clean(
            self.frames,
            self.config,
            held_out="toy-c",
            seed=42,
        )
        modified = copy.deepcopy(self.frames)
        held = modified["toy-c"]
        held["duration"] = 1e15
        held["orig_bytes"] = 1e18
        held["proto"] = "heldout-proto"
        held["service"] = "heldout-service"
        held["conn_state"] = "HELD"
        held["detailed-label"] = "Okiru-Attack"
        changed = prepare_loso_clean(
            modified,
            self.config,
            held_out="toy-c",
            seed=42,
        )
        np.testing.assert_allclose(
            baseline.preprocessor.scaler.mean_,
            changed.preprocessor.scaler.mean_,
        )
        self.assertEqual(
            baseline.preprocessor.feature_columns,
            changed.preprocessor.feature_columns,
        )
        self.assertTrue(
            torch.equal(baseline.class_weights, changed.class_weights)
        )
        self.assertEqual(
            baseline.class_to_idx,
            {label: index for index, label in enumerate(FIXED_LABELS)},
        )
        self.assertEqual(baseline.class_to_idx, changed.class_to_idx)

    def test_06_validation_rows_never_enter_training_loss(self):
        prepared = prepare_loso_clean(
            self.frames,
            self.config,
            held_out="toy-c",
            seed=42,
        )
        observed: dict[str, set[str]] = {}

        def observe(scenario, row_ids):
            observed.setdefault(scenario, set()).update(row_ids)

        result = train_prepared_clean(
            prepared,
            self.config,
            seed=42,
            epochs_override=1,
            loss_observer=observe,
        )
        for scenario, used_ids in observed.items():
            plan = prepared.split_plans[scenario]
            self.assertEqual(used_ids, set(plan.train_ids))
            self.assertTrue(used_ids.isdisjoint(plan.val_ids))
        self.assertEqual(result.best_epoch, 1)
        self.assertTrue(math.isfinite(result.validation_metric))
        self.assertEqual(result.final_evaluation_calls, 1)

    def test_07_imbalance_mode_is_preselected_and_not_metric_selected(self):
        baseline = resolve_clean_imbalance_mode(self.config)
        with_fake_results = copy.deepcopy(self.config)
        with_fake_results["historical_test_metrics"] = {
            "none": 1.0,
            "undersample": 0.99,
            "class_weight": 0.01,
        }
        self.assertEqual(baseline, CLEAN_IMBALANCE_MODE)
        self.assertEqual(
            resolve_clean_imbalance_mode(with_fake_results),
            CLEAN_IMBALANCE_MODE,
        )
        invalid = copy.deepcopy(self.config)
        invalid["phase1_clean"]["imbalance_mode"] = "none"
        with self.assertRaises(CleanProtocolError):
            resolve_clean_imbalance_mode(invalid)

    def test_08_bundle_is_exact_and_rejects_schema_or_label_drift(self):
        prepared = prepare_transductive_clean(
            self.frames,
            self.config,
            seed=42,
        )
        result = train_prepared_clean(
            prepared,
            self.config,
            seed=42,
            epochs_override=1,
        )
        with tempfile.TemporaryDirectory(prefix="phase1-clean-test-") as temp:
            bundle = write_clean_bundle(
                Path(temp) / "bundle",
                prepared,
                result,
                self.config,
                seed=42,
                repository_root=REPOSITORY_ROOT,
            )
            expected = {
                "model.pt",
                "preprocessor.pkl",
                "schema.json",
                "labels.json",
                "metadata.json",
                "metrics.json",
                "split_manifest.json",
                "predictions.csv",
            }
            self.assertEqual(
                {path.name for path in bundle.iterdir()},
                expected,
            )
            validate_clean_bundle(bundle)
            predictions = pd.read_csv(bundle / "predictions.csv")
            self.assertEqual(set(predictions["split"]), {"test"})
            self.assertEqual(set(predictions["protocol"]), {"pooled"})
            self.assertEqual(set(predictions["seed"]), {42})
            self.assertIn("true_label", predictions)
            self.assertIn("predicted_label", predictions)
            self.assertIn("true_class_train_support", predictions)
            for label in FIXED_LABELS:
                self.assertIn(f"probability::{label}", predictions)
                self.assertIn(f"logit::{label}", predictions)

            schema_path = bundle / "schema.json"
            original_schema = json.loads(
                schema_path.read_text(encoding="utf-8")
            )
            wrong_schema = copy.deepcopy(original_schema)
            wrong_schema["feature_columns"] = list(
                reversed(wrong_schema["feature_columns"])
            )
            wrong_schema["digest"] = _canonical_digest(
                {
                    "bundle_schema_version": wrong_schema[
                        "bundle_schema_version"
                    ],
                    "feature_count": wrong_schema["feature_count"],
                    "feature_columns": wrong_schema["feature_columns"],
                }
            )
            schema_path.write_text(
                json.dumps(wrong_schema), encoding="utf-8"
            )
            with self.assertRaises(ModelContractError):
                validate_clean_bundle(bundle)
            schema_path.write_text(
                json.dumps(original_schema), encoding="utf-8"
            )

            wrong_count = copy.deepcopy(original_schema)
            wrong_count["feature_count"] += 1
            wrong_count["digest"] = _canonical_digest(
                {
                    "bundle_schema_version": wrong_count[
                        "bundle_schema_version"
                    ],
                    "feature_count": wrong_count["feature_count"],
                    "feature_columns": wrong_count["feature_columns"],
                }
            )
            schema_path.write_text(
                json.dumps(wrong_count), encoding="utf-8"
            )
            with self.assertRaises(ModelContractError):
                validate_clean_bundle(bundle)
            schema_path.write_text(
                json.dumps(original_schema), encoding="utf-8"
            )

            labels_path = bundle / "labels.json"
            wrong_labels = json.loads(
                labels_path.read_text(encoding="utf-8")
            )
            wrong_labels["class_to_idx"]["Attack"] = 1
            wrong_labels["class_to_idx"]["Benign"] = 0
            wrong_labels["digest"] = _canonical_digest(
                {
                    "bundle_schema_version": wrong_labels[
                        "bundle_schema_version"
                    ],
                    "labels": wrong_labels["labels"],
                    "class_to_idx": wrong_labels["class_to_idx"],
                }
            )
            labels_path.write_text(
                json.dumps(wrong_labels), encoding="utf-8"
            )
            with self.assertRaises(ModelContractError):
                validate_clean_bundle(bundle)


if __name__ == "__main__":
    unittest.main()
