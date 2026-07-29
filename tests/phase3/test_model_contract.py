from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

from phase3_monitoring.inference_service.model_loader import (
    ModelContractError,
    load_runtime_bundle,
    validate_model_contract,
)
from src.model import EGraphSAGE
from src.preprocess import Preprocessor
from src.train import save_checkpoint


def _checkpoint(feature_columns=None):
    columns = ["feature_a", "feature_b"] if feature_columns is None else feature_columns
    return {
        "state_dict": {},
        "cfg": {"model": {"hidden_dim": 4, "num_layers": 1, "dropout": 0.0}},
        "feature_dim": len(columns),
        "feature_columns": columns,
        "num_classes": 2,
        "class_to_idx": {"Benign": 0, "C&C": 1},
    }


class ModelContractTests(unittest.TestCase):
    def test_compatible_artifacts_load_end_to_end(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint_path = root / "model.pt"
            preprocessor_path = root / "preprocessor.pkl"
            preprocessor = Preprocessor(
                resp_port_categories=[],
                proto_categories=[],
                service_categories=[],
                conn_state_categories=[],
                history_flag_chars=[],
                numeric_columns=[],
                missing_flag_columns=[],
                scaler=None,
                feature_columns=["feature_a", "feature_b"],
            )
            preprocessor.save(str(preprocessor_path))
            model = EGraphSAGE(
                edge_dim=2,
                num_classes=2,
                node_in_dim=1,
                hidden_dim=4,
                num_layers=1,
                dropout=0.0,
            )
            save_checkpoint(
                model,
                str(checkpoint_path),
                class_to_idx={"Benign": 0, "C&C": 1},
                cfg={
                    "model": {
                        "hidden_dim": 4,
                        "num_layers": 1,
                        "dropout": 0.0,
                    }
                },
                feature_dim=2,
                num_classes=2,
                imbalance_mode="none",
                val_macro_f1=0.5,
                history_meta={},
                feature_columns=["feature_a", "feature_b"],
            )
            runtime = load_runtime_bundle(
                device="cpu",
                checkpoint_path=checkpoint_path,
                preprocessor_path=preprocessor_path,
            )

        self.assertEqual(runtime.feature_columns, ("feature_a", "feature_b"))
        self.assertEqual(runtime.class_to_idx, {"Benign": 0, "C&C": 1})
        self.assertEqual(len(runtime.feature_schema_digest), 64)
        self.assertFalse(runtime.model.training)

    def test_phase1_checkpoint_export_includes_feature_schema(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.pt"
            model = torch.nn.Linear(2, 2)
            save_checkpoint(
                model,
                str(path),
                class_to_idx={"Benign": 0, "C&C": 1},
                cfg={},
                feature_dim=2,
                num_classes=2,
                imbalance_mode="none",
                val_macro_f1=0.5,
                history_meta={},
                feature_columns=["feature_a", "feature_b"],
            )
            checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
        self.assertEqual(
            checkpoint["feature_columns"],
            ["feature_a", "feature_b"],
        )

    def test_matching_feature_and_label_contract_is_accepted(self):
        preprocessor = SimpleNamespace(
            feature_columns=["feature_a", "feature_b"]
        )
        columns, mapping, digest = validate_model_contract(
            _checkpoint(),
            preprocessor,
        )
        self.assertEqual(columns, ("feature_a", "feature_b"))
        self.assertEqual(mapping, {"Benign": 0, "C&C": 1})
        self.assertEqual(len(digest), 64)

    def test_position_blind_feature_padding_is_rejected(self):
        preprocessor = SimpleNamespace(feature_columns=["feature_a"])
        with self.assertRaisesRegex(
            ModelContractError,
            "Position-blind padding is not permitted",
        ):
            validate_model_contract(_checkpoint(), preprocessor)

    def test_legacy_checkpoint_without_feature_schema_is_rejected(self):
        checkpoint = _checkpoint()
        checkpoint.pop("feature_columns")
        preprocessor = SimpleNamespace(
            feature_columns=["feature_a", "feature_b"]
        )
        with self.assertRaisesRegex(
            ModelContractError,
            "dimension-only legacy checkpoints are rejected",
        ):
            validate_model_contract(checkpoint, preprocessor)

    def test_non_contiguous_label_indices_are_rejected(self):
        checkpoint = _checkpoint()
        checkpoint["class_to_idx"] = {"Benign": 0, "C&C": 2}
        preprocessor = SimpleNamespace(
            feature_columns=["feature_a", "feature_b"]
        )
        with self.assertRaisesRegex(ModelContractError, "contiguous"):
            validate_model_contract(checkpoint, preprocessor)


if __name__ == "__main__":
    unittest.main()
