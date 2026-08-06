from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from pydantic import ValidationError
from torch_geometric.data import Data

from src.application.api.schema import ProductionInferenceRequest
from src.application.inference.bundle_loader import (
    InferenceBundleError,
    load_inference_bundle,
)
from src.application.inference.router import TrustedRoutingError
from src.application.inference.runtime import CentralizedFedPerRuntime
from src.core.model import EGraphSAGE
from src.federated.contracts.artifacts import ContractBundle
from src.federated.contracts.schema import (
    FeatureSchema,
    GraphSchema,
    LabelSchema,
    ModelSpec,
)
from src.federated.core.state import torch_state_to_arrays
from src.federated.exports.fedper_bundle import (
    EXPECTED_CLIENTS,
    export_fedper_bundle,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FedPerBundleRuntimeTests(unittest.TestCase):
    def _bundle(self, root: Path) -> tuple[Path, dict[str, np.ndarray]]:
        torch.manual_seed(42)
        model = EGraphSAGE(
            edge_dim=2,
            num_classes=2,
            node_in_dim=1,
            hidden_dim=4,
            num_layers=1,
            dropout=0.0,
        )
        state = torch_state_to_arrays(model.state_dict())
        feature_schema = FeatureSchema.from_names(("feature_a", "feature_b"))
        label_schema = LabelSchema(("Benign", "Attack"))
        graph_schema = GraphSchema(
            feature_schema_digest=feature_schema.digest,
            label_schema_digest=label_schema.digest,
        )
        model_spec = ModelSpec.from_state(
            family="egraphsage",
            model_version=1,
            feature_dim=2,
            num_classes=2,
            node_feature_dim=1,
            hyperparameters={"hidden_dim": 4, "num_layers": 1, "dropout": 0.0},
            state=model.state_dict(),
        )
        prepared = root / "prepared"
        ContractBundle(
            feature_schema=feature_schema,
            label_schema=label_schema,
            graph_schema=graph_schema,
            model_spec=model_spec,
            categories={
                "numeric_columns": ("duration",),
                "missing_flags": (),
            },
            learned_arrays={
                "scaler_mean": np.array([0.0], dtype=np.float64),
                "scaler_scale": np.array([1.0], dtype=np.float64),
                "scaler_var": np.array([1.0], dtype=np.float64),
            },
        ).write(prepared / "contract")
        _write_json(
            prepared / "manifest.json",
            {"dataset_id": "synthetic", "graph_protocol": "test-graph-v1"},
        )

        flower_run_id = "77"
        run_root = root / "run"
        shared = {
            name: value for name, value in state.items() if not name.startswith("head.")
        }
        private = {
            name: value for name, value in state.items() if name.startswith("head.")
        }
        (run_root / "checkpoints").mkdir(parents=True)
        np.savez(run_root / "checkpoints" / "best_model.npz", **shared)
        _write_json(
            run_root / "run.json",
            {
                "status": "completed",
                "strategy": "fedper",
                "best_round": 3,
                "run_id": "fedper-test",
                "dataset_digest": "d" * 64,
                "model_digest": model_spec.digest,
            },
        )
        _write_json(
            run_root / "metrics" / "summary.json",
            {"best_round": 3, "flower_run_id": flower_run_id},
        )

        heads_root = root / "heads"
        for index, client_id in enumerate(EXPECTED_CLIENTS):
            client_root = heads_root / client_id
            client_root.mkdir(parents=True)
            client_state = {
                name: value + np.float32(index / 100) for name, value in private.items()
            }
            state_path = client_root / "head-0003.npz"
            np.savez(state_path, **client_state)
            _write_json(
                client_root / "metadata.json",
                {
                    "client_id": client_id,
                    "run_id": flower_run_id,
                    "model_digest": model_spec.digest,
                    "completed_rounds": 3,
                    "ready": True,
                    "cold_start": False,
                    "personalized_prefixes": ["head."],
                    "state_file": state_path.name,
                    "state_sha256": _sha256(state_path),
                },
            )
        destination = root / "bundle"
        export_fedper_bundle(
            run_root=run_root,
            heads_root=heads_root,
            prepared_root=prepared,
            destination=destination,
        )
        return destination, state

    def test_exact_bundle_loads_and_routes_without_flower(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle_path, _ = self._bundle(Path(temporary))
            bundle = load_inference_bundle(bundle_path)
            directory_mode = bundle_path.stat().st_mode & 0o777
            manifest_mode = (bundle_path / "manifest.json").stat().st_mode & 0o777

        self.assertEqual(set(bundle.heads), set(EXPECTED_CLIENTS))
        self.assertEqual(directory_mode, 0o555)
        self.assertEqual(manifest_mode, 0o444)
        self.assertEqual(bundle.router.route("sensor-34-1"), "34-1")
        with self.assertRaises(TrustedRoutingError):
            bundle.router.route("untrusted")

    def test_research_bundle_is_not_live_serving_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle_path, _ = self._bundle(Path(temporary))
            with self.assertRaisesRegex(InferenceBundleError, "research-only"):
                load_inference_bundle(bundle_path, require_serving_ready=True)

    def test_routed_prediction_matches_full_model_forward(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle_path, _ = self._bundle(Path(temporary))
            bundle = load_inference_bundle(bundle_path)
            graph = Data(
                x=torch.ones((2, 1), dtype=torch.float32),
                edge_index=torch.tensor([[0], [1]], dtype=torch.long),
                edge_attr=torch.tensor([[0.25, -0.5]], dtype=torch.float32),
                edge_index_mp=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                edge_attr_mp=torch.tensor(
                    [[0.25, -0.5], [0.25, -0.5]], dtype=torch.float32
                ),
            )
            result = CentralizedFedPerRuntime(bundle).predict_graph(
                sensor_id="sensor-34-1", graph=graph
            )
            with torch.no_grad():
                encoded = bundle.encoder.encode_nodes(graph)
                representation = bundle.encoder.edge_representations(graph, encoded)
                expected = torch.softmax(bundle.heads["34-1"](representation), dim=-1)

        self.assertEqual(result.client_id, "34-1")
        torch.testing.assert_close(result.probabilities, expected)

    def test_corrupt_head_fails_before_deserialization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle_path, _ = self._bundle(Path(temporary))
            head = bundle_path / "heads" / "1-1.npz"
            head.chmod(0o644)
            head.write_bytes(head.read_bytes() + b"corrupt")
            with self.assertRaisesRegex(InferenceBundleError, "Digest mismatch"):
                load_inference_bundle(bundle_path)

    def test_projected_secret_symlinks_preserve_logical_head_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_path, _ = self._bundle(root)
            projection = root / "projection"
            version = projection / "..2026_08_06_00_00_00"
            projection.mkdir()
            shutil.copytree(bundle_path, version)
            for path in sorted(version.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(version)
                link = projection / relative
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(path.resolve())
            (projection / "..data").symlink_to(version.name)

            bundle = load_inference_bundle(projection)

        self.assertEqual(set(bundle.heads), set(EXPECTED_CLIENTS))

    def test_production_schema_rejects_ground_truth_label(self) -> None:
        with self.assertRaises(ValidationError):
            ProductionInferenceRequest.model_validate(
                {
                    "sensor_id": "sensor-34-1",
                    "window_id": "window-1",
                    "flows": [
                        {
                            "ts": 1.0,
                            "id.orig_h": "10.0.0.1",
                            "id.orig_p": 12345,
                            "id.resp_h": "10.0.0.2",
                            "id.resp_p": 80,
                            "proto": "tcp",
                            "conn_state": "S0",
                            "detailed-label": "C&C",
                        }
                    ],
                }
            )


def build_test_bundle(root: Path) -> tuple[Path, dict[str, np.ndarray]]:
    """Create the synthetic bundle fixture without relying on local artifacts."""
    return FedPerBundleRuntimeTests()._bundle(root)


if __name__ == "__main__":
    unittest.main()
