from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.federated.contracts.artifacts import ContractBundle
from src.federated.contracts.schema import (
    ContractError,
    FeatureSchema,
    GraphSchema,
    LabelSchema,
    ModelSpec,
)
from src.federated.data.manifest import MANIFEST_VERSION, PreparedDatasetManifest
from src.federated.data.storage import (
    GraphArrays,
    checksum_index_digest,
    sha256_file,
    write_graph_arrays,
)


def _graph(*, feature_dim: int = 2) -> GraphArrays:
    return GraphArrays(
        edge_index=np.array([[0, 1, 2, 0], [1, 2, 0, 2]], dtype=np.int64),
        edge_attr=np.arange(4 * feature_dim, dtype=np.float32).reshape(
            4, feature_dim
        ),
        edge_label=np.array([0, 1, 0, 1], dtype=np.int64),
        train_mask=np.array([True, True, False, False]),
        val_mask=np.array([False, False, True, False]),
        test_mask=np.array([False, False, False, True]),
        num_nodes=3,
    )


def _prepared_fixture(root: Path, *, client_feature_dim: int = 2) -> Path:
    features = FeatureSchema.from_names(("f0", "f1"))
    labels = LabelSchema(("benign", "attack"))
    graph_schema = GraphSchema(
        feature_schema_digest=features.digest,
        label_schema_digest=labels.digest,
    )
    initial_state = {
        "weight": np.zeros((2, 2), dtype=np.float32),
        "bias": np.zeros((2,), dtype=np.float32),
    }
    model_spec = ModelSpec.from_state(
        family="linear",
        model_version=1,
        feature_dim=2,
        num_classes=2,
        node_feature_dim=1,
        hyperparameters={},
        state=initial_state,
    )
    contract_root = ContractBundle(
        feature_schema=features,
        label_schema=labels,
        graph_schema=graph_schema,
        model_spec=model_spec,
    ).write(root / "contract")
    initial_path = root / "initial_state.npz"
    np.savez(initial_path, **initial_state)
    client_root = write_graph_arrays(
        root / "clients" / "client-a",
        _graph(feature_dim=client_feature_dim),
        metadata={
            "client_id": "client-a",
            "graph_protocol": "transductive_edge_mask",
        },
    )
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "dataset_id": "fixture",
        "graph_protocol": "transductive_edge_mask",
        "contract_path": "contract",
        "contract_digest": checksum_index_digest(contract_root),
        "initial_state_path": "initial_state.npz",
        "initial_state_sha256": sha256_file(initial_path),
        "clients": [
            {
                "client_id": "client-a",
                "path": "clients/client-a",
                "num_edges": 4,
                "artifact_digest": checksum_index_digest(client_root),
            }
        ],
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return root


class PreparedManifestTests(unittest.TestCase):
    def test_parent_digest_rejects_locally_rechecksummed_client_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = _prepared_fixture(Path(temporary))
            client_root = root / "clients" / "client-a"
            edge_attr_path = client_root / "edge_attr.npy"
            np.save(
                edge_attr_path,
                np.full((4, 2), 99.0, dtype=np.float32),
                allow_pickle=False,
            )
            checksums_path = client_root / "checksums.json"
            checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
            checksums["edge_attr.npy"] = sha256_file(edge_attr_path)
            checksums_path.write_text(json.dumps(checksums), encoding="utf-8")

            with self.assertRaisesRegex(ContractError, "artifact digest mismatch"):
                PreparedDatasetManifest.load(root, verify=True)

    def test_client_feature_dimension_must_match_shared_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = _prepared_fixture(Path(temporary), client_feature_dim=3)
            with self.assertRaisesRegex(ContractError, "shared feature_dim=2"):
                PreparedDatasetManifest.load(root, verify=True)

    def test_manifest_edge_count_must_match_client_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = _prepared_fixture(Path(temporary))
            path = root / "manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["clients"][0]["num_edges"] = 5
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "num_edges"):
                PreparedDatasetManifest.load(root, verify=True)


if __name__ == "__main__":
    unittest.main()
