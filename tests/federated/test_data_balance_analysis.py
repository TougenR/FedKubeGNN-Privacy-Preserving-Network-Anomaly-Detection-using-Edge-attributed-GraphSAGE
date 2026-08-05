from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.federated.analysis import analyze_prepared_dataset
from src.federated.contracts.artifacts import ContractBundle
from src.federated.contracts.schema import (
    FeatureSchema,
    GraphSchema,
    LabelSchema,
    ModelSpec,
)
from src.federated.data.manifest import MANIFEST_VERSION
from src.federated.data.storage import (
    GraphArrays,
    checksum_index_digest,
    sha256_file,
    write_graph_arrays,
)


def _prepared_fixture(root: Path) -> Path:
    root.mkdir(parents=True)
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
    contract = ContractBundle(
        feature_schema=features,
        label_schema=labels,
        graph_schema=graph_schema,
        model_spec=model_spec,
    ).write(root / "contract")
    initial_path = root / "initial_state.npz"
    np.savez(initial_path, **initial_state)
    graph = GraphArrays(
        edge_index=np.array([[0, 1, 2, 0], [1, 2, 0, 2]], dtype=np.int64),
        edge_attr=np.arange(8, dtype=np.float32).reshape(4, 2),
        edge_label=np.array([0, 1, 0, 1], dtype=np.int64),
        train_mask=np.array([True, True, False, False]),
        val_mask=np.array([False, False, True, False]),
        test_mask=np.array([False, False, False, True]),
        num_nodes=3,
    )
    client = write_graph_arrays(
        root / "clients" / "client-a",
        graph,
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
        "contract_digest": checksum_index_digest(contract),
        "initial_state_path": "initial_state.npz",
        "initial_state_sha256": sha256_file(initial_path),
        "clients": [
            {
                "client_id": "client-a",
                "path": "clients/client-a",
                "num_edges": 4,
                "artifact_digest": checksum_index_digest(client),
            }
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


class DataBalanceAnalysisTests(unittest.TestCase):
    def test_writes_integrity_bound_balance_feature_and_topology_tables(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = _prepared_fixture(root / "dataset")
            output = analyze_prepared_dataset(
                dataset, root / "analysis", render_figures=False
            )

            report = json.loads(
                (output / "data_balance.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["dataset_id"], "fixture")
            self.assertTrue(report["severe_imbalance"])
            self.assertEqual(
                report["global_class_support"], {"attack": 2, "benign": 2}
            )
            self.assertTrue(
                report["integrity"]["manifest_and_all_client_checksums_verified"]
            )

            with (output / "class_support_by_client.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                support = list(csv.DictReader(handle))
            self.assertEqual(len(support), 2 * 4 * 2)
            validation_attack = next(
                row
                for row in support
                if row["scope"] == "global"
                and row["split"] == "validation"
                and row["class_name"] == "attack"
            )
            self.assertEqual(validation_attack["support"], "0")
            self.assertEqual(validation_attack["zero_support"], "True")

            with (output / "graph_topology.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                topology = list(csv.DictReader(handle))
            self.assertEqual(topology[0]["num_nodes"], "3")
            self.assertEqual(topology[0]["num_edges"], "4")
            self.assertEqual(topology[0]["unique_directed_node_pairs"], "4")
            self.assertLessEqual(
                float(topology[0]["unique_directed_pair_density"]), 1.0
            )

            artifact_manifest = json.loads(
                (output / "analysis_manifest.json").read_text(encoding="utf-8")
            )
            self.assertIn("data_balance.csv", artifact_manifest["files"])
            self.assertIn("feature_distribution.csv", artifact_manifest["files"])
            report_text = (output / "report.md").read_text(encoding="utf-8")
            self.assertIn("severe imbalance", report_text)
            self.assertIn("exact prepared-data centralized reference", report_text)

    def test_refuses_to_overwrite_existing_analysis(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = _prepared_fixture(root / "dataset")
            output = root / "analysis"
            analyze_prepared_dataset(dataset, output, render_figures=False)
            with self.assertRaisesRegex(FileExistsError, "not empty"):
                analyze_prepared_dataset(dataset, output, render_figures=False)


if __name__ == "__main__":
    unittest.main()
