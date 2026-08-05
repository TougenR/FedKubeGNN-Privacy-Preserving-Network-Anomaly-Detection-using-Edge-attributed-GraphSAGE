from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.federated.adapters.phase1_iot23 import Phase1IoT23Task
from src.federated.data.manifest import MANIFEST_VERSION
from src.federated.data.storage import (
    GraphArrays,
    checksum_index_digest,
    sha256_file,
    write_graph_arrays,
)
from src.federated.tasks.iot23 import ManifestIoT23Task


class FakeGraph:
    def __init__(self):
        self.x = torch.ones((3, 1))
        self.edge_index = torch.tensor([[0, 1, 2, 0, 1, 2], [1, 2, 0, 2, 0, 1]])
        self.edge_attr = torch.tensor([[-2.0], [-1.0], [2.0], [1.0], [-0.5], [0.5]])
        self.edge_label = torch.tensor([0, 0, 1, 1, 0, 1])
        self.edge_index_mp = torch.cat(
            [self.edge_index, self.edge_index.flip(0)], dim=1
        )
        self.edge_attr_mp = torch.cat([self.edge_attr, self.edge_attr])
        self.train_mask = torch.tensor([1, 1, 1, 1, 0, 0], dtype=torch.bool)
        self.val_mask = torch.tensor([0, 0, 0, 0, 1, 0], dtype=torch.bool)
        self.test_mask = torch.tensor([0, 0, 0, 0, 0, 1], dtype=torch.bool)
        self.feature_dim = 1
        self.num_classes = 2
        self.class_to_idx = {"benign": 0, "attack": 1}
        self.num_nodes = 3

    def to(self, device):
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))
        return self


class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 2)

    def forward(self, graph):
        return self.layer(graph.edge_attr)


class ManifestTaskTests(unittest.TestCase):
    def test_server_init_loads_no_graph_and_client_load_is_lazy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "prepared"
            root.mkdir()
            torch.manual_seed(42)
            source_task = Phase1IoT23Task(
                client_graphs={"a": FakeGraph(), "b": FakeGraph()},
                feature_columns=["f0"],
                class_to_idx={"benign": 0, "attack": 1},
                model_factory=lambda _: Linear(),
                model_family="linear",
                model_hyperparameters={},
                device="cpu",
            )
            contract_root = source_task.contract_bundle().write(root / "contract")
            np.savez(root / "initial_state.npz", **source_task.initial_state())
            clients = []
            for client_id in ("a", "b"):
                relative = Path("clients") / client_id
                client_root = write_graph_arrays(
                    root / relative,
                    GraphArrays.from_graph(FakeGraph()),
                    metadata={
                        "client_id": client_id,
                        "graph_protocol": "transductive_edge_mask",
                    },
                )
                clients.append(
                    {
                        "client_id": client_id,
                        "path": str(relative),
                        "num_edges": 6,
                        "artifact_digest": checksum_index_digest(client_root),
                    }
                )
            manifest = {
                "manifest_version": MANIFEST_VERSION,
                "dataset_id": "fixture",
                "graph_protocol": "transductive_edge_mask",
                "contract_path": "contract",
                "contract_digest": checksum_index_digest(contract_root),
                "initial_state_path": "initial_state.npz",
                "initial_state_sha256": sha256_file(root / "initial_state.npz"),
                "clients": clients,
            }
            (root / "manifest.json").write_text(json.dumps(manifest))
            calls = []

            def loader(path):
                calls.append(Path(path).name)
                return FakeGraph()

            task = ManifestIoT23Task(
                root,
                model_factory=lambda _: Linear(),
                device="cpu",
                graph_loader=loader,
            )
            self.assertEqual(calls, [])
            task.initial_state()
            self.assertEqual(calls, [])
            task.evaluate_local("a", task.initial_state(), split="validation")
            self.assertEqual(calls, ["a"])
            self.assertEqual(task.metadata()["loaded_clients"], ["a"])

            global_task = ManifestIoT23Task(
                root,
                model_factory=lambda _: Linear(),
                class_weight_scope="global",
                device="cpu",
                graph_loader=loader,
            )
            self.assertEqual(global_task.metadata()["loaded_clients"], [])
            np.testing.assert_allclose(
                global_task.metadata()["global_class_weights"], [1.0, 1.0]
            )


if __name__ == "__main__":
    unittest.main()
