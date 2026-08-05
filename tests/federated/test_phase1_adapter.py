from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn

from src.federated.adapters.phase1_iot23 import Phase1IoT23Task
from src.federated.contracts.schema import ContractError
from src.federated.contracts.task import LocalTrainConfig
from src.federated.core.simulation import run_federated_simulation


FEATURES = ("f0", "f1")
CLASSES = {"benign": 0, "attack": 1}


class _FakeGraph:
    def __init__(self, *, invert: bool = False) -> None:
        # Twelve linearly separable edge examples with both labels in every split.
        values = torch.tensor(
            [
                [-2.0, -1.0],
                [-1.5, -0.5],
                [1.5, 0.5],
                [2.0, 1.0],
                [-1.2, -1.1],
                [1.2, 1.1],
                [-0.8, -1.4],
                [0.8, 1.4],
                [-1.0, -0.7],
                [1.0, 0.7],
                [-1.7, -0.8],
                [1.7, 0.8],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long)
        if invert:
            # Keep the model task the same but vary local order/distribution.
            order = torch.arange(len(labels) - 1, -1, -1)
            values = values[order]
            labels = labels[order]
        num_edges = len(labels)
        src = torch.arange(num_edges) % 4
        dst = (src + 1) % 4
        self.x = torch.ones((4, 1), dtype=torch.float32)
        self.edge_index = torch.stack([src, dst])
        self.edge_attr = values
        self.edge_label = labels
        self.edge_index_mp = torch.cat(
            [self.edge_index, self.edge_index.flip(0)], dim=1
        )
        self.edge_attr_mp = torch.cat([values, values], dim=0)
        self.train_mask = torch.zeros(num_edges, dtype=torch.bool)
        self.val_mask = torch.zeros(num_edges, dtype=torch.bool)
        self.test_mask = torch.zeros(num_edges, dtype=torch.bool)
        self.train_mask[:8] = True
        self.val_mask[8:10] = True
        self.test_mask[10:] = True
        self.feature_dim = 2
        self.num_classes = 2
        self.class_to_idx = dict(CLASSES)

    def to(self, device: torch.device) -> "_FakeGraph":
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))
        return self


class _EdgeLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(2, 2)

    def forward(self, graph: _FakeGraph) -> torch.Tensor:
        return self.classifier(graph.edge_attr)


def _model_factory(_: _FakeGraph) -> nn.Module:
    return _EdgeLinear()


class Phase1AdapterTests(unittest.TestCase):
    def _task(self) -> Phase1IoT23Task:
        torch.manual_seed(42)
        return Phase1IoT23Task(
            client_graphs={
                "scenario-a": _FakeGraph(),
                "scenario-b": _FakeGraph(invert=True),
            },
            feature_columns=FEATURES,
            class_to_idx=CLASSES,
            model_factory=_model_factory,
            model_family="fake-edge-linear",
            model_hyperparameters={"hidden_dim": 0},
            device="cpu",
        )

    def test_adapter_runs_through_public_federated_core(self) -> None:
        task = self._task()
        result = run_federated_simulation(
            task,
            num_rounds=5,
            train_config=LocalTrainConfig(
                local_epochs=2,
                learning_rate=0.1,
                optimizer="sgd",
                seed=42,
            ),
        )
        self.assertGreaterEqual(result.rounds[-1].global_metrics["macro_f1"], 0.5)
        self.assertEqual(result.rounds[-1].train_examples, 16)
        self.assertEqual(result.rounds[-1].evaluation_examples, 4)

    def test_one_client_fedavg_matches_direct_local_update(self) -> None:
        task = self._task()
        config = LocalTrainConfig(
            local_epochs=2, learning_rate=0.1, optimizer="sgd", seed=42
        )
        direct = task.train_local("scenario-a", task.initial_state(), config)
        federated = run_federated_simulation(
            task,
            num_rounds=1,
            train_config=config,
            client_ids=["scenario-a"],
        )
        for name in direct.state:
            np.testing.assert_allclose(
                federated.final_state[name], direct.state[name], rtol=0, atol=1e-7
            )

    def test_adapter_rejects_overlapping_masks(self) -> None:
        graph = _FakeGraph()
        graph.val_mask[0] = True
        with self.assertRaisesRegex(ContractError, "masks overlap"):
            Phase1IoT23Task(
                client_graphs={"bad": graph},
                feature_columns=FEATURES,
                class_to_idx=CLASSES,
                model_factory=_model_factory,
                device="cpu",
            )

    def test_adapter_rejects_feature_contract_drift(self) -> None:
        graph = _FakeGraph()
        with self.assertRaisesRegex(ContractError, "edge_attr shape"):
            Phase1IoT23Task(
                client_graphs={"bad": graph},
                feature_columns=("f0", "f1", "unexpected"),
                class_to_idx=CLASSES,
                model_factory=_model_factory,
                device="cpu",
            )

    def test_portable_bundle_does_not_store_phase1_object(self) -> None:
        class FakeScaler:
            mean_ = np.array([1.0, 2.0])
            scale_ = np.array([0.5, 0.25])
            var_ = np.array([0.25, 0.0625])

        class FakePreprocessor:
            proto_categories = ["tcp", "udp"]
            feature_columns = list(FEATURES)
            scaler = FakeScaler()

        bundle = self._task().contract_bundle(
            preprocessor=FakePreprocessor(), metadata={"audit": "pending"}
        )
        self.assertEqual(bundle.categories["proto"], ("tcp", "udp"))
        np.testing.assert_array_equal(
            bundle.learned_arrays["scaler_mean"], np.array([1.0, 2.0])
        )
        self.assertEqual(bundle.metadata["audit"], "pending")

    def test_fixed_global_class_weights_replace_local_weights(self) -> None:
        graph = _FakeGraph()
        task = Phase1IoT23Task(
            client_graphs={"scenario-a": graph},
            feature_columns=FEATURES,
            class_to_idx=CLASSES,
            model_factory=_model_factory,
            fixed_class_weights=[0.25, 3.5],
            device="cpu",
        )
        np.testing.assert_allclose(
            task._local_class_weights((graph,)).cpu().numpy(),
            np.array([0.25, 3.5], dtype=np.float32),
        )
        self.assertEqual(task.metadata()["class_weight_scope"], "global")

    def test_fixed_global_class_weights_validate_schema(self) -> None:
        with self.assertRaisesRegex(ContractError, "one value per class"):
            Phase1IoT23Task(
                client_graphs={"scenario-a": _FakeGraph()},
                feature_columns=FEATURES,
                class_to_idx=CLASSES,
                model_factory=_model_factory,
                fixed_class_weights=[1.0],
                device="cpu",
            )


if __name__ == "__main__":
    unittest.main()
