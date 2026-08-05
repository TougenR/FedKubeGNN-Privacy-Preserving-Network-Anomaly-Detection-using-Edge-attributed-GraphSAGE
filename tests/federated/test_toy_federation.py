from __future__ import annotations

import sys
import unittest

from src.federated.adapters.toy import ToyFederatedTask
from src.federated.contracts.task import FederatedTask, LocalTrainConfig
from src.federated.core.simulation import run_federated_simulation


class ToyFederationTests(unittest.TestCase):
    def test_toy_task_satisfies_runtime_protocol(self) -> None:
        self.assertIsInstance(ToyFederatedTask(), FederatedTask)

    def test_toy_federation_runs_without_phase1_or_pyg(self) -> None:
        modules_before = set(sys.modules)
        task = ToyFederatedTask(seed=42)
        result = run_federated_simulation(
            task,
            num_rounds=8,
            train_config=LocalTrainConfig(
                local_epochs=1,
                learning_rate=0.15,
                optimizer="sgd",
                seed=42,
            ),
        )
        self.assertEqual(len(result.rounds), 8)
        self.assertGreater(result.rounds[-1].global_metrics["macro_f1"], 0.90)
        self.assertEqual(result.rounds[-1].evaluation_examples, 120)
        self.assertGreater(result.rounds[-1].upload_bytes, 0)
        self.assertGreaterEqual(result.best_round, 1)
        task.model_spec.validate_state(result.best_state)
        self.assertEqual(
            set(result.rounds[0].client_diagnostics), set(task.client_ids)
        )
        for diagnostics in result.rounds[0].client_diagnostics.values():
            self.assertGreaterEqual(diagnostics["update_l2"], 0.0)
            self.assertGreaterEqual(
                diagnostics["distance_to_aggregate_l2"], 0.0
            )
        modules_loaded_by_toy = set(sys.modules) - modules_before
        self.assertNotIn("src.model", modules_loaded_by_toy)
        self.assertNotIn("torch_geometric", modules_loaded_by_toy)

    def test_fedprox_path_runs(self) -> None:
        task = ToyFederatedTask(seed=7)
        result = run_federated_simulation(
            task,
            num_rounds=2,
            train_config=LocalTrainConfig(
                local_epochs=2,
                learning_rate=0.1,
                optimizer="sgd",
                proximal_mu=0.01,
                seed=7,
            ),
            diagnose_local_states=True,
        )
        self.assertEqual(result.rounds[-1].train_examples, 180)
        diagnostics = result.rounds[-1].client_diagnostics
        self.assertIn("local_state_own_client_metrics", next(iter(diagnostics.values())))
        self.assertIn("local_state_global_metrics", next(iter(diagnostics.values())))


if __name__ == "__main__":
    unittest.main()
