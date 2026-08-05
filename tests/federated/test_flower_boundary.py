from __future__ import annotations

import importlib.util
import tempfile
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


FLOWER_AVAILABLE = importlib.util.find_spec("flwr") is not None
ROOT = Path(__file__).resolve().parents[2]


class FlowerConfigTests(unittest.TestCase):
    def test_flower_cli_declares_every_runtime_override_key(self) -> None:
        from src.federated.flower.config import DEFAULT_RUN_CONFIG

        with (ROOT / "pyproject.toml").open("rb") as handle:
            declared = tomllib.load(handle)["tool"]["flwr"]["app"]["config"]

        self.assertEqual(set(DEFAULT_RUN_CONFIG) - set(declared), set())

    def test_direct_execution_has_complete_defaults(self) -> None:
        from src.federated.flower.config import resolve_run_config

        resolved = resolve_run_config({})
        self.assertEqual(resolved["num-server-rounds"], 3)
        self.assertEqual(resolved["local-epochs"], 1)
        self.assertEqual(resolved["learning-rate"], 0.15)

    def test_iot23_uses_authoritative_phase2_hyperparameters(self) -> None:
        from src.federated.flower.config import resolve_run_config

        resolved = resolve_run_config(
            {
                "task": "iot23_manifest",
                "phase2-config": str(
                    ROOT / "configs/phase2/iot23-federated.yaml"
                ),
                "strategy": "fedprox",
            }
        )
        self.assertEqual(resolved["num-server-rounds"], 30)
        self.assertEqual(resolved["local-epochs"], 5)
        self.assertEqual(resolved["optimizer"], "adam")
        self.assertEqual(resolved["learning-rate"], 0.001)
        self.assertEqual(resolved["proximal-mu"], 0.01)
        self.assertEqual(resolved["evaluate-split"], "val")
        self.assertEqual(resolved["final-split"], "test")
        self.assertEqual(len(resolved["benchmark-config-digest"]), 64)

    def test_iot23_allows_fedper_with_private_state_defaults(self) -> None:
        from src.federated.flower.config import resolve_run_config

        resolved = resolve_run_config(
            {
                "task": "iot23_manifest",
                "phase2-config": str(
                    ROOT / "configs/phase2/iot23-federated.yaml"
                ),
                "strategy": "fedper",
            }
        )
        self.assertEqual(resolved["strategy"], "fedper")
        self.assertEqual(resolved["proximal-mu"], 0.0)
        self.assertEqual(resolved["personalized-prefixes"], "head.")
        self.assertIn("personalized-state-root", resolved)

    def test_iot23_preserves_explicit_durable_output_root(self) -> None:
        from src.federated.flower.config import resolve_run_config

        resolved = resolve_run_config(
            {
                "task": "iot23_manifest",
                "phase2-config": str(
                    ROOT / "configs/phase2/iot23-federated.yaml"
                ),
                "strategy": "fedavg",
                "flower-output-root": "/artifacts/fedavg/runs",
            }
        )

        self.assertEqual(
            resolved["flower-output-root"], "/artifacts/fedavg/runs"
        )
        self.assertEqual(resolved["num-server-rounds"], 30)
        self.assertEqual(resolved["local-epochs"], 5)

    def test_iot23_client_builds_local_train_config_from_phase2_yaml(self) -> None:
        from src.federated.flower.client_app import _train_config

        run_config = {
            "task": "iot23_manifest",
            "phase2-config": str(ROOT / "configs/phase2/iot23-federated.yaml"),
            "strategy": "fedprox",
        }
        message = SimpleNamespace(content={"config": {"lr": 0.001}})
        local = _train_config(message, SimpleNamespace(run_config=run_config))
        self.assertEqual(local.local_epochs, 5)
        self.assertEqual(local.optimizer, "adam")
        self.assertEqual(local.learning_rate, 0.001)
        self.assertEqual(local.proximal_mu, 0.01)


@unittest.skipUnless(FLOWER_AVAILABLE, "Flower is an optional Phase 2 dependency")
class FlowerBoundaryTests(unittest.TestCase):
    def test_current_message_api_apps_are_constructible(self) -> None:
        from flwr.clientapp import ClientApp
        from flwr.serverapp import ServerApp

        from src.federated.flower.toy_client_app import app as client_app
        from src.federated.flower.toy_server_app import app as server_app

        self.assertIsInstance(client_app, ClientApp)
        self.assertIsInstance(server_app, ServerApp)

    def test_confusion_matrix_callback_computes_global_metrics(self) -> None:
        from flwr.app import MetricRecord, RecordDict

        from src.federated.flower.metrics import aggregate_evaluation_records

        records = [
            RecordDict(
                {
                    "metrics": MetricRecord(
                        {
                            "num-examples": 6,
                            "loss": 0.4,
                            "num-classes": 2,
                            "confusion-matrix": [3, 1, 0, 2],
                        }
                    )
                }
            ),
            RecordDict(
                {
                    "metrics": MetricRecord(
                        {
                            "num-examples": 4,
                            "loss": 0.2,
                            "num-classes": 2,
                            "confusion-matrix": [1, 0, 1, 2],
                        }
                    )
                }
            ),
        ]
        aggregated = aggregate_evaluation_records(records, "num-examples")
        self.assertEqual(aggregated["num-examples"], 10)
        self.assertEqual(aggregated["confusion-matrix"], [4, 1, 1, 4])
        self.assertAlmostEqual(aggregated["loss"], 0.32)
        self.assertAlmostEqual(aggregated["accuracy"], 0.8)
        self.assertAlmostEqual(aggregated["macro-f1"], 0.8)

    def test_arrayrecord_preserves_named_state_schema(self) -> None:
        from flwr.app import ArrayRecord

        from src.federated.adapters.toy import ToyFederatedTask
        from src.federated.core.state import (
            arrays_to_torch_state,
            torch_state_to_arrays,
        )

        task = ToyFederatedTask()
        original = task.initial_state()
        record = ArrayRecord(
            torch_state_dict=arrays_to_torch_state(original)
        )
        restored = torch_state_to_arrays(record.to_torch_state_dict())
        task.model_spec.validate_state(restored)
        for name in original:
            np.testing.assert_array_equal(original[name], restored[name])

    def test_tracking_strategy_composes_with_pinned_flower_fedavg(self) -> None:
        from flwr.serverapp.strategy import FedAvg

        from src.federated.adapters.toy import ToyFederatedTask
        from src.federated.flower.server_app import _TrackingStrategyMixin
        from src.federated.flower.tracking import FlowerBestTracker
        from src.federated.observability.events import NoopObserver
        from src.federated.observability.run_store import RunStore

        class TrackingFedAvg(_TrackingStrategyMixin, FedAvg):
            pass

        task = ToyFederatedTask(seed=42)
        with tempfile.TemporaryDirectory() as temporary:
            store = RunStore.create(
                temporary,
                strategy="fedavg",
                config_digest="config",
                dataset_digest="dataset",
                model_digest=task.model_spec.digest,
                config_snapshot={"fixture": True},
            )
            tracker = FlowerBestTracker(
                store=store,
                model_spec=task.model_spec,
                observer=NoopObserver(),
                flower_run_id="flower-fixture",
            )
            strategy = TrackingFedAvg(
                min_train_nodes=2,
                min_evaluate_nodes=2,
                min_available_nodes=2,
                tracker=tracker,
            )
            self.assertIs(strategy._tracker, tracker)


if __name__ == "__main__":
    unittest.main()
