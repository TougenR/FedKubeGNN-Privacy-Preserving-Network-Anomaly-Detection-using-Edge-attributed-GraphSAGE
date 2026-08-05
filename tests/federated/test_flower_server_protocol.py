from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


FLOWER_AVAILABLE = importlib.util.find_spec("flwr") is not None


@unittest.skipUnless(FLOWER_AVAILABLE, "Flower is an optional Phase 2 dependency")
class FlowerServerProtocolTests(unittest.TestCase):
    def test_fedper_server_transports_only_shared_state(self) -> None:
        from flwr.app import Context, RecordDict

        from src.federated.adapters.toy import ToyFederatedTask
        from src.federated.core.state import torch_state_to_arrays
        from src.federated.flower.client_app import build_client_app
        from src.federated.flower.server_app import build_server_app

        task = ToyFederatedTask(seed=42)
        client_app = build_client_app(lambda _: task, log_message_sizes=False)

        class LocalGrid:
            def __init__(self, run_config):
                self.node_ids = (101, 102)
                self.contexts = {
                    node_id: Context(
                        run_id=8,
                        node_id=node_id,
                        node_config={"client-id": client_id},
                        state=RecordDict(),
                        run_config=run_config,
                    )
                    for node_id, client_id in zip(
                        self.node_ids, task.client_ids, strict=True
                    )
                }
                self.transported_parameter_names: list[tuple[str, ...]] = []

            def get_node_ids(self):
                return self.node_ids

            def send_and_receive(self, messages, timeout=None):
                del timeout
                replies = []
                for message in messages:
                    if "arrays" in message.content:
                        state = torch_state_to_arrays(
                            message.content["arrays"].to_torch_state_dict()
                        )
                        self.transported_parameter_names.append(tuple(state))
                    message_type = message.metadata.message_type
                    handler = client_app._registered_funcs[
                        f"{message_type}.default"
                    ]
                    reply = handler(
                        message,
                        self.contexts[message.metadata.dst_node_id],
                    )
                    if "arrays" in reply.content:
                        state = torch_state_to_arrays(
                            reply.content["arrays"].to_torch_state_dict()
                        )
                        self.transported_parameter_names.append(tuple(state))
                    replies.append(reply)
                return replies

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "runs"
            run_config = {
                "task": "toy",
                "strategy": "fedper",
                "personalized-prefixes": "classifier.bias",
                "personalized-state-root": str(root / "private"),
                "num-server-rounds": 2,
                "local-epochs": 1,
                "learning-rate": 0.15,
                "optimizer": "sgd",
                "fraction-evaluate": 1.0,
                "flower-output-root": str(output_root),
                "save-model": False,
                "seed": 42,
            }
            app = build_server_app(lambda _: task)
            server_context = Context(
                run_id=8,
                node_id=0,
                node_config={},
                state=RecordDict(),
                run_config=run_config,
            )
            grid = LocalGrid(run_config)
            app._main(grid, server_context)

            run_root = next(path for path in output_root.iterdir() if path.is_dir())
            summary = json.loads((run_root / "metrics/summary.json").read_text())
            self.assertEqual(summary["personalization"], "fedper_head")
            self.assertEqual(summary["shared_parameter_names"], ["classifier.weight"])
            self.assertEqual(summary["client_head_ownership"], "edge-local")
            self.assertEqual(summary["personalized_parameter_prefixes"], ["classifier.bias"])
            self.assertTrue(grid.transported_parameter_names)
            self.assertEqual(
                set(grid.transported_parameter_names), {("classifier.weight",)}
            )
            with np.load(
                run_root / "checkpoints/best_model.npz", allow_pickle=False
            ) as archive:
                self.assertEqual(archive.files, ["classifier.weight"])
            for client_id in task.client_ids:
                metadata_paths = list(
                    (root / "private" / client_id / "8").glob("*/metadata.json")
                )
                self.assertEqual(len(metadata_paths), 1)
                metadata = json.loads(metadata_paths[0].read_text())
                self.assertEqual(metadata["completed_rounds"], 2)

    def test_server_selects_validation_best_then_tests_once(self) -> None:
        from flwr.app import (
            ArrayRecord,
            Context,
            Message,
            MetricRecord,
            RecordDict,
        )

        from src.federated.adapters.toy import ToyFederatedTask
        from src.federated.contracts.task import LocalTrainConfig
        from src.federated.core.state import (
            arrays_to_torch_state,
            torch_state_to_arrays,
        )
        from src.federated.flower.server_app import build_server_app

        task = ToyFederatedTask(seed=42)

        class LocalGrid:
            def __init__(self) -> None:
                self.node_ids = (101, 102)
                self.client_by_node = dict(zip(self.node_ids, task.client_ids))
                self.evaluate_splits: list[str] = []
                self.validation_states: dict[int, dict[str, np.ndarray]] = {}
                self.test_state: dict[str, np.ndarray] | None = None

            def get_node_ids(self):
                return self.node_ids

            def send_and_receive(self, messages, timeout=None):
                del timeout
                replies = []
                for message in messages:
                    client_id = self.client_by_node[message.metadata.dst_node_id]
                    state = torch_state_to_arrays(
                        message.content["arrays"].to_torch_state_dict()
                    )
                    if message.metadata.message_type == "train":
                        result = task.train_local(
                            client_id,
                            state,
                            LocalTrainConfig(
                                local_epochs=1,
                                learning_rate=float(message.content["config"]["lr"]),
                                optimizer="sgd",
                                seed=42,
                            ),
                        )
                        content = RecordDict(
                            {
                                "arrays": ArrayRecord(
                                    torch_state_dict=arrays_to_torch_state(result.state)
                                ),
                                "metrics": MetricRecord(
                                    {
                                        "num-examples": result.num_examples,
                                        "train-loss": float(
                                            result.metrics["train_loss"]
                                        ),
                                    }
                                ),
                            }
                        )
                    else:
                        split = str(message.content["config"]["split"])
                        self.evaluate_splits.append(split)
                        server_round = int(
                            message.content["config"]["server-round"]
                        )
                        if split == "val":
                            self.validation_states.setdefault(
                                server_round,
                                {
                                    name: np.asarray(value).copy()
                                    for name, value in state.items()
                                },
                            )
                            matrix = (
                                np.array([[15, 0], [0, 15]], dtype=np.int64)
                                if server_round == 1
                                else np.array([[0, 15], [15, 0]], dtype=np.int64)
                            )
                        else:
                            self.test_state = {
                                name: np.asarray(value).copy()
                                for name, value in state.items()
                            }
                            matrix = np.array(
                                [[12, 3], [4, 11]], dtype=np.int64
                            )
                        content = RecordDict(
                            {
                                "metrics": MetricRecord(
                                    {
                                        "num-examples": 30,
                                        "loss": 0.1,
                                        "num-classes": task.label_schema.num_classes,
                                        "confusion-matrix": matrix.reshape(-1).tolist(),
                                    }
                                )
                            }
                        )
                    replies.append(Message(content=content, reply_to=message))
                return replies

        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "runs"
            app = build_server_app(lambda _: task)
            context = Context(
                run_id=7,
                node_id=0,
                node_config={},
                state=RecordDict(),
                run_config={
                    "task": "toy",
                    "strategy": "fedavg",
                    "num-server-rounds": 2,
                    "local-epochs": 1,
                    "learning-rate": 0.15,
                    "flower-output-root": str(output_root),
                    "save-model": False,
                },
            )
            grid = LocalGrid()
            app._main(grid, context)

            run_root = next(path for path in output_root.iterdir() if path.is_dir())
            status = json.loads((run_root / "run.json").read_text())
            summary = json.loads((run_root / "metrics/summary.json").read_text())
            self.assertEqual(status["status"], "completed")
            self.assertEqual(summary["best_round"], 1)
            self.assertEqual(summary["class_names"], list(task.label_schema.classes))
            self.assertIn("test_metrics", summary)
            self.assertEqual(grid.evaluate_splits.count("val"), 4)
            self.assertEqual(grid.evaluate_splits.count("test"), 2)
            self.assertTrue(
                (run_root / "checkpoints/best_model.npz").is_file()
            )
            self.assertIsNotNone(grid.test_state)
            assert grid.test_state is not None
            for name, value in grid.validation_states[1].items():
                np.testing.assert_array_equal(grid.test_state[name], value)
            self.assertTrue(
                any(
                    not np.array_equal(
                        grid.validation_states[1][name],
                        grid.validation_states[2][name],
                    )
                    for name in grid.validation_states[1]
                )
            )


if __name__ == "__main__":
    unittest.main()
