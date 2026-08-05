from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest

import numpy as np

from src.federated.flower.personalized_state import (
    PersonalizedStateError,
    PersonalizedStateStore,
)


FLOWER_AVAILABLE = importlib.util.find_spec("flwr") is not None


class PersonalizedStateStoreTests(unittest.TestCase):
    def _store(self, root: str, *, run_id: str = "run-1"):
        return PersonalizedStateStore(
            root,
            client_id="edge/01",
            run_id=run_id,
            model_digest="a" * 64,
            personalized_prefixes=("head.",),
            initial_state={
                "head.weight": np.zeros((2, 3), dtype=np.float32),
                "head.bias": np.zeros((2,), dtype=np.float32),
            },
        )

    def test_cold_start_then_versioned_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = self._store(temporary)
            initial, metadata = store.load(require_ready=False)
            self.assertTrue(metadata["cold_start"])
            updated = {name: value + 1 for name, value in initial.items()}
            first = store.save(updated)
            self.assertEqual(first["completed_rounds"], 1)

            restored, restored_metadata = self._store(temporary).load(
                require_ready=True
            )
            self.assertEqual(restored_metadata["state_file"], "head-0001.npz")
            for name in updated:
                np.testing.assert_array_equal(restored[name], updated[name])

            second = store.save({name: value + 1 for name, value in updated.items()})
            self.assertEqual(second["state_file"], "head-0002.npz")
            self.assertTrue((store.root / "head-0001.npz").is_file())
            self.assertTrue((store.root / "head-0002.npz").is_file())

    def test_evaluation_fails_closed_before_first_round(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                PersonalizedStateError, "not inference-ready"
            ):
                self._store(temporary).load(require_ready=True)

    def test_digest_corruption_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = self._store(temporary)
            state, _ = store.load(require_ready=False)
            metadata = store.save(state)
            path = store.root / metadata["state_file"]
            path.write_bytes(path.read_bytes() + b"corrupt")
            with self.assertRaisesRegex(
                PersonalizedStateError, "digest mismatch"
            ):
                store.load(require_ready=True)

    def test_provenance_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = self._store(temporary)
            state, _ = store.load(require_ready=False)
            store.save(state)
            metadata = json.loads(store.metadata_path.read_text())
            metadata["client_id"] = "different"
            store.metadata_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(
                PersonalizedStateError, "provenance mismatch"
            ):
                store.load(require_ready=True)


@unittest.skipUnless(FLOWER_AVAILABLE, "Flower is an optional Phase 2 dependency")
class FlowerPersonalizedClientTests(unittest.TestCase):
    def test_client_uploads_only_shared_state_and_recovers_head(self) -> None:
        from flwr.app import ArrayRecord, ConfigRecord, Context, Message, RecordDict

        from src.federated.adapters.toy import ToyFederatedTask
        from src.federated.core.simulation import split_personalized_state
        from src.federated.core.state import (
            arrays_to_torch_state,
            torch_state_to_arrays,
        )
        from src.federated.flower.client_app import build_client_app

        task = ToyFederatedTask(seed=42)
        app = build_client_app(lambda _: task, log_message_sizes=False)
        with tempfile.TemporaryDirectory() as temporary:
            run_config = {
                "strategy": "fedper",
                "personalized-prefixes": "classifier.bias",
                "personalized-state-root": temporary,
                "local-epochs": 1,
                "learning-rate": 0.1,
                "optimizer": "sgd",
                "seed": 42,
            }
            context = Context(
                run_id=7,
                node_id=1,
                node_config={"client-id": "toy-client-0"},
                state=RecordDict(),
                run_config=run_config,
            )
            shared, _ = split_personalized_state(
                task.initial_state(),
                personalized_prefixes=("classifier.bias",),
            )
            train_message = Message(
                dst_node_id=1,
                message_type="train",
                content=RecordDict(
                    {
                        "arrays": ArrayRecord(
                            torch_state_dict=arrays_to_torch_state(shared)
                        ),
                        "config": ConfigRecord({"lr": 0.1}),
                    }
                ),
            )
            train_reply = app._registered_funcs["train.default"](
                train_message, context
            )
            uploaded = torch_state_to_arrays(
                train_reply.content["arrays"].to_torch_state_dict()
            )
            self.assertEqual(set(uploaded), {"classifier.weight"})
            self.assertEqual(
                train_reply.content["metrics"]["personalized-rounds"], 1
            )

            restarted_context = Context(
                run_id=7,
                node_id=1,
                node_config={"client-id": "toy-client-0"},
                state=RecordDict(),
                run_config=run_config,
            )
            evaluate_message = Message(
                dst_node_id=1,
                message_type="evaluate",
                content=RecordDict(
                    {
                        "arrays": train_reply.content["arrays"],
                        "config": ConfigRecord({"split": "val"}),
                    }
                ),
            )
            evaluate_reply = app._registered_funcs["evaluate.default"](
                evaluate_message, restarted_context
            )
            self.assertEqual(
                evaluate_reply.content["metrics"]["num-examples"], 30
            )

            untrained_context = Context(
                run_id=8,
                node_id=1,
                node_config={"client-id": "toy-client-0"},
                state=RecordDict(),
                run_config=run_config,
            )
            with self.assertRaisesRegex(
                PersonalizedStateError, "not inference-ready"
            ):
                app._registered_funcs["evaluate.default"](
                    evaluate_message, untrained_context
                )


if __name__ == "__main__":
    unittest.main()
