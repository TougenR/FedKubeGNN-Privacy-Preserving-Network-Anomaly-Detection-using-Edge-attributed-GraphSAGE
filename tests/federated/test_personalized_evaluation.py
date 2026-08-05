from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.federated.adapters.toy import ToyFederatedTask
from src.federated.contracts.task import LocalTrainConfig
from src.federated.core.simulation import run_fedper_simulation
from src.federated.experiments.evaluation import (
    evaluate_personalized_checkpoint,
)


class _DatasetToyTask:
    def __init__(self, task: ToyFederatedTask) -> None:
        self._task = task

    def __getattr__(self, name):
        return getattr(self._task, name)

    def metadata(self):
        return {**self._task.metadata(), "dataset_id": "toy-personalized"}


class PersonalizedEvaluationTests(unittest.TestCase):
    def test_evaluates_each_client_with_its_own_private_state(self) -> None:
        task = _DatasetToyTask(ToyFederatedTask(seed=42))
        run = run_fedper_simulation(
            task,
            num_rounds=2,
            train_config=LocalTrainConfig(
                local_epochs=1,
                learning_rate=0.1,
                optimizer="sgd",
                seed=42,
            ),
            personalized_prefixes=("classifier.bias",),
            evaluate_split="val",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared = root / "shared.npz"
            heads = root / "heads"
            heads.mkdir()
            np.savez(shared, **run.best_shared_state)
            for client_id, state in run.best_personalized_states.items():
                np.savez(heads / f"{client_id}.npz", **state)

            with patch(
                "src.federated.experiments.evaluation.manifest_task",
                return_value=task,
            ):
                result = evaluate_personalized_checkpoint(
                    object(),
                    root,
                    shared,
                    heads,
                    split="test",
                )

        self.assertEqual(result["dataset_id"], "toy-personalized")
        self.assertEqual(set(result["per_client"]), set(task.client_ids))
        self.assertEqual(result["metrics"]["num_examples"], 120)

    def test_missing_client_head_fails_closed(self) -> None:
        task = _DatasetToyTask(ToyFederatedTask(seed=42))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared = root / "shared.npz"
            np.savez(shared, **task.initial_state())
            with patch(
                "src.federated.experiments.evaluation.manifest_task",
                return_value=task,
            ):
                with self.assertRaisesRegex(
                    FileNotFoundError, "Missing personalized checkpoint"
                ):
                    evaluate_personalized_checkpoint(
                        object(), root, shared, root, split="test"
                    )


if __name__ == "__main__":
    unittest.main()
