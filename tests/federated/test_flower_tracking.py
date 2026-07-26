from __future__ import annotations

import tempfile
import unittest

import numpy as np

from src.federated.adapters.toy import ToyFederatedTask
from src.federated.flower.tracking import FlowerBestTracker
from src.federated.observability.events import NoopObserver
from src.federated.observability.run_store import RunStore


class FlowerBestTrackerTests(unittest.TestCase):
    def test_retains_validation_best_state_instead_of_last_round(self):
        task = ToyFederatedTask(seed=42)
        first = task.initial_state()
        second = {
            name: np.asarray(value).copy() for name, value in first.items()
        }
        first_name = next(iter(second))
        second[first_name] = second[first_name] + 1
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
            tracker.record_train_state(1, first)
            tracker.record_validation(1, {"macro-f1": 0.8})
            tracker.record_train_state(2, second)
            tracker.record_validation(2, {"macro-f1": 0.7})

            self.assertEqual(tracker.best_round, 1)
            best = tracker.best_state()
            for name in first:
                np.testing.assert_array_equal(best[name], first[name])
            with np.load(
                store.root / "checkpoints" / "best_model.npz",
                allow_pickle=False,
            ) as archive:
                np.testing.assert_array_equal(archive[first_name], first[first_name])


if __name__ == "__main__":
    unittest.main()
