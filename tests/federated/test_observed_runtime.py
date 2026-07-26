import json
import tempfile
import unittest
from pathlib import Path

from src.federated.adapters.toy import ToyFederatedTask
from src.federated.contracts.task import LocalTrainConfig
from src.federated.observability import RunStore
from src.federated.runtimes.inprocess import run_observed_inprocess
from src.federated.strategies.fedavg import FedAvgPolicy
from src.federated.strategies.fedprox import FedProxPolicy


class SixClientToyTask:
    """Six non-IID logical clients sharing the tested toy task implementation."""

    def __init__(self):
        self.base = ToyFederatedTask(seed=42)
        self._client_ids = tuple(f"scenario-{index}" for index in range(6))

    @property
    def client_ids(self):
        return self._client_ids

    @property
    def task_id(self):
        return "six-client-toy"

    @property
    def feature_schema(self):
        return self.base.feature_schema

    @property
    def label_schema(self):
        return self.base.label_schema

    @property
    def graph_schema(self):
        return self.base.graph_schema

    @property
    def model_spec(self):
        return self.base.model_spec

    def initial_state(self):
        return self.base.initial_state()

    def _mapped(self, client_id):
        return self.base.client_ids[int(client_id.rsplit("-", 1)[1]) % 2]

    def train_local(self, client_id, state, config):
        return self.base.train_local(self._mapped(client_id), state, config)

    def evaluate_local(self, client_id, state, *, split):
        return self.base.evaluate_local(self._mapped(client_id), state, split=split)

    def metadata(self):
        return {"dataset_id": "six-scenario-fixture", "graph_protocol": "fixture"}


class ObservedRuntimeTests(unittest.TestCase):
    def test_fedavg_and_fedprox_emit_checkpoints_metrics_and_test_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            for policy in (FedAvgPolicy(), FedProxPolicy(0.01)):
                result = run_observed_inprocess(
                    SixClientToyTask(),
                    policy=policy,
                    num_rounds=2,
                    train_config=LocalTrainConfig(
                        local_epochs=1, learning_rate=0.1, optimizer="sgd"
                    ),
                    output_root=temporary,
                    config_digest="config",
                    config_snapshot={"fixture": True},
                )
                status = json.loads((result.run_root / "run.json").read_text())
                summary = json.loads(
                    (result.run_root / "metrics/summary.json").read_text()
                )
                self.assertEqual(status["status"], "completed")
                self.assertEqual(
                    len(
                        (result.run_root / "metrics/rounds.jsonl")
                        .read_text()
                        .splitlines()
                    ),
                    2,
                )
                self.assertTrue(
                    (result.run_root / "checkpoints/best_model.npz").is_file()
                )
                self.assertGreater(summary["total_upload_bytes"], 0)
                self.assertIn("test_metrics", summary)
                rounds = [
                    json.loads(line)
                    for line in (result.run_root / "metrics/rounds.jsonl")
                    .read_text()
                    .splitlines()
                ]
                self.assertEqual(rounds[0]["train_examples"], 540)
                self.assertEqual(rounds[0]["validation_examples"], 180)

    def test_failed_run_resumes_from_latest_compatible_round(self):
        class FailingTask(SixClientToyTask):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def train_local(self, client_id, state, config):
                self.calls += 1
                if self.calls == 7:
                    raise RuntimeError("synthetic interruption")
                return super().train_local(client_id, state, config)

        with tempfile.TemporaryDirectory() as temporary:
            kwargs = dict(
                policy=FedAvgPolicy(),
                num_rounds=2,
                train_config=LocalTrainConfig(
                    local_epochs=1, learning_rate=0.1, optimizer="sgd"
                ),
                output_root=temporary,
                config_digest="resume-config",
                config_snapshot={"fixture": True},
            )
            with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                run_observed_inprocess(FailingTask(), **kwargs)
            run_root = next(path for path in Path(temporary).iterdir() if path.is_dir())
            failed = json.loads((run_root / "run.json").read_text())
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["latest_round"], 1)
            with self.assertRaisesRegex(ValueError, "strategy"):
                run_observed_inprocess(
                    SixClientToyTask(),
                    resume_root=run_root,
                    **{**kwargs, "policy": FedProxPolicy(0.01)},
                )
            result = run_observed_inprocess(
                SixClientToyTask(), resume_root=run_root, **kwargs
            )
            completed = json.loads((result.run_root / "run.json").read_text())
            self.assertEqual(completed["status"], "completed")
            self.assertNotIn("failure", completed)
            self.assertEqual(completed["previous_failure"], "failure.json")
            self.assertEqual(
                len((run_root / "metrics/rounds.jsonl").read_text().splitlines()), 2
            )

    def test_orphan_checkpoint_before_commit_marker_is_safely_replayed(self):
        with tempfile.TemporaryDirectory() as temporary:
            task = SixClientToyTask()
            store = RunStore.create(
                temporary,
                strategy="fedavg",
                config_digest="orphan-config",
                dataset_digest="six-scenario-fixture",
                model_digest=task.model_spec.digest,
                config_snapshot={"fixture": True},
            )
            store.checkpoint(
                task.initial_state(), round_number=1, mark_latest=False
            )
            store.fail(RuntimeError("crash before round commit"))

            result = run_observed_inprocess(
                task,
                policy=FedAvgPolicy(),
                num_rounds=1,
                train_config=LocalTrainConfig(
                    local_epochs=1, learning_rate=0.1, optimizer="sgd"
                ),
                output_root=temporary,
                config_digest="orphan-config",
                config_snapshot={"fixture": True},
                resume_root=store.root,
            )
            status = json.loads((result.run_root / "run.json").read_text())
            self.assertEqual(status["status"], "completed")
            self.assertEqual(status["latest_round"], 1)
            self.assertTrue(
                (result.run_root / "metrics/rounds/round-0001.json").is_file()
            )
            self.assertEqual(
                len(
                    (result.run_root / "metrics/rounds.jsonl")
                    .read_text()
                    .splitlines()
                ),
                1,
            )


if __name__ == "__main__":
    unittest.main()
