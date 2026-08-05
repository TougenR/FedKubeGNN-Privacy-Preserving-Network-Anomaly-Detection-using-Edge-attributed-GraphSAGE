import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.federated.config import Phase2ConfigError, load_phase2_config
from src.federated.observability import JsonlObserver, RunStore
from src.federated.observability.events import ObservabilityError
from src.federated.registry import ComponentRegistry, RegistryError, builtin_registry


ROOT = Path(__file__).resolve().parents[2]


class ConfigTests(unittest.TestCase):
    def test_repository_config_is_strict_and_stable(self):
        config = load_phase2_config(ROOT / "configs/phase2/iot23-federated.yaml")
        self.assertEqual(len(config.data.scenarios), 6)
        self.assertEqual(config.federation.strategies, ("fedavg", "fedprox"))
        self.assertEqual(config.training.class_weight_scope, "local")
        self.assertEqual(len(config.digest), 64)

        global_config = load_phase2_config(
            ROOT / "configs/phase3/ablations/global-class-weights.yaml"
        )
        self.assertEqual(global_config.training.class_weight_scope, "global")
        self.assertNotEqual(config.digest, global_config.digest)

    def test_unknown_key_is_rejected(self):
        source = (ROOT / "configs/phase2/iot23-federated.yaml").read_text()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.yaml"
            path.write_text(source + "unknown: true\n")
            with self.assertRaises(Phase2ConfigError):
                load_phase2_config(path)


class RegistryTests(unittest.TestCase):
    def test_unknown_and_duplicate_names_fail_closed(self):
        registry = ComponentRegistry()
        registry.register("task", "toy", lambda: "ok")
        self.assertEqual(registry.resolve("task", "toy")(), "ok")
        with self.assertRaises(RegistryError):
            registry.register("task", "toy", lambda: "duplicate")
        with self.assertRaises(RegistryError):
            registry.resolve("task", "missing")

    def test_builtins_cover_every_configured_extension_point(self):
        config = load_phase2_config(ROOT / "configs/phase2/iot23-federated.yaml")
        components = builtin_registry()
        for kind, name in config.components.__dict__.items():
            self.assertIn(name, components.names(kind))


class ObservabilityTests(unittest.TestCase):
    def test_jsonl_event_and_sensitive_field_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            observer = JsonlObserver(path)
            observer.emit("round.completed", run_id="r1", round=1, macro_f1=0.5)
            event = json.loads(path.read_text())
            self.assertEqual(event["event"], "round.completed")
            self.assertEqual(event["round"], 1)
            with self.assertRaises(ObservabilityError):
                observer.emit("bad", raw_ip="10.0.0.1")

    def test_run_store_atomic_status_checkpoint_and_resume_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = RunStore.create(
                temporary,
                strategy="fedavg",
                config_digest="c",
                dataset_digest="d",
                model_digest="m",
                config_snapshot={"version": 1},
            )
            checkpoint = store.checkpoint(
                {"weight": np.array([1.0])}, round_number=1, best=True
            )
            self.assertTrue(checkpoint.is_file())
            self.assertTrue((store.root / "checkpoints/best_model.npz").is_file())
            RunStore.resume(
                store.root,
                strategy="fedavg",
                config_digest="c",
                dataset_digest="d",
                model_digest="m",
            )
            with self.assertRaises(ValueError):
                RunStore.resume(
                    store.root,
                    strategy="fedavg",
                    config_digest="wrong",
                    dataset_digest="d",
                    model_digest="m",
                )
            with self.assertRaises(ValueError):
                RunStore.resume(
                    store.root,
                    strategy="fedprox",
                    config_digest="c",
                    dataset_digest="d",
                    model_digest="m",
                )
            store.complete(best_round=1)
            status = json.loads((store.root / "run.json").read_text())
            self.assertEqual(status["status"], "completed")


if __name__ == "__main__":
    unittest.main()
