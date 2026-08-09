from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.application.api.schema import ProductionFlow
from src.application.collection.delivery import ObservationDispatcher
from src.application.collection.transport import ServiceRequestError
from src.application.collection.zeek_shipper import (
    follow_rotating_file,
    production_flow_from_zeek,
)
from src.application.evaluation.replay_demo import (
    execute_replay_case,
    load_scientific_replay,
)
from src.core.preprocess import clean_flows, fit_preprocessor, transform


ROOT = Path(__file__).resolve().parents[2]
REPLAY = ROOT / "configs/application/scientific-replay.json"


class LiveRemediationTests(unittest.TestCase):
    def test_nullable_production_numeric_values_recreate_missing_flags(self) -> None:
        flow = ProductionFlow.model_validate(
            {
                "ts": 1.0,
                "id.orig_h": "source",
                "id.orig_p": 40000,
                "id.resp_h": "target",
                "id.resp_p": 80,
                "proto": "tcp",
                "service": "-",
                "conn_state": "S0",
                "history": "S",
            }
        ).model_dump(by_alias=True)
        frame = pd.DataFrame([flow])
        frame["label"] = "-"
        frame["detailed-label"] = "-"
        cleaned = clean_flows(frame)
        self.assertEqual(cleaned.loc[0, "duration_missing"], 1)
        self.assertEqual(cleaned.loc[0, "orig_bytes_missing"], 1)
        self.assertEqual(cleaned.loc[0, "resp_bytes_missing"], 1)
        # A self-contained train-only fit proves that serialization through the
        # production schema preserves the missing flags; CI intentionally has
        # no access to the private serving bundle.
        preprocessor = fit_preprocessor(cleaned)
        features = transform(cleaned, preprocessor)
        self.assertEqual(features.loc[0, "duration_missing"], 1)
        self.assertEqual(features.loc[0, "orig_bytes_missing"], 1)
        self.assertEqual(features.loc[0, "resp_bytes_missing"], 1)

    def test_delivery_queue_retries_and_reports_run_counters(self) -> None:
        async def exercise() -> None:
            calls = 0

            def sender(_url, _document):
                nonlocal calls
                calls += 1
                if calls < 3:
                    raise ServiceRequestError("transient")
                return {"accepted": True}

            dispatcher = ObservationDispatcher(
                endpoint="http://collector/observe",
                queue_size=2,
                retry_attempts=3,
                retry_backoff_seconds=0,
                sender=sender,
            )
            await dispatcher.start()
            self.assertTrue(dispatcher.enqueue({"flow": {}}, run_id="demo-run"))
            await dispatcher.stop()
            self.assertEqual(
                dispatcher.metrics("demo-run"),
                {
                    "enqueued": 1,
                    "delivered": 1,
                    "retries": 2,
                    "terminal_failures": 0,
                    "queue_dropped": 0,
                    "queue_depth": 0,
                    "queue_capacity": 2,
                },
            )

        asyncio.run(exercise())

    def test_scientific_replay_is_validation_only_and_label_free_on_wire(self) -> None:
        catalog = load_scientific_replay(REPLAY)
        self.assertEqual(len(catalog.cases), 7)
        captured = {}
        case = catalog.case("ddos")

        def sender(url, document):
            captured["url"] = url
            captured["document"] = document
            predictions = [
                {
                    "flow_id": flow["uid"],
                    "predicted_label": case.expected_class,
                    "confidence": 0.99,
                    "entropy": 0.01,
                    "probabilities": {case.expected_class: 0.99, "Benign": 0.01},
                }
                for flow in document["flows"]
            ]
            return {
                "client_id": case.client_id,
                "model_digest": "a" * 64,
                "head_digest": "b" * 64,
                "schema_digest": "c" * 64,
                "predictions": predictions,
            }

        result = execute_replay_case(
            case=case, inference_url="http://inference/predict", sender=sender
        )
        self.assertTrue(result["correct"])
        self.assertFalse(result["request_contains_ground_truth"])
        self.assertEqual(captured["url"], "http://inference/predict")
        self.assertNotIn("expected_class", captured["document"])
        for flow in captured["document"]["flows"]:
            self.assertNotIn("label", flow)
            self.assertNotIn("detailed-label", flow)
            self.assertNotRegex(flow["id.orig_h"], r"^\d+\.\d+\.\d+\.\d+$")

    def test_zeek_normalizer_preserves_missing_numeric_values_and_forbids_labels(self) -> None:
        record = {
            "ts": 1.0,
            "uid": "C1",
            "id.orig_h": "10.0.0.1",
            "id.orig_p": 40000,
            "id.resp_h": "10.0.0.2",
            "id.resp_p": 80,
            "proto": "tcp",
            "conn_state": "S0",
        }
        flow = production_flow_from_zeek(record)
        self.assertIsNone(flow["duration"])
        self.assertIsNone(flow["orig_bytes"])
        self.assertIsNone(flow["resp_bytes"])
        self.assertNotIn("label", flow)

    def test_zeek_follower_reopens_rotated_conn_log(self) -> None:
        async def exercise() -> None:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "conn.log"
                path.write_text("first\n", encoding="utf-8")
                lines = follow_rotating_file(path, poll_seconds=0.001)
                self.assertEqual(await anext(lines), "first\n")
                path.rename(path.with_suffix(".log.1"))
                path.write_text("second\n", encoding="utf-8")
                self.assertEqual(await anext(lines), "second\n")
                await lines.aclose()

        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
