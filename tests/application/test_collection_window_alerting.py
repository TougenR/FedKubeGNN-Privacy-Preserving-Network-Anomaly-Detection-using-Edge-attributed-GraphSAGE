from __future__ import annotations

import unittest
import json
import asyncio
from datetime import datetime, timezone
from io import BytesIO
from unittest.mock import patch

import pandas as pd

from src.application.alerting.event import DetectionEvent, entity_hash, numeric_bucket
from src.application.alerting.policy import AlertPolicy
from src.application.alerting.privacy import validate_elasticsearch_document
from src.application.alerting.elasticsearch import (
    ElasticsearchSettings,
    ElasticsearchSink,
)
from src.application.collection.zeek_reader import ZeekRecordError, parse_zeek_json
from src.application.collection.app import (
    CollectorObservation,
    app_state as collector_state,
    observe,
)
from src.application.graph_window.buffer import (
    RollingWindowBuffer,
    RollingWindowConfig,
)
from src.application.graph_window.graph_builder import build_inference_graph


class CollectionWindowAlertingTests(unittest.TestCase):
    def tearDown(self) -> None:
        collector_state.clear()

    def test_collector_emits_label_free_window_to_inference(self) -> None:
        collector_state.update(
            {
                "window_config": RollingWindowConfig(
                    duration_seconds=5,
                    max_flows=50,
                    emit_stride_flows=1,
                    allowed_lateness_seconds=1,
                ),
                "buffers": {},
                "inference_url": "http://inference/predict",
                "alert_router_url": None,
                "policy": None,
                "inference_latency_ms": [],
                "observations": 0,
                "windows": 0,
                "events": 0,
            }
        )
        observation = CollectorObservation.model_validate(
            {
                "sensor_id": "sensor-34-1",
                "source": "zeek-json-v1",
                "flow": {
                    "ts": 1.0,
                    "uid": "flow-1",
                    "id.orig_h": "10.0.0.1",
                    "id.orig_p": 12345,
                    "id.resp_h": "10.0.0.2",
                    "id.resp_p": 80,
                    "proto": "tcp",
                    "conn_state": "S0",
                },
            }
        )
        captured = {}

        def fake_post(url, document):
            captured["url"] = url
            captured["document"] = document
            return {
                "client_id": "34-1",
                "model_digest": "a" * 64,
                "head_digest": "b" * 64,
                "schema_digest": "c" * 64,
                "graph_protocol": "rolling-window-v1",
                "predictions": [
                    {
                        "flow_id": "flow-1",
                        "predicted_label": "Benign",
                        "confidence": 0.99,
                        "entropy": 0.01,
                        "probabilities": {"Benign": 0.99},
                    }
                ],
            }

        with patch("src.application.collection.app.post_json", side_effect=fake_post):
            response = asyncio.run(observe(observation))
        self.assertTrue(response["window_emitted"])
        self.assertEqual(captured["url"], "http://inference/predict")
        self.assertNotIn("label", captured["document"]["flows"][0])
        self.assertNotIn("detailed-label", captured["document"]["flows"][0])
        monitor_event = list(collector_state["monitor_events"])[0]
        self.assertEqual(monitor_event["predicted_class"], "Benign")
        self.assertEqual(monitor_event["sensor_id"], "sensor-34-1")
        self.assertNotIn("id.orig_h", monitor_event)
        self.assertNotIn("probabilities", monitor_event)
        self.assertNotIn("ground_truth", monitor_event)

    def test_elasticsearch_sink_sends_only_validated_document(self) -> None:
        captured = {}

        class Response:
            status = 201

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return None

            def read(self):
                return BytesIO(b'{"_id":"event-1"}').read()

        def opener(request, *, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return Response()

        sink = ElasticsearchSink(
            ElasticsearchSettings(
                endpoint="http://elasticsearch:9200",
                index="fedper-detections",
                api_key="secret-not-in-document",
            ),
            opener=opener,
        )
        document = {"event_type": "fedper_detection"}
        self.assertEqual(sink.index_event(document), "event-1")
        self.assertEqual(json.loads(captured["request"].data), document)
        self.assertNotIn("secret-not-in-document", captured["request"].data.decode())

    def test_zeek_reader_forbids_evaluation_labels(self) -> None:
        base = (
            '{"ts":1,"id.orig_h":"10.0.0.1","id.orig_p":1,'
            '"id.resp_h":"10.0.0.2","id.resp_p":2,'
            '"proto":"tcp","conn_state":"S0"}'
        )
        self.assertEqual(parse_zeek_json(base)["proto"], "tcp")
        with self.assertRaises(ZeekRecordError):
            parse_zeek_json(base[:-1] + ',"detailed-label":"C&C"}')

    def test_rolling_window_is_event_time_ordered_and_reports_late_drop(self) -> None:
        config = RollingWindowConfig(
            duration_seconds=10,
            max_flows=3,
            emit_stride_flows=2,
            allowed_lateness_seconds=2,
        )
        buffer = RollingWindowBuffer(sensor_id="sensor-1", config=config)
        self.assertIsNone(buffer.add({"ts": 10.0, "uid": "a"}))
        snapshot = buffer.add({"ts": 9.0, "uid": "b"})
        self.assertEqual([flow["uid"] for flow in snapshot.flows], ["b", "a"])
        self.assertEqual(snapshot.emission_indices, (0, 1))
        self.assertIsNone(buffer.add({"ts": 7.0, "uid": "too-late"}))
        self.assertEqual(buffer.late_drop_count, 1)

    def test_flush_emits_only_tail_and_capacity_eviction_is_not_input_drop(self) -> None:
        buffer = RollingWindowBuffer(
            sensor_id="sensor-1",
            config=RollingWindowConfig(
                duration_seconds=60,
                max_flows=2,
                emit_stride_flows=2,
                allowed_lateness_seconds=0,
            ),
        )
        self.assertIsNone(buffer.add({"ts": 1.0, "uid": "a"}))
        first = buffer.add({"ts": 2.0, "uid": "b"})
        self.assertEqual(first.emission_indices, (0, 1))
        self.assertIsNone(buffer.add({"ts": 3.0, "uid": "c"}))
        tail = buffer.flush()
        self.assertEqual([flow["uid"] for flow in tail.flows], ["b", "c"])
        self.assertEqual(tail.emission_indices, (1,))
        self.assertEqual(buffer.flow_drop_rate, 0.0)
        self.assertGreater(buffer.capacity_eviction_rate, 0.0)

    def test_graph_namespaces_nodes_by_sensor(self) -> None:
        frame = pd.DataFrame(
            {
                "id.orig_h": ["10.0.0.1"],
                "id.resp_h": ["10.0.0.2"],
                "feature_a": [1.0],
            }
        )
        graph = build_inference_graph(frame, ["feature_a"], sensor_id="sensor-1")
        self.assertEqual(
            graph.node_ids, ["sensor-1::10.0.0.1", "sensor-1::10.0.0.2"]
        )
        self.assertFalse(hasattr(graph, "edge_label"))

    def test_detection_event_contains_no_raw_network_or_tensor_fields(self) -> None:
        digest = "a" * 64
        document = DetectionEvent(
            **{
                "@timestamp": datetime.now(timezone.utc),
                "sensor_id": "sensor-1",
                "client_id": "1-1",
                "window_id": "window-1",
                "predicted_class": "C&C",
                "is_alert": True,
                "severity": "medium",
                "confidence_bucket": numeric_bucket(0.92, (0.0, 0.9, 0.95, 1.0)),
                "entropy_bucket": numeric_bucket(0.2, (0.0, 0.5, 1.0)),
                "flow_count": 10,
                "entity_hash": entity_hash(
                    key=b"test-key", sensor_id="sensor-1", entity="10.0.0.1"
                ),
                "model_digest": digest,
                "head_digest": digest,
                "schema_digest": digest,
            }
        ).model_dump(by_alias=True, mode="json")
        validate_elasticsearch_document(document)
        document["id.orig_h"] = "10.0.0.1"
        with self.assertRaises(ValueError):
            validate_elasticsearch_document(document)

    def test_policy_indexes_all_decisions_but_only_qualifies_selected_alerts(self) -> None:
        policy = AlertPolicy(
            confidence_threshold=0.85,
            confidence_boundaries=(0.0, 0.5, 0.85, 0.95, 1.0),
            entropy_boundaries=(0.0, 0.25, 0.5, 1.0),
            class_severity={"C&C": "high"},
        )
        common = {
            "sensor_id": "sensor-34-1",
            "window_id": "window-1",
            "entity": "10.0.0.1",
            "entity_key": b"test-key",
            "flow_count": 10,
            "response": {
                "client_id": "34-1",
                "model_digest": "a" * 64,
                "head_digest": "b" * 64,
                "schema_digest": "c" * 64,
            },
        }
        benign = policy.detection_event(
            **common,
            prediction={"predicted_label": "Benign", "confidence": 0.99, "entropy": 0.01},
        )
        self.assertFalse(benign.is_alert)
        self.assertEqual(benign.severity, "none")
        self.assertIsNone(
            policy.event_for_prediction(
                **common,
                prediction={"predicted_label": "Benign", "confidence": 0.99, "entropy": 0.01},
            )
        )
        attack = policy.detection_event(
            **common,
            prediction={"predicted_label": "C&C", "confidence": 0.92, "entropy": 0.2},
        )
        self.assertTrue(attack.is_alert)
        self.assertEqual(attack.severity, "high")


if __name__ == "__main__":
    unittest.main()
