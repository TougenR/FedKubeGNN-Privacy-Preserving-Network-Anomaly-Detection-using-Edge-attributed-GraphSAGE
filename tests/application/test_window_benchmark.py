from __future__ import annotations

import unittest
from types import SimpleNamespace

import pandas as pd

from src.application.evaluation.window_benchmark import (
    alert_threshold_tradeoff,
    preprocess_validation_frames,
    select_candidate,
)
from src.application.graph_window.graph_builder import preprocess_production_flows
from src.application.evaluation.locked_window_test import validate_locked_protocol
from src.core.preprocess import clean_flows, fit_preprocessor


class WindowBenchmarkTests(unittest.TestCase):
    def test_locked_test_requires_exact_validation_selected_protocol(self) -> None:
        report = {
            "kind": "rolling_window_validation_selection",
            "bundle_id": "research",
            "dataset_digest": "d" * 64,
            "selected": {
                "selection_split": "validation",
                "graph_protocol": "rolling-window-v1:test",
                "duration_seconds": 60,
                "max_flows": 50,
                "emit_stride_flows": 1,
                "allowed_lateness_seconds": 1,
            },
        }
        manifest = {
            "source_research_bundle": {"bundle_id": "research"},
            "dataset_digest": "d" * 64,
            "graph_protocol": "rolling-window-v1:test",
            "rolling_window_protocol": {
                "duration_seconds": 60,
                "max_flows": 50,
                "emit_stride_flows": 1,
                "allowed_lateness_seconds": 1,
            },
        }

        protocol = validate_locked_protocol(
            serving_manifest=manifest, window_report=report
        )
        self.assertEqual(protocol["max_flows"], 50)
        manifest["graph_protocol"] = "rolling-window-v1:changed"
        with self.assertRaisesRegex(Exception, "differs from validation lock"):
            validate_locked_protocol(serving_manifest=manifest, window_report=report)

    def test_alert_tradeoff_reports_false_alert_and_detection_rates(self) -> None:
        rows = alert_threshold_tradeoff(
            truth=[0, 0, 1, 2],
            predictions=[0, 1, 1, 0],
            confidence=[0.99, 0.80, 0.90, 0.95],
            benign_index=0,
            thresholds=[0.0, 0.85],
        )

        self.assertEqual(rows[0]["benign_false_alert_rate"], 0.5)
        self.assertEqual(rows[0]["malicious_alert_recall"], 0.5)
        self.assertEqual(rows[0]["correct_class_alert_recall"], 0.5)
        self.assertEqual(rows[1]["benign_false_alert_rate"], 0.0)
        self.assertEqual(rows[1]["malicious_alert_recall"], 0.5)

    def test_one_pass_preprocessing_matches_per_window_transform(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "uid": "test-flow",
                    "ts": 1.0,
                    "id.orig_h": "10.0.0.1",
                    "id.orig_p": 12345,
                    "id.resp_h": "10.0.0.2",
                    "id.resp_p": 80,
                    "proto": "tcp",
                    "service": "-",
                    "duration": 2.0,
                    "orig_bytes": 10,
                    "resp_bytes": 20,
                    "conn_state": "S0",
                    "local_orig": "-",
                    "local_resp": "-",
                    "missed_bytes": 0,
                    "history": "S",
                    "orig_pkts": 1,
                    "orig_ip_bytes": 20,
                    "resp_pkts": 2,
                    "resp_ip_bytes": 40,
                    "tunnel_parents": "-",
                    "label": "Malicious",
                    "detailed-label": "Attack",
                    "source_edge_index": 17,
                }
            ]
        )
        preprocessor = fit_preprocessor(clean_flows(frame))
        bundle = SimpleNamespace(preprocessor=preprocessor)
        expected = preprocess_production_flows(
            frame.to_dict(orient="records"), bundle.preprocessor
        )
        actual = preprocess_validation_frames(bundle=bundle, frames={"1-1": frame})[
            "1-1"
        ]

        model_columns = [
            "id.orig_h",
            "id.resp_h",
            "ts",
            *preprocessor.feature_columns,
        ]
        pd.testing.assert_frame_equal(
            actual[model_columns], expected[model_columns], check_dtype=True
        )
        self.assertEqual(actual["detailed-label"].tolist(), ["Attack"])
        self.assertEqual(actual["source_edge_index"].tolist(), [17])

    def test_selection_uses_validation_macro_f1_then_latency(self) -> None:
        base = {
            "duration_seconds": 5.0,
            "max_flows": 50,
            "emit_stride_flows": 1,
            "allowed_lateness_seconds": 1.0,
            "metrics": {"macro_f1_fixed": 0.9},
            "inference_latency_ms": {"p95": 12.0},
            "late_event_probe": {},
        }
        better_latency = {
            **base,
            "duration_seconds": 15.0,
            "inference_latency_ms": {"p95": 8.0},
        }
        lower_score = {
            **base,
            "metrics": {"macro_f1_fixed": 0.89},
            "inference_latency_ms": {"p95": 1.0},
        }
        selected = select_candidate([base, lower_score, better_latency])
        self.assertEqual(selected["duration_seconds"], 15.0)
        self.assertEqual(selected["selection_split"], "validation")
        self.assertEqual(
            selected["graph_protocol"],
            "rolling-window-v1:duration=15s:max-flows=50:stride=1:lateness=1s",
        )


if __name__ == "__main__":
    unittest.main()
