from __future__ import annotations

import unittest

from src.application.evaluation.window_benchmark import select_candidate


class WindowBenchmarkTests(unittest.TestCase):
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
