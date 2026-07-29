from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch
from fastapi import HTTPException

from phase3_monitoring.inference_service.app import (
    app_state,
    health_ready,
    predict_flows,
)
from phase3_monitoring.inference_service.model_loader import RuntimeBundle
from phase3_monitoring.inference_service.schema import FlowBatchRequest


class _Graph:
    def to(self, _device):
        return self


class _Model:
    def __call__(self, _graph):
        return torch.tensor([[0.1, 1.9]], dtype=torch.float32)


def _runtime():
    return RuntimeBundle(
        model=_Model(),
        preprocessor=SimpleNamespace(feature_columns=["feature_a"]),
        class_to_idx={"Benign": 0, "C&C": 1},
        idx_to_class={0: "Benign", 1: "C&C"},
        feature_columns=("feature_a",),
        feature_schema_digest="a" * 64,
        model_version="test-model",
        checkpoint_path="/tmp/model.pt",
        preprocessor_path="/tmp/preprocessor.pkl",
        device="cpu",
    )


def _request():
    return FlowBatchRequest(
        flows=[
            {
                "ts": 1.0,
                "uid": "flow-1",
                "id.orig_h": "10.0.0.1",
                "id.orig_p": 12345,
                "id.resp_h": "10.0.0.2",
                "id.resp_p": 80,
                "proto": "tcp",
                "conn_state": "S0",
            }
        ]
    )


class InferenceApiTests(unittest.TestCase):
    def tearDown(self):
        app_state.clear()

    def test_readiness_fails_closed_without_validated_runtime(self):
        app_state["load_error"] = "feature contract mismatch"
        with self.assertRaises(HTTPException) as raised:
            asyncio.run(health_ready())
        self.assertEqual(raised.exception.status_code, 503)
        self.assertIn("feature contract mismatch", raised.exception.detail)

    def test_prediction_includes_model_and_schema_provenance(self):
        app_state["runtime"] = _runtime()
        transformed = pd.DataFrame(
            {
                "id.orig_h": ["10.0.0.1"],
                "id.resp_h": ["10.0.0.2"],
                "feature_a": [1.0],
                "detailed-label": ["Benign"],
            }
        )
        with (
            patch(
                "phase3_monitoring.inference_service.app.clean_flows",
                return_value=pd.DataFrame(),
            ),
            patch(
                "phase3_monitoring.inference_service.app.transform",
                return_value=transformed,
            ),
            patch(
                "phase3_monitoring.inference_service.app.build_graph",
                return_value=_Graph(),
            ),
        ):
            response = asyncio.run(predict_flows(_request()))

        self.assertEqual(response.model_version, "test-model")
        self.assertEqual(response.feature_schema_digest, "a" * 64)
        self.assertEqual(response.graph_protocol, "batch_local_graph")
        self.assertEqual(len(response.predictions), 1)
        self.assertEqual(response.predictions[0].predicted_label, "C&C")
        self.assertGreater(response.predictions[0].entropy, 0.0)


if __name__ == "__main__":
    unittest.main()
