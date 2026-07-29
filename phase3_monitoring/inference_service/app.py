"""FastAPI inference boundary for the Phase 3 Minikube PoC."""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.graph_build import build_graph
from src.preprocess import clean_flows, transform

from .model_loader import ModelContractError, RuntimeBundle, load_runtime_bundle
from .schema import BatchPredictionResponse, FlowBatchRequest, FlowPrediction


logger = logging.getLogger(__name__)
GRAPH_PROTOCOL = "batch_local_graph"
app_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    app_state.clear()
    try:
        app_state["runtime"] = load_runtime_bundle()
        logger.info(
            "Inference runtime ready: model=%s schema=%s device=%s",
            app_state["runtime"].model_version,
            app_state["runtime"].feature_schema_digest,
            app_state["runtime"].device,
        )
    except (ModelContractError, KeyError, RuntimeError, ValueError) as error:
        # Keep the process live so /health/ready exposes a diagnosable failure,
        # but fail closed for every prediction.
        app_state["load_error"] = str(error)
        logger.error("Inference runtime is not ready: %s", error)
    yield
    app_state.clear()


app = FastAPI(lifespan=lifespan, title="FedKubeGNN Inference API")


def _runtime_or_503() -> RuntimeBundle:
    runtime = app_state.get("runtime")
    if isinstance(runtime, RuntimeBundle):
        return runtime
    detail = app_state.get("load_error", "Model runtime has not been initialized.")
    raise HTTPException(status_code=503, detail=detail)


@app.get("/health/live")
async def health_live():
    return {"status": "live"}


@app.get("/health/ready")
async def health_ready():
    runtime = _runtime_or_503()
    return {
        "status": "ready",
        "model_version": runtime.model_version,
        "feature_dim": len(runtime.feature_columns),
        "num_classes": len(runtime.class_to_idx),
        "feature_schema_digest": runtime.feature_schema_digest,
        "graph_protocol": GRAPH_PROTOCOL,
        "device": runtime.device,
    }


@app.post("/predict", response_model=BatchPredictionResponse)
async def predict_flows(request: FlowBatchRequest):
    if not request.flows:
        raise HTTPException(status_code=400, detail="Empty batch")

    runtime = _runtime_or_503()
    records = [
        {
            "ts": flow.ts,
            "uid": flow.uid,
            "id.orig_h": flow.id_orig_h,
            "id.orig_p": flow.id_orig_p,
            "id.resp_h": flow.id_resp_h,
            "id.resp_p": flow.id_resp_p,
            "proto": flow.proto,
            "service": flow.service,
            "duration": flow.duration,
            "orig_bytes": flow.orig_bytes,
            "resp_bytes": flow.resp_bytes,
            "conn_state": flow.conn_state,
            "local_orig": flow.local_orig,
            "local_resp": flow.local_resp,
            "missed_bytes": flow.missed_bytes,
            "history": flow.history,
            "orig_pkts": flow.orig_pkts,
            "orig_ip_bytes": flow.orig_ip_bytes,
            "resp_pkts": flow.resp_pkts,
            "resp_ip_bytes": flow.resp_ip_bytes,
            "tunnel_parents": flow.tunnel_parents,
            "label": flow.label,
            "detailed-label": flow.detailed_label,
        }
        for flow in request.flows
    ]

    try:
        clean = clean_flows(pd.DataFrame(records))
        features = transform(clean, runtime.preprocessor)
    except Exception as error:
        raise HTTPException(
            status_code=422,
            detail=f"Preprocessing failed: {error}",
        ) from error

    if tuple(runtime.preprocessor.feature_columns) != runtime.feature_columns:
        # The bundle is immutable by contract, but retain this guard against
        # accidental in-process mutation.
        raise HTTPException(
            status_code=503,
            detail="Runtime feature schema changed after startup.",
        )

    try:
        graph = build_graph(
            features,
            runtime.class_to_idx,
            runtime.feature_columns,
        ).to(runtime.device)
    except Exception as error:
        raise HTTPException(
            status_code=422,
            detail=f"Graph build failed: {error}",
        ) from error

    predictions: list[FlowPrediction] = []
    with torch.no_grad():
        probabilities = torch.softmax(runtime.model(graph), dim=-1).cpu()
        for index, flow in enumerate(request.flows):
            flow_probabilities = probabilities[index]
            entropy = -torch.sum(
                flow_probabilities * torch.log(flow_probabilities + 1e-12)
            ).item()
            confidence, predicted_index = torch.max(
                flow_probabilities,
                dim=-1,
            )
            predicted_label = runtime.idx_to_class[predicted_index.item()]
            probability_map = {
                runtime.idx_to_class[class_index]: float(
                    flow_probabilities[class_index].item()
                )
                for class_index in range(len(flow_probabilities))
            }
            flow_id = (
                flow.uid
                if flow.uid and flow.uid != "-"
                else (
                    f"{flow.id_orig_h}:{flow.id_orig_p}"
                    f"->{flow.id_resp_h}:{flow.id_resp_p}"
                )
            )
            predictions.append(
                FlowPrediction(
                    flow_id=flow_id,
                    predicted_label=predicted_label,
                    confidence=float(confidence.item()),
                    probabilities=probability_map,
                    entropy=float(entropy),
                )
            )

    return BatchPredictionResponse(
        model_version=runtime.model_version,
        feature_schema_digest=runtime.feature_schema_digest,
        graph_protocol=GRAPH_PROTOCOL,
        predictions=predictions,
    )
