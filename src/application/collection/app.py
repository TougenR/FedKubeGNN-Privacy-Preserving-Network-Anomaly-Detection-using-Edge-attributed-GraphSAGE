"""Flow collector and rolling-window orchestration boundary for the PoC."""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from src.application.alerting.policy import AlertPolicy, parse_boundaries
from src.application.api.schema import ProductionFlow
from src.application.collection.transport import ServiceRequestError, post_json
from src.application.graph_window.buffer import RollingWindowBuffer, RollingWindowConfig


class CollectorObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sensor_id: str
    source: Literal["zeek-json-v1", "ingress-adapter-v1"]
    flow: ProductionFlow


app_state: dict[str, Any] = {}


def _required_float(name: str) -> float:
    return float(os.environ[name])


def _required_int(name: str) -> int:
    return int(os.environ[name])


def _load_policy() -> AlertPolicy | None:
    if os.environ.get("ALERT_POLICY_ENABLED", "false").lower() != "true":
        return None
    return AlertPolicy(
        confidence_threshold=float(os.environ["ALERT_CONFIDENCE_THRESHOLD"]),
        confidence_boundaries=parse_boundaries(
            json.loads(os.environ["ALERT_CONFIDENCE_BOUNDARIES"])
        ),
        entropy_boundaries=parse_boundaries(
            json.loads(os.environ["ALERT_ENTROPY_BOUNDARIES"])
        ),
        class_severity=json.loads(os.environ["ALERT_CLASS_SEVERITY"]),
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    app_state.clear()
    try:
        app_state["window_config"] = RollingWindowConfig(
            duration_seconds=_required_float("WINDOW_DURATION_SECONDS"),
            max_flows=_required_int("WINDOW_MAX_FLOWS"),
            emit_stride_flows=_required_int("WINDOW_EMIT_STRIDE_FLOWS"),
            allowed_lateness_seconds=_required_float("WINDOW_ALLOWED_LATENESS_SECONDS"),
        )
        app_state["inference_url"] = os.environ["INFERENCE_URL"]
        app_state["alert_router_url"] = os.environ.get("ALERT_ROUTER_URL")
        app_state["policy"] = _load_policy()
        if app_state["policy"] is not None:
            app_state["entity_key"] = os.environ["ENTITY_HASH_KEY"].encode("utf-8")
            if not app_state["alert_router_url"]:
                raise ValueError("ALERT_ROUTER_URL is required when alerting is enabled.")
        app_state["buffers"] = {}
        app_state["inference_latency_ms"] = []
        app_state["observations"] = 0
        app_state["windows"] = 0
        app_state["events"] = 0
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        app_state["load_error"] = str(exc)
    yield
    app_state.clear()


app = FastAPI(title="FedKube Detection Flow Collector", lifespan=lifespan)


def _ready() -> None:
    if "window_config" not in app_state:
        raise HTTPException(
            status_code=503,
            detail=app_state.get("load_error", "Collector is not initialized."),
        )


@app.get("/health/live")
async def health_live() -> dict[str, str]:
    return {"status": "live"}


@app.get("/health/ready")
async def health_ready() -> dict[str, Any]:
    _ready()
    config = app_state["window_config"]
    return {
        "status": "ready",
        "window_seconds": config.duration_seconds,
        "max_flows": config.max_flows,
        "alerting": app_state["policy"] is not None,
    }


@app.get("/metrics")
async def metrics() -> dict[str, Any]:
    _ready()
    buffers = app_state["buffers"].values()
    latency = sorted(app_state["inference_latency_ms"])

    def percentile(fraction: float) -> float | None:
        if not latency:
            return None
        return latency[round((len(latency) - 1) * fraction)]
    dropped = sum(buffer.late_drop_count for buffer in buffers)
    evicted = sum(buffer.capacity_drop_count for buffer in buffers)
    return {
        "observations": app_state["observations"],
        "windows": app_state["windows"],
        "events": app_state["events"],
        "dropped_flows": dropped,
        "window_context_evictions": evicted,
        "inference_latency_ms_p50": percentile(0.50),
        "inference_latency_ms_p95": percentile(0.95),
    }


@app.post("/observe")
async def observe(observation: CollectorObservation) -> dict[str, Any]:
    _ready()
    buffers = app_state["buffers"]
    buffer = buffers.setdefault(
        observation.sensor_id,
        RollingWindowBuffer(
            sensor_id=observation.sensor_id,
            config=app_state["window_config"],
        ),
    )
    flow = observation.flow.model_dump(by_alias=True)
    snapshot = buffer.add(flow)
    app_state["observations"] += 1
    if snapshot is None:
        return {
            "accepted": True,
            "window_emitted": False,
            "flow_drop_rate": buffer.flow_drop_rate,
        }
    inference_request = {
        "sensor_id": snapshot.sensor_id,
        "window_id": snapshot.window_id,
        "flows": list(snapshot.flows),
    }
    started = time.perf_counter()
    try:
        response = post_json(app_state["inference_url"], inference_request)
    except ServiceRequestError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    latency_ms = (time.perf_counter() - started) * 1000
    app_state["inference_latency_ms"].append(latency_ms)
    app_state["windows"] += 1

    emitted = 0
    policy = app_state["policy"]
    if policy is not None:
        for index in snapshot.emission_indices:
            source_flow = snapshot.flows[index]
            prediction = response["predictions"][index]
            event = policy.event_for_prediction(
                sensor_id=snapshot.sensor_id,
                window_id=snapshot.window_id,
                entity=str(source_flow["id.orig_h"]),
                entity_key=app_state["entity_key"],
                flow_count=len(snapshot.flows),
                response=response,
                prediction=prediction,
            )
            if event is not None:
                try:
                    post_json(
                        app_state["alert_router_url"],
                        event.model_dump(by_alias=True, mode="json"),
                    )
                except ServiceRequestError as exc:
                    raise HTTPException(status_code=502, detail=str(exc)) from exc
                emitted += 1
    app_state["events"] += emitted
    predicted_counts: dict[str, int] = {}
    for prediction in response["predictions"]:
        label = str(prediction["predicted_label"])
        predicted_counts[label] = predicted_counts.get(label, 0) + 1
    return {
        "accepted": True,
        "window_emitted": True,
        "window_id": snapshot.window_id,
        "predicted_counts": predicted_counts,
        "events_emitted": emitted,
        "inference_latency_ms": latency_ms,
        "flow_drop_rate": buffer.flow_drop_rate,
    }
