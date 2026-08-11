"""Flow collector and rolling-window orchestration boundary for the PoC."""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.application.alerting.event import numeric_bucket
from src.application.alerting.policy import AlertPolicy, parse_boundaries
from src.application.api.schema import ProductionFlow
from src.application.collection.transport import ServiceRequestError, post_json
from src.application.graph_window.buffer import RollingWindowBuffer, RollingWindowConfig
from src.application.inference.fusion import policy_digest


class CollectorObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sensor_id: str
    source: Literal["zeek-json-v1", "ingress-adapter-v1"]
    run_id: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")
    scenario_id: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")
    flow: ProductionFlow


class LabRunRegistration(BaseModel):
    """Internal correlation metadata; it never becomes a model feature."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")
    scenario_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")
    sensor_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")


app_state: dict[str, Any] = {}


def _run_evidence_store() -> defaultdict[str, deque]:
    return app_state.setdefault(
        "run_zeek_evidence", defaultdict(lambda: deque(maxlen=50))
    )


def _append_zeek_evidence(run_id: str, flow: dict[str, Any]) -> None:
    """Retain bounded, IP-free UI evidence outside the model contract."""
    response_seen = bool((flow.get("resp_pkts") or 0) or (flow.get("resp_bytes") or 0))
    evidence = {
        "timestamp": float(flow["ts"]),
        "source": "attacker-vm",
        "target": "fixed-private-lab-target",
        "port": int(flow["id.resp_p"]),
        "protocol": str(flow["proto"]),
        "service": str(flow.get("service") or "-"),
        "connection_state": str(flow["conn_state"]),
        "history": str(flow.get("history") or "-"),
        "response_behavior": "response-observed" if response_seen else "no-response",
        "orig_packets": flow.get("orig_pkts"),
        "resp_packets": flow.get("resp_pkts"),
        "orig_bytes": flow.get("orig_bytes"),
        "resp_bytes": flow.get("resp_bytes"),
    }
    evidence_buffer = _run_evidence_store()[run_id]
    evidence_buffer.append(evidence)
    evidence["density"] = len(evidence_buffer)


def _increment_metric(values: dict[str, int], key: str) -> None:
    values[key] = values.get(key, 0) + 1


def _required_float(name: str) -> float:
    return float(os.environ[name])


def _required_int(name: str) -> int:
    return int(os.environ[name])


def _load_policy() -> AlertPolicy | None:
    if os.environ.get("ALERT_POLICY_ENABLED", "false").lower() != "true":
        return None
    decision_source = os.environ.get("ALERT_DECISION_SOURCE", "fusion")
    if decision_source not in {"fusion", "trusted-shadow"}:
        raise ValueError("ALERT_DECISION_SOURCE must be fusion or trusted-shadow.")
    class_thresholds = None
    if decision_source == "fusion":
        fusion_document = json.loads(
            Path(os.environ["FUSION_POLICY_PATH"]).read_text(encoding="utf-8")
        )
        if fusion_document.get("policy_digest") != policy_digest(fusion_document):
            raise ValueError("Fusion policy digest mismatch in collector.")
        class_thresholds = fusion_document.get("class_alert_thresholds")
        if not isinstance(class_thresholds, dict):
            raise ValueError("Fusion policy has no class alert thresholds.")
    return AlertPolicy(
        confidence_threshold=float(os.environ["ALERT_CONFIDENCE_THRESHOLD"]),
        class_confidence_thresholds=class_thresholds,
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
        app_state["alert_decision_source"] = os.environ.get(
            "ALERT_DECISION_SOURCE", "fusion"
        )
        if app_state["policy"] is not None:
            app_state["entity_key"] = os.environ["ENTITY_HASH_KEY"].encode("utf-8")
            if not app_state["alert_router_url"]:
                raise ValueError(
                    "ALERT_ROUTER_URL is required when alerting is enabled."
                )
        app_state["buffers"] = {}
        app_state["inference_latency_ms"] = []
        app_state["observations"] = 0
        app_state["windows"] = 0
        app_state["events"] = 0
        monitor_size = int(os.environ.get("MONITOR_BUFFER_SIZE", "500"))
        if not 10 <= monitor_size <= 1000:
            raise ValueError("MONITOR_BUFFER_SIZE must be between 10 and 1000.")
        app_state["monitor_events"] = deque(maxlen=monitor_size)
        app_state["monitor_sequence"] = 0
        app_state["run_metrics"] = defaultdict(
            lambda: {
                "gateway_received": 0,
                "received": 0,
                "accepted": 0,
                "windowed": 0,
                "predicted": 0,
                "routed": 0,
                "late_dropped": 0,
                "inference_failures": 0,
                "alert_sink_failures": 0,
                "duplicates": 0,
            }
        )
        app_state["run_zeek_evidence"] = defaultdict(lambda: deque(maxlen=50))
        app_state["flow_runs"] = defaultdict(dict)
        app_state["completed_uids"] = deque(maxlen=5000)
        app_state["active_runs"] = {}
        token_path = os.environ.get("OBSERVATION_TOKEN_FILE")
        if token_path:
            token = Path(token_path).read_text(encoding="utf-8").strip()
            if len(token) < 32:
                raise ValueError(
                    "Observation token must contain at least 32 characters."
                )
            app_state["observation_token"] = token
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


@app.get("/monitor/events")
async def monitor_events(after: int = 0, limit: int = 100) -> dict[str, Any]:
    _ready()
    if after < 0 or not 1 <= limit <= 200:
        raise HTTPException(status_code=422, detail="invalid monitor cursor or limit")
    selected = [
        event for event in app_state["monitor_events"] if int(event["sequence"]) > after
    ][:limit]
    metric_values = await metrics()
    return {
        "events": selected,
        "next_cursor": selected[-1]["sequence"] if selected else after,
        "metrics": metric_values,
    }


@app.get("/runs/{run_id}/metrics")
async def run_metrics(run_id: str) -> dict[str, Any]:
    _ready()
    if not run_id or len(run_id) > 64:
        raise HTTPException(status_code=422, detail="invalid run ID")
    metrics_by_run = app_state["run_metrics"]
    values = dict(metrics_by_run.get(run_id, {}))
    return {
        "run_id": run_id,
        "available": bool(values),
        **values,
        "zeek_evidence": list(_run_evidence_store().get(run_id, ())),
    }


def _require_observation_token(supplied: str | None) -> None:
    expected = app_state.get("observation_token")
    if not isinstance(expected, str) or not hmac.compare_digest(
        supplied or "", expected
    ):
        raise HTTPException(status_code=401, detail="invalid observation token")


@app.post("/runs/register")
async def register_run(registration: LabRunRegistration) -> dict[str, str]:
    """Associate subsequent label-free Zeek flows with the one active lab run."""
    _ready()
    app_state.setdefault("active_runs", {})[registration.sensor_id] = {
        "run_id": registration.run_id,
        "scenario_id": registration.scenario_id,
    }
    # A run is an independent experiment. Reusing sensor-local graph context
    # would mix earlier scenario flows into the first windows of the new run.
    app_state.setdefault("buffers", {}).pop(registration.sensor_id, None)
    app_state.setdefault("flow_runs", defaultdict(dict)).pop(
        registration.sensor_id, None
    )
    # Materialize zero-valued metrics so the console can distinguish an active
    # run with no captured flows from an unavailable metrics endpoint.
    app_state["run_metrics"][registration.run_id]
    _run_evidence_store()[registration.run_id].clear()
    return {
        "status": "registered",
        "run_id": registration.run_id,
        "sensor_id": registration.sensor_id,
    }


@app.post("/private/runs/register")
async def private_register_run(
    registration: LabRunRegistration,
    x_fedkube_observation_token: str | None = Header(default=None),
) -> dict[str, str]:
    _require_observation_token(x_fedkube_observation_token)
    return await register_run(registration)


@app.get("/private/runs/{run_id}/metrics")
async def private_run_metrics(
    run_id: str,
    x_fedkube_observation_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_observation_token(x_fedkube_observation_token)
    return await run_metrics(run_id)


@app.post("/observe")
async def observe(
    observation: CollectorObservation,
    x_fedkube_observation_token: str | None = Header(default=None),
    x_fedkube_gateway_hop: str | None = Header(default=None),
) -> dict[str, Any]:
    _ready()
    expected_token = app_state.get("observation_token")
    if isinstance(expected_token, str):
        _require_observation_token(x_fedkube_observation_token)
    active = app_state.setdefault("active_runs", {}).get(observation.sensor_id, {})
    resolved_run_id = observation.run_id or active.get("run_id")
    run_values = (
        app_state["run_metrics"][resolved_run_id]
        if resolved_run_id is not None
        else None
    )
    if run_values is not None:
        _increment_metric(run_values, "received")
        if x_fedkube_gateway_hop == "internal-nginx-v1":
            _increment_metric(run_values, "gateway_received")
    buffers = app_state["buffers"]
    buffer = buffers.setdefault(
        observation.sensor_id,
        RollingWindowBuffer(
            sensor_id=observation.sensor_id,
            config=app_state["window_config"],
        ),
    )
    flow = observation.flow.model_dump(by_alias=True)
    if resolved_run_id is not None and observation.source == "zeek-json-v1":
        _append_zeek_evidence(resolved_run_id, flow)
    uid = str(flow.get("uid", "-"))
    completed_uids = app_state.setdefault("completed_uids", deque(maxlen=5000))
    if uid != "-" and uid in completed_uids:
        if run_values is not None:
            _increment_metric(run_values, "duplicates")
        return {"accepted": True, "duplicate": True, "window_emitted": False}
    late_before = buffer.late_drop_count
    snapshot = buffer.add(flow)
    app_state["observations"] += 1
    if buffer.late_drop_count > late_before:
        if run_values is not None:
            _increment_metric(run_values, "late_dropped")
        return {
            "accepted": False,
            "window_emitted": False,
            "late_dropped": True,
            "flow_drop_rate": buffer.flow_drop_rate,
        }
    if run_values is not None:
        _increment_metric(run_values, "accepted")
    if uid != "-":
        app_state.setdefault("flow_runs", defaultdict(dict))[observation.sensor_id][
            uid
        ] = resolved_run_id
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
    for index in snapshot.emission_indices:
        emitted_uid = str(snapshot.flows[index].get("uid", "-"))
        emitted_run = app_state["flow_runs"][snapshot.sensor_id].get(emitted_uid)
        if emitted_run is not None:
            values = app_state["run_metrics"][emitted_run]
            _increment_metric(values, "windowed")
    started = time.perf_counter()
    try:
        response = await asyncio.to_thread(
            post_json, app_state["inference_url"], inference_request
        )
    except ServiceRequestError as exc:
        for index in snapshot.emission_indices:
            emitted_uid = str(snapshot.flows[index].get("uid", "-"))
            emitted_run = app_state["flow_runs"][snapshot.sensor_id].pop(
                emitted_uid, None
            )
            if emitted_run is not None:
                _increment_metric(
                    app_state["run_metrics"][emitted_run], "inference_failures"
                )
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    latency_ms = (time.perf_counter() - started) * 1000
    app_state["inference_latency_ms"].append(latency_ms)
    app_state["windows"] += 1

    emitted = 0
    alerted_indices: set[int] = set()
    policy = app_state["policy"]
    if policy is not None:
        for index in snapshot.emission_indices:
            source_flow = snapshot.flows[index]
            prediction = response["predictions"][index]
            policy_prediction = prediction
            if app_state["alert_decision_source"] == "trusted-shadow":
                policy_prediction = {
                    **prediction["trusted_prediction"],
                    "trusted_prediction": prediction["trusted_prediction"],
                    "fused_predicted_label": prediction["predicted_label"],
                    "head_disagreement_count": prediction["head_disagreement_count"],
                    "alert_decision_source": "trusted-shadow",
                }
            else:
                policy_prediction = {
                    **prediction,
                    "fused_predicted_label": prediction["predicted_label"],
                    "alert_decision_source": "fusion",
                }
            event = policy.detection_event(
                sensor_id=snapshot.sensor_id,
                window_id=snapshot.window_id,
                entity=str(source_flow["id.orig_h"]),
                entity_key=app_state["entity_key"],
                flow_count=len(snapshot.flows),
                response=response,
                prediction=policy_prediction,
            )
            try:
                await asyncio.to_thread(
                    post_json,
                    app_state["alert_router_url"],
                    event.model_dump(by_alias=True, mode="json"),
                )
            except ServiceRequestError:
                emitted_uid = str(source_flow.get("uid", "-"))
                emitted_run = app_state["flow_runs"][snapshot.sensor_id].get(
                    emitted_uid
                )
                if emitted_run is not None:
                    _increment_metric(
                        app_state["run_metrics"][emitted_run],
                        "alert_sink_failures",
                    )
                # Prediction already succeeded. Do not make an upstream retry
                # add this flow to the rolling buffer a second time merely
                # because the downstream evidence sink was unavailable.
                continue
            emitted_uid = str(source_flow.get("uid", "-"))
            emitted_run = app_state["flow_runs"][snapshot.sensor_id].get(emitted_uid)
            if emitted_run is not None:
                values = app_state["run_metrics"][emitted_run]
                _increment_metric(values, "routed")
            if event.is_alert:
                emitted += 1
                alerted_indices.add(index)
    app_state["events"] += emitted
    for index in snapshot.emission_indices:
        source_flow = snapshot.flows[index]
        emitted_uid = str(source_flow.get("uid", "-"))
        emitted_run = app_state["flow_runs"][snapshot.sensor_id].pop(emitted_uid, None)
        if emitted_run is not None:
            _increment_metric(app_state["run_metrics"][emitted_run], "predicted")
        if emitted_uid != "-":
            completed_uids.append(emitted_uid)
        prediction = response["predictions"][index]
        app_state["monitor_sequence"] = app_state.get("monitor_sequence", 0) + 1
        predicted_class = str(prediction["predicted_label"])
        monitor_event = {
            "sequence": app_state["monitor_sequence"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sensor_id": snapshot.sensor_id,
            "client_id": str(response["client_id"]),
            "window_id": snapshot.window_id,
            "predicted_class": predicted_class,
            "is_alert": index in alerted_indices,
            "severity": (
                str(policy.class_severity.get(predicted_class, "none"))
                if policy is not None
                else "not-configured"
            ),
            "confidence_bucket": (
                numeric_bucket(
                    float(prediction["confidence"]), policy.confidence_boundaries
                )
                if policy is not None
                else "not-configured"
            ),
            "entropy_bucket": (
                numeric_bucket(float(prediction["entropy"]), policy.entropy_boundaries)
                if policy is not None
                else "not-configured"
            ),
            "flow_count": len(snapshot.flows),
            "inference_latency_ms": round(latency_ms, 3),
            "model_digest": str(response["model_digest"]),
            "head_digest": str(response["head_digest"]),
            "schema_digest": str(response["schema_digest"]),
            "decision_mode": str(response.get("decision_mode", "trusted-head-v1")),
            "fusion_policy_digest": response.get("fusion_policy_digest"),
            "alert_decision_source": app_state.get("alert_decision_source", "fusion"),
            "trusted_predicted_class": str(
                prediction.get("trusted_prediction", {}).get(
                    "predicted_label", predicted_class
                )
            ),
            "head_disagreement_count": int(
                prediction.get("head_disagreement_count", 0)
            ),
            "head_predictions": {
                str(head): {
                    "predicted_label": str(value["predicted_label"]),
                    "confidence_bucket": (
                        numeric_bucket(
                            float(value["confidence"]),
                            policy.confidence_boundaries,
                        )
                        if policy is not None
                        else "not-configured"
                    ),
                }
                for head, value in prediction.get("head_predictions", {}).items()
            },
        }
        app_state.setdefault("monitor_events", deque(maxlen=500)).append(monitor_event)
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
