"""Attacker-side console without access to model predictions or alerts."""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from src.application.collection.transport import (
    ServiceRequestError,
    delete_json,
    get_json,
    post_json,
)


class StartTrafficRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    events: int = Field(ge=1, le=50)
    interval_ms: int = Field(ge=4, le=60000)


state: dict[str, Any] = {}


def _read_token(variable: str) -> str:
    token = Path(os.environ[variable]).read_text(encoding="utf-8").strip()
    if len(token) < 32:
        raise ValueError(f"{variable} must contain at least 32 characters.")
    return token


@asynccontextmanager
async def lifespan(_: FastAPI):
    state.clear()
    try:
        state.update(
            agent_url=os.environ["TRAFFIC_AGENT_URL"].rstrip("/"),
            agent_token=_read_token("TRAFFIC_AGENT_TOKEN_FILE"),
            control_url=os.environ["COLLECTOR_CONTROL_URL"].rstrip("/"),
            observation_token=_read_token("OBSERVATION_TOKEN_FILE"),
            sensor_id=os.environ["TRAFFIC_SENSOR_ID"],
            identity={
                "generator_name": os.environ["GENERATOR_NAME"],
                "generator_source_ipv4": os.environ["GENERATOR_SOURCE_IPV4"],
                "generator_zone": os.environ["GENERATOR_ZONE"],
                "target_name": os.environ["TARGET_NAME"],
                "target_ipv4": os.environ["TARGET_IPV4"],
                "sensor_id": os.environ["TRAFFIC_SENSOR_ID"],
            },
        )
    except (KeyError, OSError, ValueError) as exc:
        state["load_error"] = str(exc)
    yield
    state.clear()


app = FastAPI(title="FedKube Private Attacker Console", lifespan=lifespan)


def _ready() -> None:
    if "identity" not in state:
        raise HTTPException(
            status_code=503,
            detail=state.get("load_error", "Attacker Console chưa sẵn sàng"),
        )


def _agent_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {state['agent_token']}"}


def _control_headers() -> dict[str, str]:
    return {"X-FedKube-Observation-Token": state["observation_token"]}


def _stage_status(
    *, count: int, active: bool, failed: int = 0, upstream: int = 0
) -> str:
    if failed:
        return "error"
    if count:
        return "acknowledged"
    if active or upstream:
        return "waiting"
    return "idle"


def _pipeline_snapshot(
    record: dict[str, Any], metrics: dict[str, Any]
) -> dict[str, Any]:
    """Translate private counters without exposing model or policy output."""
    active = record.get("status") in {"waiting-for-release", "running"}
    sent = int(record.get("succeeded", 0))
    execution_failures = int(record.get("failed", 0))
    evidence = list(metrics.get("zeek_evidence", []))[:50]
    observed = len(evidence)
    gateway = int(metrics.get("gateway_received", 0))
    received = int(metrics.get("received", 0))
    accepted = int(metrics.get("accepted", 0))
    windowed = int(metrics.get("windowed", 0))
    inferred = int(metrics.get("predicted", 0))
    stored = int(metrics.get("routed", 0))
    dropped = int(metrics.get("late_dropped", 0))
    duplicates = int(metrics.get("duplicates", 0))
    inference_failures = int(metrics.get("inference_failures", 0))
    sink_failures = int(metrics.get("alert_sink_failures", 0))

    def stage(
        key: str,
        label: str,
        count: int,
        *,
        upstream: int = 0,
        failed: int = 0,
    ) -> dict[str, Any]:
        return {
            "key": key,
            "label": label,
            "count": count,
            "status": _stage_status(
                count=count, active=active, failed=failed, upstream=upstream
            ),
        }

    return {
        "stages": [
            stage("agent", "Traffic Agent", sent, failed=execution_failures),
            stage("zeek", "Zeek", observed, upstream=sent),
            stage("shipper", "Shipper", gateway, upstream=observed),
            stage("gateway", "Internal NGINX", gateway, upstream=observed),
            stage(
                "collector",
                "Collector",
                accepted,
                upstream=received or gateway,
                failed=dropped + duplicates,
            ),
            stage("window", "Rolling Window", windowed, upstream=accepted),
            stage(
                "inference",
                "FedPer Inference",
                inferred,
                upstream=windowed,
                failed=inference_failures,
            ),
            stage(
                "router",
                "Alert Router / SOC",
                stored,
                upstream=inferred,
                failed=sink_failures,
            ),
        ],
        "counters": {
            "sent": sent,
            "observed": observed,
            "gateway": gateway,
            "received": received,
            "accepted": accepted,
            "windowed": windowed,
            "inferred": inferred,
            "stored": stored,
        },
        "failures": {
            "send": execution_failures,
            "dropped": dropped,
            "duplicates": duplicates,
            "inference": inference_failures,
            "sink": sink_failures,
        },
        "zeek_evidence": evidence,
    }


@app.get("/health/live")
async def health_live() -> dict[str, str]:
    return {"status": "live"}


@app.get("/health/ready")
async def health_ready() -> dict[str, str]:
    _ready()
    return {"status": "ready"}


@app.get("/api/config")
async def config() -> dict[str, Any]:
    _ready()
    # Deliberately excludes tokens, model identity, predictions and alerts.
    return {
        "console_schema_version": 1,
        "identity": dict(state["identity"]),
        "access_boundary": "attacker-only",
    }


@app.get("/api/profiles")
async def profiles() -> dict[str, Any]:
    _ready()
    try:
        return await asyncio.to_thread(
            get_json,
            f"{state['agent_url']}/v1/profiles",
            headers=_agent_headers(),
        )
    except ServiceRequestError as exc:
        raise HTTPException(
            status_code=502, detail="Traffic agent không khả dụng"
        ) from exc


@app.post("/api/runs/{profile_id}", status_code=202)
async def start_run(profile_id: str, request: StartTrafficRequest) -> dict[str, Any]:
    _ready()
    run_id: str | None = None
    try:
        record = await asyncio.to_thread(
            post_json,
            f"{state['agent_url']}/v1/runs",
            {"profile_id": profile_id, **request.model_dump()},
            headers=_agent_headers(),
        )
        run_id = str(record["run_id"])
        await asyncio.to_thread(
            post_json,
            f"{state['control_url']}/runs/register",
            {
                "run_id": run_id,
                "scenario_id": profile_id,
                "sensor_id": state["sensor_id"],
            },
            headers=_control_headers(),
        )
        return await asyncio.to_thread(
            post_json,
            f"{state['agent_url']}/v1/runs/{run_id}/release",
            {},
            headers=_agent_headers(),
        )
    except (KeyError, ServiceRequestError) as exc:
        if run_id is not None:
            try:
                await asyncio.to_thread(
                    delete_json,
                    f"{state['agent_url']}/v1/runs/current",
                    headers=_agent_headers(),
                )
            except ServiceRequestError:
                pass
        raise HTTPException(
            status_code=502, detail="Không thể bắt đầu traffic run"
        ) from exc


@app.get("/api/runs/current")
async def current_run() -> dict[str, Any]:
    _ready()
    try:
        body = await asyncio.to_thread(
            get_json,
            f"{state['agent_url']}/v1/runs/current",
            headers=_agent_headers(),
        )
        record = body.get("run")
        if not isinstance(record, dict):
            return {"run": None}
        metrics = await asyncio.to_thread(
            get_json,
            f"{state['control_url']}/runs/{record['run_id']}/metrics",
            headers=_control_headers(),
        )
        return {"run": {**record, "pipeline": _pipeline_snapshot(record, metrics)}}
    except (KeyError, ServiceRequestError) as exc:
        raise HTTPException(
            status_code=502, detail="Không đọc được trạng thái run"
        ) from exc


@app.delete("/api/runs/current")
async def stop_run() -> dict[str, Any]:
    _ready()
    try:
        return await asyncio.to_thread(
            delete_json,
            f"{state['agent_url']}/v1/runs/current",
            headers=_agent_headers(),
        )
    except ServiceRequestError as exc:
        raise HTTPException(
            status_code=502, detail="Không thể dừng traffic run"
        ) from exc


STATIC_DIR = Path(__file__).with_name("static")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
