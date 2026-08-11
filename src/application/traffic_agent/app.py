"""Authenticated private API for the bounded traffic agent."""

from __future__ import annotations

import hmac
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.application.traffic_agent.catalog import (
    load_profile_catalog,
    load_target_catalog,
)
from src.application.traffic_agent.executor import (
    TrafficExecutor,
    TrafficProfileDisabledError,
    TrafficRunConflictError,
)


class StartTrafficRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")
    events: int | None = Field(default=None, ge=1, le=50)
    interval_ms: int | None = Field(default=None, ge=4, le=60000)


state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    state.clear()
    try:
        catalog = load_profile_catalog(
            os.environ.get(
                "TRAFFIC_PROFILE_CONFIG",
                "/app/configs/application/scientific-traffic-profiles.yaml",
            )
        )
        targets = load_target_catalog(os.environ["TRAFFIC_TARGET_CONFIG"])
        token = (
            Path(os.environ["TRAFFIC_AGENT_TOKEN_FILE"])
            .read_text(encoding="utf-8")
            .strip()
        )
        if len(token) < 32:
            raise ValueError("Traffic-agent token must contain at least 32 characters.")
        state["token"] = token
        state["executor"] = TrafficExecutor(catalog=catalog, targets=targets)
    except (KeyError, OSError, ValueError) as exc:
        state["load_error"] = str(exc)
    yield
    executor = state.get("executor")
    if isinstance(executor, TrafficExecutor):
        await executor.stop()
    state.clear()


app = FastAPI(title="FedKube Private Traffic Agent", lifespan=lifespan)


def _ready() -> TrafficExecutor:
    executor = state.get("executor")
    if not isinstance(executor, TrafficExecutor):
        raise HTTPException(
            status_code=503,
            detail=state.get("load_error", "traffic agent is not ready"),
        )
    return executor


def require_token(authorization: str | None = Header(default=None)) -> None:
    expected = state.get("token")
    supplied = authorization.removeprefix("Bearer ") if authorization else ""
    if not isinstance(expected, str) or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="invalid traffic-agent token")


@app.get("/health/live")
async def health_live() -> dict[str, str]:
    return {"status": "live"}


@app.get("/health/ready")
async def health_ready() -> dict[str, str]:
    _ready()
    return {"status": "ready"}


@app.get("/v1/profiles", dependencies=[Depends(require_token)])
async def profiles() -> dict[str, Any]:
    executor = _ready()
    profile_documents = []
    for profile in executor.catalog.profiles:
        document = profile.model_dump()
        document["fixed_targets"] = [
            {
                "alias": (
                    "internal-gateway"
                    if profile.target_group in {"gateway-http", "ssh-emulator"}
                    else "internal-gateway/blackhole"
                    if profile.target_group == "irc-emulator"
                    else "fixed-lab-sink/blackhole"
                ),
                "endpoint": endpoint,
            }
            for endpoint in executor.targets.groups[profile.target_group].endpoints
        ]
        profile_documents.append(document)
    return {
        "reference_digest": executor.catalog.reference_digest,
        "dataset_digest": executor.catalog.dataset_digest,
        "graph_protocol": executor.catalog.graph_protocol,
        "claim_boundary": executor.catalog.claim_boundary,
        "profiles": profile_documents,
    }


@app.post("/v1/runs", status_code=202, dependencies=[Depends(require_token)])
async def start_run(request: StartTrafficRun) -> dict[str, Any]:
    executor = _ready()
    try:
        return (
            await executor.start(
                request.profile_id,
                events=request.events,
                interval_ms=request.interval_ms,
            )
        ).public()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="unknown traffic profile") from exc
    except TrafficProfileDisabledError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except TrafficRunConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post(
    "/v1/runs/{run_id}/release",
    dependencies=[Depends(require_token)],
)
async def release_run(run_id: str) -> dict[str, Any]:
    executor = _ready()
    try:
        return executor.release(run_id).public()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="unknown traffic run") from exc
    except TrafficRunConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/v1/runs/current", dependencies=[Depends(require_token)])
async def current_run() -> dict[str, Any]:
    record = _ready().current()
    return {"run": record.public() if record else None}


@app.delete("/v1/runs/current", dependencies=[Depends(require_token)])
async def stop_run() -> dict[str, Any]:
    record = await _ready().stop()
    return {"run": record.public() if record else None}
