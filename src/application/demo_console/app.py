"""Same-origin UI/API boundary for bounded lab scenarios and live monitoring."""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from src.application.collection.transport import ServiceRequestError, get_json
from src.application.scenario_runner.catalog import load_catalog
from src.application.scenario_runner.executor import (
    ScenarioConflictError,
    ScenarioExecutor,
    scan_urls_from_json,
)


class StartRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    parameters: dict[str, Any]


state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    state.clear()
    try:
        catalog = load_catalog(
            os.environ.get(
                "DEMO_SCENARIO_CONFIG",
                "/app/configs/application/demo-scenarios.yaml",
            )
        )
        state["catalog"] = catalog
        state["collector_url"] = os.environ["COLLECTOR_URL"].rstrip("/")
        state["executor"] = ScenarioExecutor(
            catalog=catalog,
            target_url=os.environ["DEMO_TARGET_URL"],
            scan_urls=scan_urls_from_json(os.environ["DEMO_SCAN_URLS"]),
        )
    except (KeyError, OSError, ValueError) as exc:
        state["load_error"] = str(exc)
    yield
    executor = state.get("executor")
    if isinstance(executor, ScenarioExecutor):
        await executor.stop()
    state.clear()


app = FastAPI(title="FedKube Phase 4 Demo Console", lifespan=lifespan)


def _ready() -> tuple[Any, ScenarioExecutor]:
    catalog = state.get("catalog")
    executor = state.get("executor")
    if catalog is None or not isinstance(executor, ScenarioExecutor):
        raise HTTPException(status_code=503, detail=state.get("load_error", "console is not ready"))
    return catalog, executor


@app.get("/health/live")
async def health_live() -> dict[str, str]:
    return {"status": "live"}


@app.get("/health/ready")
async def health_ready() -> dict[str, str]:
    _ready()
    return {"status": "ready"}


@app.get("/api/config")
async def config() -> dict[str, Any]:
    catalog, _ = _ready()
    return catalog.model_dump()


@app.post("/api/runs", status_code=202)
async def start_run(request: StartRunRequest) -> dict[str, Any]:
    _, executor = _ready()
    try:
        record = await executor.start(request.scenario_id, request.parameters)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ScenarioConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return record.public()


@app.get("/api/runs/current")
async def current_run() -> dict[str, Any]:
    _, executor = _ready()
    record = executor.current()
    return {"run": record.public() if record else None}


@app.delete("/api/runs/current")
async def stop_run() -> dict[str, Any]:
    _, executor = _ready()
    record = await executor.stop()
    return {"run": record.public() if record else None}


@app.get("/api/monitor")
async def monitor(
    after: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=200),
) -> dict[str, Any]:
    _ready()
    try:
        return await asyncio.to_thread(
            get_json,
            f"{state['collector_url']}/monitor/events?after={after}&limit={limit}",
        )
    except ServiceRequestError as exc:
        raise HTTPException(status_code=502, detail="collector monitor unavailable") from exc


static_root = Path(__file__).with_name("static")
app.mount("/", StaticFiles(directory=static_root, html=True), name="console")
