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

from src.application.collection.transport import ServiceRequestError, get_json, post_json
from src.application.evaluation.replay_demo import (
    execute_replay_case,
    load_scientific_replay,
    public_catalog,
)
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
        state["observation_mode"] = os.environ.get(
            "OBSERVATION_MODE", "ingress-adapter"
        )
        if state["observation_mode"] not in {"ingress-adapter", "zeek"}:
            raise ValueError("OBSERVATION_MODE must be ingress-adapter or zeek")
        target_url = os.environ["DEMO_TARGET_URL"].rstrip("/")
        state["target_status_url"] = os.environ.get(
            "DEMO_TARGET_STATUS_URL", target_url
        ).rstrip("/")
        inference_url = os.environ.get("INFERENCE_URL")
        state["inference_url"] = inference_url.rstrip("/") if inference_url else None
        state["scientific_replay"] = load_scientific_replay(
            os.environ.get(
                "SCIENTIFIC_REPLAY_CONFIG",
                "/app/configs/application/scientific-replay.json",
            )
        )
        state["executor"] = ScenarioExecutor(
            catalog=catalog,
            target_url=target_url,
            scan_urls=scan_urls_from_json(os.environ["DEMO_SCAN_URLS"]),
        )
    except (KeyError, OSError, ValueError) as exc:
        state["load_error"] = str(exc)
    yield
    executor = state.get("executor")
    if isinstance(executor, ScenarioExecutor):
        await executor.stop()
    state.clear()


app = FastAPI(title="Bảng điều khiển phát hiện FedKube - Giai đoạn 4", lifespan=lifespan)


def _ready() -> tuple[Any, ScenarioExecutor]:
    catalog = state.get("catalog")
    executor = state.get("executor")
    if catalog is None or not isinstance(executor, ScenarioExecutor):
        raise HTTPException(
            status_code=503,
            detail=state.get("load_error", "bảng điều khiển chưa sẵn sàng"),
        )
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
    return {
        **catalog.model_dump(),
        "observation_mode": state["observation_mode"],
    }


@app.get("/api/scientific-replay")
async def scientific_replay_catalog() -> dict[str, Any]:
    _ready()
    return public_catalog(state["scientific_replay"])


@app.post("/api/scientific-replay/{case_id}")
async def run_scientific_replay(case_id: str) -> dict[str, Any]:
    _ready()
    inference_url = state.get("inference_url")
    if not isinstance(inference_url, str):
        raise HTTPException(
            status_code=503,
            detail="bản triển khai này chưa được cấu hình phát lại khoa học",
        )
    try:
        case = state["scientific_replay"].case(case_id)
        return await asyncio.to_thread(
            execute_replay_case,
            case=case,
            inference_url=inference_url,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="không tìm thấy mẫu phát lại") from exc
    except (ServiceRequestError, ValueError) as exc:
        raise HTTPException(status_code=502, detail="phát lại khoa học thất bại") from exc


@app.post("/api/runs", status_code=202)
async def start_run(request: StartRunRequest) -> dict[str, Any]:
    _, executor = _ready()
    try:
        record = await executor.start(
            request.scenario_id, request.parameters, gated=True
        )
        await asyncio.to_thread(
            post_json,
            f"{state['collector_url']}/runs/register",
            {
                "run_id": record.run_id,
                "scenario_id": record.scenario_id,
                "sensor_id": record.sensor_id,
            },
        )
        executor.release()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ScenarioConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ServiceRequestError as exc:
        executor.release()
        await executor.stop()
        raise HTTPException(
            status_code=502, detail="collector run registration failed"
        ) from exc
    return record.public()


@app.get("/api/runs/current")
async def current_run() -> dict[str, Any]:
    _, executor = _ready()
    record = executor.current()
    if record is None:
        return {"run": None}
    public = record.public()

    async def optional_metrics(url: str) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(get_json, url)
        except ServiceRequestError:
            return {"available": False}

    delivery, collector = await asyncio.gather(
        optional_metrics(f"{state['target_status_url']}/observations/runs/{record.run_id}"),
        optional_metrics(f"{state['collector_url']}/runs/{record.run_id}/metrics"),
    )
    public["pipeline"] = {"delivery": delivery, "collector": collector}
    return {"run": public}


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
