"""Small HTTP target for benign and bounded lab traffic patterns."""

from __future__ import annotations

import logging
import os
import re
import time

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from asyncio import sleep

from src.application.collection.ingress_adapter import (
    INGRESS_ADAPTER_PROTOCOL,
    observed_http_flow,
)
from src.application.collection.delivery import ObservationDispatcher


logger = logging.getLogger(__name__)
app_state: dict[str, object] = {}
LAB_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


@asynccontextmanager
async def lifespan(_: FastAPI):
    app_state.clear()
    endpoint = os.environ.get("COLLECTOR_OBSERVE_URL")
    sensor_id = os.environ.get("DEMO_SENSOR_ID")
    if endpoint and sensor_id:
        dispatcher = ObservationDispatcher(
            endpoint=endpoint,
            queue_size=int(os.environ.get("OBSERVATION_QUEUE_SIZE", "1000")),
            workers=int(os.environ.get("OBSERVATION_WORKERS", "1")),
            retry_attempts=int(os.environ.get("OBSERVATION_RETRY_ATTEMPTS", "3")),
            retry_backoff_seconds=float(
                os.environ.get("OBSERVATION_RETRY_BACKOFF_SECONDS", "0.25")
            ),
        )
        await dispatcher.start()
        app_state["dispatcher"] = dispatcher
        app_state["sensor_id"] = sensor_id
    yield
    dispatcher = app_state.get("dispatcher")
    if isinstance(dispatcher, ObservationDispatcher):
        await dispatcher.stop()
    app_state.clear()


app = FastAPI(title="FedKube Detection Demo Target", lifespan=lifespan)


def _lab_id(request: Request, name: str) -> str | None:
    value = request.headers.get(name)
    return value if value and LAB_ID.fullmatch(value) else None


def _enqueue_observation(
    *, request: Request, response_bytes: int, started_at: float
) -> None:
    dispatcher = app_state.get("dispatcher")
    sensor_id = app_state.get("sensor_id")
    if (
        not isinstance(dispatcher, ObservationDispatcher)
        or not isinstance(sensor_id, str)
        or request.client is None
    ):
        return
    flow = observed_http_flow(
        source_host=request.client.host,
        source_port=request.client.port,
        target_host=os.environ.get("DEMO_TARGET_ENTITY", "demo-target"),
        target_port=request.url.port or int(os.environ.get("DEMO_TARGET_PORT", "8080")),
        request_bytes=int(request.headers.get("content-length", "0")),
        response_bytes=response_bytes,
        started_at=started_at,
    )
    run_id = _lab_id(request, "x-fedkube-demo-run")
    document = {
        "sensor_id": sensor_id,
        "source": INGRESS_ADAPTER_PROTOCOL,
        "run_id": run_id,
        "scenario_id": _lab_id(request, "x-fedkube-demo-scenario"),
        "flow": flow,
    }
    if not dispatcher.enqueue(document, run_id=run_id):
        logger.warning("Observation queue is full; flow was not enqueued.")


@app.get("/")
async def root(request: Request) -> dict[str, str]:
    started_at = time.time()
    response = {"service": "fedkube-demo-target", "status": "ok"}
    _enqueue_observation(
        request=request,
        response_bytes=len(str(response).encode("utf-8")),
        started_at=started_at,
    )
    return response


@app.get("/health")
async def health() -> dict[str, object]:
    dispatcher = app_state.get("dispatcher")
    return {
        "status": "healthy",
        "observation_delivery": (
            dispatcher.metrics() if isinstance(dispatcher, ObservationDispatcher) else None
        ),
    }


@app.get("/observations/runs/{run_id}")
async def run_observations(run_id: str) -> dict[str, object]:
    if not LAB_ID.fullmatch(run_id):
        return {"run_id": run_id, "available": False}
    dispatcher = app_state.get("dispatcher")
    if not isinstance(dispatcher, ObservationDispatcher):
        return {"run_id": run_id, "available": False}
    return {"run_id": run_id, "available": True, **dispatcher.metrics(run_id)}


@app.get("/probe")
async def probe(request: Request) -> dict[str, str]:
    started_at = time.time()
    response = {"status": "open", "service": "fedkube-demo-target"}
    _enqueue_observation(
        request=request,
        response_bytes=len(str(response).encode("utf-8")),
        started_at=started_at,
    )
    return response


@app.get("/payload/{size}")
async def payload(size: int, request: Request) -> dict[str, str | int]:
    started_at = time.time()
    bounded = max(0, min(size, 65536))
    response = {"requested": size, "returned": bounded, "payload": "x" * bounded}
    _enqueue_observation(
        request=request,
        response_bytes=bounded,
        started_at=started_at,
    )
    return response


@app.get("/slow/{delay_ms}")
async def slow(delay_ms: int, request: Request) -> dict[str, int | str]:
    started_at = time.time()
    bounded = max(0, min(delay_ms, 10000))
    await sleep(bounded / 1000)
    response = {"status": "completed", "delay_ms": bounded}
    _enqueue_observation(
        request=request,
        response_bytes=len(str(response).encode("utf-8")),
        started_at=started_at,
    )
    return response
