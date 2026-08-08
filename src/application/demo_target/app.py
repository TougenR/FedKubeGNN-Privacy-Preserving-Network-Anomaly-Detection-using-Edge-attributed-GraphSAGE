"""Small HTTP target for benign and bounded lab traffic patterns."""

from __future__ import annotations

import logging
import os
import time

from fastapi import BackgroundTasks, FastAPI, Request
from asyncio import sleep

from src.application.collection.ingress_adapter import (
    INGRESS_ADAPTER_PROTOCOL,
    observed_http_flow,
)
from src.application.collection.transport import ServiceRequestError, post_json


app = FastAPI(title="FedKube Detection Demo Target")
logger = logging.getLogger(__name__)


def _emit_observation(
    *, request: Request, response_bytes: int, started_at: float
) -> None:
    endpoint = os.environ.get("COLLECTOR_OBSERVE_URL")
    sensor_id = os.environ.get("DEMO_SENSOR_ID")
    if not endpoint or not sensor_id or request.client is None:
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
    try:
        post_json(
            endpoint,
            {
                "sensor_id": sensor_id,
                "source": INGRESS_ADAPTER_PROTOCOL,
                "flow": flow,
            },
        )
    except (ServiceRequestError, ValueError) as exc:
        logger.warning("Collector observation failed: %s", exc)


@app.get("/")
async def root(
    request: Request, background_tasks: BackgroundTasks
) -> dict[str, str]:
    started_at = time.time()
    response = {"service": "fedkube-demo-target", "status": "ok"}
    background_tasks.add_task(
        _emit_observation,
        request=request,
        response_bytes=len(str(response).encode("utf-8")),
        started_at=started_at,
    )
    return response


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/probe")
async def probe(
    request: Request, background_tasks: BackgroundTasks
) -> dict[str, str]:
    started_at = time.time()
    response = {"status": "open", "service": "fedkube-demo-target"}
    background_tasks.add_task(
        _emit_observation,
        request=request,
        response_bytes=len(str(response).encode("utf-8")),
        started_at=started_at,
    )
    return response


@app.get("/payload/{size}")
async def payload(
    size: int, request: Request, background_tasks: BackgroundTasks
) -> dict[str, str | int]:
    started_at = time.time()
    bounded = max(0, min(size, 65536))
    response = {"requested": size, "returned": bounded, "payload": "x" * bounded}
    background_tasks.add_task(
        _emit_observation,
        request=request,
        response_bytes=bounded,
        started_at=started_at,
    )
    return response


@app.get("/slow/{delay_ms}")
async def slow(
    delay_ms: int, request: Request, background_tasks: BackgroundTasks
) -> dict[str, int | str]:
    started_at = time.time()
    bounded = max(0, min(delay_ms, 10000))
    await sleep(bounded / 1000)
    response = {"status": "completed", "delay_ms": bounded}
    background_tasks.add_task(
        _emit_observation,
        request=request,
        response_bytes=len(str(response).encode("utf-8")),
        started_at=started_at,
    )
    return response
