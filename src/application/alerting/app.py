"""Structured alert-router API with fail-closed privacy validation."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from src.application.alerting.elasticsearch import (
    ElasticsearchSettings,
    ElasticsearchSink,
    ElasticsearchSinkError,
)
from src.application.alerting.event import DetectionEvent
from src.application.alerting.privacy import validate_elasticsearch_document


logger = logging.getLogger(__name__)
app_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    app_state.clear()
    mode = os.environ.get("ALERT_SINK_MODE", "stdout")
    if mode == "stdout":
        app_state["mode"] = mode
    elif mode == "elasticsearch":
        try:
            app_state["sink"] = ElasticsearchSink(
                ElasticsearchSettings(
                    endpoint=os.environ["ELASTICSEARCH_ENDPOINT"],
                    index=os.environ.get("ELASTICSEARCH_INDEX", "fedper-detections"),
                    username=os.environ.get("ELASTICSEARCH_USERNAME"),
                    password=os.environ.get("ELASTICSEARCH_PASSWORD"),
                    api_key=os.environ.get("ELASTICSEARCH_API_KEY"),
                )
            )
            app_state["mode"] = mode
        except (KeyError, ValueError) as exc:
            app_state["load_error"] = str(exc)
    else:
        app_state["load_error"] = f"Unsupported ALERT_SINK_MODE '{mode}'."
    yield
    app_state.clear()


app = FastAPI(title="FedKube Detection Alert Router", lifespan=lifespan)


@app.get("/health/live")
async def health_live() -> dict[str, str]:
    return {"status": "live"}


@app.get("/health/ready")
async def health_ready() -> dict[str, str]:
    if "mode" not in app_state:
        raise HTTPException(
            status_code=503,
            detail=app_state.get("load_error", "Alert router is not initialized."),
        )
    return {"status": "ready", "sink": str(app_state["mode"])}


@app.post("/events")
async def route_event(event: DetectionEvent) -> dict[str, str | bool]:
    if "mode" not in app_state:
        raise HTTPException(status_code=503, detail="Alert router is not ready.")
    document = event.model_dump(by_alias=True, mode="json")
    validate_elasticsearch_document(document)
    if app_state["mode"] == "stdout":
        logger.info("fedper_detection=%s", document)
        return {"accepted": True, "sink": "stdout"}
    try:
        document_id = app_state["sink"].index_event(document)
    except ElasticsearchSinkError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"accepted": True, "sink": "elasticsearch", "document_id": document_id}
