"""FastAPI boundary for centralized, trusted-routed FedPer inference."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from src.application.api.schema import (
    FlowPrediction,
    ProductionInferenceRequest,
    ProductionInferenceResponse,
)
from src.application.graph_window.graph_builder import (
    build_inference_graph,
    preprocess_production_flows,
)
from src.application.inference.bundle_loader import (
    FedPerServingBundle,
    InferenceBundleError,
    load_inference_bundle,
)
from src.application.inference.router import TrustedRoutingError
from src.application.inference.runtime import CentralizedFedPerRuntime


logger = logging.getLogger(__name__)
app_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    app_state.clear()
    bundle_path = os.environ.get("MODEL_BUNDLE_PATH")
    if not bundle_path:
        app_state["load_error"] = "MODEL_BUNDLE_PATH is required."
    else:
        try:
            bundle = load_inference_bundle(
                bundle_path,
                device=os.environ.get("INFERENCE_DEVICE", "cpu"),
                require_serving_ready=True,
            )
            app_state["bundle"] = bundle
            app_state["runtime"] = CentralizedFedPerRuntime(bundle)
        except (InferenceBundleError, KeyError, OSError, ValueError) as exc:
            app_state["load_error"] = str(exc)
            logger.error("Detection runtime is not ready: %s", exc)
    yield
    app_state.clear()


app = FastAPI(title="FedKube Centralized FedPer Detection", lifespan=lifespan)


def _ready() -> tuple[FedPerServingBundle, CentralizedFedPerRuntime]:
    bundle = app_state.get("bundle")
    runtime = app_state.get("runtime")
    if not isinstance(bundle, FedPerServingBundle) or not isinstance(
        runtime, CentralizedFedPerRuntime
    ):
        raise HTTPException(
            status_code=503,
            detail=app_state.get("load_error", "Runtime is not initialized."),
        )
    return bundle, runtime


@app.get("/health/live")
async def health_live() -> dict[str, str]:
    return {"status": "live"}


@app.get("/health/ready")
async def health_ready() -> dict[str, Any]:
    bundle, _ = _ready()
    return {
        "status": "ready",
        "bundle_id": bundle.manifest["bundle_id"],
        "model_digest": bundle.manifest["model_digest"],
        "best_round": bundle.manifest["best_round"],
        "graph_protocol": bundle.manifest["graph_protocol"],
        "feature_schema_digest": bundle.manifest["feature_schema_digest"],
        "label_schema_digest": bundle.manifest["label_schema_digest"],
        "clients": sorted(bundle.heads),
    }


@app.post("/predict", response_model=ProductionInferenceResponse)
async def predict(request: ProductionInferenceRequest) -> ProductionInferenceResponse:
    if not request.flows:
        raise HTTPException(status_code=400, detail="Empty graph window.")
    bundle, runtime = _ready()
    try:
        records = [flow.model_dump(by_alias=True) for flow in request.flows]
        features = preprocess_production_flows(records, bundle.preprocessor)
        graph = build_inference_graph(
            features,
            bundle.preprocessor.feature_columns,
            sensor_id=request.sensor_id,
        )
        result = runtime.predict_graph(sensor_id=request.sensor_id, graph=graph)
    except TrustedRoutingError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (KeyError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    labels = bundle.idx_to_class
    predictions: list[FlowPrediction] = []
    for index, flow in enumerate(request.flows):
        probabilities = result.probabilities[index]
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
                predicted_label=labels[int(result.predicted_indices[index])],
                confidence=float(result.confidence[index]),
                entropy=float(result.entropy[index]),
                probabilities={
                    labels[class_index]: float(probabilities[class_index])
                    for class_index in range(len(labels))
                },
            )
        )
    return ProductionInferenceResponse(
        client_id=result.client_id,
        model_digest=bundle.manifest["model_digest"],
        head_digest=bundle.manifest["head_digests"][result.client_id],
        schema_digest=bundle.manifest["feature_schema_digest"],
        graph_protocol=bundle.manifest["graph_protocol"],
        predictions=predictions,
    )
