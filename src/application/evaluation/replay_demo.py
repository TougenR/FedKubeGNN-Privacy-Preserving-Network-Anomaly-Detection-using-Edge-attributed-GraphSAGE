"""Validation-only, label-separated replay for the Phase 4 demo console."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.application.api.schema import ProductionFlow
from src.application.collection.transport import post_json


class ScientificReplayCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")
    display_name: str
    sensor_id: str
    client_id: str
    expected_class: str
    selection_occurrence: int = Field(ge=1)
    source_edge_index: int = Field(ge=0)
    target_index: int = Field(ge=0)
    flows: list[ProductionFlow] = Field(min_length=1, max_length=50)

    @model_validator(mode="after")
    def validate_target(self) -> "ScientificReplayCase":
        if self.target_index >= len(self.flows):
            raise ValueError("Replay target_index is outside the flow window.")
        return self


class ScientificReplayCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    kind: str
    selection_split: str
    selection_rule: str
    dataset_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    graph_protocol: str
    disclaimer: str
    cases: list[ScientificReplayCase]

    @model_validator(mode="after")
    def validate_contract(self) -> "ScientificReplayCatalog":
        if self.schema_version != 1:
            raise ValueError("Unsupported scientific replay schema version.")
        if self.kind != "validation-only-scientific-replay":
            raise ValueError("Scientific replay kind is invalid.")
        if self.selection_split != "validation":
            raise ValueError("Scientific replay must never use the test split.")
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("Scientific replay case IDs must be unique.")
        return self

    def case(self, case_id: str) -> ScientificReplayCase:
        for item in self.cases:
            if item.id == case_id:
                return item
        raise KeyError(case_id)


def load_scientific_replay(path: str | Path) -> ScientificReplayCatalog:
    return ScientificReplayCatalog.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def public_catalog(catalog: ScientificReplayCatalog) -> dict[str, Any]:
    return {
        "schema_version": catalog.schema_version,
        "kind": catalog.kind,
        "selection_split": catalog.selection_split,
        "selection_rule": catalog.selection_rule,
        "dataset_digest": catalog.dataset_digest,
        "graph_protocol": catalog.graph_protocol,
        "disclaimer": catalog.disclaimer,
        "cases": [
            {
                "id": case.id,
                "display_name": case.display_name,
                "sensor_id": case.sensor_id,
                "client_id": case.client_id,
                "expected_class": case.expected_class,
                "window_flows": len(case.flows),
                "selection_occurrence": case.selection_occurrence,
            }
            for case in catalog.cases
        ],
    }


def execute_replay_case(
    *,
    case: ScientificReplayCase,
    inference_url: str,
    sender: Callable[..., dict[str, Any]] = post_json,
) -> dict[str, Any]:
    # Expected class and source identity deliberately remain outside this
    # production request. ProductionFlow also rejects label fields.
    request = {
        "sensor_id": case.sensor_id,
        "window_id": f"scientific-validation-{case.id}",
        "flows": [flow.model_dump(by_alias=True) for flow in case.flows],
    }
    response = sender(inference_url, request)
    if response.get("client_id") != case.client_id:
        raise ValueError("Trusted replay route selected an unexpected client head.")
    predictions = response.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != len(case.flows):
        raise ValueError("Inference response does not align with the replay window.")
    target = predictions[case.target_index]
    probabilities = target.get("probabilities", {})
    if not isinstance(probabilities, dict):
        raise ValueError("Replay target probabilities are invalid.")
    top3 = sorted(
        ((str(label), float(value)) for label, value in probabilities.items()),
        key=lambda item: item[1],
        reverse=True,
    )[:3]
    predicted_class = str(target["predicted_label"])
    return {
        "case_id": case.id,
        "selection_split": "validation",
        "expected_class": case.expected_class,
        "predicted_class": predicted_class,
        "correct": predicted_class == case.expected_class,
        "confidence": float(target["confidence"]),
        "entropy": float(target["entropy"]),
        "top3": [{"class": label, "probability": value} for label, value in top3],
        "sensor_id": case.sensor_id,
        "client_id": str(response["client_id"]),
        "window_flows": len(case.flows),
        "model_digest": str(response["model_digest"]),
        "head_digest": str(response["head_digest"]),
        "schema_digest": str(response["schema_digest"]),
        "request_contains_ground_truth": False,
    }
