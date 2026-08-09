"""Validation-only, label-separated replay for the Phase 4 demo console."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.application.api.schema import ProductionFlow
from src.application.collection.transport import post_json


class ReplayPolicyError(ValueError):
    """Raised when replay alert decisions cannot be tied to a trusted policy."""


@dataclass(frozen=True)
class ReplayAlertPolicy:
    policy_digest: str
    class_alert_thresholds: Mapping[str, float]


class ReplayClassProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class_name: str
    behavior: str
    indicators: list[str] = Field(min_length=1)
    limitation: str


def _policy_digest(document: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in document.items() if key != "policy_digest"}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    class_profiles: list[ReplayClassProfile]
    cases: list[ScientificReplayCase]

    @model_validator(mode="after")
    def validate_contract(self) -> "ScientificReplayCatalog":
        if self.schema_version != 2:
            raise ValueError("Unsupported scientific replay schema version.")
        if self.kind != "validation-only-scientific-replay":
            raise ValueError("Scientific replay kind is invalid.")
        if self.selection_split != "validation":
            raise ValueError("Scientific replay must never use the test split.")
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("Scientific replay case IDs must be unique.")
        case_classes = [case.expected_class for case in self.cases]
        profile_classes = [profile.class_name for profile in self.class_profiles]
        if len(case_classes) != len(set(case_classes)):
            raise ValueError("Scientific replay must contain one case per class.")
        if len(profile_classes) != len(set(profile_classes)):
            raise ValueError("Scientific replay class profiles must be unique.")
        if set(profile_classes) != set(case_classes):
            raise ValueError("Class profiles must cover every replay class exactly.")
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


def load_replay_alert_policy(path: str | Path) -> ReplayAlertPolicy:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayPolicyError(f"Cannot read replay alert policy: {exc}") from exc
    if not isinstance(document, dict):
        raise ReplayPolicyError("Replay alert policy must be a JSON object.")
    if document.get("selection_split") != "validation":
        raise ReplayPolicyError("Replay alert policy must be validation-selected.")
    expected_digest = _policy_digest(document)
    if document.get("policy_digest") != expected_digest:
        raise ReplayPolicyError("Replay alert policy digest mismatch.")
    classes = document.get("classes")
    thresholds = document.get("class_alert_thresholds")
    if not isinstance(classes, list) or not isinstance(thresholds, dict):
        raise ReplayPolicyError("Replay alert policy class thresholds are missing.")
    attack_classes = {str(value) for value in classes} - {"Benign"}
    if set(thresholds) != attack_classes:
        raise ReplayPolicyError("Replay alert thresholds must cover attack classes.")
    parsed = {str(name): float(value) for name, value in thresholds.items()}
    if any(not 0.0 <= value <= 1.0 for value in parsed.values()):
        raise ReplayPolicyError("Replay alert thresholds must be in [0, 1].")
    return ReplayAlertPolicy(
        policy_digest=expected_digest,
        class_alert_thresholds=parsed,
    )


def _sample_characteristics(case: ScientificReplayCase) -> dict[str, Any]:
    def values(attribute: str) -> list[str]:
        return sorted(
            {
                str(getattr(flow, attribute))
                for flow in case.flows
                if getattr(flow, attribute) is not None
            }
        )

    durations = [flow.duration for flow in case.flows if flow.duration is not None]
    return {
        "flow_count": len(case.flows),
        "protocols": values("proto"),
        "services": values("service"),
        "connection_states": values("conn_state"),
        "destination_ports": sorted({flow.id_resp_p for flow in case.flows}),
        "duration_seconds": (
            {"min": min(durations), "max": max(durations)} if durations else None
        ),
        "total_origin_packets": sum(flow.orig_pkts or 0 for flow in case.flows),
        "total_response_packets": sum(flow.resp_pkts or 0 for flow in case.flows),
        "total_origin_bytes": sum(flow.orig_bytes or 0 for flow in case.flows),
        "total_response_bytes": sum(flow.resp_bytes or 0 for flow in case.flows),
    }


def public_catalog(catalog: ScientificReplayCatalog) -> dict[str, Any]:
    profiles = {profile.class_name: profile for profile in catalog.class_profiles}
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
                "profile": profiles[case.expected_class].model_dump(),
                "sample_characteristics": _sample_characteristics(case),
            }
            for case in catalog.cases
        ],
    }


def execute_replay_case(
    *,
    case: ScientificReplayCase,
    inference_url: str,
    alert_policy: ReplayAlertPolicy,
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
    if response.get("fusion_policy_digest") != alert_policy.policy_digest:
        raise ReplayPolicyError(
            "Inference response does not match the replay alert policy digest."
        )
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
    confidence = float(target["confidence"])
    if predicted_class == "Benign":
        threshold = None
        is_alert = False
        decision_status = "benign"
    else:
        try:
            threshold = alert_policy.class_alert_thresholds[predicted_class]
        except KeyError as exc:
            raise ReplayPolicyError(
                f"No alert threshold exists for '{predicted_class}'."
            ) from exc
        is_alert = confidence >= threshold
        decision_status = "alert" if is_alert else "below-threshold"
    return {
        "case_id": case.id,
        "selection_split": "validation",
        "expected_class": case.expected_class,
        "predicted_class": predicted_class,
        "correct": predicted_class == case.expected_class,
        "confidence": confidence,
        "entropy": float(target["entropy"]),
        "is_alert": is_alert,
        "decision_status": decision_status,
        "alert_threshold": threshold,
        "fusion_policy_digest": alert_policy.policy_digest,
        "top3": [{"class": label, "probability": value} for label, value in top3],
        "sensor_id": case.sensor_id,
        "client_id": str(response["client_id"]),
        "window_flows": len(case.flows),
        "model_digest": str(response["model_digest"]),
        "head_digest": str(response["head_digest"]),
        "schema_digest": str(response["schema_digest"]),
        "request_contains_ground_truth": False,
    }
