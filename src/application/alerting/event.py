"""Structured Elasticsearch event that forbids raw network/model data."""

from __future__ import annotations

import hashlib
import hmac
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DetectionEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    timestamp: datetime = Field(alias="@timestamp")
    event_type: str = "fedper_detection"
    sensor_id: str
    client_id: str
    window_id: str
    predicted_class: str
    is_alert: bool
    severity: str
    confidence_bucket: str
    entropy_bucket: str
    flow_count: int
    entity_hash: str
    model_digest: str
    head_digest: str
    schema_digest: str
    decision_mode: str = "trusted-head-v1"
    fusion_policy_digest: str | None = None
    trusted_predicted_class: str | None = None
    head_disagreement_count: int = Field(default=0, ge=0, le=6)
    fusion_predicted_class: str | None = None
    alert_decision_source: str = Field(
        default="fusion", pattern=r"^(fusion|trusted-shadow)$"
    )

    @field_validator(
        "entity_hash",
        "model_digest",
        "head_digest",
        "schema_digest",
        "fusion_policy_digest",
    )
    @classmethod
    def validate_digest(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("Digest fields must be lowercase SHA-256 hex.")
        return value


def entity_hash(*, key: bytes, sensor_id: str, entity: str) -> str:
    if not key:
        raise ValueError("Entity hash key cannot be empty.")
    return hmac.new(
        key,
        f"{sensor_id}\0{entity}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def numeric_bucket(value: float, boundaries: tuple[float, ...]) -> str:
    if not boundaries or tuple(sorted(boundaries)) != boundaries:
        raise ValueError("Bucket boundaries must be a sorted non-empty tuple.")
    if value < boundaries[0]:
        return f"<{boundaries[0]:g}"
    for lower, upper in zip(boundaries, boundaries[1:]):
        if lower <= value < upper:
            return f"{lower:g}-{upper:g}"
    return f">={boundaries[-1]:g}"
