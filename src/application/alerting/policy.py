"""Explicit, validation-selected policy for turning predictions into events."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from src.application.alerting.event import (
    DetectionEvent,
    entity_hash,
    numeric_bucket,
)


@dataclass(frozen=True)
class AlertPolicy:
    confidence_threshold: float
    confidence_boundaries: tuple[float, ...]
    entropy_boundaries: tuple[float, ...]
    class_severity: Mapping[str, str]

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1].")
        numeric_bucket(0.0, self.confidence_boundaries)
        numeric_bucket(0.0, self.entropy_boundaries)
        if "Benign" in self.class_severity:
            raise ValueError("Benign must not have an alert severity.")
        if not self.class_severity:
            raise ValueError("class_severity must contain selected attack classes.")

    def event_for_prediction(
        self,
        *,
        sensor_id: str,
        window_id: str,
        entity: str,
        entity_key: bytes,
        flow_count: int,
        response: Mapping[str, Any],
        prediction: Mapping[str, Any],
    ) -> DetectionEvent | None:
        predicted_class = str(prediction["predicted_label"])
        confidence = float(prediction["confidence"])
        if predicted_class == "Benign" or confidence < self.confidence_threshold:
            return None
        severity = self.class_severity.get(predicted_class)
        if severity is None:
            return None
        return DetectionEvent(
            **{
                "@timestamp": datetime.now(timezone.utc),
                "sensor_id": sensor_id,
                "client_id": str(response["client_id"]),
                "window_id": window_id,
                "predicted_class": predicted_class,
                "severity": severity,
                "confidence_bucket": numeric_bucket(
                    confidence, self.confidence_boundaries
                ),
                "entropy_bucket": numeric_bucket(
                    float(prediction["entropy"]), self.entropy_boundaries
                ),
                "flow_count": flow_count,
                "entity_hash": entity_hash(
                    key=entity_key, sensor_id=sensor_id, entity=entity
                ),
                "model_digest": str(response["model_digest"]),
                "head_digest": str(response["head_digest"]),
                "schema_digest": str(response["schema_digest"]),
            }
        )


def parse_boundaries(value: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(item) for item in value)
