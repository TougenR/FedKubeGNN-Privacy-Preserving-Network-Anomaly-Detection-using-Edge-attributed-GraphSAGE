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
    class_confidence_thresholds: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1].")
        if self.class_confidence_thresholds is not None:
            if set(self.class_confidence_thresholds) != set(self.class_severity):
                raise ValueError(
                    "class_confidence_thresholds must cover selected attack classes."
                )
            if any(
                not 0.0 <= float(value) <= 1.0
                for value in self.class_confidence_thresholds.values()
            ):
                raise ValueError("Class confidence thresholds must be in [0, 1].")
        numeric_bucket(0.0, self.confidence_boundaries)
        numeric_bucket(0.0, self.entropy_boundaries)
        if "Benign" in self.class_severity:
            raise ValueError("Benign must not have an alert severity.")
        if not self.class_severity:
            raise ValueError("class_severity must contain selected attack classes.")

    def detection_event(
        self,
        *,
        sensor_id: str,
        window_id: str,
        entity: str,
        entity_key: bytes,
        flow_count: int,
        response: Mapping[str, Any],
        prediction: Mapping[str, Any],
    ) -> DetectionEvent:
        predicted_class = str(prediction["predicted_label"])
        confidence = float(prediction["confidence"])
        configured_severity = self.class_severity.get(predicted_class)
        selected_threshold = (
            float(self.class_confidence_thresholds[predicted_class])
            if self.class_confidence_thresholds is not None
            and predicted_class in self.class_confidence_thresholds
            else self.confidence_threshold
        )
        is_alert = (
            predicted_class != "Benign"
            and confidence >= selected_threshold
            and configured_severity is not None
        )
        return DetectionEvent(
            **{
                "@timestamp": datetime.now(timezone.utc),
                "sensor_id": sensor_id,
                "client_id": str(response["client_id"]),
                "window_id": window_id,
                "predicted_class": predicted_class,
                "is_alert": is_alert,
                "severity": configured_severity if is_alert else "none",
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
                "decision_mode": str(
                    response.get("decision_mode", "trusted-head-v1")
                ),
                "fusion_policy_digest": response.get("fusion_policy_digest"),
                "trusted_predicted_class": prediction.get(
                    "trusted_prediction", {}
                ).get("predicted_label"),
                "head_disagreement_count": int(
                    prediction.get("head_disagreement_count", 0)
                ),
                "fusion_predicted_class": prediction.get(
                    "fused_predicted_label", predicted_class
                ),
                "alert_decision_source": str(
                    prediction.get("alert_decision_source", "fusion")
                ),
            }
        )

    def event_for_prediction(
        self,
        **kwargs: Any,
    ) -> DetectionEvent | None:
        """Return only policy-qualified alerts for compatibility with evaluators."""
        event = self.detection_event(**kwargs)
        return event if event.is_alert else None


def parse_boundaries(value: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(item) for item in value)
