"""Validation-selected probability fusion for the six exact FedPer heads."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from src.application.inference.bundle_loader import FedPerServingBundle
from src.application.inference.runtime import RoutedPrediction


class FusionPolicyError(ValueError):
    """Raised when a fusion policy cannot be trusted for the loaded bundle."""


def _canonical_digest(document: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in document.items() if key != "policy_digest"}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class FusedPrediction:
    probabilities: torch.Tensor
    predicted_indices: torch.Tensor
    confidence: torch.Tensor
    entropy: torch.Tensor


@dataclass(frozen=True)
class MultiHeadFusionPolicy:
    policy_id: str
    policy_digest: str
    heads: tuple[str, ...]
    classes: tuple[str, ...]
    method: str
    class_head_weights: torch.Tensor | None
    feature_transform: str | None
    feature_mean: torch.Tensor | None
    feature_scale: torch.Tensor | None
    coefficients: torch.Tensor | None
    intercept: torch.Tensor | None
    class_alert_thresholds: Mapping[str, float]
    provenance: Mapping[str, Any]

    def fuse(
        self, predictions: Mapping[str, RoutedPrediction]
    ) -> FusedPrediction:
        if set(predictions) != set(self.heads):
            raise FusionPolicyError("Runtime head set does not match fusion policy.")
        stacked = torch.stack(
            [predictions[head].probabilities for head in self.heads], dim=0
        )
        if self.method == "class-f1-weighted-probability":
            assert self.class_head_weights is not None
            weights = self.class_head_weights.to(dtype=stacked.dtype)
            scores = torch.sum(stacked * weights[:, None, :], dim=0)
            denominator = scores.sum(dim=-1, keepdim=True)
            if torch.any(denominator <= 0) or not torch.isfinite(scores).all():
                raise FusionPolicyError("Fusion produced invalid class scores.")
            probabilities = scores / denominator
        elif self.method == "multinomial-logistic-stacking":
            assert self.feature_mean is not None
            assert self.feature_scale is not None
            assert self.coefficients is not None
            assert self.intercept is not None
            features = stacked.permute(1, 0, 2).reshape(stacked.shape[1], -1)
            if self.feature_transform == "log-probability":
                features = torch.log(features.clamp_min(1e-12))
            features = (
                features - self.feature_mean.to(dtype=features.dtype)
            ) / self.feature_scale.to(dtype=features.dtype)
            logits = features @ self.coefficients.to(dtype=features.dtype).T
            logits = logits + self.intercept.to(dtype=features.dtype)
            probabilities = torch.softmax(logits, dim=-1)
        else:  # pragma: no cover - the loader rejects this state
            raise FusionPolicyError(f"Unsupported fusion method '{self.method}'.")
        confidence, predicted_indices = probabilities.max(dim=-1)
        entropy = -torch.sum(
            probabilities * torch.log(probabilities + 1e-12), dim=-1
        )
        return FusedPrediction(
            probabilities=probabilities,
            predicted_indices=predicted_indices,
            confidence=confidence,
            entropy=entropy,
        )


def load_fusion_policy(
    path: str | Path, bundle: FedPerServingBundle
) -> MultiHeadFusionPolicy:
    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FusionPolicyError(f"Cannot read fusion policy: {exc}") from exc
    if not isinstance(document, dict):
        raise FusionPolicyError("Fusion policy must be a JSON object.")
    if document.get("schema_version") != 1:
        raise FusionPolicyError("Unsupported fusion policy schema version.")
    if document.get("kind") != "validation-selected-multi-head-probability-fusion":
        raise FusionPolicyError("Unsupported fusion policy kind.")
    if document.get("selection_split") != "validation":
        raise FusionPolicyError("Fusion policy must be selected on validation.")
    expected_digest = _canonical_digest(document)
    if document.get("policy_digest") != expected_digest:
        raise FusionPolicyError("Fusion policy digest mismatch.")

    manifest = bundle.manifest
    exact_fields = {
        "bundle_id": manifest["bundle_id"],
        "model_digest": manifest["model_digest"],
        "dataset_digest": manifest["dataset_digest"],
        "graph_protocol": manifest["graph_protocol"],
    }
    for field, expected in exact_fields.items():
        if document.get(field) != expected:
            raise FusionPolicyError(
                f"Fusion policy {field} does not match serving bundle."
            )
    if document.get("head_digests") != manifest["head_digests"]:
        raise FusionPolicyError("Fusion policy head digests do not match bundle.")

    heads = tuple(str(value) for value in document.get("heads", ()))
    classes = tuple(str(value) for value in document.get("classes", ()))
    if not heads or set(heads) != set(bundle.heads):
        raise FusionPolicyError("Fusion policy must cover every exact head once.")
    if not classes or classes != tuple(bundle.class_to_idx):
        raise FusionPolicyError("Fusion policy class order does not match bundle.")

    method = str(document.get("method"))
    weights: torch.Tensor | None = None
    feature_transform: str | None = None
    feature_mean: torch.Tensor | None = None
    feature_scale: torch.Tensor | None = None
    coefficients: torch.Tensor | None = None
    intercept: torch.Tensor | None = None
    if method == "class-f1-weighted-probability":
        raw_weights = document.get("class_head_weights")
        if not isinstance(raw_weights, dict) or set(raw_weights) != set(classes):
            raise FusionPolicyError("Fusion policy weights must cover every class.")
        columns: list[list[float]] = []
        for class_name in classes:
            values = raw_weights[class_name]
            if not isinstance(values, dict) or set(values) != set(heads):
                raise FusionPolicyError(
                    f"Fusion weights for '{class_name}' must cover every head."
                )
            column = [float(values[head]) for head in heads]
            if any(
                value < 0 or not torch.isfinite(torch.tensor(value))
                for value in column
            ):
                raise FusionPolicyError(
                    "Fusion weights must be finite and non-negative."
                )
            total = sum(column)
            if total <= 0 or abs(total - 1.0) > 1e-6:
                raise FusionPolicyError(
                    "Fusion weights for each class must sum to one."
                )
            columns.append(column)
        weights = (
            torch.tensor(columns, dtype=torch.float64).transpose(0, 1).contiguous()
        )
    elif method == "multinomial-logistic-stacking":
        stacking = document.get("stacking")
        if not isinstance(stacking, dict):
            raise FusionPolicyError("Fusion stacking parameters are missing.")
        feature_transform = str(stacking.get("feature_transform"))
        if feature_transform not in {"probability", "log-probability"}:
            raise FusionPolicyError("Unsupported stacking feature transform.")
        feature_count = len(heads) * len(classes)
        try:
            feature_mean = torch.tensor(stacking["feature_mean"], dtype=torch.float64)
            feature_scale = torch.tensor(stacking["feature_scale"], dtype=torch.float64)
            coefficients = torch.tensor(stacking["coefficients"], dtype=torch.float64)
            intercept = torch.tensor(stacking["intercept"], dtype=torch.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise FusionPolicyError("Invalid stacking parameter arrays.") from exc
        if feature_mean.shape != (feature_count,) or feature_scale.shape != (
            feature_count,
        ):
            raise FusionPolicyError(
                "Stacking scaler shape does not match heads/classes."
            )
        if coefficients.shape != (len(classes), feature_count) or intercept.shape != (
            len(classes),
        ):
            raise FusionPolicyError(
                "Stacking coefficient shape does not match policy."
            )
        tensors = (feature_mean, feature_scale, coefficients, intercept)
        if any(not torch.isfinite(value).all() for value in tensors) or torch.any(
            feature_scale <= 0
        ):
            raise FusionPolicyError(
                "Stacking parameters must be finite with positive scale."
            )
    else:
        raise FusionPolicyError(f"Unsupported fusion method '{method}'.")

    raw_thresholds = document.get("class_alert_thresholds")
    attack_classes = set(classes) - {"Benign"}
    if not isinstance(raw_thresholds, dict) or set(raw_thresholds) != attack_classes:
        raise FusionPolicyError("Fusion alert thresholds must cover attack classes.")
    thresholds = {name: float(value) for name, value in raw_thresholds.items()}
    if any(not 0.0 <= value <= 1.0 for value in thresholds.values()):
        raise FusionPolicyError("Fusion alert thresholds must be in [0, 1].")

    provenance = document.get("provenance")
    if not isinstance(provenance, dict) or not provenance.get("validation_report_sha256"):
        raise FusionPolicyError("Fusion policy validation provenance is incomplete.")
    return MultiHeadFusionPolicy(
        policy_id=str(document["policy_id"]),
        policy_digest=expected_digest,
        heads=heads,
        classes=classes,
        method=method,
        class_head_weights=weights,
        feature_transform=feature_transform,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        coefficients=coefficients,
        intercept=intercept,
        class_alert_thresholds=thresholds,
        provenance=provenance,
    )


def policy_digest(document: Mapping[str, Any]) -> str:
    """Return the canonical digest used by the selector and fail-closed loader."""
    return _canonical_digest(document)
