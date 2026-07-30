#!/usr/bin/env python3
"""Create report-ready clean Phase 1 tables and figures without retraining.

The analyzer treats run bundles as immutable inputs.  It never fabricates
missing predictions/history, never changes stored metrics, and keeps historical
leakage-prone results outside clean aggregates.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.phase1_contract import FIXED_LABELS


NOT_AVAILABLE = "NOT_AVAILABLE"
SCENARIO_ORDER = ("1-1", "3-1", "9-1", "34-1", "36-1", "39-1")
PROBABILITY_COLUMNS = tuple(
    f"probability::{label}" for label in FIXED_LABELS
)
LOGIT_COLUMNS = tuple(f"logit::{label}" for label in FIXED_LABELS)
PREDICTION_IDENTITY_COLUMNS = (
    "true_label",
    "predicted_label",
    "scenario",
    "split",
    "seed",
)
REQUIRED_BUNDLE_FILES = (
    "model.pt",
    "metadata.json",
    "metrics.json",
    "split_manifest.json",
    "predictions.csv",
)


def fixed_class_macro_f1(
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
) -> float:
    return float(
        f1_score(
            true_labels,
            predicted_labels,
            labels=list(FIXED_LABELS),
            average="macro",
            zero_division=0,
        )
    )


def seen_class_macro_f1(
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
    train_support: Mapping[str, int],
) -> float | None:
    """Macro-F1 over evaluated classes with positive training support."""

    present_true = {str(label) for label in true_labels}
    seen = [
        label
        for label in FIXED_LABELS
        if int(train_support.get(label, 0)) > 0 and label in present_true
    ]
    if not seen:
        return None
    return float(
        f1_score(
            true_labels,
            predicted_labels,
            labels=seen,
            average="macro",
            zero_division=0,
        )
    )


def collapse_binary(labels: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [
            "Benign" if str(label) == "Benign" else "Malicious"
            for label in labels
        ],
        dtype=object,
    )


def binary_metrics(
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
) -> dict[str, float]:
    true_binary = collapse_binary(true_labels)
    predicted_binary = collapse_binary(predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_binary,
        predicted_binary,
        labels=["Malicious"],
        average=None,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(true_binary, predicted_binary)),
        "malicious_precision": float(precision[0]),
        "malicious_recall": float(recall[0]),
        "malicious_f1": float(f1[0]),
    }


def probabilities_and_entropy(
    predictions: pd.DataFrame,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    """Use fixed-order probabilities first and logits as a safe fallback."""

    if set(PROBABILITY_COLUMNS).issubset(predictions.columns):
        probabilities = predictions.loc[:, PROBABILITY_COLUMNS].to_numpy(
            dtype=float
        )
        source = "probabilities"
        if (
            not np.isfinite(probabilities).all()
            or np.any(probabilities < 0)
            or np.any(probabilities > 1)
            or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
        ):
            return None, None, (
                "Invalid probability columns: require finite [0,1] values "
                "whose rows sum to 1."
            )
    elif set(LOGIT_COLUMNS).issubset(predictions.columns):
        logits = predictions.loc[:, LOGIT_COLUMNS].to_numpy(dtype=float)
        source = "logits"
        if not np.isfinite(logits).all():
            return None, None, "Invalid logit columns: values must be finite."
        shifted = logits - logits.max(axis=1, keepdims=True)
        exponentiated = np.exp(shifted)
        probabilities = exponentiated / exponentiated.sum(
            axis=1, keepdims=True
        )
    else:
        return None, None, (
            "predictions.csv requires all probability::<fixed-label> columns "
            "or all logit::<fixed-label> columns."
        )
    entropy = -np.sum(
        probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)),
        axis=1,
    )
    return probabilities, entropy, source


def _describe_values(values: np.ndarray) -> dict[str, Any]:
    row: dict[str, Any] = {
        "count": int(len(values)),
        "mean": NOT_AVAILABLE,
        "std": NOT_AVAILABLE,
        "median": NOT_AVAILABLE,
        "q25": NOT_AVAILABLE,
        "q75": NOT_AVAILABLE,
        "min": NOT_AVAILABLE,
        "max": NOT_AVAILABLE,
    }
    if len(values):
        row.update(
            {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=0)),
                "median": float(np.median(values)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        )
    return row


def entropy_analysis(
    predictions: pd.DataFrame,
    train_support: Mapping[str, int],
) -> tuple[pd.DataFrame, float | None, str | None]:
    """Backward-compatible three-group entropy summary and absent AUROC."""

    _, entropy, source_or_error = probabilities_and_entropy(predictions)
    if entropy is None:
        return pd.DataFrame(), None, source_or_error
    frame = predictions.copy()
    frame["_entropy"] = entropy
    absent = frame["true_label"].map(
        lambda label: int(train_support.get(str(label), 0)) == 0
    )
    correct = (
        frame["true_label"].astype(str)
        == frame["predicted_label"].astype(str)
    )
    groups = {
        "known-correct": (~absent) & correct,
        "known-incorrect": (~absent) & (~correct),
        "class-absent-from-train": absent,
    }
    rows: list[dict[str, Any]] = []
    for group_name, mask in groups.items():
        values = frame.loc[mask, "_entropy"].to_numpy(dtype=float)
        described = _describe_values(values)
        rows.append(
            {
                "group": group_name,
                **described,
                "q05": (
                    float(np.quantile(values, 0.05))
                    if len(values)
                    else NOT_AVAILABLE
                ),
                "q95": (
                    float(np.quantile(values, 0.95))
                    if len(values)
                    else NOT_AVAILABLE
                ),
                "entropy_source": source_or_error,
            }
        )
    eligible = groups["known-correct"] | groups["class-absent-from-train"]
    targets = absent[eligible].astype(int).to_numpy()
    scores = frame.loc[eligible, "_entropy"].to_numpy(dtype=float)
    if len(np.unique(targets)) == 2:
        return (
            pd.DataFrame(rows),
            float(roc_auc_score(targets, scores)),
            None,
        )
    return pd.DataFrame(rows), None, (
        "AUROC requires at least one known-correct row and one row whose "
        "true class has train support 0."
    )


def aggregate_seed_metrics(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    metric_columns: Sequence[str],
) -> pd.DataFrame:
    """Aggregate numeric metrics using population standard deviation."""

    if frame.empty:
        columns = list(group_columns) + ["seed_count"]
        for metric in metric_columns:
            columns.extend(
                [
                    f"{metric}_mean",
                    f"{metric}_std",
                    f"{metric}_min",
                    f"{metric}_max",
                    f"{metric}_mean_std",
                ]
            )
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(
        list(group_columns), dropna=False, sort=True
    )
    for group_key, group in grouped:
        keys = group_key if isinstance(group_key, tuple) else (group_key,)
        row = dict(zip(group_columns, keys))
        row["seed_count"] = int(group["seed"].nunique())
        for metric in metric_columns:
            numeric = pd.to_numeric(group[metric], errors="coerce").dropna()
            if numeric.empty:
                for suffix in ("mean", "std", "min", "max", "mean_std"):
                    row[f"{metric}_{suffix}"] = NOT_AVAILABLE
            else:
                mean = float(numeric.mean())
                std = float(numeric.std(ddof=0))
                row.update(
                    {
                        f"{metric}_mean": mean,
                        f"{metric}_std": std,
                        f"{metric}_min": float(numeric.min()),
                        f"{metric}_max": float(numeric.max()),
                        f"{metric}_mean_std": f"{mean:.6f} ± {std:.6f}",
                    }
                )
        rows.append(row)
    return pd.DataFrame(rows)


def discover_bundles(inputs: Iterable[Path], output_dir: Path) -> list[Path]:
    bundles: set[Path] = set()
    output_resolved = output_dir.resolve()
    for input_path in inputs:
        if not input_path.exists():
            continue
        candidates = (
            [input_path]
            if input_path.is_dir()
            and (
                (input_path / "metadata.json").is_file()
                or (input_path / "metrics.json").is_file()
            )
            else [path.parent for path in input_path.rglob("metadata.json")]
        )
        for candidate in candidates:
            resolved = candidate.resolve()
            if (
                resolved == output_resolved
                or output_resolved in resolved.parents
            ):
                continue
            bundles.add(resolved)
    return sorted(bundles)


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, f"Missing {path.name}."
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"Cannot read {path.name}: {exc}"
    if not isinstance(value, dict):
        return None, f"{path.name} must contain a JSON object."
    return value, None


def _prediction_audit(
    predictions: pd.DataFrame,
    *,
    held_out: str,
    train_support: Mapping[str, int],
) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    """Derive only fields whose source is unambiguous."""

    frame = predictions.copy()
    audit: dict[str, str] = {}
    notes: list[str] = []
    for column in (
        "seed",
        "protocol",
        "scenario",
        "split",
        "true_label",
        "predicted_label",
    ):
        audit[column] = (
            "EXPORTED" if column in frame.columns else NOT_AVAILABLE
        )
    if "held_out" in frame.columns:
        audit["held_out"] = "EXPORTED"
    else:
        frame["held_out"] = held_out
        audit["held_out"] = "DERIVED_FROM_METADATA"
    if "class_present_in_train" in frame.columns:
        audit["class_present_in_train"] = "EXPORTED"
    elif "true_class_absent_from_train" in frame.columns:
        frame["class_present_in_train"] = ~frame[
            "true_class_absent_from_train"
        ].astype(bool)
        audit["class_present_in_train"] = (
            "DERIVED_FROM_EXPORTED_ABSENCE_FLAG"
        )
    elif train_support and "true_label" in frame.columns:
        frame["class_present_in_train"] = frame["true_label"].map(
            lambda label: int(train_support.get(str(label), 0)) > 0
        )
        audit["class_present_in_train"] = "DERIVED_FROM_CLASS_SUPPORT"
    else:
        audit["class_present_in_train"] = NOT_AVAILABLE
        notes.append(
            "class_present_in_train requires true_label plus "
            "metadata.json:class_support.train."
        )
    if set(PROBABILITY_COLUMNS).issubset(frame.columns):
        audit["probabilities_or_logits"] = "PROBABILITIES_EXPORTED"
    elif set(LOGIT_COLUMNS).issubset(frame.columns):
        audit["probabilities_or_logits"] = "LOGITS_EXPORTED"
    else:
        audit["probabilities_or_logits"] = NOT_AVAILABLE
    audit["entropy"] = (
        "EXPORTED" if "entropy" in frame.columns else "DERIVABLE_FROM_SCORES"
    )
    return frame, audit, notes


def _per_class_frame(
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=list(FIXED_LABELS),
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "class": FIXED_LABELS,
            "precision": precision.astype(float),
            "recall": recall.astype(float),
            "f1": f1.astype(float),
            "support": support.astype(int),
        }
    )


def _entropy_products(
    predictions: pd.DataFrame,
    train_support: Mapping[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Return per-sample entropy, five-group summary, and detection metrics."""

    _, entropy, source_or_error = probabilities_and_entropy(predictions)
    if entropy is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [
            f"Entropy NOT_AVAILABLE: {source_or_error}"
        ]
    frame = predictions.copy()
    frame["_entropy"] = entropy
    absent = frame["true_label"].map(
        lambda label: int(train_support.get(str(label), 0)) == 0
    )
    correct = (
        frame["true_label"].astype(str)
        == frame["predicted_label"].astype(str)
    )
    benign = frame["true_label"].astype(str) == "Benign"
    masks = {
        "known_correct": (~absent) & correct,
        "known_incorrect": (~absent) & (~correct),
        "absent_from_train": absent,
        "benign": benign,
        "malicious": ~benign,
    }
    summary_rows = [
        {
            "group": group,
            **_describe_values(
                frame.loc[mask, "_entropy"].to_numpy(dtype=float)
            ),
            "entropy_source": source_or_error,
        }
        for group, mask in masks.items()
    ]
    comparisons = (
        (
            "absent_from_train vs known_correct",
            masks["absent_from_train"],
            masks["known_correct"],
            "higher_entropy_indicates_absent_from_train",
        ),
        (
            "known_incorrect vs known_correct",
            masks["known_incorrect"],
            masks["known_correct"],
            "higher_entropy_indicates_known_incorrect",
        ),
        (
            "malicious vs benign",
            masks["malicious"],
            masks["benign"],
            "higher_entropy_indicates_malicious",
        ),
    )
    detection_rows: list[dict[str, Any]] = []
    notes: list[str] = []
    for comparison, positive, negative, direction in comparisons:
        eligible = positive | negative
        targets = positive[eligible].astype(int).to_numpy()
        scores = frame.loc[eligible, "_entropy"].to_numpy(dtype=float)
        row: dict[str, Any] = {
            "comparison": comparison,
            "auroc": NOT_AVAILABLE,
            "auprc": NOT_AVAILABLE,
            "direction": direction,
            "positive_count": int(positive.sum()),
            "negative_count": int(negative.sum()),
            "status": NOT_AVAILABLE,
            "reason": "",
        }
        if len(np.unique(targets)) == 2:
            row.update(
                {
                    "auroc": float(roc_auc_score(targets, scores)),
                    "auprc": float(average_precision_score(targets, scores)),
                    "status": "AVAILABLE",
                }
            )
        else:
            row["reason"] = (
                "Requires at least one positive and one negative sample."
            )
            notes.append(f"{comparison}: {row['reason']}")
        detection_rows.append(row)
    frame["_absent_from_train"] = absent.astype(bool)
    frame["_known_correct"] = masks["known_correct"].astype(bool)
    return (
        frame,
        pd.DataFrame(summary_rows),
        pd.DataFrame(detection_rows),
        notes,
    )


def analyze_bundle(bundle: Path) -> dict[str, Any]:
    metadata, metadata_error = _read_json(bundle / "metadata.json")
    metrics, metrics_error = _read_json(bundle / "metrics.json")
    split_manifest, split_error = _read_json(bundle / "split_manifest.json")
    notes = [
        note
        for note in (metadata_error, metrics_error, split_error)
        if note is not None
    ]
    metadata = metadata or {}
    metrics = metrics or {}
    final = metrics.get("final", {})
    protocol = str(metadata.get("protocol", NOT_AVAILABLE))
    seed = metadata.get("seed", NOT_AVAILABLE)
    held_value = metadata.get("held_out")
    held_out = str(held_value) if held_value is not None else "ALL"
    support = metadata.get("class_support", {})
    train_support = support.get("train", {})
    if not isinstance(train_support, dict):
        train_support = {}
        notes.append(
            "Seen/absent analysis requires metadata.json:class_support.train."
        )
    inventory = {
        name: (bundle / name).is_file() for name in REQUIRED_BUNDLE_FILES
    }
    run: dict[str, Any] = {
        "bundle": str(bundle),
        "protocol": protocol,
        "seed": seed,
        "held_out": held_out,
        "accuracy": final.get("accuracy", NOT_AVAILABLE),
        "weighted_f1": final.get("weighted_f1", NOT_AVAILABLE),
        "fixed_8_macro_f1": final.get("macro_f1", NOT_AVAILABLE),
        "seen_class_macro_f1": NOT_AVAILABLE,
        "validation_macro_f1": metrics.get(
            "validation_macro_f1",
            metadata.get("validation_metric", NOT_AVAILABLE),
        ),
        "best_epoch": metrics.get(
            "best_epoch", metadata.get("best_epoch", NOT_AVAILABLE)
        ),
        "train_time": metadata.get("train_time", NOT_AVAILABLE),
        "history": (
            metrics.get("history", [])
            if isinstance(metrics.get("history", []), list)
            else []
        ),
        "prediction_status": NOT_AVAILABLE,
        "prediction_audit": {},
        "notes": notes,
        "class_support": support if isinstance(support, dict) else {},
        "zero_train_support_classes": sorted(
            label
            for label in FIXED_LABELS
            if int(train_support.get(label, 0)) == 0
        ),
        "binary": None,
        "per_class": pd.DataFrame(),
        "entropy_samples": pd.DataFrame(),
        "entropy_summary": pd.DataFrame(),
        "entropy_detection": pd.DataFrame(),
        "confusion_matrix": None,
        "predictions": pd.DataFrame(),
        "inventory": inventory,
        "split_manifest_available": split_manifest is not None,
    }
    prediction_path = bundle / "predictions.csv"
    if not prediction_path.is_file():
        run["notes"].append(
            "Per-sample analysis requires predictions.csv with identity and "
            "all probability::<fixed-label> or logit::<fixed-label> columns."
        )
        return run
    try:
        predictions = pd.read_csv(prediction_path)
    except (OSError, pd.errors.ParserError) as exc:
        run["notes"].append(f"Cannot read predictions.csv: {exc}")
        return run
    predictions, audit, audit_notes = _prediction_audit(
        predictions,
        held_out=held_out,
        train_support=train_support,
    )
    run["prediction_audit"] = audit
    run["notes"].extend(audit_notes)
    missing_identity = sorted(
        set(PREDICTION_IDENTITY_COLUMNS) - set(predictions.columns)
    )
    if missing_identity or predictions.empty:
        run["notes"].append(
            "predictions.csv missing required identity fields "
            f"{missing_identity} or contains no rows."
        )
        return run
    true_labels = predictions["true_label"].astype(str).tolist()
    predicted_labels = predictions["predicted_label"].astype(str).tolist()
    unknown = sorted(
        (set(true_labels) | set(predicted_labels)) - set(FIXED_LABELS)
    )
    if unknown:
        run["notes"].append(
            "Predictions contain labels outside fixed taxonomy: "
            + ", ".join(unknown)
        )
        return run
    run["prediction_status"] = "AVAILABLE"
    run["accuracy"] = float(accuracy_score(true_labels, predicted_labels))
    run["weighted_f1"] = float(
        f1_score(
            true_labels,
            predicted_labels,
            average="weighted",
            zero_division=0,
        )
    )
    run["fixed_8_macro_f1"] = fixed_class_macro_f1(
        true_labels, predicted_labels
    )
    seen = seen_class_macro_f1(
        true_labels, predicted_labels, train_support
    )
    run["seen_class_macro_f1"] = (
        seen if seen is not None else NOT_AVAILABLE
    )
    run["binary"] = binary_metrics(true_labels, predicted_labels)
    run["per_class"] = _per_class_frame(true_labels, predicted_labels)
    run["confusion_matrix"] = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(FIXED_LABELS),
    )
    entropy_samples, entropy_summary, detection, entropy_notes = (
        _entropy_products(predictions, train_support)
    )
    run["entropy_samples"] = entropy_samples
    run["entropy_summary"] = entropy_summary
    run["entropy_detection"] = detection
    run["notes"].extend(entropy_notes)
    run["predictions"] = predictions
    return run


def _run_frame(runs: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    columns = [
        "bundle",
        "protocol",
        "seed",
        "held_out",
        "accuracy",
        "weighted_f1",
        "fixed_8_macro_f1",
        "seen_class_macro_f1",
        "validation_macro_f1",
        "best_epoch",
        "train_time",
        "zero_train_support_classes",
        "prediction_status",
        "notes",
    ]
    rows = []
    for run in runs:
        rows.append(
            {
                "bundle": run["bundle"],
                "protocol": run["protocol"],
                "seed": run["seed"],
                "held_out": run["held_out"],
                "accuracy": run["accuracy"],
                "weighted_f1": run["weighted_f1"],
                "fixed_8_macro_f1": run["fixed_8_macro_f1"],
                "seen_class_macro_f1": run["seen_class_macro_f1"],
                "validation_macro_f1": run["validation_macro_f1"],
                "best_epoch": run["best_epoch"],
                "train_time": run["train_time"],
                "zero_train_support_classes": "|".join(
                    run["zero_train_support_classes"]
                ),
                "prediction_status": run["prediction_status"],
                "notes": " | ".join(run["notes"]),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _pooled_metrics_frame(runs: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    columns = [
        "seed",
        "accuracy",
        "weighted_f1",
        "fixed_macro_f1",
        "validation_macro_f1",
        "best_epoch",
        "train_time",
    ]
    return pd.DataFrame(
        [
            {
                "seed": run["seed"],
                "accuracy": run["accuracy"],
                "weighted_f1": run["weighted_f1"],
                "fixed_macro_f1": run["fixed_8_macro_f1"],
                "validation_macro_f1": run["validation_macro_f1"],
                "best_epoch": run["best_epoch"],
                "train_time": run["train_time"],
            }
            for run in runs
            if run["protocol"] == "pooled"
        ],
        columns=columns,
    ).sort_values("seed", ignore_index=True)


def _loso_metrics_frame(runs: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    columns = [
        "seed",
        "held_out",
        "fixed_macro_f1",
        "seen_class_macro_f1",
        "binary_precision",
        "binary_recall",
        "binary_f1",
        "accuracy",
        "validation_macro_f1",
        "best_epoch",
    ]
    rows = []
    for run in runs:
        if run["protocol"] != "loso":
            continue
        binary = run["binary"] or {}
        rows.append(
            {
                "seed": run["seed"],
                "held_out": run["held_out"],
                "fixed_macro_f1": run["fixed_8_macro_f1"],
                "seen_class_macro_f1": run["seen_class_macro_f1"],
                "binary_precision": binary.get(
                    "malicious_precision", NOT_AVAILABLE
                ),
                "binary_recall": binary.get(
                    "malicious_recall", NOT_AVAILABLE
                ),
                "binary_f1": binary.get(
                    "malicious_f1", NOT_AVAILABLE
                ),
                "accuracy": run["accuracy"],
                "validation_macro_f1": run["validation_macro_f1"],
                "best_epoch": run["best_epoch"],
            }
        )
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return frame
    order = {scenario: index for index, scenario in enumerate(SCENARIO_ORDER)}
    frame["_order"] = frame["held_out"].map(
        lambda value: order.get(str(value), len(order))
    )
    return frame.sort_values(
        ["_order", "seed"]
    ).drop(columns="_order").reset_index(drop=True)


def _class_support_frame(
    runs: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    columns = [
        "seed",
        "protocol",
        "held_out",
        "class",
        "train_support",
        "validation_support",
        "test_support",
        "present_in_train",
        "private_to_held_out",
    ]
    rows: list[dict[str, Any]] = []
    for run in runs:
        support = run["class_support"]
        for label in FIXED_LABELS:
            train = support.get("train", {}).get(label, NOT_AVAILABLE)
            validation = support.get("validation", {}).get(
                label, NOT_AVAILABLE
            )
            test = support.get("test", {}).get(label, NOT_AVAILABLE)
            present = (
                int(train) > 0
                if train != NOT_AVAILABLE
                else NOT_AVAILABLE
            )
            private = (
                run["protocol"] == "loso"
                and present is False
                and test != NOT_AVAILABLE
                and int(test) > 0
            )
            rows.append(
                {
                    "seed": run["seed"],
                    "protocol": run["protocol"],
                    "held_out": run["held_out"],
                    "class": label,
                    "train_support": train,
                    "validation_support": validation,
                    "test_support": test,
                    "present_in_train": present,
                    "private_to_held_out": private,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _per_class_metrics_frame(
    runs: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    frames = []
    for run in runs:
        frame = run["per_class"]
        if frame.empty:
            continue
        frame = frame.copy()
        frame.insert(0, "held_out", run["held_out"])
        frame.insert(0, "seed", run["seed"])
        frame.insert(0, "protocol", run["protocol"])
        frames.append(frame)
    columns = [
        "protocol",
        "seed",
        "held_out",
        "class",
        "precision",
        "recall",
        "f1",
        "support",
    ]
    return (
        pd.concat(frames, ignore_index=True).reindex(columns=columns)
        if frames
        else pd.DataFrame(columns=columns)
    )


def _entropy_summary_frame(
    runs: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    frames = []
    for run in runs:
        frame = run["entropy_summary"]
        if frame.empty:
            continue
        frame = frame.copy()
        frame.insert(0, "held_out", run["held_out"])
        frame.insert(0, "seed", run["seed"])
        frame.insert(0, "protocol", run["protocol"])
        frames.append(frame)
    columns = [
        "protocol",
        "seed",
        "held_out",
        "group",
        "count",
        "mean",
        "std",
        "median",
        "q25",
        "q75",
        "min",
        "max",
        "entropy_source",
    ]
    return (
        pd.concat(frames, ignore_index=True).reindex(columns=columns)
        if frames
        else pd.DataFrame(columns=columns)
    )


def _entropy_detection_frame(
    runs: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    frames = []
    for run in runs:
        frame = run["entropy_detection"]
        if frame.empty:
            continue
        frame = frame.copy()
        frame.insert(0, "held_out", run["held_out"])
        frame.insert(0, "seed", run["seed"])
        frame.insert(0, "protocol", run["protocol"])
        frames.append(frame)
    columns = [
        "protocol",
        "seed",
        "held_out",
        "comparison",
        "auroc",
        "auprc",
        "direction",
        "positive_count",
        "negative_count",
        "status",
        "reason",
    ]
    return (
        pd.concat(frames, ignore_index=True).reindex(columns=columns)
        if frames
        else pd.DataFrame(columns=columns)
    )


def _prediction_audit_frame(
    runs: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    rows = []
    for run in runs:
        for field, status in run["prediction_audit"].items():
            rows.append(
                {
                    "bundle": run["bundle"],
                    "protocol": run["protocol"],
                    "seed": run["seed"],
                    "held_out": run["held_out"],
                    "field": field,
                    "status": status,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "bundle",
            "protocol",
            "seed",
            "held_out",
            "field",
            "status",
        ],
    )


def _inventory_frame(
    runs: Sequence[Mapping[str, Any]],
    input_paths: Sequence[Path],
    output_dir: Path,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    rows = []
    for run in runs:
        for artifact, present in run["inventory"].items():
            rows.append(
                {
                    "bundle": run["bundle"],
                    "protocol": run["protocol"],
                    "seed": run["seed"],
                    "held_out": run["held_out"],
                    "artifact": artifact,
                    "present": present,
                }
            )
    existing_figures: list[str] = []
    existing_analysis_files: list[str] = []
    output_resolved = output_dir.resolve()
    for root in input_paths:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if (
                resolved == output_resolved
                or output_resolved in resolved.parents
            ):
                continue
            if path.suffix.lower() in {".png", ".pdf", ".svg"}:
                existing_figures.append(str(path))
            elif "analysis" in path.parts and path.suffix.lower() in {
                ".csv",
                ".md",
            }:
                existing_analysis_files.append(str(path))
    return (
        pd.DataFrame(
            rows,
            columns=[
                "bundle",
                "protocol",
                "seed",
                "held_out",
                "artifact",
                "present",
            ],
        ),
        sorted(set(existing_figures)),
        sorted(set(existing_analysis_files)),
    )


def _historical_vs_clean(
    run_frame: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "protocol",
        "historical_metric",
        "clean_seed42_metric",
        "clean_mean_metric",
        "note",
    ]
    root = REPOSITORY_ROOT / "artifacts/phase1_results"
    records = []
    definitions = (
        (
            "pooled",
            root / "phase_a_pooled_egraphsage_3modes.csv",
        ),
        (
            "loso",
            root / "phase_a_loso_egraphsage_3modes.csv",
        ),
    )
    for protocol, path in definitions:
        if not path.is_file():
            continue
        try:
            historical = pd.read_csv(path)
        except (OSError, pd.errors.ParserError):
            continue
        if not {
            "model",
            "imbalance_mode",
            "macro_f1",
        }.issubset(historical.columns):
            continue
        selected = historical[
            (historical["model"] == "egraphsage")
            & (historical["imbalance_mode"] == "class_weight")
        ]
        historical_values = pd.to_numeric(
            selected.get("macro_f1"), errors="coerce"
        ).dropna()
        if historical_values.empty:
            continue
        historical_metric = float(
            historical_values.iloc[0]
            if protocol == "pooled"
            else historical_values.mean()
        )
        clean = run_frame[run_frame["protocol"] == protocol].copy()
        if protocol == "loso" and not clean.empty:
            clean = (
                clean.groupby("seed", as_index=False)["fixed_8_macro_f1"]
                .apply(
                    lambda group: pd.to_numeric(
                        group, errors="coerce"
                    ).mean()
                )
                .rename(columns={None: "fixed_8_macro_f1"})
            )
        clean_values = pd.to_numeric(
            clean.get("fixed_8_macro_f1"), errors="coerce"
        ).dropna()
        seed42 = run_frame[
            (run_frame["protocol"] == protocol)
            & (pd.to_numeric(run_frame["seed"], errors="coerce") == 42)
        ]
        seed42_values = pd.to_numeric(
            seed42.get("fixed_8_macro_f1"), errors="coerce"
        ).dropna()
        records.append(
            {
                "protocol": protocol,
                "historical_metric": historical_metric,
                "clean_seed42_metric": (
                    float(seed42_values.mean())
                    if not seed42_values.empty
                    else NOT_AVAILABLE
                ),
                "clean_mean_metric": (
                    float(clean_values.mean())
                    if not clean_values.empty
                    else NOT_AVAILABLE
                ),
                "note": (
                    "Historical result is leakage-prone/context-only and is "
                    "not included in clean aggregation."
                ),
            }
        )
    return pd.DataFrame(records, columns=columns)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return NOT_AVAILABLE
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for _, row in frame.iterrows():
        values = [
            str(row[column]).replace("|", "\\|").replace("\n", " ")
            for column in columns
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


@dataclass
class FigureResult:
    name: str
    status: str
    reason: str = ""


def _plot_runtime() -> tuple[Any, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return matplotlib, plt


def _save_figure(
    figure: Any,
    figures_dir: Path,
    stem: str,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        figures_dir / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
    )
    figure.savefig(
        figures_dir / f"{stem}.pdf",
        bbox_inches="tight",
    )


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _aggregate_loso_for_figures(
    loso: pd.DataFrame,
) -> pd.DataFrame:
    if loso.empty:
        return pd.DataFrame()
    rows = []
    for held_out, group in loso.groupby("held_out", sort=False):
        row = {"held_out": str(held_out), "seed_count": group["seed"].nunique()}
        for metric in (
            "fixed_macro_f1",
            "seen_class_macro_f1",
            "binary_f1",
        ):
            values = _numeric(group, metric).dropna()
            row[f"{metric}_mean"] = (
                float(values.mean()) if not values.empty else np.nan
            )
            row[f"{metric}_std"] = (
                float(values.std(ddof=0)) if not values.empty else np.nan
            )
        rows.append(row)
    frame = pd.DataFrame(rows)
    order = {scenario: index for index, scenario in enumerate(SCENARIO_ORDER)}
    frame["_order"] = frame["held_out"].map(
        lambda value: order.get(str(value), len(order))
    )
    return frame.sort_values("_order").drop(columns="_order")


def _plot_confusion(
    matrix: np.ndarray,
    *,
    title: str,
    normalized: bool,
    stem: str,
    figures_dir: Path,
) -> None:
    _, plt = _plot_runtime()
    values = matrix.astype(float)
    if normalized:
        denominators = values.sum(axis=1, keepdims=True)
        values = np.divide(
            values,
            denominators,
            out=np.zeros_like(values),
            where=denominators != 0,
        )
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(values, cmap="Blues", vmin=0)
    figure.colorbar(image, ax=axis)
    axis.set_xticks(
        range(len(FIXED_LABELS)), FIXED_LABELS, rotation=45, ha="right"
    )
    axis.set_yticks(range(len(FIXED_LABELS)), FIXED_LABELS)
    axis.set_xlabel("Nhãn dự đoán (Predicted)")
    axis.set_ylabel("Nhãn thật (True)")
    axis.set_title(title)
    threshold = float(values.max()) / 2 if values.size else 0
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            text = (
                f"{values[row, column]:.2f}"
                if normalized
                else str(int(matrix[row, column]))
            )
            axis.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                fontsize=7,
                color="white" if values[row, column] > threshold else "black",
            )
    _save_figure(figure, figures_dir, stem)
    plt.close(figure)


def _representative_run(
    runs: Sequence[Mapping[str, Any]],
    *,
    protocol: str,
    held_out: str | None = None,
) -> Mapping[str, Any] | None:
    selected = [
        run
        for run in runs
        if run["protocol"] == protocol
        and (held_out is None or run["held_out"] == held_out)
        and run["confusion_matrix"] is not None
    ]
    if not selected:
        return None
    return sorted(
        selected,
        key=lambda run: (
            0 if str(run["seed"]) == "42" else 1,
            str(run["seed"]),
        ),
    )[0]


def _generate_figures(
    runs: Sequence[Mapping[str, Any]],
    pooled: pd.DataFrame,
    loso: pd.DataFrame,
    support: pd.DataFrame,
    per_class: pd.DataFrame,
    figures_dir: Path,
) -> list[FigureResult]:
    _, plt = _plot_runtime()
    results: list[FigureResult] = []
    loso_agg = _aggregate_loso_for_figures(loso)
    seed_count = int(
        pd.to_numeric(
            pd.Series([run["seed"] for run in runs]), errors="coerce"
        ).dropna().nunique()
    )

    def unavailable(stem: str, reason: str) -> None:
        results.append(FigureResult(stem, NOT_AVAILABLE, reason))

    # 01 pooled vs LOSO fixed macro-F1.
    pooled_values = _numeric(pooled, "fixed_macro_f1").dropna()
    loso_seed = (
        loso.assign(
            fixed_macro_f1=pd.to_numeric(
                loso["fixed_macro_f1"], errors="coerce"
            )
        )
        .groupby("seed")["fixed_macro_f1"]
        .mean()
        .dropna()
        if not loso.empty
        else pd.Series(dtype=float)
    )
    if pooled_values.empty or loso_seed.empty:
        unavailable(
            "fig01_pooled_vs_loso_macro_f1",
            "Requires pooled and LOSO fixed macro-F1.",
        )
    else:
        figure, axis = plt.subplots(figsize=(7, 5))
        means = [float(pooled_values.mean()), float(loso_seed.mean())]
        errors = [
            float(pooled_values.std(ddof=0)),
            float(loso_seed.std(ddof=0)),
        ]
        axis.bar(
            ["Pooled", "LOSO trung bình"],
            means,
            yerr=errors if seed_count > 1 else None,
            capsize=5,
            color=["#4C78A8", "#F58518"],
        )
        axis.set_ylim(0, 1)
        axis.set_ylabel("Fixed 8-class macro-F1")
        axis.set_title("Pooled so với khả năng tổng quát hóa LOSO")
        if seed_count == 1:
            axis.text(
                0.5,
                0.96,
                f"Một seed: {pooled.iloc[0]['seed']}",
                transform=axis.transAxes,
                ha="center",
            )
        _save_figure(figure, figures_dir, "fig01_pooled_vs_loso_macro_f1")
        plt.close(figure)
        results.append(FigureResult("fig01_pooled_vs_loso_macro_f1", "CREATED"))

    # 02 LOSO by scenario.
    if loso_agg.empty:
        unavailable("fig02_loso_by_scenario", "No LOSO metrics.")
    else:
        figure, axis = plt.subplots(figsize=(9, 5))
        x = np.arange(len(loso_agg))
        means = loso_agg["fixed_macro_f1_mean"].to_numpy(dtype=float)
        errors = loso_agg["fixed_macro_f1_std"].to_numpy(dtype=float)
        bars = axis.bar(
            x,
            means,
            yerr=errors if seed_count > 1 else None,
            capsize=4,
            color="#4C78A8",
        )
        private_folds = set(
            support.loc[
                (support["protocol"] == "loso")
                & (support["private_to_held_out"] == True),  # noqa: E712
                "held_out",
            ].astype(str)
        )
        for index, (bar, scenario) in enumerate(
            zip(bars, loso_agg["held_out"].astype(str))
        ):
            if scenario in private_folds:
                bar.set_color("#E45756")
                axis.text(
                    index,
                    means[index] + 0.02,
                    "private class",
                    rotation=90,
                    ha="center",
                    fontsize=8,
                )
        axis.set_xticks(x, loso_agg["held_out"])
        axis.set_ylim(0, 1)
        axis.set_ylabel("Fixed 8-class macro-F1")
        axis.set_xlabel("Held-out scenario")
        axis.set_title("LOSO macro-F1 theo scenario")
        _save_figure(figure, figures_dir, "fig02_loso_by_scenario")
        plt.close(figure)
        results.append(FigureResult("fig02_loso_by_scenario", "CREATED"))

    # 03 decomposition.
    if loso_agg.empty:
        unavailable(
            "fig03_loso_metric_decomposition", "No LOSO decomposition metrics."
        )
    else:
        figure, axis = plt.subplots(figsize=(11, 5))
        x = np.arange(len(loso_agg))
        width = 0.25
        for offset, (column, label, color) in enumerate(
            (
                ("fixed_macro_f1_mean", "Fixed macro-F1", "#4C78A8"),
                ("seen_class_macro_f1_mean", "Seen-class macro-F1", "#F58518"),
                ("binary_f1_mean", "Binary malicious F1", "#54A24B"),
            )
        ):
            axis.bar(
                x + (offset - 1) * width,
                loso_agg[column],
                width,
                label=label,
                color=color,
            )
        axis.set_xticks(x, loso_agg["held_out"])
        axis.set_ylim(0, 1)
        axis.set_ylabel("F1")
        axis.set_xlabel("Held-out scenario")
        axis.set_title("Phân rã chỉ số LOSO")
        axis.legend()
        _save_figure(
            figure, figures_dir, "fig03_loso_metric_decomposition"
        )
        plt.close(figure)
        results.append(
            FigureResult("fig03_loso_metric_decomposition", "CREATED")
        )

    # 04 availability heatmap.
    loso_support = support[support["protocol"] == "loso"].copy()
    if loso_support.empty:
        unavailable(
            "fig04_train_class_availability_heatmap",
            "Requires LOSO class support.",
        )
    else:
        availability = (
            loso_support.assign(
                present=pd.to_numeric(
                    loso_support["present_in_train"], errors="coerce"
                )
            )
            .groupby(["held_out", "class"])["present"]
            .mean()
            .unstack("class")
            .reindex(index=SCENARIO_ORDER, columns=FIXED_LABELS)
        )
        figure, axis = plt.subplots(figsize=(12, 5))
        image = axis.imshow(
            availability.to_numpy(dtype=float),
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            aspect="auto",
        )
        figure.colorbar(image, ax=axis, label="Tỷ lệ seed có class trong train")
        axis.set_xticks(
            range(len(FIXED_LABELS)), FIXED_LABELS, rotation=45, ha="right"
        )
        axis.set_yticks(range(len(SCENARIO_ORDER)), SCENARIO_ORDER)
        for row in range(availability.shape[0]):
            for column in range(availability.shape[1]):
                value = availability.iloc[row, column]
                if pd.notna(value):
                    text = (
                        "✓" if value == 1 else ("×" if value == 0 else f"{value:.2f}")
                    )
                    axis.text(column, row, text, ha="center", va="center")
        axis.set_title("Class có/không có trong tập train của từng LOSO fold")
        _save_figure(
            figure,
            figures_dir,
            "fig04_train_class_availability_heatmap",
        )
        plt.close(figure)
        results.append(
            FigureResult(
                "fig04_train_class_availability_heatmap", "CREATED"
            )
        )

    # 05 train/test support heatmap, log10(1 + support).
    if loso_support.empty:
        unavailable(
            "fig05_train_test_support_heatmap",
            "Requires LOSO class support.",
        )
    else:
        support_mean = (
            loso_support.groupby(["held_out", "class"])[
                ["train_support", "test_support"]
            ]
            .mean(numeric_only=True)
            .reindex(
                pd.MultiIndex.from_product(
                    [SCENARIO_ORDER, FIXED_LABELS],
                    names=["held_out", "class"],
                )
            )
        )
        train_matrix = support_mean["train_support"].unstack("class")
        test_matrix = support_mean["test_support"].unstack("class")
        combined = np.vstack(
            [
                np.log10(1 + train_matrix.to_numpy(dtype=float)),
                np.log10(1 + test_matrix.to_numpy(dtype=float)),
            ]
        )
        ylabels = [
            *(f"{scenario} train" for scenario in SCENARIO_ORDER),
            *(f"{scenario} test" for scenario in SCENARIO_ORDER),
        ]
        figure, axis = plt.subplots(figsize=(12, 8))
        image = axis.imshow(combined, cmap="viridis", aspect="auto")
        figure.colorbar(image, ax=axis, label="log10(1 + support)")
        axis.set_xticks(
            range(len(FIXED_LABELS)), FIXED_LABELS, rotation=45, ha="right"
        )
        axis.set_yticks(range(len(ylabels)), ylabels)
        axis.set_title("Support train/test theo class và LOSO fold")
        _save_figure(
            figure, figures_dir, "fig05_train_test_support_heatmap"
        )
        plt.close(figure)
        results.append(
            FigureResult("fig05_train_test_support_heatmap", "CREATED")
        )

    # 06 entropy boxplot.
    entropy_values: dict[str, list[float]] = {
        "known-correct": [],
        "known-incorrect": [],
        "absent-from-train": [],
    }
    for run in runs:
        samples = run["entropy_samples"]
        if samples.empty:
            continue
        absent = samples["_absent_from_train"].astype(bool)
        correct = samples["_known_correct"].astype(bool)
        entropy_values["known-correct"].extend(
            samples.loc[correct, "_entropy"].astype(float).tolist()
        )
        entropy_values["known-incorrect"].extend(
            samples.loc[(~absent) & (~correct), "_entropy"].astype(float).tolist()
        )
        entropy_values["absent-from-train"].extend(
            samples.loc[absent, "_entropy"].astype(float).tolist()
        )
    if not any(entropy_values.values()):
        unavailable(
            "fig06_entropy_boxplot",
            "Requires probabilities/logits in predictions.csv.",
        )
    else:
        labels = list(entropy_values)
        data = [entropy_values[label] for label in labels]
        figure, axis = plt.subplots(figsize=(9, 5))
        axis.boxplot(
            [values if values else [np.nan] for values in data],
            tick_labels=[
                f"{label}\n(n={len(values)})"
                for label, values in zip(labels, data)
            ],
            showfliers=False,
        )
        axis.set_ylabel("Entropy")
        axis.set_title(
            "Phân bố entropy theo nhóm (phân tích bất định, không phải claim zero-day)"
        )
        _save_figure(figure, figures_dir, "fig06_entropy_boxplot")
        plt.close(figure)
        results.append(FigureResult("fig06_entropy_boxplot", "CREATED"))

    # 07 combined absent-vs-known-correct ROC.
    roc_targets: list[int] = []
    roc_scores: list[float] = []
    for run in runs:
        samples = run["entropy_samples"]
        if samples.empty:
            continue
        eligible = (
            samples["_absent_from_train"].astype(bool)
            | samples["_known_correct"].astype(bool)
        )
        roc_targets.extend(
            samples.loc[eligible, "_absent_from_train"].astype(int).tolist()
        )
        roc_scores.extend(
            samples.loc[eligible, "_entropy"].astype(float).tolist()
        )
    if len(set(roc_targets)) != 2:
        reason = (
            "Không đủ cả absent-from-train và known-correct để tính ROC."
        )
        (figures_dir / "fig07_entropy_roc_NOT_AVAILABLE.md").write_text(
            "# Entropy ROC — NOT_AVAILABLE\n\n" + reason + "\n",
            encoding="utf-8",
        )
        unavailable("fig07_entropy_roc", reason)
    else:
        fpr, tpr, _ = roc_curve(roc_targets, roc_scores)
        auc = roc_auc_score(roc_targets, roc_scores)
        figure, axis = plt.subplots(figsize=(6, 6))
        axis.plot(fpr, tpr, label=f"Entropy AUROC = {auc:.3f}")
        axis.plot([0, 1], [0, 1], "--", color="gray", label="Ngẫu nhiên")
        axis.set_xlabel("False positive rate")
        axis.set_ylabel("True positive rate")
        axis.set_title("Entropy: absent-from-train vs known-correct")
        axis.legend()
        _save_figure(figure, figures_dir, "fig07_entropy_roc")
        plt.close(figure)
        results.append(FigureResult("fig07_entropy_roc", "CREATED"))

    # 08 pooled learning curve.
    pooled_runs = [
        run
        for run in runs
        if run["protocol"] == "pooled" and run["history"]
    ]
    if not pooled_runs:
        unavailable(
            "fig08_pooled_learning_curve",
            "Requires metrics.json:history for pooled run.",
        )
    else:
        figure, loss_axis = plt.subplots(figsize=(9, 5))
        validation_axis = loss_axis.twinx()
        for run in sorted(pooled_runs, key=lambda item: str(item["seed"])):
            history = pd.DataFrame(run["history"])
            if not {"epoch", "train_loss", "validation_macro_f1"}.issubset(
                history.columns
            ):
                continue
            loss_axis.plot(
                history["epoch"],
                history["train_loss"],
                alpha=0.8,
                label=f"Loss seed {run['seed']}",
            )
            validation_axis.plot(
                history["epoch"],
                history["validation_macro_f1"],
                linestyle="--",
                alpha=0.8,
                label=f"Val F1 seed {run['seed']}",
            )
            validation_axis.axvline(
                float(run["best_epoch"]), color="gray", alpha=0.2
            )
        loss_axis.set_xlabel("Epoch")
        loss_axis.set_ylabel("Train loss")
        validation_axis.set_ylabel("Validation macro-F1")
        validation_axis.set_ylim(0, 1)
        loss_axis.set_title("Learning curve — pooled")
        handles1, labels1 = loss_axis.get_legend_handles_labels()
        handles2, labels2 = validation_axis.get_legend_handles_labels()
        loss_axis.legend(handles1 + handles2, labels1 + labels2, fontsize=8)
        _save_figure(figure, figures_dir, "fig08_pooled_learning_curve")
        plt.close(figure)
        results.append(
            FigureResult("fig08_pooled_learning_curve", "CREATED")
        )

    hardest = (
        str(
            loso_agg.loc[
                loso_agg["fixed_macro_f1_mean"].idxmin(), "held_out"
            ]
        )
        if not loso_agg.empty
        else None
    )
    # 09 hardest learning curve.
    hardest_runs = [
        run
        for run in runs
        if run["protocol"] == "loso"
        and run["held_out"] == hardest
        and run["history"]
    ]
    if not hardest_runs:
        unavailable(
            "fig09_hardest_loso_learning_curve",
            "Requires LOSO metrics/history.",
        )
    else:
        figure, loss_axis = plt.subplots(figsize=(9, 5))
        validation_axis = loss_axis.twinx()
        for run in sorted(hardest_runs, key=lambda item: str(item["seed"])):
            history = pd.DataFrame(run["history"])
            loss_axis.plot(
                history["epoch"],
                history["train_loss"],
                label=f"Loss seed {run['seed']}",
            )
            validation_axis.plot(
                history["epoch"],
                history["validation_macro_f1"],
                "--",
                label=f"Val F1 seed {run['seed']}",
            )
            validation_axis.axvline(
                float(run["best_epoch"]), color="gray", alpha=0.2
            )
        loss_axis.set_xlabel("Epoch")
        loss_axis.set_ylabel("Train loss")
        validation_axis.set_ylabel("Validation macro-F1")
        validation_axis.set_ylim(0, 1)
        loss_axis.set_title(f"Learning curve — hardest LOSO {hardest}")
        handles1, labels1 = loss_axis.get_legend_handles_labels()
        handles2, labels2 = validation_axis.get_legend_handles_labels()
        loss_axis.legend(handles1 + handles2, labels1 + labels2, fontsize=8)
        _save_figure(
            figure, figures_dir, "fig09_hardest_loso_learning_curve"
        )
        plt.close(figure)
        results.append(
            FigureResult("fig09_hardest_loso_learning_curve", "CREATED")
        )

    # 10-13 representative fixed-order confusion matrices.
    representative_pooled = _representative_run(runs, protocol="pooled")
    representative_hardest = _representative_run(
        runs, protocol="loso", held_out=hardest
    )
    confusion_specs = (
        (
            representative_pooled,
            False,
            "fig10_pooled_confusion_matrix_counts",
            "Confusion matrix pooled — counts",
        ),
        (
            representative_pooled,
            True,
            "fig11_pooled_confusion_matrix_normalized",
            "Confusion matrix pooled — normalized by true class",
        ),
        (
            representative_hardest,
            False,
            "fig12_hardest_loso_confusion_matrix_counts",
            f"Confusion matrix hardest LOSO {hardest} — counts",
        ),
        (
            representative_hardest,
            True,
            "fig13_hardest_loso_confusion_matrix_normalized",
            f"Confusion matrix hardest LOSO {hardest} — normalized by true class",
        ),
    )
    for run, normalized, stem, title in confusion_specs:
        if run is None:
            unavailable(stem, "Requires predictions for representative run.")
            continue
        _plot_confusion(
            np.asarray(run["confusion_matrix"], dtype=int),
            title=f"{title} (seed {run['seed']})",
            normalized=normalized,
            stem=stem,
            figures_dir=figures_dir,
        )
        results.append(FigureResult(stem, "CREATED"))

    # 14 recall pooled vs hardest.
    pooled_class = per_class[per_class["protocol"] == "pooled"]
    hardest_class = per_class[
        (per_class["protocol"] == "loso")
        & (per_class["held_out"].astype(str) == str(hardest))
    ]
    if pooled_class.empty or hardest_class.empty:
        unavailable(
            "fig14_per_class_recall_pooled_vs_hardest",
            "Requires per-class predictions for pooled and hardest LOSO.",
        )
    else:
        pooled_recall = (
            pooled_class.groupby("class")["recall"]
            .mean()
            .reindex(FIXED_LABELS)
        )
        hardest_recall = (
            hardest_class.groupby("class")["recall"]
            .mean()
            .reindex(FIXED_LABELS)
        )
        figure, axis = plt.subplots(figsize=(12, 5))
        x = np.arange(len(FIXED_LABELS))
        width = 0.38
        axis.bar(
            x - width / 2,
            pooled_recall,
            width,
            label="Pooled",
            color="#4C78A8",
        )
        axis.bar(
            x + width / 2,
            hardest_recall,
            width,
            label=f"LOSO {hardest}",
            color="#E45756",
        )
        axis.set_xticks(x, FIXED_LABELS, rotation=45, ha="right")
        axis.set_ylim(0, 1)
        axis.set_ylabel("Recall")
        axis.set_title("Recall theo class: pooled vs hardest LOSO")
        axis.legend()
        _save_figure(
            figure,
            figures_dir,
            "fig14_per_class_recall_pooled_vs_hardest",
        )
        plt.close(figure)
        results.append(
            FigureResult(
                "fig14_per_class_recall_pooled_vs_hardest", "CREATED"
            )
        )

    # 15 stability only with >= 3 seeds.
    if seed_count < 3:
        unavailable(
            "fig15_seed_stability",
            f"Requires at least 3 seeds; found {seed_count}.",
        )
    else:
        pooled_seed = (
            pooled.assign(
                fixed_macro_f1=pd.to_numeric(
                    pooled["fixed_macro_f1"], errors="coerce"
                )
            )
            .set_index("seed")["fixed_macro_f1"]
        )
        loso_per_seed = (
            loso.assign(
                fixed_macro_f1=pd.to_numeric(
                    loso["fixed_macro_f1"], errors="coerce"
                )
            )
            .groupby("seed")["fixed_macro_f1"]
            .mean()
        )
        seeds = sorted(set(pooled_seed.index) & set(loso_per_seed.index))
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(
            seeds,
            pooled_seed.reindex(seeds),
            marker="o",
            label="Pooled macro-F1",
        )
        axis.plot(
            seeds,
            loso_per_seed.reindex(seeds),
            marker="o",
            label="LOSO mean macro-F1",
        )
        axis.axhline(
            pooled_seed.reindex(seeds).mean(),
            color="#4C78A8",
            linestyle=":",
        )
        axis.axhline(
            loso_per_seed.reindex(seeds).mean(),
            color="#F58518",
            linestyle=":",
        )
        axis.set_xticks(seeds)
        axis.set_ylim(0, 1)
        axis.set_xlabel("Seed")
        axis.set_ylabel("Fixed macro-F1")
        axis.set_title("Độ ổn định theo seed")
        axis.legend()
        _save_figure(figure, figures_dir, "fig15_seed_stability")
        plt.close(figure)
        results.append(FigureResult("fig15_seed_stability", "CREATED"))
    return results


def _write_figure_selection(
    output_dir: Path,
    results: Sequence[FigureResult],
) -> None:
    created = {result.name for result in results if result.status == "CREATED"}
    preferred = [
        "fig01_pooled_vs_loso_macro_f1",
        "fig02_loso_by_scenario",
        "fig03_loso_metric_decomposition",
        "fig04_train_class_availability_heatmap",
        "fig11_pooled_confusion_matrix_normalized",
        "fig13_hardest_loso_confusion_matrix_normalized",
    ]
    main = [name for name in preferred if name in created][:6]
    appendix = sorted(created - set(main))
    unavailable = [
        result for result in results if result.status != "CREATED"
    ]
    lines = [
        "# Phase 1 Figure Selection",
        "",
        "## Báo cáo chính (tối đa 6 hình)",
        "",
        *(
            [f"- `figures/{name}.png`" for name in main]
            if main
            else [f"- {NOT_AVAILABLE}"]
        ),
        "",
        "## Phụ lục",
        "",
        *(
            [f"- `figures/{name}.png`" for name in appendix]
            if appendix
            else ["- Không có hình bổ sung."]
        ),
        "",
        "## Không thể tạo",
        "",
        *(
            [
                f"- `{item.name}`: {item.reason}"
                for item in unavailable
            ]
            if unavailable
            else ["- Không có."]
        ),
    ]
    (output_dir / "FIGURE_SELECTION.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_analysis(
    runs: Sequence[Mapping[str, Any]],
    output_dir: Path,
    input_paths: Sequence[Path],
) -> dict[str, Any]:
    historical = (REPOSITORY_ROOT / "artifacts/phase1_results").resolve()
    resolved_output = output_dir.resolve()
    if resolved_output == historical or historical in resolved_output.parents:
        raise ValueError("Analysis output cannot be historical Phase 1 artifacts.")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    run_frame = _run_frame(runs)
    pooled = _pooled_metrics_frame(runs)
    loso = _loso_metrics_frame(runs)
    class_support = _class_support_frame(runs)
    per_class = _per_class_metrics_frame(runs)
    entropy = _entropy_summary_frame(runs)
    entropy_detection = _entropy_detection_frame(runs)
    prediction_audit = _prediction_audit_frame(runs)
    inventory, existing_figures, existing_analysis = _inventory_frame(
        runs, input_paths, output_dir
    )
    historical_comparison = _historical_vs_clean(run_frame)
    summary = aggregate_seed_metrics(
        run_frame,
        group_columns=("protocol", "held_out"),
        metric_columns=(
            "accuracy",
            "weighted_f1",
            "fixed_8_macro_f1",
            "seen_class_macro_f1",
        ),
    )
    loso_aggregate = aggregate_seed_metrics(
        loso,
        group_columns=("held_out",),
        metric_columns=(
            "fixed_macro_f1",
            "seen_class_macro_f1",
            "binary_f1",
        ),
    )
    loso_aggregate = loso_aggregate.rename(
        columns={
            column: column.replace(
                "seen_class_macro_f1_", "seen_macro_f1_"
            )
            for column in loso_aggregate.columns
            if column.startswith("seen_class_macro_f1_")
        }
    )
    binary_rows = []
    for run in runs:
        row = {
            "bundle": run["bundle"],
            "protocol": run["protocol"],
            "seed": run["seed"],
            "held_out": run["held_out"],
        }
        row.update(
            run["binary"]
            or {
                "accuracy": NOT_AVAILABLE,
                "malicious_precision": NOT_AVAILABLE,
                "malicious_recall": NOT_AVAILABLE,
                "malicious_f1": NOT_AVAILABLE,
            }
        )
        binary_rows.append(row)
    binary = pd.DataFrame(
        binary_rows,
        columns=[
            "bundle",
            "protocol",
            "seed",
            "held_out",
            "accuracy",
            "malicious_precision",
            "malicious_recall",
            "malicious_f1",
        ],
    )

    outputs = {
        "summary.csv": summary,
        "pooled_summary.csv": run_frame[
            run_frame["protocol"] == "pooled"
        ],
        "loso_summary.csv": run_frame[run_frame["protocol"] == "loso"],
        "pooled_metrics_by_seed.csv": pooled,
        "loso_metrics_by_seed.csv": loso,
        "loso_aggregate.csv": loso_aggregate,
        "class_support.csv": class_support,
        "per_class_metrics.csv": per_class,
        "binary_metrics.csv": binary,
        "entropy_summary.csv": entropy,
        "entropy_detection_metrics.csv": entropy_detection.rename(
            columns={"auroc": "AUROC", "auprc": "AUPRC"}
        ),
        "historical_vs_clean.csv": historical_comparison,
        "prediction_export_audit.csv": prediction_audit,
        "artifact_inventory.csv": inventory,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)

    figure_results = _generate_figures(
        runs, pooled, loso, class_support, per_class, figures_dir
    )
    _write_figure_selection(output_dir, figure_results)
    seeds = sorted(
        {
            int(seed)
            for seed in pd.to_numeric(
                run_frame.get("seed", pd.Series(dtype=float)),
                errors="coerce",
            ).dropna()
        }
    )
    single_seed = len(seeds) == 1
    uncertainty = NOT_AVAILABLE if len(seeds) <= 1 else "AVAILABLE"
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(
            {
                "seeds": seeds,
                "seed_count": len(seeds),
                "single_seed": single_seed,
                "statistical_uncertainty": uncertainty,
                "bundle_count": len(runs),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    created_figures = [
        result.name
        for result in figure_results
        if result.status == "CREATED"
    ]
    missing_figures = [
        result for result in figure_results if result.status != "CREATED"
    ]
    loso_fixed = pd.to_numeric(
        loso.get("fixed_macro_f1", pd.Series(dtype=float)), errors="coerce"
    )
    loso_seen = pd.to_numeric(
        loso.get("seen_class_macro_f1", pd.Series(dtype=float)),
        errors="coerce",
    )
    loso_binary = pd.to_numeric(
        loso.get("binary_f1", pd.Series(dtype=float)), errors="coerce"
    )
    hardest = (
        str(
            loso_aggregate.loc[
                pd.to_numeric(
                    loso_aggregate["fixed_macro_f1_mean"],
                    errors="coerce",
                ).idxmin(),
                "held_out",
            ]
        )
        if not loso_aggregate.empty
        else NOT_AVAILABLE
    )
    entropy_auc_rows = entropy_detection[
        entropy_detection.get("comparison", pd.Series(dtype=str))
        == "absent_from_train vs known_correct"
    ]
    report_lines = [
        "# Phase 1 Clean Report-Ready Analysis",
        "",
        "This report aggregates immutable clean bundles only. It does not train "
        "a model, alter stored metrics, or infer unavailable samples.",
        "",
        "## 1. Run inventory",
        "",
        f"- Inputs: {', '.join(f'`{path}`' for path in input_paths)}",
        f"- Bundles: **{len(runs)}**",
        f"- Seeds: **{len(seeds)}** ({seeds})",
        f"- `single_seed={str(single_seed).lower()}`",
        "- `statistical_uncertainty="
        + uncertainty
        + "`",
        f"- Existing input figures discovered: **{len(existing_figures)}**",
        f"- Existing input analysis CSV/MD discovered: **{len(existing_analysis)}**",
        "",
        "The prior seven figures, when present, are one count confusion matrix "
        "for pooled plus one for each of six LOSO bundles. The report-ready "
        "set below adds normalized matrices, aggregate comparisons, support, "
        "entropy, and learning curves.",
        "",
        "## 2. Pooled result",
        "",
        _markdown_table(pooled),
        "",
        "## 3. LOSO result",
        "",
        _markdown_table(loso_aggregate),
        "",
        "## 4. Hardest scenarios",
        "",
        f"- Hardest fold by clean fixed macro-F1 mean: **{hardest}**.",
        "- Hardness is reported from observed held-out performance; private "
        "classes are identified separately from class support.",
        "",
        "## 5. Class absence explanation",
        "",
        "A class with `train_support=0` remains in the fixed eight-class output "
        "space but was not learned in that fold. `private_to_held_out=true` "
        "means it appears in held-out test support while absent from train.",
        "",
        "## 6. Fixed vs seen vs binary evaluation",
        "",
        f"- LOSO fixed macro-F1 mean over supplied runs: "
        f"{loso_fixed.mean() if not loso_fixed.dropna().empty else NOT_AVAILABLE}",
        f"- LOSO seen-class macro-F1 mean: "
        f"{loso_seen.mean() if not loso_seen.dropna().empty else NOT_AVAILABLE}",
        f"- LOSO binary F1 mean: "
        f"{loso_binary.mean() if not loso_binary.dropna().empty else NOT_AVAILABLE}",
        "",
        "Fixed-class macro-F1 is the primary closed-set score. Seen-class and "
        "binary F1 explain failure modes but do not replace it.",
        "",
        "## 7. Entropy analysis",
        "",
        _markdown_table(entropy_auc_rows),
        "",
        "Entropy is uncertainty analysis only. No result here is described as "
        "zero-day detection; AUROC/AUPRC and direction must support any narrower "
        "claim.",
        "",
        "## 8. Historical vs clean comparison",
        "",
        _markdown_table(historical_comparison),
        "",
        "Historical leakage-prone values are context only and are never mixed "
        "into clean means.",
        "",
        "## 9. Statistical limitations",
        "",
        (
            "- Only one seed is available; standard deviation is descriptive "
            "zero and statistical uncertainty is NOT_AVAILABLE."
            if single_seed
            else "- Multi-seed mean/std/min/max are reported with population "
            "standard deviation (`ddof=0`). Three seeds quantify seed variation "
            "but do not establish external-dataset generalization."
        ),
        "- `train_time` is NOT_AVAILABLE unless explicitly stored in metadata; "
        "it is never reconstructed from timestamps.",
        "",
        "## 10. Figure recommendation",
        "",
        "See `FIGURE_SELECTION.md`. Main-report figures are capped at six; "
        "learning curves, entropy diagnostics, count matrices, and stability "
        "details belong in the appendix unless central to the argument.",
        "",
        "## Availability",
        "",
        *[
            f"- `{run['bundle']}`: {note}"
            for run in runs
            for note in run["notes"]
        ],
        *(
            [
                f"- `{item.name}`: {item.reason}"
                for item in missing_figures
            ]
            if missing_figures
            else ["- All requested figures were created."]
        ),
    ]
    (output_dir / "report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    absent_auc_values = pd.to_numeric(
        entropy_auc_rows.get("auroc", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    absent_auc = (
        float(absent_auc_values.mean())
        if not absent_auc_values.empty
        else NOT_AVAILABLE
    )
    return {
        "bundles": len(runs),
        "seeds": len(seeds),
        "single_seed": single_seed,
        "statistical_uncertainty": uncertainty,
        "pooled_runs": int((run_frame["protocol"] == "pooled").sum()),
        "loso_runs": int((run_frame["protocol"] == "loso").sum()),
        "figures_created": len(created_figures),
        "figures": len(created_figures),
        "entropy_available": not entropy.empty,
        "figures_unavailable": {
            result.name: result.reason for result in missing_figures
        },
        "entropy_absent_vs_known_correct_auroc": absent_auc,
        "loso_mean_fixed_f1": (
            float(loso_fixed.mean())
            if not loso_fixed.dropna().empty
            else NOT_AVAILABLE
        ),
        "loso_mean_seen_f1": (
            float(loso_seen.mean())
            if not loso_seen.dropna().empty
            else NOT_AVAILABLE
        ),
        "loso_mean_binary_f1": (
            float(loso_binary.mean())
            if not loso_binary.dropna().empty
            else NOT_AVAILABLE
        ),
        "report": str((output_dir / "report.md").resolve()),
        "figure_selection": str(
            (output_dir / "FIGURE_SELECTION.md").resolve()
        ),
        "output_dir": str(output_dir.resolve()),
    }


def _default_inputs() -> list[Path]:
    return sorted(Path("artifacts/phase1_clean").glob("seed-*-full"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create report-ready analysis from clean Phase 1 bundles."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="One or more seed roots or individual clean bundle directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/phase1_clean/report_analysis"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    inputs = args.inputs or _default_inputs()
    bundles = discover_bundles(inputs, args.output_dir)
    runs = [analyze_bundle(bundle) for bundle in bundles]
    result = write_analysis(runs, args.output_dir, inputs)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not bundles:
        print(
            "No clean bundles found. Outputs contain headers/NOT_AVAILABLE only.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
