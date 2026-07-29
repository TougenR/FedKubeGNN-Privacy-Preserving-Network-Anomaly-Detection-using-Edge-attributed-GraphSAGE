"""Evaluate every labeled replay prediction; entropy remains a score, not an alert."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import requests
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from phase3_monitoring.flow_feeder.replay_flows import (
    SAMPLE_FILE,
    load_sample_dataframe,
    row_to_flow,
)


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {}

    def percentile(fraction: float) -> float:
        index = round((len(ordered) - 1) * fraction)
        return float(ordered[index])

    return {
        "min": float(ordered[0]),
        "mean": float(statistics.fmean(ordered)),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": float(ordered[-1]),
    }


def evaluate(
    *,
    url: str,
    sample_path: Path,
    batch_size: int,
) -> dict:
    dataframe = load_sample_dataframe(sample_path)
    true_labels = [
        "Benign" if value in {"-", "(empty)"} else str(value)
        for value in dataframe["detailed-label"]
    ]
    predicted_labels: list[str] = []
    entropies: list[float] = []
    confidences: list[float] = []
    batch_latencies_ms: list[float] = []
    model_version = None
    schema_digest = None
    fixed_labels = None

    for offset in range(0, len(dataframe), batch_size):
        rows = dataframe.iloc[offset : offset + batch_size]
        flows = [row_to_flow(row) for _, row in rows.iterrows()]
        started = time.perf_counter()
        response = requests.post(
            url,
            json={"flows": flows},
            timeout=30,
        )
        batch_latencies_ms.append((time.perf_counter() - started) * 1000)
        response.raise_for_status()
        payload = response.json()
        predictions = payload["predictions"]
        if len(predictions) != len(flows):
            raise RuntimeError(
                f"Expected {len(flows)} predictions, got {len(predictions)}."
            )

        current_version = payload["model_version"]
        current_digest = payload["feature_schema_digest"]
        if model_version not in {None, current_version}:
            raise RuntimeError("Model version changed during evaluation.")
        if schema_digest not in {None, current_digest}:
            raise RuntimeError("Feature schema changed during evaluation.")
        model_version = current_version
        schema_digest = current_digest

        if predictions and fixed_labels is None:
            fixed_labels = list(predictions[0]["probabilities"])
        predicted_labels.extend(p["predicted_label"] for p in predictions)
        entropies.extend(float(p["entropy"]) for p in predictions)
        confidences.extend(float(p["confidence"]) for p in predictions)

    fixed_labels = fixed_labels or sorted(set(true_labels) | set(predicted_labels))
    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=fixed_labels,
        zero_division=0,
    )
    per_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(per_class_f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(fixed_labels)
    }
    return {
        "sample_path": str(sample_path.resolve()),
        "num_flows": len(true_labels),
        "model_version": model_version,
        "feature_schema_digest": schema_digest,
        "labels": fixed_labels,
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "macro_f1_present": float(
            f1_score(true_labels, predicted_labels, average="macro", zero_division=0)
        ),
        "macro_f1_fixed": float(
            f1_score(
                true_labels,
                predicted_labels,
                labels=fixed_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                true_labels,
                predicted_labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(
            true_labels,
            predicted_labels,
            labels=fixed_labels,
        ).tolist(),
        "entropy": _summary(entropies),
        "confidence": _summary(confidences),
        "batch_latency_ms": _summary(batch_latencies_ms),
        "note": (
            "Entropy is reported as an uncertainty score only. No zero-day "
            "or alert-threshold claim is made by this evaluation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/predict")
    parser.add_argument("--sample", type=Path, default=SAMPLE_FILE)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(
        url=args.url,
        sample_path=args.sample,
        batch_size=args.batch_size,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
