"""Evaluate a portable checkpoint against one prepared split."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.federated.config.schema import Phase2Config
from src.federated.core.metrics import (
    aggregate_confusion_matrices,
    classification_metrics,
)
from src.federated.experiments.factory import manifest_task
from src.federated.observability.events import NoopObserver, Observer
from src.federated.observability.run_store import atomic_json


def evaluate_checkpoint(
    config: Phase2Config,
    dataset_root: str | Path,
    checkpoint: str | Path,
    *,
    split: str = "test",
    output: str | Path | None = None,
    observer: Observer | None = None,
) -> dict[str, Any]:
    if split not in {"validation", "test"}:
        raise ValueError("Evaluation split must be validation or test.")
    observer = observer or NoopObserver()
    task = manifest_task(config, dataset_root, observer=observer)
    with np.load(checkpoint, allow_pickle=False) as archive:
        state = {name: np.asarray(archive[name]).copy() for name in archive.files}
    task.model_spec.validate_state(state)
    results = [
        task.evaluate_local(client_id, state, split=split)
        for client_id in task.client_ids
    ]
    matrix = aggregate_confusion_matrices(
        (result.confusion_matrix for result in results),
        num_classes=task.label_schema.num_classes,
    )
    metrics = classification_metrics(matrix, class_names=task.label_schema.classes)
    total = sum(result.num_examples for result in results)
    metrics["loss"] = (
        sum(result.loss * result.num_examples for result in results) / total
        if total
        else 0.0
    )
    document = {
        "dataset_id": task.metadata()["dataset_id"],
        "model_digest": task.model_spec.digest,
        "checkpoint": str(checkpoint),
        "split": split,
        "metrics": metrics,
        "confusion_matrix": matrix.tolist(),
    }
    if output is not None:
        atomic_json(Path(output), document)
    observer.emit(
        "checkpoint.evaluated",
        component="evaluation",
        dataset_id=str(document["dataset_id"]),
        split=split,
        examples=int(matrix.sum()),
        macro_f1=float(metrics["macro_f1"]),
        loss=float(metrics["loss"]),
        output=str(output) if output else None,
    )
    return document
