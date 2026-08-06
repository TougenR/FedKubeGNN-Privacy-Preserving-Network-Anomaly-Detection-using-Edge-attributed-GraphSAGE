"""Correctly routed, cross-head, and validation-selected oracle evaluation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.application.evaluation.graphs import load_client_graph
from src.application.evaluation.metrics import (
    classification_metrics,
    confusion_matrix,
    numeric_summary,
)
from src.application.inference.bundle_loader import FedPerServingBundle
from src.application.inference.runtime import CentralizedFedPerRuntime


VALID_SPLITS = {"validation": "val_mask", "test": "test_mask"}


def _mask(graph, split: str) -> np.ndarray:
    try:
        name = VALID_SPLITS[split]
    except KeyError as exc:
        raise ValueError("Evaluation split must be validation or test.") from exc
    return getattr(graph, name).detach().cpu().numpy().astype(bool)


def _clients(bundle: FedPerServingBundle) -> tuple[str, ...]:
    return tuple(bundle.manifest["client_head_mapping"])


def _source_graph(dataset_root: Path, client_id: str):
    return load_client_graph(dataset_root / "clients" / client_id)


def evaluate_correctly_routed(
    bundle: FedPerServingBundle,
    dataset_root: str | Path,
    *,
    split: str,
) -> dict[str, Any]:
    runtime = CentralizedFedPerRuntime(bundle)
    classes = tuple(bundle.class_to_idx)
    total_matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    confidence: list[float] = []
    entropy: list[float] = []
    latencies: list[float] = []
    per_client: dict[str, Any] = {}
    for client_id in _clients(bundle):
        graph = _source_graph(Path(dataset_root), client_id)
        mask = _mask(graph, split)
        started = time.perf_counter()
        result = runtime.predict_graph_for_client(client_id=client_id, graph=graph)
        latencies.append((time.perf_counter() - started) * 1000)
        truth = graph.edge_label.detach().cpu().numpy()[mask]
        predicted = result.predicted_indices.numpy()[mask]
        matrix = confusion_matrix(truth, predicted, num_classes=len(classes))
        metrics = classification_metrics(matrix, class_names=classes)
        total_matrix += matrix
        confidence.extend(result.confidence.numpy()[mask].tolist())
        entropy.extend(result.entropy.numpy()[mask].tolist())
        per_client[client_id] = {
            "head_client_id": client_id,
            "metrics": metrics,
            "confusion_matrix": matrix.tolist(),
        }
    return {
        "kind": "correctly_routed_fedper",
        "split": split,
        "bundle_id": bundle.manifest["bundle_id"],
        "model_digest": bundle.manifest["model_digest"],
        "class_names": list(classes),
        "label_mapping": dict(bundle.class_to_idx),
        "metrics": classification_metrics(total_matrix, class_names=classes),
        "confusion_matrix": total_matrix.tolist(),
        "per_client": per_client,
        "confidence": numeric_summary(confidence),
        "entropy": numeric_summary(entropy),
        "latency_ms_per_client_graph": numeric_summary(latencies),
    }


def evaluate_cross_head(
    bundle: FedPerServingBundle,
    dataset_root: str | Path,
    *,
    split: str,
) -> dict[str, Any]:
    runtime = CentralizedFedPerRuntime(bundle)
    classes = tuple(bundle.class_to_idx)
    heads = _clients(bundle)
    aggregate = {
        head: np.zeros((len(classes), len(classes)), dtype=np.int64)
        for head in heads
    }
    detailed: dict[str, dict[str, Any]] = {}
    latencies: list[float] = []
    for source_client in heads:
        graph = _source_graph(Path(dataset_root), source_client)
        mask = _mask(graph, split)
        truth = graph.edge_label.detach().cpu().numpy()[mask]
        started = time.perf_counter()
        predictions = runtime.predict_graph_all_heads(graph)
        latencies.append((time.perf_counter() - started) * 1000)
        detailed[source_client] = {}
        for head, result in predictions.items():
            matrix = confusion_matrix(
                truth,
                result.predicted_indices.numpy()[mask],
                num_classes=len(classes),
            )
            aggregate[head] += matrix
            detailed[source_client][head] = {
                "metrics": classification_metrics(matrix, class_names=classes),
                "confusion_matrix": matrix.tolist(),
            }
    aggregate_documents = {
        head: {
            "metrics": classification_metrics(matrix, class_names=classes),
            "confusion_matrix": matrix.tolist(),
        }
        for head, matrix in aggregate.items()
    }
    matrix_6x7 = {
        head: {
            class_name: aggregate_documents[head]["metrics"]["per_class"][
                class_name
            ]["f1"]
            for class_name in classes
        }
        for head in heads
    }
    return {
        "kind": "cross_head_fedper",
        "split": split,
        "bundle_id": bundle.manifest["bundle_id"],
        "class_names": list(classes),
        "label_mapping": dict(bundle.class_to_idx),
        "aggregate_by_head": aggregate_documents,
        "source_client_by_head": detailed,
        "head_by_class_f1": matrix_6x7,
        "latency_ms_all_heads_per_client_graph": numeric_summary(latencies),
    }


def select_oracle_mapping(validation_cross_head: Mapping[str, Any]) -> dict[str, str]:
    if validation_cross_head.get("split") != "validation":
        raise ValueError("Oracle mapping must be selected from validation only.")
    table = validation_cross_head.get("head_by_class_f1")
    if not isinstance(table, dict) or not table:
        raise ValueError("Validation cross-head report has no head/class matrix.")
    heads = sorted(str(head) for head in table)
    classes = tuple(next(iter(table.values())))
    mapping: dict[str, str] = {}
    for class_name in classes:
        candidates = [
            (table[head].get(class_name), head)
            for head in heads
            if table[head].get(class_name) is not None
        ]
        if not candidates:
            raise ValueError(f"No validation support for class '{class_name}'.")
        # Resolve equal validation scores deterministically by client ID without
        # consulting test results.
        mapping[class_name] = max(candidates, key=lambda item: (item[0], item[1]))[1]
    return mapping


def evaluate_oracle_once(
    bundle: FedPerServingBundle,
    dataset_root: str | Path,
    *,
    class_head_mapping: Mapping[str, str],
) -> dict[str, Any]:
    runtime = CentralizedFedPerRuntime(bundle)
    classes = tuple(bundle.class_to_idx)
    if set(class_head_mapping) != set(classes):
        raise ValueError("Oracle mapping must contain every fixed class.")
    total_matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for source_client in _clients(bundle):
        graph = _source_graph(Path(dataset_root), source_client)
        mask = _mask(graph, "test")
        truth = graph.edge_label.detach().cpu().numpy()[mask]
        predictions = runtime.predict_graph_all_heads(graph)
        selected = np.empty_like(truth)
        for class_name, class_index in bundle.class_to_idx.items():
            class_mask = truth == class_index
            head = str(class_head_mapping[class_name])
            selected[class_mask] = predictions[head].predicted_indices.numpy()[mask][
                class_mask
            ]
        total_matrix += confusion_matrix(
            truth, selected, num_classes=len(classes)
        )
    return {
        "kind": "validation_selected_oracle_upper_bound",
        "split": "test",
        "selection_split": "validation",
        "class_head_mapping": dict(class_head_mapping),
        "class_names": list(classes),
        "label_mapping": dict(bundle.class_to_idx),
        "metrics": classification_metrics(total_matrix, class_names=classes),
        "confusion_matrix": total_matrix.tolist(),
        "warning": "Uses ground-truth class for routing; not a production policy.",
    }


def write_report(path: str | Path, document: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
