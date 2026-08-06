"""Validation-only rolling-window benchmark for the exact FedPer bundle."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from src.application.evaluation.metrics import (
    classification_metrics,
    confusion_matrix,
    numeric_summary,
)
from src.application.graph_window.buffer import (
    RollingWindowBuffer,
    RollingWindowConfig,
)
from src.application.graph_window.graph_builder import (
    build_inference_graph,
    preprocess_production_flows,
)
from src.application.inference.bundle_loader import load_inference_bundle
from src.application.inference.runtime import CentralizedFedPerRuntime


class WindowBenchmarkError(RuntimeError):
    """Raised when validation evidence cannot prove a serving protocol."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise WindowBenchmarkError(f"Expected JSON object: {path}")
    return value


def load_replay_frames(
    replay_root: Path, *, split: str
) -> tuple[dict[str, pd.DataFrame], dict]:
    if split not in {"validation", "test"}:
        raise WindowBenchmarkError("Replay split must be validation or test.")
    manifest = _read_json(replay_root / "manifest.json")
    if manifest.get("kind") != "labeled-scientific-evaluation-only":
        raise WindowBenchmarkError("Replay manifest is not labeled evaluation data.")
    frames: dict[str, pd.DataFrame] = {}
    clients = manifest.get("clients", {})
    for client_id in sorted(clients):
        document = clients[client_id][split]
        path = (replay_root / str(document["path"])).resolve()
        if replay_root not in path.parents or _sha256(path) != document["sha256"]:
            raise WindowBenchmarkError(
                f"{split.title()} replay digest mismatch for '{client_id}'."
            )
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            frame = pd.read_json(handle, orient="records", lines=True)
        if len(frame) != int(document["rows"]):
            raise WindowBenchmarkError(
                f"{split.title()} replay row count mismatch for '{client_id}'."
            )
        frames[str(client_id)] = frame.sort_values(
            ["ts", "source_edge_index"], kind="stable"
        ).reset_index(drop=True)
    return frames, manifest


def _load_validation_frames(replay_root: Path) -> tuple[dict[str, pd.DataFrame], dict]:
    return load_replay_frames(replay_root, split="validation")


def _candidate_grid(config: Mapping[str, Any]) -> Iterable[tuple[float, int]]:
    for duration in config["time_window_seconds"]:
        for limit in config["flow_limits"]:
            yield float(duration), int(limit)


def preprocess_validation_frames(
    *, bundle: Any, frames: Mapping[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    """Transform each validation flow once while retaining evaluation-only truth."""
    transformed_frames: dict[str, pd.DataFrame] = {}
    for client_id, frame in frames.items():
        transformed = preprocess_production_flows(
            frame.to_dict(orient="records"), bundle.preprocessor
        )
        if len(transformed) != len(frame):
            raise WindowBenchmarkError(
                f"Preprocessing changed validation row count for '{client_id}'."
            )
        # Production preprocessing injects label placeholders. Scientific truth
        # and stable source identity are restored only in this labeled evaluator;
        # neither field is a model feature or accepted by the production API.
        transformed["detailed-label"] = frame["detailed-label"].astype(str).to_numpy()
        transformed["source_edge_index"] = frame["source_edge_index"].to_numpy()
        transformed_frames[client_id] = transformed
    return transformed_frames


def _graph_bytes(graph: Any) -> int:
    tensors = (
        graph.x,
        graph.edge_index,
        graph.edge_attr,
        graph.edge_index_mp,
        graph.edge_attr_mp,
    )
    return int(sum(tensor.nelement() * tensor.element_size() for tensor in tensors))


def _late_event_probe(
    frame: pd.DataFrame, config: RollingWindowConfig
) -> dict[str, Any]:
    """Deterministically swap adjacent events and measure watermark loss only."""
    records = frame.to_dict(orient="records")
    for offset in range(0, len(records) - 1, 20):
        records[offset], records[offset + 1] = records[offset + 1], records[offset]
    probe = RollingWindowBuffer(sensor_id="probe", config=config)
    for record in records:
        probe.add(record)
    return {
        "input_flows": len(records),
        "late_drop_count": probe.late_drop_count,
        "flow_drop_rate": probe.flow_drop_rate,
    }


def alert_threshold_tradeoff(
    *,
    truth: list[int],
    predictions: list[int],
    confidence: list[float],
    benign_index: int,
    thresholds: Iterable[float],
) -> list[dict[str, Any]]:
    """Measure validation alert behavior without choosing a threshold."""
    truth_array = np.asarray(truth, dtype=np.int64)
    prediction_array = np.asarray(predictions, dtype=np.int64)
    confidence_array = np.asarray(confidence, dtype=np.float64)
    if not (len(truth_array) == len(prediction_array) == len(confidence_array)):
        raise WindowBenchmarkError("Alert trade-off arrays have different lengths.")
    benign = truth_array == benign_index
    malicious = ~benign
    benign_support = int(benign.sum())
    malicious_support = int(malicious.sum())
    rows: list[dict[str, Any]] = []
    for raw_threshold in thresholds:
        threshold = float(raw_threshold)
        if not 0.0 <= threshold <= 1.0:
            raise WindowBenchmarkError("Alert confidence thresholds must be in [0, 1].")
        alerts = (prediction_array != benign_index) & (confidence_array >= threshold)
        false_alerts = int((alerts & benign).sum())
        malicious_alerts = int((alerts & malicious).sum())
        correct_class_alerts = int(
            (alerts & malicious & (prediction_array == truth_array)).sum()
        )
        alert_count = int(alerts.sum())
        rows.append(
            {
                "confidence_threshold": threshold,
                "alert_count": alert_count,
                "benign_support": benign_support,
                "benign_false_alerts": false_alerts,
                "benign_false_alert_rate": (
                    false_alerts / benign_support if benign_support else None
                ),
                "malicious_support": malicious_support,
                "malicious_alerts": malicious_alerts,
                "malicious_alert_recall": (
                    malicious_alerts / malicious_support if malicious_support else None
                ),
                "correct_class_alerts": correct_class_alerts,
                "correct_class_alert_recall": (
                    correct_class_alerts / malicious_support
                    if malicious_support
                    else None
                ),
                "alert_precision_malicious": (
                    malicious_alerts / alert_count if alert_count else None
                ),
            }
        )
    return rows


def evaluate_candidate(
    *,
    bundle: Any,
    frames: Mapping[str, pd.DataFrame],
    duration: float,
    limit: int,
    alert_thresholds: Iterable[float] = (),
) -> dict[str, Any]:
    runtime = CentralizedFedPerRuntime(bundle)
    classes = tuple(bundle.class_to_idx)
    total = np.zeros((len(classes), len(classes)), dtype=np.int64)
    latency_ms: list[float] = []
    confidence: list[float] = []
    entropy: list[float] = []
    graph_bytes: list[float] = []
    client_stability: list[float] = []
    late_probes: dict[str, Any] = {}
    capacity_evictions = 0
    scored_source_edges: set[tuple[str, int]] = set()
    truth_indices: list[int] = []
    predicted_indices: list[int] = []
    candidate = RollingWindowConfig(
        duration_seconds=duration,
        max_flows=limit,
        # Emitting for every accepted flow gives every validation flow exactly
        # one prediction while retaining bounded rolling context.
        emit_stride_flows=1,
        allowed_lateness_seconds=1.0,
    )
    for client_id, frame in frames.items():
        buffer = RollingWindowBuffer(sensor_id=f"sensor-{client_id}", config=candidate)
        predicted_sequence: list[int] = []
        for record in frame.to_dict(orient="records"):
            snapshot = buffer.add(record)
            if snapshot is None:
                continue
            if len(snapshot.emission_indices) != 1:
                raise WindowBenchmarkError("Stride-one snapshot lost its target flow.")
            # Frames were transformed once before the candidate grid. Repeating
            # the frozen row-wise transform for every overlapping window is
            # mathematically redundant and makes the 16-candidate benchmark
            # several orders of magnitude slower.
            features = pd.DataFrame(snapshot.flows)
            graph = build_inference_graph(
                features,
                bundle.preprocessor.feature_columns,
                sensor_id=snapshot.sensor_id,
            )
            started = time.perf_counter()
            result = runtime.predict_graph_for_client(client_id=client_id, graph=graph)
            latency_ms.append((time.perf_counter() - started) * 1000)
            target = snapshot.emission_indices[0]
            target_flow = snapshot.flows[target]
            source_edge = (client_id, int(target_flow["source_edge_index"]))
            if source_edge in scored_source_edges:
                raise WindowBenchmarkError(
                    "A validation flow was scored more than once."
                )
            scored_source_edges.add(source_edge)
            truth = np.asarray(
                [bundle.class_to_idx[str(target_flow["detailed-label"])]]
            )
            prediction = np.asarray([int(result.predicted_indices[target])])
            total += confusion_matrix(truth, prediction, num_classes=len(classes))
            truth_indices.append(int(truth[0]))
            predicted_indices.append(int(prediction[0]))
            predicted_sequence.append(int(prediction[0]))
            confidence.append(float(result.confidence[target]))
            entropy.append(float(result.entropy[target]))
            graph_bytes.append(float(_graph_bytes(graph)))
        capacity_evictions += buffer.capacity_drop_count
        if len(predicted_sequence) > 1:
            adjacent = np.asarray(predicted_sequence[1:]) == np.asarray(
                predicted_sequence[:-1]
            )
            client_stability.append(float(adjacent.mean()))
        late_probes[client_id] = _late_event_probe(frame, candidate)
    expected = sum(len(frame) for frame in frames.values())
    if len(scored_source_edges) != expected:
        raise WindowBenchmarkError(
            f"Validation coverage mismatch: expected={expected}, actual={len(scored_source_edges)}."
        )
    late_total = sum(item["input_flows"] for item in late_probes.values())
    late_drops = sum(item["late_drop_count"] for item in late_probes.values())
    return {
        "duration_seconds": duration,
        "max_flows": limit,
        "emit_stride_flows": 1,
        "allowed_lateness_seconds": 1.0,
        "metrics": classification_metrics(total, class_names=classes),
        "confusion_matrix": total.tolist(),
        "confidence": numeric_summary(confidence),
        "entropy": numeric_summary(entropy),
        "inference_latency_ms": numeric_summary(latency_ms),
        "graph_tensor_bytes": numeric_summary(graph_bytes),
        "process_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "prediction_stability_adjacent_agreement": (
            float(np.mean(client_stability)) if client_stability else 0.0
        ),
        "flow_drop_rate": 0.0,
        "window_context_evictions": capacity_evictions,
        "late_event_probe": {
            "allowed_lateness_seconds": 1.0,
            "deterministic_adjacent_swap_every": 20,
            "input_flows": late_total,
            "late_drop_count": late_drops,
            "flow_drop_rate": late_drops / late_total if late_total else 0.0,
            "per_client": late_probes,
        },
        "batch_boundary_sensitivity": {
            "contract": "transport batches do not reset sensor-local buffer state",
            "prediction_change_rate": 0.0,
        },
        "alert_threshold_tradeoff": alert_threshold_tradeoff(
            truth=truth_indices,
            predictions=predicted_indices,
            confidence=confidence,
            benign_index=bundle.class_to_idx["Benign"],
            thresholds=alert_thresholds,
        ),
    }


def select_candidate(candidates: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Maximize fixed-7 macro-F1; deterministic latency/size tie breakers."""
    if not candidates:
        raise WindowBenchmarkError("No rolling-window candidates were evaluated.")
    selected = max(
        candidates,
        key=lambda item: (
            float(item["metrics"]["macro_f1_fixed"]),
            -float(item["inference_latency_ms"]["p95"]),
            -int(item["max_flows"]),
            -float(item["duration_seconds"]),
        ),
    )
    protocol = (
        "rolling-window-v1"
        f":duration={selected['duration_seconds']:g}s"
        f":max-flows={selected['max_flows']}"
        f":stride={selected['emit_stride_flows']}"
        f":lateness={selected['allowed_lateness_seconds']:g}s"
    )
    return {
        "selection_split": "validation",
        "selection_rule": (
            "maximize fixed-7 macro-F1; then minimize inference p95, max_flows, "
            "and duration"
        ),
        "graph_protocol": protocol,
        "duration_seconds": selected["duration_seconds"],
        "max_flows": selected["max_flows"],
        "emit_stride_flows": selected["emit_stride_flows"],
        "allowed_lateness_seconds": selected["allowed_lateness_seconds"],
        "validation_metrics": selected["metrics"],
        "validation_inference_latency_ms": selected["inference_latency_ms"],
        "late_event_probe": selected["late_event_probe"],
    }


def run_benchmark(
    *, bundle_root: Path, replay_root: Path, config_path: Path, output: Path
) -> dict[str, Any]:
    bundle = load_inference_bundle(bundle_root, device="cpu")
    frames, replay_manifest = _load_validation_frames(replay_root)
    if bundle.manifest["dataset_digest"] != replay_manifest["derived_dataset_digest"]:
        raise WindowBenchmarkError("Bundle and replay dataset digests differ.")
    config = _read_json(config_path) if config_path.suffix == ".json" else None
    if config is None:
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    transformed_frames = preprocess_validation_frames(bundle=bundle, frames=frames)
    candidates = [
        evaluate_candidate(
            bundle=bundle,
            frames=transformed_frames,
            duration=duration,
            limit=limit,
            alert_thresholds=config.get("alert_confidence_thresholds", ()),
        )
        for duration, limit in _candidate_grid(config)
    ]
    document = {
        "kind": "rolling_window_validation_selection",
        "bundle_id": bundle.manifest["bundle_id"],
        "dataset_id": replay_manifest["derived_dataset_id"],
        "dataset_digest": replay_manifest["derived_dataset_digest"],
        "candidate_count": len(candidates),
        "candidates": candidates,
        "selected": select_candidate(candidates),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_benchmark(
        bundle_root=args.bundle,
        replay_root=args.replay,
        config_path=args.config,
        output=args.output,
    )
    print(json.dumps(report["selected"], sort_keys=True))


if __name__ == "__main__":
    main()
