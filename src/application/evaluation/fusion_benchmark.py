"""Select multi-head fusion on rolling validation and evaluate locked test once."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.application.evaluation.metrics import classification_metrics, confusion_matrix
from src.application.evaluation.window_benchmark import (
    load_replay_frames,
    preprocess_validation_frames,
)
from src.application.graph_window.buffer import RollingWindowBuffer, RollingWindowConfig
from src.application.graph_window.graph_builder import build_inference_graph
from src.application.inference.bundle_loader import load_inference_bundle
from src.application.inference.fusion import load_fusion_policy, policy_digest
from src.application.inference.runtime import CentralizedFedPerRuntime


class FusionBenchmarkError(RuntimeError):
    """Raised when fusion selection would weaken validation/test separation."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _window_config(manifest: Mapping[str, Any]) -> RollingWindowConfig:
    protocol = str(manifest["graph_protocol"])
    expected_prefix = "rolling-window-v1:"
    if not protocol.startswith(expected_prefix):
        raise FusionBenchmarkError("Serving bundle has no locked rolling protocol.")
    fields = dict(part.split("=", 1) for part in protocol[len(expected_prefix) :].split(":"))
    return RollingWindowConfig(
        duration_seconds=float(fields["duration"].removesuffix("s")),
        max_flows=int(fields["max-flows"]),
        emit_stride_flows=int(fields["stride"]),
        allowed_lateness_seconds=float(fields["lateness"].removesuffix("s")),
    )


def collect_all_head_outputs(
    *, bundle: Any, frames: Mapping[str, pd.DataFrame]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], dict[str, Any]]:
    """Return label-free head probabilities plus evaluator-only truth."""
    runtime = CentralizedFedPerRuntime(bundle)
    heads = tuple(sorted(bundle.heads))
    config = _window_config(bundle.manifest)
    truth: list[int] = []
    probabilities: list[np.ndarray] = []
    calibration_training: list[bool] = []
    per_client: dict[str, int] = {}
    latencies: list[float] = []
    for client_id, frame in frames.items():
        buffer = RollingWindowBuffer(sensor_id=f"sensor-{client_id}", config=config)
        count = 0
        training_limit = int(len(frame) * 0.70)
        for row_index, record in enumerate(frame.to_dict(orient="records")):
            snapshot = buffer.add(record)
            if snapshot is None:
                continue
            if len(snapshot.emission_indices) != 1:
                raise FusionBenchmarkError("Locked stride-one protocol changed.")
            graph = build_inference_graph(
                pd.DataFrame(snapshot.flows),
                bundle.preprocessor.feature_columns,
                sensor_id=snapshot.sensor_id,
            )
            started = time.perf_counter()
            results = runtime.predict_graph_all_heads(graph)
            latencies.append((time.perf_counter() - started) * 1000)
            target = snapshot.emission_indices[0]
            target_flow = snapshot.flows[target]
            probabilities.append(
                np.stack(
                    [results[head].probabilities[target].numpy() for head in heads],
                    axis=0,
                )
            )
            truth.append(bundle.class_to_idx[str(target_flow["detailed-label"])])
            calibration_training.append(row_index < training_limit)
            count += 1
        per_client[client_id] = count
    expected = sum(len(frame) for frame in frames.values())
    if len(truth) != expected:
        raise FusionBenchmarkError(
            f"Fusion coverage mismatch: expected={expected}, actual={len(truth)}."
        )
    latency = np.asarray(latencies, dtype=np.float64)
    return (
        np.asarray(truth, dtype=np.int64),
        np.stack(probabilities, axis=0),
        np.asarray(calibration_training, dtype=bool),
        heads,
        {
            "examples": len(truth),
            "per_client_examples": per_client,
            "calibration_train_fraction_per_client": 0.70,
            "calibration_train_examples": int(sum(calibration_training)),
            "selection_examples": int(len(calibration_training) - sum(calibration_training)),
            "all_head_latency_ms": {
                "p50": float(np.quantile(latency, 0.50)),
                "p95": float(np.quantile(latency, 0.95)),
                "mean": float(latency.mean()),
            },
        },
    )


def _head_class_f1(
    truth: np.ndarray,
    head_probabilities: np.ndarray,
    classes: tuple[str, ...],
) -> np.ndarray:
    values = np.zeros((head_probabilities.shape[1], len(classes)), dtype=np.float64)
    for head_index in range(head_probabilities.shape[1]):
        matrix = confusion_matrix(
            truth,
            head_probabilities[:, head_index, :].argmax(axis=1),
            num_classes=len(classes),
        )
        metrics = classification_metrics(matrix, class_names=classes)
        values[head_index] = [
            float(metrics["per_class"][class_name]["f1"])
            for class_name in classes
        ]
    return values


def _weights(reliability: np.ndarray, *, top_k: int, power: float) -> np.ndarray:
    head_count, class_count = reliability.shape
    result = np.zeros_like(reliability)
    for class_index in range(class_count):
        order = np.argsort(reliability[:, class_index], kind="stable")
        selected = order[-min(top_k, head_count) :]
        raw = np.power(np.maximum(reliability[selected, class_index], 1e-9), power)
        result[selected, class_index] = raw / raw.sum()
    return result


def fuse_probabilities(head_probabilities: np.ndarray, weights: np.ndarray) -> np.ndarray:
    scores = np.sum(head_probabilities * weights[None, :, :], axis=1)
    denominator = scores.sum(axis=1, keepdims=True)
    if np.any(denominator <= 0) or not np.isfinite(scores).all():
        raise FusionBenchmarkError("Fusion candidate produced invalid scores.")
    return scores / denominator


def _metrics(
    truth: np.ndarray, probabilities: np.ndarray, classes: tuple[str, ...]
) -> tuple[dict[str, Any], list[list[int]]]:
    matrix = confusion_matrix(
        truth, probabilities.argmax(axis=1), num_classes=len(classes)
    )
    return classification_metrics(matrix, class_names=classes), matrix.tolist()


def select_class_thresholds(
    *,
    truth: np.ndarray,
    probabilities: np.ndarray,
    classes: tuple[str, ...],
    maximum_benign_false_alert_rate: float,
) -> dict[str, Any]:
    """Allocate one validation FP budget across attack classes with dynamic programming."""
    benign_index = classes.index("Benign")
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    benign = truth == benign_index
    benign_support = int(benign.sum())
    maximum_false_alerts = int(
        np.floor(maximum_benign_false_alert_rate * benign_support)
    )
    attack_indices = [index for index in range(len(classes)) if index != benign_index]
    options: dict[int, list[dict[str, Any]]] = {}
    for class_index in attack_indices:
        relevant = predicted == class_index
        thresholds = np.unique(
            np.concatenate(([0.0], confidence[relevant], [1.0]))
        )
        best_by_false_alerts: dict[int, dict[str, Any]] = {}
        for threshold in thresholds:
            selected = relevant & (confidence >= float(threshold))
            false_alerts = int(np.sum(selected & benign))
            if false_alerts > maximum_false_alerts:
                continue
            malicious_alerts = int(np.sum(selected & ~benign))
            correct_class_alerts = int(
                np.sum(selected & (truth == class_index))
            )
            candidate = {
                "threshold": float(threshold),
                "benign_false_alerts": false_alerts,
                "malicious_alerts": malicious_alerts,
                "correct_class_alerts": correct_class_alerts,
            }
            previous = best_by_false_alerts.get(false_alerts)
            key = (malicious_alerts, correct_class_alerts, -float(threshold))
            if previous is None or key > (
                previous["malicious_alerts"],
                previous["correct_class_alerts"],
                -previous["threshold"],
            ):
                best_by_false_alerts[false_alerts] = candidate
        options[class_index] = list(best_by_false_alerts.values())

    states: dict[int, tuple[tuple[int, int, float], dict[int, dict[str, Any]]]] = {
        0: ((0, 0, 0.0), {})
    }
    for class_index in attack_indices:
        next_states: dict[
            int, tuple[tuple[int, int, float], dict[int, dict[str, Any]]]
        ] = {}
        for used, (score, chosen) in states.items():
            for option in options[class_index]:
                total_false = used + option["benign_false_alerts"]
                if total_false > maximum_false_alerts:
                    continue
                candidate_score = (
                    score[0] + option["malicious_alerts"],
                    score[1] + option["correct_class_alerts"],
                    score[2] - option["threshold"],
                )
                previous = next_states.get(total_false)
                if previous is None or candidate_score > previous[0]:
                    next_states[total_false] = (
                        candidate_score,
                        {**chosen, class_index: option},
                    )
        states = next_states
    _, (_, selected) = max(states.items(), key=lambda item: item[1][0])
    thresholds = {
        classes[index]: float(selected[index]["threshold"])
        for index in attack_indices
    }
    alerts = np.zeros(len(truth), dtype=bool)
    for index in attack_indices:
        alerts |= (predicted == index) & (confidence >= thresholds[classes[index]])
    false_alerts = int(np.sum(alerts & benign))
    malicious_support = int(np.sum(~benign))
    return {
        "class_alert_thresholds": thresholds,
        "maximum_benign_false_alert_rate": maximum_benign_false_alert_rate,
        "maximum_benign_false_alerts": maximum_false_alerts,
        "benign_support": benign_support,
        "benign_false_alerts": false_alerts,
        "benign_false_alert_rate": false_alerts / benign_support,
        "malicious_support": malicious_support,
        "malicious_alerts": int(np.sum(alerts & ~benign)),
        "malicious_alert_recall": float(np.sum(alerts & ~benign) / malicious_support),
        "correct_class_alerts": int(np.sum(alerts & (predicted == truth) & ~benign)),
    }


def select_policy(
    *,
    bundle: Any,
    truth: np.ndarray,
    head_probabilities: np.ndarray,
    calibration_training: np.ndarray,
    heads: tuple[str, ...],
    collection: Mapping[str, Any],
    maximum_benign_false_alert_rate: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    classes = tuple(bundle.class_to_idx)
    if calibration_training.shape != truth.shape or calibration_training.all():
        raise FusionBenchmarkError("Calibration train/selection split is invalid.")
    selection = ~calibration_training
    reliability = _head_class_f1(
        truth[calibration_training],
        head_probabilities[calibration_training],
        classes,
    )
    candidates: list[dict[str, Any]] = []
    candidate_state: dict[str, dict[str, Any]] = {}
    for top_k in (1, 2, 3, len(heads)):
        for power in (1.0, 2.0, 4.0):
            candidate_id = f"f1-weighted-top{top_k}-power{power:g}"
            weights = _weights(reliability, top_k=top_k, power=power)
            fused = fuse_probabilities(head_probabilities[selection], weights)
            metrics, matrix = _metrics(truth[selection], fused, classes)
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "method": "class-f1-weighted-probability",
                    "top_k": top_k,
                    "reliability_power": power,
                    "metrics": metrics,
                    "confusion_matrix": matrix,
                }
            )
            candidate_state[candidate_id] = {
                "method": "class-f1-weighted-probability",
                "weights": weights,
                "selection_probabilities": fused,
            }

    flat = head_probabilities.reshape(len(truth), -1)
    for feature_transform in ("probability", "log-probability"):
        features = (
            np.log(np.clip(flat, 1e-12, None))
            if feature_transform == "log-probability"
            else flat
        )
        for class_weight in (None, "balanced"):
            for regularization in (0.01, 0.1, 1.0, 10.0):
                scaler = StandardScaler().fit(features[calibration_training])
                classifier = LogisticRegression(
                    C=regularization,
                    class_weight=class_weight,
                    max_iter=2000,
                    random_state=42,
                    solver="lbfgs",
                ).fit(
                    scaler.transform(features[calibration_training]),
                    truth[calibration_training],
                )
                if not np.array_equal(classifier.classes_, np.arange(len(classes))):
                    raise FusionBenchmarkError(
                        "Calibration split does not contain the fixed seven classes."
                    )
                fused = classifier.predict_proba(
                    scaler.transform(features[selection])
                )
                weight_name = "balanced" if class_weight else "natural"
                candidate_id = (
                    f"logistic-{feature_transform}-{weight_name}-c{regularization:g}"
                )
                metrics, matrix = _metrics(truth[selection], fused, classes)
                candidates.append(
                    {
                        "candidate_id": candidate_id,
                        "method": "multinomial-logistic-stacking",
                        "feature_transform": feature_transform,
                        "class_weight": weight_name,
                        "regularization_c": regularization,
                        "metrics": metrics,
                        "confusion_matrix": matrix,
                    }
                )
                candidate_state[candidate_id] = {
                    "method": "multinomial-logistic-stacking",
                    "feature_transform": feature_transform,
                    "scaler": scaler,
                    "classifier": classifier,
                    "selection_probabilities": fused,
                }
    selected = max(
        candidates,
        key=lambda item: (
            float(item["metrics"]["macro_f1_fixed"]),
            float(item["metrics"]["accuracy"]),
            item["candidate_id"],
        ),
    )
    state = candidate_state[selected["candidate_id"]]
    probabilities = state["selection_probabilities"]
    alert_policy = select_class_thresholds(
        truth=truth[selection],
        probabilities=probabilities,
        classes=classes,
        maximum_benign_false_alert_rate=maximum_benign_false_alert_rate,
    )
    selected_policy: dict[str, Any] = {
        "method": state["method"],
        "thresholds": alert_policy["class_alert_thresholds"],
    }
    selected_details: dict[str, Any] = {}
    if state["method"] == "class-f1-weighted-probability":
        weights = state["weights"]
        weight_document = {
            class_name: {
                head: float(weights[head_index, class_index])
                for head_index, head in enumerate(heads)
            }
            for class_index, class_name in enumerate(classes)
        }
        selected_policy["weights"] = weight_document
        selected_details["class_head_weights"] = weight_document
    else:
        scaler = state["scaler"]
        classifier = state["classifier"]
        stacking = {
            "feature_transform": state["feature_transform"],
            "feature_order": "head-major-then-class",
            "feature_mean": scaler.mean_.astype(float).tolist(),
            "feature_scale": scaler.scale_.astype(float).tolist(),
            "coefficients": classifier.coef_.astype(float).tolist(),
            "intercept": classifier.intercept_.astype(float).tolist(),
        }
        selected_policy["stacking"] = stacking
        selected_details["stacking"] = stacking
    report = {
        "kind": "multi_head_fusion_validation_selection",
        "selection_split": "validation",
        "bundle_id": bundle.manifest["bundle_id"],
        "model_digest": bundle.manifest["model_digest"],
        "dataset_digest": bundle.manifest["dataset_digest"],
        "graph_protocol": bundle.manifest["graph_protocol"],
        "heads": list(heads),
        "classes": list(classes),
        "selection_rule": (
            "fit on the first 70% of each validation client and maximize "
            "fixed-7 macro-F1 on the remaining 30%; then accuracy and a "
            "deterministic candidate ID; alert thresholds maximize malicious "
            "recall on that selection partition within the approved benign "
            "false-alert budget"
        ),
        "head_by_class_validation_f1": {
            head: {
                class_name: float(reliability[head_index, class_index])
                for class_index, class_name in enumerate(classes)
            }
            for head_index, head in enumerate(heads)
        },
        "candidates": candidates,
        "selected": {
            **selected,
            **selected_details,
            "alert_policy": alert_policy,
        },
        "collection": dict(collection),
    }
    return report, {
        **selected_policy,
    }


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_selection(
    *,
    bundle_root: Path,
    replay_root: Path,
    validation_output: Path,
    policy_output: Path,
    maximum_benign_false_alert_rate: float = 0.001,
) -> tuple[dict[str, Any], dict[str, Any]]:
    bundle_root = bundle_root.resolve()
    replay_root = replay_root.resolve()
    bundle = load_inference_bundle(bundle_root, device="cpu", require_serving_ready=True)
    frames, replay_manifest = load_replay_frames(replay_root, split="validation")
    if bundle.manifest["dataset_digest"] != replay_manifest["derived_dataset_digest"]:
        raise FusionBenchmarkError("Bundle and replay dataset digests differ.")
    transformed = preprocess_validation_frames(bundle=bundle, frames=frames)
    truth, outputs, calibration_training, heads, collection = collect_all_head_outputs(
        bundle=bundle, frames=transformed
    )
    report, selected = select_policy(
        bundle=bundle,
        truth=truth,
        head_probabilities=outputs,
        calibration_training=calibration_training,
        heads=heads,
        collection=collection,
        maximum_benign_false_alert_rate=maximum_benign_false_alert_rate,
    )
    _write_json(validation_output, report)
    validation_digest = _sha256(validation_output)
    policy: dict[str, Any] = {
        "schema_version": 1,
        "kind": "validation-selected-multi-head-probability-fusion",
        "policy_id": f"fusion-{bundle.manifest['model_digest'][:12]}-{validation_digest[:10]}",
        "selection_split": "validation",
        "bundle_id": bundle.manifest["bundle_id"],
        "model_digest": bundle.manifest["model_digest"],
        "dataset_digest": bundle.manifest["dataset_digest"],
        "graph_protocol": bundle.manifest["graph_protocol"],
        "heads": list(heads),
        "classes": list(bundle.class_to_idx),
        "head_digests": bundle.manifest["head_digests"],
        "method": selected["method"],
        "class_alert_thresholds": selected["thresholds"],
        "provenance": {
            "validation_report_path": str(validation_output),
            "validation_report_sha256": validation_digest,
            "maximum_benign_false_alert_rate": maximum_benign_false_alert_rate,
        },
    }
    if selected["method"] == "class-f1-weighted-probability":
        policy["class_head_weights"] = selected["weights"]
    else:
        policy["stacking"] = selected["stacking"]
    policy["policy_digest"] = policy_digest(policy)
    _write_json(policy_output, policy)
    return report, policy


def run_locked_test(
    *, bundle_root: Path, replay_root: Path, policy_path: Path, output: Path
) -> dict[str, Any]:
    bundle_root = bundle_root.resolve()
    replay_root = replay_root.resolve()
    policy_path = policy_path.resolve()
    bundle = load_inference_bundle(bundle_root, device="cpu", require_serving_ready=True)
    policy = load_fusion_policy(policy_path, bundle)
    frames, replay_manifest = load_replay_frames(replay_root, split="test")
    if bundle.manifest["dataset_digest"] != replay_manifest["derived_dataset_digest"]:
        raise FusionBenchmarkError("Bundle and replay dataset digests differ.")
    transformed = preprocess_validation_frames(bundle=bundle, frames=frames)
    truth, outputs, _, heads, collection = collect_all_head_outputs(
        bundle=bundle, frames=transformed
    )
    if policy.method == "class-f1-weighted-probability":
        assert policy.class_head_weights is not None
        probabilities = fuse_probabilities(outputs, policy.class_head_weights.numpy())
    else:
        assert policy.feature_mean is not None
        assert policy.feature_scale is not None
        assert policy.coefficients is not None
        assert policy.intercept is not None
        features = outputs.reshape(len(truth), -1)
        if policy.feature_transform == "log-probability":
            features = np.log(np.clip(features, 1e-12, None))
        features = (
            features - policy.feature_mean.numpy()
        ) / policy.feature_scale.numpy()
        logits = features @ policy.coefficients.numpy().T + policy.intercept.numpy()
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
    metrics, matrix = _metrics(truth, probabilities, tuple(policy.classes))
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    benign_index = policy.classes.index("Benign")
    alerts = np.zeros(len(truth), dtype=bool)
    for class_name, threshold in policy.class_alert_thresholds.items():
        class_index = policy.classes.index(class_name)
        alerts |= (predicted == class_index) & (confidence >= threshold)
    benign = truth == benign_index
    malicious = ~benign
    report = {
        "kind": "locked_multi_head_fusion_test",
        "split": "test",
        "selection_split": "validation",
        "policy_id": policy.policy_id,
        "policy_digest": policy.policy_digest,
        "bundle_id": bundle.manifest["bundle_id"],
        "dataset_digest": bundle.manifest["dataset_digest"],
        "heads": list(heads),
        "metrics": metrics,
        "confusion_matrix": matrix,
        "alert_metrics": {
            "benign_support": int(benign.sum()),
            "benign_false_alerts": int(np.sum(alerts & benign)),
            "benign_false_alert_rate": float(np.mean(alerts[benign])),
            "malicious_support": int(malicious.sum()),
            "malicious_alerts": int(np.sum(alerts & malicious)),
            "malicious_alert_recall": float(np.mean(alerts[malicious])),
            "correct_class_alerts": int(np.sum(alerts & malicious & (predicted == truth))),
        },
        "collection": collection,
    }
    _write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select")
    select.add_argument("--bundle", type=Path, required=True)
    select.add_argument("--replay", type=Path, required=True)
    select.add_argument("--validation-output", type=Path, required=True)
    select.add_argument("--policy-output", type=Path, required=True)
    select.add_argument("--maximum-benign-far", type=float, default=0.001)
    test = subparsers.add_parser("test")
    test.add_argument("--bundle", type=Path, required=True)
    test.add_argument("--replay", type=Path, required=True)
    test.add_argument("--policy", type=Path, required=True)
    test.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "select":
        report, policy = run_selection(
            bundle_root=args.bundle,
            replay_root=args.replay,
            validation_output=args.validation_output,
            policy_output=args.policy_output,
            maximum_benign_false_alert_rate=args.maximum_benign_far,
        )
        print(
            json.dumps(
                {
                    "candidate": report["selected"]["candidate_id"],
                    "metrics": report["selected"]["metrics"],
                    "alert_policy": report["selected"]["alert_policy"],
                    "policy_digest": policy["policy_digest"],
                },
                sort_keys=True,
            )
        )
    else:
        report = run_locked_test(
            bundle_root=args.bundle,
            replay_root=args.replay,
            policy_path=args.policy,
            output=args.output,
        )
        print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
