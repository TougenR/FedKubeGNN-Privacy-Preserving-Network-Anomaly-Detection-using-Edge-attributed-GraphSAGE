"""Re-evaluate the validation-selected window to expose alert trade-offs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from src.application.evaluation.window_benchmark import (
    WindowBenchmarkError,
    _load_validation_frames,
    evaluate_candidate,
    preprocess_validation_frames,
)
from src.application.inference.bundle_loader import load_inference_bundle


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise WindowBenchmarkError(f"Expected JSON object: {path}")
    return value


def run_alert_policy_benchmark(
    *,
    bundle_root: Path,
    replay_root: Path,
    window_report_path: Path,
    candidate_config_path: Path,
    output: Path,
) -> dict[str, Any]:
    bundle = load_inference_bundle(bundle_root, device="cpu")
    frames, replay_manifest = _load_validation_frames(replay_root)
    window_report = _read_json(window_report_path)
    config = yaml.safe_load(candidate_config_path.read_text(encoding="utf-8"))
    selected = window_report.get("selected")
    if window_report.get(
        "kind"
    ) != "rolling_window_validation_selection" or not isinstance(selected, dict):
        raise WindowBenchmarkError("Window report has no validation selection.")
    expected = {
        "bundle_id": bundle.manifest["bundle_id"],
        "dataset_digest": replay_manifest["derived_dataset_digest"],
    }
    observed = {
        "bundle_id": window_report.get("bundle_id"),
        "dataset_digest": window_report.get("dataset_digest"),
    }
    if observed != expected or selected.get("selection_split") != "validation":
        raise WindowBenchmarkError(
            "Window report provenance/split does not match inputs."
        )
    transformed = preprocess_validation_frames(bundle=bundle, frames=frames)
    candidate = evaluate_candidate(
        bundle=bundle,
        frames=transformed,
        duration=float(selected["duration_seconds"]),
        limit=int(selected["max_flows"]),
        alert_thresholds=config.get("alert_confidence_thresholds", ()),
    )
    document = {
        "kind": "alert_policy_validation_tradeoff",
        **expected,
        "selection_split": "validation",
        "graph_protocol": selected["graph_protocol"],
        "window_metrics": candidate["metrics"],
        "confusion_matrix": candidate["confusion_matrix"],
        "confidence": candidate["confidence"],
        "entropy": candidate["entropy"],
        "alert_threshold_tradeoff": candidate["alert_threshold_tradeoff"],
        "selected_policy": None,
        "selection_blocker": (
            "No maximum acceptable benign false-alert rate has been authorized."
        ),
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
    parser.add_argument("--window-report", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = run_alert_policy_benchmark(
        bundle_root=args.bundle,
        replay_root=args.replay,
        window_report_path=args.window_report,
        candidate_config_path=args.config,
        output=args.output,
    )
    print(json.dumps({"selected_policy": document["selected_policy"]}))


if __name__ == "__main__":
    main()
