"""Expose validation-only alert trade-offs from a locked serving bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.application.evaluation.window_benchmark import (
    WindowBenchmarkError,
    _load_validation_frames,
    evaluate_candidate,
    preprocess_validation_frames,
)
from src.application.evaluation.locked_window_test import validate_locked_protocol
from src.application.inference.bundle_loader import load_inference_bundle


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise WindowBenchmarkError(f"Expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_alert_policy_provenance(
    *,
    serving_manifest: Mapping[str, Any],
    window_report: Mapping[str, Any],
    replay_manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    protocol = validate_locked_protocol(
        serving_manifest=serving_manifest, window_report=window_report
    )
    if serving_manifest.get("dataset_digest") != replay_manifest.get(
        "derived_dataset_digest"
    ):
        raise WindowBenchmarkError(
            "Serving bundle and validation replay dataset digests differ."
        )
    return protocol


def run_alert_policy_benchmark(
    *,
    bundle_root: Path,
    replay_root: Path,
    window_report_path: Path,
    candidate_config_path: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Alert policy report already exists: {output}")
    bundle = load_inference_bundle(
        bundle_root, device="cpu", require_serving_ready=True
    )
    frames, replay_manifest = _load_validation_frames(replay_root)
    window_report = _read_json(window_report_path)
    config = yaml.safe_load(candidate_config_path.read_text(encoding="utf-8"))
    selected = window_report.get("selected")
    protocol = validate_alert_policy_provenance(
        serving_manifest=bundle.manifest,
        window_report=window_report,
        replay_manifest=replay_manifest,
    )
    assert isinstance(selected, dict)
    transformed = preprocess_validation_frames(bundle=bundle, frames=frames)
    candidate = evaluate_candidate(
        bundle=bundle,
        frames=transformed,
        duration=float(protocol["duration_seconds"]),
        limit=int(protocol["max_flows"]),
        alert_thresholds=config.get("alert_confidence_thresholds", ()),
    )
    document = {
        "kind": "alert_policy_validation_tradeoff",
        "bundle_id": bundle.manifest["bundle_id"],
        "source_research_bundle_id": bundle.manifest["source_research_bundle"][
            "bundle_id"
        ],
        "dataset_id": replay_manifest["derived_dataset_id"],
        "dataset_digest": replay_manifest["derived_dataset_digest"],
        "validation_report_sha256": _sha256(window_report_path),
        "selection_split": "validation",
        "graph_protocol": bundle.manifest["graph_protocol"],
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
