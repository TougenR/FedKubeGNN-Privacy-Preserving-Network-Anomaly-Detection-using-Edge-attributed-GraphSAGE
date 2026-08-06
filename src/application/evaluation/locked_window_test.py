"""Evaluate the validation-locked rolling protocol exactly once on test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from src.application.evaluation.window_benchmark import (
    WindowBenchmarkError,
    evaluate_candidate,
    load_replay_frames,
    preprocess_validation_frames,
)
from src.application.inference.bundle_loader import load_inference_bundle


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


def validate_locked_protocol(
    *, serving_manifest: Mapping[str, Any], window_report: Mapping[str, Any]
) -> Mapping[str, Any]:
    selected = window_report.get("selected")
    source = serving_manifest.get("source_research_bundle")
    if window_report.get(
        "kind"
    ) != "rolling_window_validation_selection" or not isinstance(selected, dict):
        raise WindowBenchmarkError("Window report has no validation selection.")
    if selected.get("selection_split") != "validation":
        raise WindowBenchmarkError("Window protocol was not selected on validation.")
    if not isinstance(source, dict) or source.get("bundle_id") != window_report.get(
        "bundle_id"
    ):
        raise WindowBenchmarkError(
            "Serving bundle does not derive from window report bundle."
        )
    if serving_manifest.get("dataset_digest") != window_report.get("dataset_digest"):
        raise WindowBenchmarkError("Serving and window report dataset digests differ.")
    if serving_manifest.get("graph_protocol") != selected.get("graph_protocol"):
        raise WindowBenchmarkError(
            "Serving graph protocol differs from validation lock."
        )
    protocol = serving_manifest.get("rolling_window_protocol")
    if not isinstance(protocol, dict):
        raise WindowBenchmarkError("Serving bundle has no rolling window protocol.")
    expected = {
        "duration_seconds": float(selected["duration_seconds"]),
        "max_flows": int(selected["max_flows"]),
        "emit_stride_flows": int(selected["emit_stride_flows"]),
        "allowed_lateness_seconds": float(selected["allowed_lateness_seconds"]),
    }
    observed = {
        "duration_seconds": float(protocol.get("duration_seconds", -1)),
        "max_flows": int(protocol.get("max_flows", -1)),
        "emit_stride_flows": int(protocol.get("emit_stride_flows", -1)),
        "allowed_lateness_seconds": float(protocol.get("allowed_lateness_seconds", -1)),
    }
    if observed != expected:
        raise WindowBenchmarkError(
            "Serving rolling parameters differ from validation lock."
        )
    return protocol


def run_locked_test(
    *,
    bundle_root: Path,
    replay_root: Path,
    window_report_path: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Locked test report already exists: {output}")
    bundle = load_inference_bundle(
        bundle_root, device="cpu", require_serving_ready=True
    )
    window_report = _read_json(window_report_path)
    protocol = validate_locked_protocol(
        serving_manifest=bundle.manifest, window_report=window_report
    )
    frames, replay_manifest = load_replay_frames(replay_root, split="test")
    if bundle.manifest["dataset_digest"] != replay_manifest["derived_dataset_digest"]:
        raise WindowBenchmarkError("Serving bundle and replay dataset digests differ.")
    transformed = preprocess_validation_frames(bundle=bundle, frames=frames)
    candidate = evaluate_candidate(
        bundle=bundle,
        frames=transformed,
        duration=float(protocol["duration_seconds"]),
        limit=int(protocol["max_flows"]),
    )
    document = {
        "kind": "rolling_window_locked_test",
        "evaluation_split": "test",
        "selection_split": "validation",
        "bundle_id": bundle.manifest["bundle_id"],
        "source_research_bundle_id": bundle.manifest["source_research_bundle"][
            "bundle_id"
        ],
        "dataset_id": replay_manifest["derived_dataset_id"],
        "dataset_digest": replay_manifest["derived_dataset_digest"],
        "graph_protocol": bundle.manifest["graph_protocol"],
        "validation_report_sha256": _sha256(window_report_path),
        "result": candidate,
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = run_locked_test(
        bundle_root=args.bundle,
        replay_root=args.replay,
        window_report_path=args.window_report,
        output=args.output,
    )
    print(json.dumps(document["result"]["metrics"], sort_keys=True))


if __name__ == "__main__":
    main()
