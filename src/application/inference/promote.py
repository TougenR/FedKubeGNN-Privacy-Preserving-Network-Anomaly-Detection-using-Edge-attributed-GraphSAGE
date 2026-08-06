"""Promote a research FedPer bundle after validation selects graph protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.application.inference.bundle_loader import load_inference_bundle


class ServingPromotionError(RuntimeError):
    """Raised before publishing a serving bundle without sufficient evidence."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ServingPromotionError(f"Expected JSON object: {path}")
    return value


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def promote_serving_bundle(
    *, research_root: str | Path, validation_report: str | Path, destination: str | Path
) -> Path:
    research = Path(research_root).resolve()
    report_path = Path(validation_report).resolve()
    output = Path(destination).resolve()
    if output.exists():
        raise FileExistsError(f"Serving bundle destination exists: {output}")
    source_manifest_path = research / "manifest.json"
    source_manifest_digest = _sha256(source_manifest_path)
    manifest = _read_json(source_manifest_path)
    report = _read_json(report_path)
    selected = report.get("selected")
    if report.get("kind") != "rolling_window_validation_selection" or not isinstance(
        selected, dict
    ):
        raise ServingPromotionError("Report is not rolling-window validation selection.")
    if selected.get("selection_split") != "validation":
        raise ServingPromotionError("Serving protocol must be selected on validation.")
    if manifest.get("serving_ready") or manifest.get("graph_protocol"):
        raise ServingPromotionError("Source must be an unpromoted research bundle.")
    if manifest.get("bundle_id") != report.get("bundle_id"):
        raise ServingPromotionError("Bundle ID differs from validation evidence.")
    if manifest.get("dataset_digest") != report.get("dataset_digest"):
        raise ServingPromotionError("Dataset digest differs from validation evidence.")
    graph_protocol = selected.get("graph_protocol")
    if not isinstance(graph_protocol, str) or not graph_protocol.startswith(
        "rolling-window-v1:"
    ):
        raise ServingPromotionError("Selected graph protocol is invalid.")
    identity = {
        "research_bundle_id": manifest["bundle_id"],
        "research_manifest_sha256": source_manifest_digest,
        "validation_report_sha256": _sha256(report_path),
        "graph_protocol": graph_protocol,
    }
    suffix = _canonical_digest(identity)[:10]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        shutil.rmtree(temporary)
        shutil.copytree(research, temporary)
        for path in temporary.rglob("*"):
            path.chmod(0o755 if path.is_dir() else 0o644)
        promoted = dict(manifest)
        promoted.update(
            {
                "bundle_id": f"{manifest['bundle_id']}-serving-{suffix}",
                "purpose": "centralized-fedper-research-demo-serving",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "serving_ready": True,
                "serving_readiness_reason": "rolling protocol selected on validation",
                "graph_protocol": graph_protocol,
                "rolling_window_protocol": {
                    key: selected[key]
                    for key in (
                        "duration_seconds",
                        "max_flows",
                        "emit_stride_flows",
                        "allowed_lateness_seconds",
                    )
                },
                "validation_selection": {
                    "selection_rule": selected["selection_rule"],
                    "validation_metrics": selected["validation_metrics"],
                    "validation_inference_latency_ms": selected[
                        "validation_inference_latency_ms"
                    ],
                    "report_sha256": identity["validation_report_sha256"],
                },
                "source_research_bundle": {
                    "bundle_id": manifest["bundle_id"],
                    "manifest_sha256": source_manifest_digest,
                },
            }
        )
        (temporary / "manifest.json").write_text(
            json.dumps(promoted, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        load_inference_bundle(temporary, device="cpu", require_serving_ready=True)
        if _sha256(source_manifest_path) != source_manifest_digest:
            raise ServingPromotionError("Research source manifest changed during promotion.")
        _make_read_only(temporary)
        os.replace(temporary, output)
        return output
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-bundle", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(
        promote_serving_bundle(
            research_root=args.research_bundle,
            validation_report=args.validation_report,
            destination=args.destination,
        )
    )


if __name__ == "__main__":
    main()
