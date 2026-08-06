from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.application.inference.bundle_loader import load_inference_bundle
from src.application.inference.promote import promote_serving_bundle


ROOT = Path(__file__).resolve().parents[2]
RESEARCH = (
    ROOT
    / "artifacts"
    / "federated"
    / "exports"
    / "fedper-gke-14339380272482304688-r0030-42642e4cc839-b02"
)


class ServingPromotionTests(unittest.TestCase):
    def test_promotion_is_derived_immutable_and_strictly_loadable(self) -> None:
        source_digest = hashlib.sha256((RESEARCH / "manifest.json").read_bytes()).hexdigest()
        source_manifest = json.loads((RESEARCH / "manifest.json").read_text())
        report = {
            "kind": "rolling_window_validation_selection",
            "bundle_id": source_manifest["bundle_id"],
            "dataset_digest": source_manifest["dataset_digest"],
            "selected": {
                "selection_split": "validation",
                "selection_rule": "test rule",
                "graph_protocol": (
                    "rolling-window-v1:duration=15s:max-flows=100:stride=1:lateness=1s"
                ),
                "duration_seconds": 15.0,
                "max_flows": 100,
                "emit_stride_flows": 1,
                "allowed_lateness_seconds": 1.0,
                "validation_metrics": {"macro_f1_fixed": 0.9},
                "validation_inference_latency_ms": {"p95": 12.0},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "validation.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            destination = root / "serving"
            promote_serving_bundle(
                research_root=RESEARCH,
                validation_report=report_path,
                destination=destination,
            )
            bundle = load_inference_bundle(
                destination, device="cpu", require_serving_ready=True
            )
            self.assertTrue(bundle.manifest["serving_ready"])
            self.assertEqual(bundle.manifest["rolling_window_protocol"]["max_flows"], 100)
            self.assertEqual(destination.stat().st_mode & 0o777, 0o555)
        self.assertEqual(
            hashlib.sha256((RESEARCH / "manifest.json").read_bytes()).hexdigest(),
            source_digest,
        )


if __name__ == "__main__":
    unittest.main()
