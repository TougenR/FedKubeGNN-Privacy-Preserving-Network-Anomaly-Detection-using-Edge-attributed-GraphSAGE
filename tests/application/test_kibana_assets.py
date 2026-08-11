from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SAVED_OBJECTS = ROOT / "deploy/application/helm/detection-stack/files/kibana.ndjson"


class KibanaAssetsTests(unittest.TestCase):
    def test_strict_mapping_accepts_privacy_reduced_run_correlation(self) -> None:
        template = (
            ROOT
            / "deploy/application/helm/detection-stack/templates/elastic-bootstrap.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn('"run_id": {"type": "keyword"}', template)
        self.assertNotIn('"zeek_evidence"', template)

    def test_dashboard_references_version_controlled_objects(self) -> None:
        objects = [
            json.loads(line)
            for line in SAVED_OBJECTS.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        object_keys = {(item["type"], item["id"]) for item in objects}
        self.assertEqual(len(objects), 8)
        self.assertEqual(len(object_keys), len(objects))

        dashboard = next(
            item
            for item in objects
            if item["type"] == "dashboard" and item["id"] == "fedper-detection-overview"
        )
        panels = json.loads(dashboard["attributes"]["panelsJSON"])
        self.assertEqual(len(panels), 6)
        self.assertEqual(len(dashboard["references"]), 6)
        for reference in dashboard["references"]:
            self.assertIn((reference["type"], reference["id"]), object_keys)

    def test_dashboard_does_not_display_forbidden_payload_fields(self) -> None:
        objects = [
            json.loads(line)
            for line in SAVED_OBJECTS.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        displayed_fields: set[str] = set()
        for item in objects:
            attributes = item.get("attributes", {})
            displayed_fields.update(attributes.get("columns", []))
            if "visState" in attributes:
                vis_state = json.loads(attributes["visState"])
                displayed_fields.update(
                    aggregation.get("params", {}).get("field", "")
                    for aggregation in vis_state.get("aggs", [])
                )
        for forbidden in (
            "id.orig_h",
            "id.resp_h",
            "probabilities",
            "ground_truth",
            "tensor",
            "raw_feature",
        ):
            self.assertNotIn(forbidden, displayed_fields)


if __name__ == "__main__":
    unittest.main()
