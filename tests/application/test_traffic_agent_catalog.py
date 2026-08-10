from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from src.application.traffic_agent.catalog import (
    TrafficTargetCatalog,
    load_profile_catalog,
    load_target_catalog,
)


ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "configs" / "application" / "scientific-traffic-profiles.yaml"
TARGETS = ROOT / "configs" / "application" / "traffic-targets.gcp.yaml"


class TrafficAgentCatalogTests(unittest.TestCase):
    def test_catalog_covers_seven_classes_and_blocks_unready_profiles(self) -> None:
        catalog = load_profile_catalog(CATALOG)
        self.assertEqual(len(catalog.profiles), 7)
        self.assertEqual(len({item.reference_class for item in catalog.profiles}), 7)
        self.assertTrue(catalog.profile("command-control-heartbeat").execution_enabled)
        self.assertFalse(catalog.profile("ddos").execution_enabled)
        self.assertEqual(
            catalog.profile("ddos").scientific_status,
            "unsupported-dataset-artifact",
        )

    def test_catalog_cannot_enable_unsupported_profile(self) -> None:
        document = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
        ddos = next(item for item in document["profiles"] if item["id"] == "ddos")
        ddos["execution_enabled"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.yaml"
            path.write_text(yaml.safe_dump(document), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_profile_catalog(path)

    def test_targets_must_be_literal_private_addresses(self) -> None:
        deployed = load_target_catalog(TARGETS)
        self.assertEqual(deployed.source_ipv4, "10.10.0.20")
        self.assertEqual(
            deployed.groups["multi-blackhole"].endpoints,
            ["10.20.0.20", "10.20.0.21", "10.20.0.22"],
        )
        valid = TrafficTargetCatalog.model_validate(
            {
                "schema_version": 1,
                "source_ipv4": "10.10.0.20",
                "groups": {
                    "gateway-http": {"endpoints": ["http://10.10.0.5/target/"]},
                    "blackholes": {"endpoints": ["10.20.0.20", "10.20.0.21"]},
                },
            }
        )
        self.assertEqual(len(valid.groups), 2)
        for endpoint in (
            "https://10.10.0.5",
            "http://example.com",
            "8.8.8.8",
            "127.0.0.1",
        ):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                TrafficTargetCatalog.model_validate(
                    {
                        "schema_version": 1,
                        "source_ipv4": "10.10.0.20",
                        "groups": {"invalid": {"endpoints": [endpoint]}},
                    }
                )


if __name__ == "__main__":
    unittest.main()
