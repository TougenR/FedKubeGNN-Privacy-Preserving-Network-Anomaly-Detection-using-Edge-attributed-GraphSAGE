from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.verify_phase1_dataset import (
    REQUIRED_SCENARIOS,
    build_manifest,
    parse_scenario_arguments,
)
from src.phase1_clean import FIXED_LABELS


def _zeek_fixture() -> str:
    header = (
        "#fields\tts\tuid\tid.orig_h\tid.orig_p\tid.resp_h\tid.resp_p"
        "\tproto\tservice\tduration\torig_bytes\tresp_bytes\tconn_state"
        "\tlocal_orig\tlocal_resp\tmissed_bytes\thistory\torig_pkts"
        "\torig_ip_bytes\tresp_pkts\tresp_ip_bytes"
        "\ttunnel_parents   label   detailed-label\n"
    )
    rows = []
    for index, label in enumerate(FIXED_LABELS):
        binary = "Benign" if label == "Benign" else "Malicious"
        fields = [
            str(index),
            f"C{index}",
            f"10.0.0.{index + 1}",
            "50000",
            f"172.16.0.{index + 1}",
            "443",
            "tcp",
            "http",
            "1.0",
            "10",
            "20",
            "SF",
            "-",
            "-",
            "0",
            "ShAD",
            "1",
            "10",
            "1",
            "20",
            f"- {binary} {label}",
        ]
        rows.append("\t".join(fields))
    return header + "\n".join(rows) + "\n"


class DatasetVerifierTests(unittest.TestCase):
    def test_name_path_parser_matches_clean_cli_format(self):
        self.assertEqual(
            parse_scenario_arguments(["1-1=/tmp/a", "3-1=/tmp/b"]),
            {"1-1": "/tmp/a", "3-1": "/tmp/b"},
        )
        with self.assertRaises(ValueError):
            parse_scenario_arguments(["not-a-mapping"])

    def test_manifest_requires_all_six_scenarios_and_stream_parses_rows(self):
        with tempfile.TemporaryDirectory(prefix="dataset-verify-") as temp:
            path = Path(temp) / "conn.log.labeled"
            path.write_text(_zeek_fixture(), encoding="utf-8")
            complete = build_manifest(
                {name: str(path) for name in REQUIRED_SCENARIOS},
                chunksize=3,
                cache_dir=Path(temp) / "cache",
            )
            self.assertTrue(complete["complete"])
            for record in complete["scenarios"]:
                self.assertEqual(record["raw_data_rows"], 8)
                self.assertEqual(record["parsed_rows"], 8)
                self.assertEqual(record["parse_error_rows"], 0)
                self.assertEqual(record["missing_expected_labels"], [])
                self.assertEqual(record["unexpected_labels"], [])
                self.assertEqual(record["unique_source_ips"], 8)
                self.assertEqual(record["unique_destination_ips"], 8)
                self.assertEqual(len(record["sha256"]), 64)

            incomplete = build_manifest(
                {"3-1": str(path)},
                chunksize=3,
                cache_dir=Path(temp) / "cache",
            )
            self.assertFalse(incomplete["complete"])
            missing = {
                record["scenario"]
                for record in incomplete["scenarios"]
                if record["error"] == "SCENARIO_NOT_SUPPLIED"
            }
            self.assertEqual(
                missing,
                set(REQUIRED_SCENARIOS) - {"3-1"},
            )


if __name__ == "__main__":
    unittest.main()
