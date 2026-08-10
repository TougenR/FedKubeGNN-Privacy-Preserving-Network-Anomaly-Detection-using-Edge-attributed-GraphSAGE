from __future__ import annotations

import unittest

import pandas as pd

from src.application.evaluation.traffic_profile_comparator import (
    TrafficProfileComparisonError,
    _validate_dataset_digest,
    compare_candidate_frame,
)


CLASSES = (
    "Benign",
    "Attack",
    "C&C",
    "C&C-HeartBeat",
    "DDoS",
    "Okiru",
    "PartOfAHorizontalPortScan",
)


def frame(port: int, state: str = "S0") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "proto": "tcp",
                "service": "-",
                "conn_state": state,
                "history": "S",
                "id.resp_p": port,
                "id.resp_h": f"node-{index % 2}",
                "duration": None,
                "orig_bytes": None,
                "resp_bytes": None,
                "orig_pkts": 1.0,
                "orig_ip_bytes": 60.0,
                "resp_pkts": 0.0,
                "resp_ip_bytes": 0.0,
            }
            for index in range(20)
        ]
    )


class TrafficProfileComparatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.references = {
            class_name: frame(10000 + index) for index, class_name in enumerate(CLASSES)
        }

    def test_exact_candidate_passes_deterministic_envelope_and_nearest_class(
        self,
    ) -> None:
        candidate = self.references["Okiru"].iloc[:3].copy()
        result = compare_candidate_frame(
            candidate=candidate,
            references=self.references,
            expected_class="Okiru",
            profile_id="okiru",
            reference_digest="a" * 64,
            scientific_status="candidate",
            bootstrap_iterations=100,
        )
        self.assertTrue(result["accepted"])
        self.assertEqual(result["nearest_reference"]["nearest_classes"], ["Okiru"])
        self.assertFalse(result["locked_test_read"])

    def test_control_can_match_but_cannot_be_accepted_as_class_equivalent(self) -> None:
        result = compare_candidate_frame(
            candidate=self.references["Benign"].iloc[:2].copy(),
            references=self.references,
            expected_class="Benign",
            profile_id="benign-control",
            reference_digest="b" * 64,
            scientific_status="control-not-class-equivalent",
            bootstrap_iterations=100,
        )
        self.assertFalse(result["accepted"])
        self.assertEqual(result["result"], "control-only")

    def test_single_flow_is_rejected_before_scientific_interpretation(self) -> None:
        with self.assertRaises(TrafficProfileComparisonError):
            compare_candidate_frame(
                candidate=self.references["Okiru"].iloc[:1].copy(),
                references=self.references,
                expected_class="Okiru",
                profile_id="okiru",
                reference_digest="a" * 64,
                scientific_status="candidate",
                bootstrap_iterations=100,
            )

    def test_catalog_is_bound_to_derived_training_dataset_digest(self) -> None:
        manifest = {
            "source_dataset_digest": "source-digest",
            "derived_dataset_digest": "derived-digest",
        }
        _validate_dataset_digest(manifest, "derived-digest")
        with self.assertRaises(TrafficProfileComparisonError):
            _validate_dataset_digest(manifest, "source-digest")


if __name__ == "__main__":
    unittest.main()
