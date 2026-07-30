from __future__ import annotations

import math
import unittest

import pandas as pd

from src.imbalance import compute_class_weights, undersample_majority
from src.preprocess import clean_flows, fit_preprocessor, transform


def _clean_frame() -> pd.DataFrame:
    rows = []
    for uid, proto, service, conn_state, duration, detailed_label in (
        ("train-1", "tcp", "http", "SF", "1.0", "Benign"),
        ("train-2", "tcp", "http", "SF", "3.0", "Attack"),
        ("held-1", "quic", "novel-service", "ZZ", "101.0", "DDoS"),
    ):
        rows.append(
            {
                "ts": "1.0",
                "uid": uid,
                "id.orig_h": "10.0.0.1",
                "id.orig_p": "50000",
                "id.resp_h": "10.0.0.2",
                "id.resp_p": "443",
                "proto": proto,
                "service": service,
                "duration": duration,
                "orig_bytes": "10",
                "resp_bytes": "20",
                "conn_state": conn_state,
                "local_orig": "-",
                "local_resp": "-",
                "missed_bytes": "0",
                "history": "ShAD",
                "orig_pkts": "1",
                "orig_ip_bytes": "10",
                "resp_pkts": "1",
                "resp_ip_bytes": "20",
                "tunnel_parents": "-",
                "label": "Malicious",
                "detailed-label": detailed_label,
            }
        )
    return clean_flows(pd.DataFrame(rows))


class PreprocessingContractTests(unittest.TestCase):
    def test_fit_only_learns_categories_and_scaler_statistics_from_input_train_rows(self):
        frame = _clean_frame()
        train = frame.iloc[:2].copy()
        held_out = frame.iloc[2:].copy()

        preprocessor = fit_preprocessor(train)
        held_transformed = transform(held_out, preprocessor)

        self.assertEqual(preprocessor.proto_categories, ["tcp"])
        self.assertEqual(preprocessor.service_categories, ["http"])
        self.assertEqual(preprocessor.conn_state_categories, ["SF"])
        self.assertAlmostEqual(
            float(preprocessor.scaler.mean_[0]),
            (math.log1p(1.0) + math.log1p(3.0)) / 2,
        )
        self.assertNotIn("proto_quic", preprocessor.feature_columns)
        self.assertEqual(
            list(held_transformed.columns[3:-2]), preprocessor.feature_columns
        )
        self.assertEqual(int(held_transformed.filter(like="proto_").sum(axis=1).iloc[0]), 0)
        self.assertEqual(int(held_transformed["service_other"].iloc[0]), 1)

    def test_imbalance_helpers_use_only_the_rows_passed_by_the_caller(self):
        frame = _clean_frame()
        train = frame.iloc[:2].copy()
        held_out_before = frame.iloc[2:].copy(deep=True)

        weights, mapping, _ = compute_class_weights(train["detailed-label"])
        undersampled = undersample_majority(train, verbose=False)

        self.assertEqual(set(weights), {"Attack", "Benign"})
        self.assertEqual(mapping, {"Attack": 0, "Benign": 1})
        self.assertEqual(set(undersampled["detailed-label"]), {"Attack", "Benign"})
        pd.testing.assert_frame_equal(frame.iloc[2:].reset_index(drop=True), held_out_before.reset_index(drop=True))


if __name__ == "__main__":
    unittest.main()
