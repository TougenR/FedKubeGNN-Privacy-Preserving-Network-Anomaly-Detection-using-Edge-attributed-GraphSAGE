from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.application.evaluation.iot23_replay import (
    ReplayPreparationError,
    load_source_contract,
    split_sampled_frame,
)


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "configs" / "application" / "iot23-replay-sources.json"


class IoT23ReplayTests(unittest.TestCase):
    def test_locked_contract_has_six_official_digest_pinned_sources(self) -> None:
        document = load_source_contract(CONTRACT)
        self.assertEqual(len(document["sources"]), 6)
        self.assertEqual(sum(source["size"] for source in document["sources"]), 13_842_289_244)
        self.assertTrue(
            all(len(source["sha256"]) == 64 for source in document["sources"])
        )

    def test_source_contract_rejects_non_official_host(self) -> None:
        document = json.loads(CONTRACT.read_text(encoding="utf-8"))
        document["sources"][0]["url"] = "https://example.com/conn.log.labeled"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(ReplayPreparationError):
                load_source_contract(path)

    def test_split_drops_only_removed_class_and_never_returns_train(self) -> None:
        contract = {
            "seed": 42,
            "split": {"train": 0.7, "validation": 0.1, "test": 0.2},
            "dropped_class": "Okiru-Attack",
        }
        labels = ["Benign"] * 10 + ["C&C"] * 10 + ["Okiru-Attack"] * 10
        frame = pd.DataFrame(
            {
                "ts": list(range(30)),
                "id.orig_h": ["10.0.0.1"] * 30,
                "id.resp_h": ["10.0.0.2"] * 30,
                "detailed-label": labels,
            }
        )
        splits = split_sampled_frame(frame, contract=contract)
        self.assertEqual(set(splits), {"validation", "test"})
        self.assertEqual(sum(len(value) for value in splits.values()), 6)
        self.assertTrue(
            all(
                "Okiru-Attack" not in set(value["detailed-label"])
                for value in splits.values()
            )
        )
        self.assertTrue(
            all(set(value["evaluation_split"]) == {name} for name, value in splits.items())
        )


if __name__ == "__main__":
    unittest.main()
