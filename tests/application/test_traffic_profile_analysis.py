from __future__ import annotations

import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.application.evaluation.traffic_profile_analysis import (
    TrafficProfileAnalysisError,
    analyze_validation_profiles,
    write_reference,
)


CLASSES = [
    "Benign",
    "Attack",
    "C&C",
    "C&C-HeartBeat",
    "DDoS",
    "Okiru",
    "PartOfAHorizontalPortScan",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TrafficProfileAnalysisTests(unittest.TestCase):
    def _replay(self, root: Path, *, split: str = "validation") -> Path:
        clients: dict[str, dict] = {}
        for index, class_name in enumerate(CLASSES):
            client_id = f"client-{index}"
            relative = Path("clients") / client_id / f"{split}.jsonl.gz"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            rows = [
                {
                    "source_edge_index": index * 10 + offset,
                    "evaluation_split": split,
                    "ts": 1000.0 + offset,
                    "id.orig_h": f"10.0.{index}.1",
                    "id.orig_p": 40000 + offset,
                    "id.resp_h": f"10.1.{index}.{offset + 1}",
                    "id.resp_p": 80 if class_name != "Attack" else 22,
                    "proto": "tcp",
                    "service": "http" if class_name != "Attack" else "ssh",
                    "duration": float(offset + 1),
                    "orig_bytes": float(10 + offset),
                    "resp_bytes": float(20 + offset),
                    "conn_state": "SF",
                    "missed_bytes": 0.0,
                    "history": "ShADadFf",
                    "orig_pkts": 2.0,
                    "orig_ip_bytes": 100.0,
                    "resp_pkts": 2.0,
                    "resp_ip_bytes": 120.0,
                    "duration_missing": 0,
                    "orig_bytes_missing": 0,
                    "resp_bytes_missing": 0,
                    "label": "Malicious" if class_name != "Benign" else "Benign",
                    "detailed-label": class_name,
                }
                for offset in range(2)
            ]
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            counts = [2 if value == class_name else 0 for value in CLASSES]
            clients[client_id] = {
                split: {
                    "path": str(relative),
                    "rows": 2,
                    "sha256": _sha256(path),
                    "class_counts": counts,
                }
            }
        manifest = {
            "schema_version": 1,
            "kind": "labeled-scientific-evaluation-only",
            "classes": CLASSES,
            "source_dataset_id": "source",
            "source_dataset_digest": "a" * 64,
            "derived_dataset_id": "derived",
            "derived_dataset_digest": "b" * 64,
            "clients": clients,
        }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return root

    def test_analysis_is_validation_only_and_covers_all_classes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            document = analyze_validation_profiles(
                replay_root=self._replay(Path(directory))
            )
        self.assertEqual(document["selection_split"], "validation")
        self.assertFalse(document["locked_test_read"])
        self.assertEqual(document["classes"], CLASSES)
        self.assertEqual(document["dataset"]["validation_rows"], 14)
        self.assertFalse(
            document["sampling_contract"]["generator_timing_authoritative"]
        )
        self.assertEqual(document["profiles"]["Attack"]["support"], 2)
        self.assertEqual(
            document["profiles"]["Attack"]["candidate_reference"][
                "dominant_destination_ports_90pct"
            ],
            ["22"],
        )
        self.assertEqual(len(document["reference_digest"]), 64)

    def test_analysis_rejects_non_validation_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._replay(Path(directory), split="validation")
            first = next(iter(json.loads((root / "manifest.json").read_text())["clients"].values()))
            path = root / first["validation"]["path"]
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
            rows[0]["evaluation_split"] = "test"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            first["validation"]["sha256"] = _sha256(path)
            manifest = json.loads((root / "manifest.json").read_text())
            next(iter(manifest["clients"].values()))["validation"]["sha256"] = _sha256(path)
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(TrafficProfileAnalysisError):
                analyze_validation_profiles(replay_root=root)

    def test_write_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            replay = self._replay(root / "replay")
            output = root / "reference.json"
            write_reference(replay_root=replay, output=output)
            with self.assertRaises(FileExistsError):
                write_reference(replay_root=replay, output=output)


if __name__ == "__main__":
    unittest.main()
