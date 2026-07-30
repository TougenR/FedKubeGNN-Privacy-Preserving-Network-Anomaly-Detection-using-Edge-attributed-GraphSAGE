from __future__ import annotations

import copy
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.phase1_clean import (
    ROW_ID_COLUMN,
    prepare_loso_clean,
    prepare_transductive_clean,
)
from src.phase1_data_cache import (
    cache_fingerprint,
    load_canonical_scenario,
    validate_cache_columns,
)
from src.multi_scenario import _read_chunked_with_cap, load_all_scenarios


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _config() -> dict:
    with (REPOSITORY_ROOT / "config.yaml").open(encoding="utf-8") as handle:
        return copy.deepcopy(yaml.safe_load(handle))


def _zeek_fixture(labels: tuple[str, ...], rows: int = 72) -> str:
    header = (
        "#fields\tts\tuid\tid.orig_h\tid.orig_p\tid.resp_h\tid.resp_p"
        "\tproto\tservice\tduration\torig_bytes\tresp_bytes\tconn_state"
        "\tlocal_orig\tlocal_resp\tmissed_bytes\thistory\torig_pkts"
        "\torig_ip_bytes\tresp_pkts\tresp_ip_bytes"
        "\ttunnel_parents   label   detailed-label\n"
    )
    output = []
    for index in range(rows):
        detailed = labels[index % len(labels)]
        binary = "Benign" if detailed == "Benign" else "Malicious"
        service = "-" if index == rows - 1 else ("http" if index % 2 else "dns")
        duration = "-" if index == rows - 2 else str(index % 7 + 1)
        fields = [
            str(index),
            f"C{index}",
            f"10.0.{index % 4}.{index % 20 + 1}",
            str(40000 + index),
            f"172.16.{index % 3}.{index % 18 + 1}",
            ("22", "80", "443")[index % 3],
            ("tcp", "udp")[index % 2],
            service,
            duration,
            str(index + 1),
            str(index + 2),
            ("SF", "S0")[index % 2],
            "-",
            "-",
            "0",
            ("ShAD", "D")[index % 2],
            str(index % 5 + 1),
            str(index + 20),
            str(index % 4 + 1),
            str(index + 30),
            f"- {binary} {detailed}",
        ]
        output.append("\t".join(fields))
    return header + "\n".join(output) + "\n"


def _preprocessor_signature(preprocessor) -> dict:
    return {
        "resp_port_categories": preprocessor.resp_port_categories,
        "proto_categories": preprocessor.proto_categories,
        "service_categories": preprocessor.service_categories,
        "conn_state_categories": preprocessor.conn_state_categories,
        "history_flag_chars": preprocessor.history_flag_chars,
        "numeric_columns": preprocessor.numeric_columns,
        "missing_flag_columns": preprocessor.missing_flag_columns,
        "feature_columns": preprocessor.feature_columns,
    }


class Phase1CanonicalCacheTests(unittest.TestCase):
    def _paths(self, root: Path) -> dict[str, str]:
        paths = {}
        fixtures = {
            "toy-a": ("Benign", "Attack", "C&C"),
            "toy-b": ("Benign", "DDoS", "PartOfAHorizontalPortScan"),
            "toy-c": ("Benign", "Okiru", "C&C-HeartBeat"),
        }
        for scenario, labels in fixtures.items():
            path = root / scenario / "conn.log.labeled"
            path.parent.mkdir(parents=True)
            path.write_text(_zeek_fixture(labels), encoding="utf-8")
            paths[scenario] = str(path)
        return paths

    def test_raw_miss_hit_are_equal_and_hit_does_not_open_raw(self):
        with tempfile.TemporaryDirectory(prefix="phase1-cache-") as temporary:
            root = Path(temporary)
            path = next(iter(self._paths(root).values()))
            cache_dir = root / "cache"
            raw, raw_report = load_canonical_scenario(
                path,
                "toy-a",
                cache_enabled=False,
                cache_dir=cache_dir,
                chunksize=11,
            )
            miss, miss_report = load_canonical_scenario(
                path,
                "toy-a",
                cache_dir=cache_dir,
                chunksize=11,
            )
            hit, hit_report = load_canonical_scenario(
                path,
                "toy-a",
                cache_dir=cache_dir,
                chunksize=11,
            )
            pd.testing.assert_frame_equal(raw, miss)
            pd.testing.assert_frame_equal(miss, hit)
            self.assertEqual(raw_report.raw_open_count, 1)
            self.assertEqual(miss_report.cache_status, "MISS")
            self.assertEqual(miss_report.raw_open_count, 1)
            self.assertEqual(hit_report.cache_status, "HIT")
            self.assertEqual(hit_report.raw_open_count, 0)
            self.assertEqual(hit_report.fingerprint, miss_report.fingerprint)

    def test_cap_variants_reuse_cache_and_cache_has_no_learned_fields(self):
        with tempfile.TemporaryDirectory(prefix="phase1-cache-") as temporary:
            root = Path(temporary)
            path = next(iter(self._paths(root).values()))
            cache_dir = root / "cache"
            small, first = load_canonical_scenario(
                path,
                "toy-a",
                cache_dir=cache_dir,
                chunksize=13,
                cap_per_class=3,
            )
            large, second = load_canonical_scenario(
                path,
                "toy-a",
                cache_dir=cache_dir,
                chunksize=13,
                cap_per_class=7,
            )
            self.assertEqual(first.cache_status, "MISS")
            self.assertEqual(second.cache_status, "HIT")
            self.assertEqual(first.fingerprint, second.fingerprint)
            self.assertLess(len(small), len(large))
            cache = Path(first.cache_path)
            columns = pd.read_parquet(cache).columns.tolist()
            validate_cache_columns(columns)
            self.assertIn(ROW_ID_COLUMN, columns)
            for forbidden in (
                "train_mask",
                "split",
                "scaler",
                "category_vocabulary",
                "class_weights",
                "transformed_features",
            ):
                self.assertNotIn(forbidden, columns)

    def test_cached_cap_matches_existing_raw_parser_semantics(self):
        with tempfile.TemporaryDirectory(prefix="phase1-cache-") as temporary:
            root = Path(temporary)
            path = next(iter(self._paths(root).values()))
            historical = _read_chunked_with_cap(
                path,
                cap_per_class=20,
                chunksize=17,
            ).reset_index(drop=True)
            cached, _ = load_canonical_scenario(
                path,
                "toy-a",
                cache_dir=root / "cache",
                chunksize=17,
                cap_per_class=20,
            )
            pd.testing.assert_frame_equal(
                historical,
                cached.drop(columns=[ROW_ID_COLUMN]),
            )

    def test_raw_identity_change_invalidates_fingerprint(self):
        with tempfile.TemporaryDirectory(prefix="phase1-cache-") as temporary:
            root = Path(temporary)
            path = next(iter(self._paths(root).values()))
            first, _ = cache_fingerprint(path)
            time.sleep(0.002)
            with Path(path).open("a", encoding="utf-8") as handle:
                handle.write("#close\tfixture\n")
            second, _ = cache_fingerprint(path)
            self.assertNotEqual(first, second)

    def test_cache_on_off_preserves_splits_preprocessor_and_weights(self):
        with tempfile.TemporaryDirectory(prefix="phase1-cache-") as temporary:
            root = Path(temporary)
            paths = self._paths(root)
            reports_raw: list[dict] = []
            reports_cached: list[dict] = []
            raw_frames = load_all_scenarios(
                paths,
                cap_per_class=20,
                chunksize=17,
                cache_enabled=False,
                cache_dir=str(root / "cache"),
                load_reports=reports_raw,
            )
            cached_frames = load_all_scenarios(
                paths,
                cap_per_class=20,
                chunksize=17,
                cache_enabled=True,
                cache_dir=str(root / "cache"),
                load_reports=reports_cached,
            )
            hit_frames = load_all_scenarios(
                paths,
                cap_per_class=20,
                chunksize=17,
                cache_enabled=True,
                cache_dir=str(root / "cache"),
            )
            for scenario in paths:
                pd.testing.assert_frame_equal(
                    raw_frames[scenario], cached_frames[scenario]
                )
                pd.testing.assert_frame_equal(
                    cached_frames[scenario], hit_frames[scenario]
                )
            self.assertTrue(
                all(item["raw_open_count"] == 1 for item in reports_raw)
            )
            self.assertTrue(
                all(item["cache_status"] == "MISS" for item in reports_cached)
            )

            config = _config()
            raw_pooled = prepare_transductive_clean(
                raw_frames, config, seed=42
            )
            cache_pooled = prepare_transductive_clean(
                cached_frames, config, seed=42
            )
            self.assertEqual(
                {
                    key: value.manifest_record()
                    for key, value in raw_pooled.split_plans.items()
                },
                {
                    key: value.manifest_record()
                    for key, value in cache_pooled.split_plans.items()
                },
            )
            self.assertEqual(
                _preprocessor_signature(raw_pooled.preprocessor),
                _preprocessor_signature(cache_pooled.preprocessor),
            )
            np.testing.assert_allclose(
                raw_pooled.preprocessor.scaler.mean_,
                cache_pooled.preprocessor.scaler.mean_,
            )
            np.testing.assert_allclose(
                raw_pooled.preprocessor.scaler.scale_,
                cache_pooled.preprocessor.scaler.scale_,
            )
            self.assertTrue(
                torch.equal(
                    raw_pooled.class_weights, cache_pooled.class_weights
                )
            )

            raw_loso = prepare_loso_clean(
                raw_frames, config, held_out="toy-c", seed=42
            )
            cache_loso = prepare_loso_clean(
                cached_frames, config, held_out="toy-c", seed=42
            )
            self.assertEqual(
                {
                    key: (
                        value.scenario,
                        value.seed,
                        value.protocol,
                        value.all_ids,
                        value.train_ids,
                        value.val_ids,
                        value.test_ids,
                    )
                    for key, value in raw_loso.split_plans.items()
                },
                {
                    key: (
                        value.scenario,
                        value.seed,
                        value.protocol,
                        value.all_ids,
                        value.train_ids,
                        value.val_ids,
                        value.test_ids,
                    )
                    for key, value in cache_loso.split_plans.items()
                },
            )
            self.assertEqual(
                _preprocessor_signature(raw_loso.preprocessor),
                _preprocessor_signature(cache_loso.preprocessor),
            )
            np.testing.assert_allclose(
                raw_loso.preprocessor.scaler.mean_,
                cache_loso.preprocessor.scaler.mean_,
            )
            np.testing.assert_allclose(
                raw_loso.preprocessor.scaler.scale_,
                cache_loso.preprocessor.scaler.scale_,
            )
            self.assertTrue(
                torch.equal(raw_loso.class_weights, cache_loso.class_weights)
            )


if __name__ == "__main__":
    unittest.main()
