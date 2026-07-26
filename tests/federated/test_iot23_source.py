import tempfile
import unittest
from pathlib import Path

from src.federated.data.sources.iot23 import read_clean_priority_sample
from src.preprocess import _MOCK_CONN_LOG_CLEAN


class IoT23SourceTests(unittest.TestCase):
    def test_priority_sample_is_deterministic_and_capped_across_chunks(self):
        content = (
            _MOCK_CONN_LOG_CLEAN.replace("C&C-Mirai", "Attack")
            .replace("C&C-FileDownload", "Attack")
            .replace("PartOfAHorizontalPortScan", "Attack")
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "conn.log.labeled"
            path.write_text(content, encoding="utf-8")
            first = read_clean_priority_sample(
                path, cap_per_class=2, chunk_size=2, seed=42
            )
            second = read_clean_priority_sample(
                path, cap_per_class=2, chunk_size=2, seed=42
            )
            self.assertEqual(
                first["detailed-label"].value_counts().to_dict(),
                {"Attack": 2, "Benign": 2},
            )
            self.assertEqual(first["ts"].tolist(), second["ts"].tolist())


if __name__ == "__main__":
    unittest.main()
