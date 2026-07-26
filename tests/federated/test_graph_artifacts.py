import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.federated.contracts.schema import ContractError
from src.federated.data.storage import (
    GraphArrays,
    load_graph_arrays,
    write_graph_arrays,
)


def example_graph():
    return GraphArrays(
        edge_index=np.array([[0, 1, 2, 0], [1, 2, 0, 2]], dtype=np.int64),
        edge_attr=np.arange(12, dtype=np.float32).reshape(4, 3),
        edge_label=np.array([0, 1, 0, 1], dtype=np.int64),
        train_mask=np.array([True, True, False, False]),
        val_mask=np.array([False, False, True, False]),
        test_mask=np.array([False, False, False, True]),
        num_nodes=3,
    )


class GraphArtifactTests(unittest.TestCase):
    def test_round_trip_uses_portable_arrays(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = write_graph_arrays(Path(temporary) / "client", example_graph())
            loaded = load_graph_arrays(root)
            np.testing.assert_array_equal(loaded.edge_index, example_graph().edge_index)
            self.assertEqual(loaded.feature_dim, 3)
            self.assertEqual(loaded.num_edges, 4)

    def test_checksum_detects_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = write_graph_arrays(Path(temporary) / "client", example_graph())
            with (root / "edge_label.npy").open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaises(ContractError):
                load_graph_arrays(root)

    def test_mask_overlap_is_rejected(self):
        graph = example_graph()
        graph.train_mask[2] = True
        with self.assertRaises(ContractError):
            graph.validate()


if __name__ == "__main__":
    unittest.main()
