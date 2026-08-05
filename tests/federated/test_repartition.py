from __future__ import annotations

import unittest

import numpy as np

from src.federated.data.repartition import (
    OUTPUT_BIAS,
    OUTPUT_WEIGHT,
    _iid_graph,
    project_seven_class_state,
    remap_seven_labels,
    stratified_iid_train_assignment,
)
from src.federated.data.storage import GraphArrays


def _graph(offset: int) -> GraphArrays:
    labels = np.tile(np.arange(8, dtype=np.int64), 4)
    edges = len(labels)
    edge_index = np.vstack(
        [
            np.arange(edges, dtype=np.int64),
            np.arange(1, edges + 1, dtype=np.int64),
        ]
    )
    return GraphArrays(
        edge_index=edge_index,
        edge_attr=np.arange(offset, offset + edges * 2, dtype=np.float32).reshape(
            edges, 2
        ),
        edge_label=labels,
        train_mask=np.ones(edges, dtype=np.bool_),
        val_mask=np.zeros(edges, dtype=np.bool_),
        test_mask=np.zeros(edges, dtype=np.bool_),
        num_nodes=edges + 1,
    )


class SevenClassRepartitionTests(unittest.TestCase):
    def test_label_projection_drops_six_and_remaps_seven(self):
        keep, labels = remap_seven_labels(np.arange(8, dtype=np.int64))
        np.testing.assert_array_equal(
            keep,
            np.array([True, True, True, True, True, True, False, True]),
        )
        np.testing.assert_array_equal(labels, np.arange(7, dtype=np.int64))

    def test_state_projection_preserves_non_output_tensors(self):
        state = {
            "hidden": np.arange(6, dtype=np.float32).reshape(2, 3),
            OUTPUT_WEIGHT: np.arange(32, dtype=np.float32).reshape(8, 4),
            OUTPUT_BIAS: np.arange(8, dtype=np.float32),
        }
        projected = project_seven_class_state(state)
        np.testing.assert_array_equal(projected["hidden"], state["hidden"])
        np.testing.assert_array_equal(
            projected[OUTPUT_WEIGHT], state[OUTPUT_WEIGHT][[0, 1, 2, 3, 4, 5, 7]]
        )
        np.testing.assert_array_equal(
            projected[OUTPUT_BIAS], state[OUTPUT_BIAS][[0, 1, 2, 3, 4, 5, 7]]
        )

    def test_stratified_assignment_is_balanced_unique_and_deterministic(self):
        graphs = (_graph(0), _graph(1000), _graph(2000))
        first = stratified_iid_train_assignment(
            graphs, num_clients=3, seed=42
        )
        second = stratified_iid_train_assignment(
            graphs, num_clients=3, seed=42
        )
        other_seed = stratified_iid_train_assignment(
            graphs, num_clients=3, seed=1337
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first, other_seed)
        flat = [record for assignment in first for record in assignment]
        self.assertEqual(len(flat), len(set(flat)))
        # Seven retained classes × 12 examples per class.
        self.assertEqual(len(flat), 84)
        for assignment in first:
            supports = np.zeros(7, dtype=np.int64)
            for source, edge in assignment:
                old_label = int(graphs[source].edge_label[edge])
                supports[old_label - int(old_label > 6)] += 1
            np.testing.assert_array_equal(supports, np.full(7, 4))

    def test_iid_graph_namespaces_equal_node_ids_by_source_scenario(self):
        graphs = (_graph(0), _graph(1000))
        combined = _iid_graph(
            graphs,
            ("scenario-a", "scenario-b"),
            [(0, 0), (1, 0)],
        )
        self.assertEqual(combined.num_nodes, 4)
        self.assertEqual(len(np.unique(combined.edge_index)), 4)
        self.assertFalse(
            np.array_equal(combined.edge_index[:, 0], combined.edge_index[:, 1])
        )


if __name__ == "__main__":
    unittest.main()
