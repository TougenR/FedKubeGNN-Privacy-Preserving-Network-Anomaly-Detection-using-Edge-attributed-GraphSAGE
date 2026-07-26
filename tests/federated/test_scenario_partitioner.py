import unittest

import numpy as np

from src.federated.data.partitioners.scenario import deterministic_edge_masks


class ScenarioPartitionerTests(unittest.TestCase):
    def test_masks_are_deterministic_disjoint_and_cover(self):
        labels = np.array([0] * 10 + [1] * 7 + [2])
        first = deterministic_edge_masks(
            labels, train_ratio=0.7, validation_ratio=0.1, test_ratio=0.2, seed=42
        )
        second = deterministic_edge_masks(
            labels, train_ratio=0.7, validation_ratio=0.1, test_ratio=0.2, seed=42
        )
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
        coverage = first[0].astype(int) + first[1] + first[2]
        np.testing.assert_array_equal(coverage, np.ones(len(labels)))
        singleton_index = len(labels) - 1
        self.assertTrue(first[0][singleton_index])


if __name__ == "__main__":
    unittest.main()
