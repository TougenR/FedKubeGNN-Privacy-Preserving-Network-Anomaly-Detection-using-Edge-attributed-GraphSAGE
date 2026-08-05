from __future__ import annotations

import unittest

import numpy as np

from src.federated.contracts.schema import ContractError, ModelSpec
from src.federated.contracts.task import LocalTrainResult
from src.federated.core.aggregation import (
    class_balanced_client_fedavg,
    class_balanced_client_head_fedavg,
    class_balanced_client_weights,
    class_support_head_fedavg,
    weighted_fedavg,
)


class AggregationTests(unittest.TestCase):
    def test_class_balanced_weights_equalize_global_class_influence(self) -> None:
        weights = class_balanced_client_weights(
            np.array([[90, 0], [10, 100]], dtype=np.int64)
        )
        np.testing.assert_allclose(weights, [0.45, 0.55])

    def test_class_balanced_fedavg_uses_external_client_weights(self) -> None:
        state = {"weight": np.zeros((1,), dtype=np.float32)}
        spec = ModelSpec.from_state(
            family="test",
            model_version=1,
            feature_dim=1,
            num_classes=2,
            node_feature_dim=1,
            hyperparameters={},
            state=state,
        )
        results = [
            LocalTrainResult(
                state={"weight": np.array([0.0], dtype=np.float32)},
                num_examples=90,
            ),
            LocalTrainResult(
                state={"weight": np.array([10.0], dtype=np.float32)},
                num_examples=110,
            ),
        ]
        aggregated = class_balanced_client_fedavg(
            results,
            class_support=np.array([[90, 0], [10, 100]]),
            model_spec=spec,
        )
        np.testing.assert_allclose(aggregated["weight"], [5.5])

    def test_combined_aggregation_uses_balanced_trunk_and_support_head(self) -> None:
        state = {
            "trunk": np.zeros((1,), dtype=np.float32),
            "head.3.weight": np.zeros((2, 1), dtype=np.float32),
            "head.3.bias": np.zeros((2,), dtype=np.float32),
        }
        spec = ModelSpec.from_state(
            family="test",
            model_version=1,
            feature_dim=1,
            num_classes=2,
            node_feature_dim=1,
            hyperparameters={},
            state=state,
        )
        results = [
            LocalTrainResult(
                state={
                    "trunk": np.array([0.0], dtype=np.float32),
                    "head.3.weight": np.array([[10.0], [20.0]], dtype=np.float32),
                    "head.3.bias": np.array([1.0, 2.0], dtype=np.float32),
                },
                num_examples=90,
            ),
            LocalTrainResult(
                state={
                    "trunk": np.array([10.0], dtype=np.float32),
                    "head.3.weight": np.array([[30.0], [40.0]], dtype=np.float32),
                    "head.3.bias": np.array([3.0, 4.0], dtype=np.float32),
                },
                num_examples=110,
            ),
        ]
        aggregated = class_balanced_client_head_fedavg(
            results,
            class_support=np.array([[90, 0], [10, 100]]),
            model_spec=spec,
        )
        np.testing.assert_allclose(aggregated["trunk"], [5.5])
        np.testing.assert_allclose(aggregated["head.3.weight"], [[12.0], [40.0]])
        np.testing.assert_allclose(aggregated["head.3.bias"], [1.2, 4.0])

    def test_class_support_head_excludes_absent_clients_per_output_row(self) -> None:
        state = {
            "trunk": np.zeros((1,), dtype=np.float32),
            "head.3.weight": np.zeros((2, 1), dtype=np.float32),
            "head.3.bias": np.zeros((2,), dtype=np.float32),
        }
        spec = ModelSpec.from_state(
            family="test",
            model_version=1,
            feature_dim=1,
            num_classes=2,
            node_feature_dim=1,
            hyperparameters={},
            state=state,
        )
        results = [
            LocalTrainResult(
                state={
                    "trunk": np.array([2.0], dtype=np.float32),
                    "head.3.weight": np.array([[10.0], [20.0]], dtype=np.float32),
                    "head.3.bias": np.array([1.0, 2.0], dtype=np.float32),
                },
                num_examples=1,
            ),
            LocalTrainResult(
                state={
                    "trunk": np.array([6.0], dtype=np.float32),
                    "head.3.weight": np.array([[30.0], [40.0]], dtype=np.float32),
                    "head.3.bias": np.array([3.0, 4.0], dtype=np.float32),
                },
                num_examples=3,
            ),
        ]
        aggregated = class_support_head_fedavg(
            results,
            class_support=np.array([[10, 0], [0, 10]]),
            model_spec=spec,
        )
        np.testing.assert_allclose(aggregated["trunk"], [5.0])
        np.testing.assert_allclose(aggregated["head.3.weight"], [[10.0], [40.0]])
        np.testing.assert_allclose(aggregated["head.3.bias"], [1.0, 4.0])

    def test_class_support_head_requires_every_class(self) -> None:
        state = {
            "head.3.weight": np.zeros((2, 1), dtype=np.float32),
            "head.3.bias": np.zeros((2,), dtype=np.float32),
        }
        spec = ModelSpec.from_state(
            family="test",
            model_version=1,
            feature_dim=1,
            num_classes=2,
            node_feature_dim=1,
            hyperparameters={},
            state=state,
        )
        result = LocalTrainResult(state=state, num_examples=1)
        with self.assertRaisesRegex(ContractError, "positive global support"):
            class_support_head_fedavg(
                [result],
                class_support=np.array([[1, 0]]),
                model_spec=spec,
            )

    def test_weighted_fedavg_uses_num_examples(self) -> None:
        results = [
            LocalTrainResult(
                state={"weight": np.array([1.0, 3.0], dtype=np.float32)},
                num_examples=1,
            ),
            LocalTrainResult(
                state={"weight": np.array([5.0, 7.0], dtype=np.float32)},
                num_examples=3,
            ),
        ]
        state = weighted_fedavg(results)
        np.testing.assert_allclose(state["weight"], np.array([4.0, 6.0]))

    def test_rejects_shape_mismatch(self) -> None:
        results = [
            LocalTrainResult(
                state={"weight": np.zeros((2,), dtype=np.float32)},
                num_examples=1,
            ),
            LocalTrainResult(
                state={"weight": np.zeros((3,), dtype=np.float32)},
                num_examples=1,
            ),
        ]
        with self.assertRaisesRegex(ContractError, "shape"):
            weighted_fedavg(results)

    def test_non_floating_state_must_be_identical(self) -> None:
        results = [
            LocalTrainResult(
                state={"counter": np.array(1, dtype=np.int64)},
                num_examples=1,
            ),
            LocalTrainResult(
                state={"counter": np.array(2, dtype=np.int64)},
                num_examples=1,
            ),
        ]
        with self.assertRaisesRegex(ContractError, "Non-floating"):
            weighted_fedavg(results)

    def test_model_contract_is_checked_before_aggregation(self) -> None:
        expected = {"weight": np.zeros((2,), dtype=np.float32)}
        spec = ModelSpec.from_state(
            family="test",
            model_version=1,
            feature_dim=1,
            num_classes=2,
            node_feature_dim=1,
            hyperparameters={},
            state=expected,
        )
        result = LocalTrainResult(
            state={"weight": np.zeros((2,), dtype=np.float64)},
            num_examples=1,
        )
        with self.assertRaisesRegex(ContractError, "dtype"):
            weighted_fedavg([result], model_spec=spec)


if __name__ == "__main__":
    unittest.main()
