"""Strict named-parameter implementations of federated aggregation."""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from src.federated.contracts.schema import ContractError, ModelSpec
from src.federated.contracts.task import ArrayState, LocalTrainResult


def _validate_state_against_reference(
    state: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
    *,
    client_index: int,
) -> None:
    if tuple(state.keys()) != tuple(reference.keys()):
        raise ContractError(
            f"Client {client_index} state keys/order differ from client 0."
        )
    for name, reference_value in reference.items():
        value = np.asarray(state[name])
        reference_array = np.asarray(reference_value)
        if value.shape != reference_array.shape:
            raise ContractError(
                f"Client {client_index} parameter '{name}' shape "
                f"{value.shape} != {reference_array.shape}."
            )
        if value.dtype != reference_array.dtype:
            raise ContractError(
                f"Client {client_index} parameter '{name}' dtype "
                f"{value.dtype} != {reference_array.dtype}."
            )


def weighted_fedavg(
    results: Iterable[LocalTrainResult],
    *,
    model_spec: ModelSpec | None = None,
) -> ArrayState:
    """Aggregate named states with sample weighting and fail-closed validation.

    Floating and complex values are averaged. Non-floating state entries are
    copied only when every client has exactly the same value; silently averaging
    counters or booleans would not have a well-defined model meaning.
    """
    items = list(results)
    if not items:
        raise ValueError("weighted_fedavg requires at least one client result.")
    if any(item.num_examples <= 0 for item in items):
        raise ValueError("Every client aggregation weight must be positive.")

    reference = items[0].state
    if not reference:
        raise ContractError("Cannot aggregate an empty model state.")
    for index, item in enumerate(items):
        _validate_state_against_reference(
            item.state, reference, client_index=index
        )
        if model_spec is not None:
            model_spec.validate_state(item.state)

    total_weight = float(sum(item.num_examples for item in items))
    aggregated: ArrayState = {}
    for name, reference_value in reference.items():
        reference_array = np.asarray(reference_value)
        if np.issubdtype(reference_array.dtype, np.floating):
            accumulator = np.zeros(reference_array.shape, dtype=np.float64)
            for item in items:
                value = np.asarray(item.state[name], dtype=np.float64)
                if not np.all(np.isfinite(value)):
                    raise ContractError(
                        f"Client state parameter '{name}' contains NaN/Inf."
                    )
                accumulator += value * (item.num_examples / total_weight)
            aggregated[name] = accumulator.astype(reference_array.dtype)
        elif np.issubdtype(reference_array.dtype, np.complexfloating):
            accumulator = np.zeros(reference_array.shape, dtype=np.complex128)
            for item in items:
                accumulator += np.asarray(item.state[name], dtype=np.complex128) * (
                    item.num_examples / total_weight
                )
            aggregated[name] = accumulator.astype(reference_array.dtype)
        else:
            for item in items[1:]:
                if not np.array_equal(reference_array, item.state[name]):
                    raise ContractError(
                        f"Non-floating state entry '{name}' differs across clients."
                    )
            aggregated[name] = reference_array.copy()
    if model_spec is not None:
        model_spec.validate_state(aggregated)
    return aggregated


def class_support_head_fedavg(
    results: Iterable[LocalTrainResult],
    *,
    class_support: np.ndarray,
    model_spec: ModelSpec,
    output_weight_name: str = "head.3.weight",
    output_bias_name: str = "head.3.bias",
) -> ArrayState:
    """FedAvg trunk plus class-support-weighted classifier output rows.

    Clients without training support for a class do not update that class's
    final classifier row. Shared representation parameters retain ordinary
    sample-weighted FedAvg behavior.
    """
    items = list(results)
    aggregated = weighted_fedavg(items, model_spec=model_spec)
    support = np.asarray(class_support)
    expected_shape = (len(items), model_spec.num_classes)
    if support.shape != expected_shape:
        raise ContractError(
            f"class_support shape {support.shape} does not match {expected_shape}."
        )
    if not np.all(np.isfinite(support)) or np.any(support < 0):
        raise ContractError("class_support must be finite and non-negative.")
    if np.any(support.sum(axis=0) <= 0):
        raise ContractError("Every output class requires positive global support.")

    for name in (output_weight_name, output_bias_name):
        if name not in aggregated:
            raise ContractError(f"Model state is missing classifier output '{name}'.")
        reference = np.asarray(aggregated[name])
        if reference.shape[0] != model_spec.num_classes:
            raise ContractError(
                f"Classifier output '{name}' first dimension must equal num_classes."
            )
        if not np.issubdtype(reference.dtype, np.floating):
            raise ContractError(f"Classifier output '{name}' must be floating point.")
        class_aggregated = np.empty(reference.shape, dtype=np.float64)
        for class_index in range(model_spec.num_classes):
            weights = support[:, class_index].astype(np.float64)
            weights /= weights.sum()
            accumulator = np.zeros(reference[class_index].shape, dtype=np.float64)
            for client_index, item in enumerate(items):
                value = np.asarray(item.state[name], dtype=np.float64)
                accumulator += value[class_index] * weights[client_index]
            class_aggregated[class_index] = accumulator
        aggregated[name] = class_aggregated.astype(reference.dtype)

    model_spec.validate_state(aggregated)
    return aggregated


def class_balanced_client_weights(class_support: np.ndarray) -> np.ndarray:
    """Give every globally present class equal total client-weight influence."""
    support = np.asarray(class_support, dtype=np.float64)
    if support.ndim != 2 or support.shape[0] < 1 or support.shape[1] < 1:
        raise ContractError("class_support must have shape [clients, classes].")
    if not np.all(np.isfinite(support)) or np.any(support < 0):
        raise ContractError("class_support must be finite and non-negative.")
    global_support = support.sum(axis=0)
    if np.any(global_support <= 0):
        raise ContractError("Every class requires positive global support.")
    weights = (support / global_support).mean(axis=1)
    if np.any(weights <= 0):
        raise ContractError("Every participating client requires class support.")
    return weights / weights.sum()


def class_balanced_client_fedavg(
    results: Iterable[LocalTrainResult],
    *,
    class_support: np.ndarray,
    model_spec: ModelSpec,
) -> ArrayState:
    """Aggregate the full model using train-only class-balanced client weights."""
    items = list(results)
    # Reuse the strict state/model/non-floating validation of ordinary FedAvg.
    aggregated = weighted_fedavg(items, model_spec=model_spec)
    support = np.asarray(class_support)
    expected_shape = (len(items), model_spec.num_classes)
    if support.shape != expected_shape:
        raise ContractError(
            f"class_support shape {support.shape} does not match {expected_shape}."
        )
    weights = class_balanced_client_weights(support)
    reference = items[0].state
    for name, reference_value in reference.items():
        reference_array = np.asarray(reference_value)
        if np.issubdtype(reference_array.dtype, np.floating):
            accumulator = np.zeros(reference_array.shape, dtype=np.float64)
            for client_index, item in enumerate(items):
                accumulator += (
                    np.asarray(item.state[name], dtype=np.float64)
                    * weights[client_index]
                )
            aggregated[name] = accumulator.astype(reference_array.dtype)
        elif np.issubdtype(reference_array.dtype, np.complexfloating):
            accumulator = np.zeros(reference_array.shape, dtype=np.complex128)
            for client_index, item in enumerate(items):
                accumulator += (
                    np.asarray(item.state[name], dtype=np.complex128)
                    * weights[client_index]
                )
            aggregated[name] = accumulator.astype(reference_array.dtype)
    model_spec.validate_state(aggregated)
    return aggregated


def class_balanced_client_head_fedavg(
    results: Iterable[LocalTrainResult],
    *,
    class_support: np.ndarray,
    model_spec: ModelSpec,
    output_weight_name: str = "head.3.weight",
    output_bias_name: str = "head.3.bias",
) -> ArrayState:
    """Class-balanced full-model aggregation with support-only output rows."""
    items = list(results)
    aggregated = class_balanced_client_fedavg(
        items, class_support=class_support, model_spec=model_spec
    )
    support_head = class_support_head_fedavg(
        items,
        class_support=class_support,
        model_spec=model_spec,
        output_weight_name=output_weight_name,
        output_bias_name=output_bias_name,
    )
    aggregated[output_weight_name] = support_head[output_weight_name]
    aggregated[output_bias_name] = support_head[output_bias_name]
    model_spec.validate_state(aggregated)
    return aggregated
