"""Helpers for named array states used across transport boundaries."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Mapping

import numpy as np

from src.federated.contracts.task import ArrayState


def copy_array_state(state: Mapping[str, np.ndarray]) -> ArrayState:
    """Return a deep array copy while preserving mapping order."""
    return {str(name): np.asarray(value).copy() for name, value in state.items()}


def state_nbytes(state: Mapping[str, np.ndarray]) -> int:
    """Raw payload bytes, excluding protocol and key metadata."""
    return sum(int(np.asarray(value).nbytes) for value in state.values())


def validate_array_state_like(
    state: Mapping[str, np.ndarray],
    template: Mapping[str, np.ndarray],
    *,
    label: str = "state",
) -> None:
    """Validate exact named-array schema when a full ModelSpec is unavailable."""
    actual_names = tuple(state)
    expected_names = tuple(template)
    if actual_names != expected_names:
        raise ValueError(
            f"{label} parameter names/order differ: expected {expected_names}, "
            f"got {actual_names}."
        )
    for name, expected_value in template.items():
        actual = np.asarray(state[name])
        expected = np.asarray(expected_value)
        if actual.shape != expected.shape:
            raise ValueError(
                f"{label} parameter '{name}' shape differs: expected "
                f"{expected.shape}, got {actual.shape}."
            )
        if actual.dtype != expected.dtype:
            raise ValueError(
                f"{label} parameter '{name}' dtype differs: expected "
                f"{expected.dtype}, got {actual.dtype}."
            )


def torch_state_to_arrays(state: Mapping[str, Any]) -> ArrayState:
    """Convert a PyTorch-compatible state dict without importing Phase 1."""
    arrays: ArrayState = {}
    for name, value in state.items():
        tensor = value.detach().cpu() if hasattr(value, "detach") else value
        arrays[str(name)] = np.asarray(tensor).copy()
    return arrays


def arrays_to_torch_state(
    state: Mapping[str, np.ndarray],
    *,
    template: Mapping[str, Any] | None = None,
) -> "OrderedDict[str, Any]":
    """Convert arrays to tensors, optionally matching a template's dtype."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("PyTorch is required to convert array state.") from exc

    converted: "OrderedDict[str, Any]" = OrderedDict()
    for name, value in state.items():
        array = np.asarray(value)
        tensor = torch.from_numpy(array.copy())
        if template is not None:
            if name not in template:
                raise KeyError(f"State template is missing '{name}'.")
            tensor = tensor.to(dtype=template[name].dtype)
        converted[str(name)] = tensor
    return converted
