"""Small deterministic in-process runner for proof independent of Flower."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

from src.federated.contracts.task import (
    ArrayState,
    EvaluationResult,
    FederatedTask,
    LocalTrainConfig,
    LocalTrainResult,
)
from src.federated.core.aggregation import weighted_fedavg
from src.federated.core.metrics import (
    aggregate_confusion_matrices,
    classification_metrics,
)
from src.federated.core.state import copy_array_state, state_nbytes


@dataclass
class FederatedRoundResult:
    round_number: int
    participating_clients: tuple[str, ...]
    train_examples: int
    evaluation_examples: int
    train_metrics: dict[str, float]
    global_metrics: dict[str, object]
    confusion_matrix: np.ndarray
    upload_bytes: int
    download_bytes: int
    client_diagnostics: dict[str, dict[str, object]] = field(default_factory=dict)


@dataclass
class FederatedRunResult:
    final_state: ArrayState
    best_state: ArrayState
    best_round: int
    rounds: list[FederatedRoundResult] = field(default_factory=list)


@dataclass
class PersonalizedFederatedRunResult:
    """FedPer result with one shared state and private state per client."""

    final_shared_state: ArrayState
    final_personalized_states: dict[str, ArrayState]
    best_shared_state: ArrayState
    best_personalized_states: dict[str, ArrayState]
    best_round: int
    rounds: list[FederatedRoundResult] = field(default_factory=list)


def split_personalized_state(
    state: Mapping[str, np.ndarray],
    *,
    personalized_prefixes: Sequence[str],
) -> tuple[ArrayState, ArrayState]:
    """Split a model state into server-shared and client-private parameters."""
    prefixes = tuple(prefix for prefix in personalized_prefixes if prefix)
    if not prefixes:
        raise ValueError("At least one non-empty personalized prefix is required.")
    personalized = {
        name: np.asarray(value).copy()
        for name, value in state.items()
        if name.startswith(prefixes)
    }
    shared = {
        name: np.asarray(value).copy()
        for name, value in state.items()
        if name not in personalized
    }
    if not personalized:
        raise ValueError(
            f"No model parameters match personalized prefixes {prefixes}."
        )
    if not shared:
        raise ValueError("FedPer requires at least one shared model parameter.")
    return shared, personalized


def merge_personalized_state(
    shared: Mapping[str, np.ndarray],
    personalized: Mapping[str, np.ndarray],
) -> ArrayState:
    """Build one complete client model without aliasing input arrays."""
    overlap = set(shared) & set(personalized)
    if overlap:
        raise ValueError(f"Shared and personalized states overlap: {sorted(overlap)}")
    return {
        **copy_array_state(shared),
        **copy_array_state(personalized),
    }


def _copy_personalized_states(
    states: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, ArrayState]:
    return {
        client_id: copy_array_state(state)
        for client_id, state in states.items()
    }


def _weighted_scalar_metrics(results: Sequence[LocalTrainResult]) -> dict[str, float]:
    keys = sorted(set.intersection(*(set(result.metrics) for result in results)))
    total = float(sum(result.num_examples for result in results))
    return {
        key: float(
            sum(result.metrics[key] * result.num_examples for result in results)
            / total
        )
        for key in keys
    }


def _evaluate_clients(
    task: FederatedTask,
    client_ids: Sequence[str],
    state: Mapping[str, np.ndarray],
    *,
    split: str,
) -> tuple[list[EvaluationResult], np.ndarray, dict[str, object]]:
    results = [
        task.evaluate_local(client_id, state, split=split)
        for client_id in client_ids
    ]
    matrix = aggregate_confusion_matrices(
        (result.confusion_matrix for result in results),
        num_classes=task.label_schema.num_classes,
    )
    metrics = classification_metrics(
        matrix, class_names=task.label_schema.classes
    )
    total_examples = sum(result.num_examples for result in results)
    metrics["loss"] = (
        float(
            sum(result.loss * result.num_examples for result in results)
            / total_examples
        )
        if total_examples
        else 0.0
    )
    return results, matrix, metrics


def _evaluate_personalized_clients(
    task: FederatedTask,
    client_ids: Sequence[str],
    shared_state: Mapping[str, np.ndarray],
    personalized_states: Mapping[str, Mapping[str, np.ndarray]],
    *,
    split: str,
) -> tuple[list[EvaluationResult], np.ndarray, dict[str, object]]:
    results = [
        task.evaluate_local(
            client_id,
            merge_personalized_state(
                shared_state, personalized_states[client_id]
            ),
            split=split,
        )
        for client_id in client_ids
    ]
    matrix = aggregate_confusion_matrices(
        (result.confusion_matrix for result in results),
        num_classes=task.label_schema.num_classes,
    )
    metrics = classification_metrics(
        matrix, class_names=task.label_schema.classes
    )
    total_examples = sum(result.num_examples for result in results)
    metrics["loss"] = (
        float(
            sum(result.loss * result.num_examples for result in results)
            / total_examples
        )
        if total_examples
        else 0.0
    )
    return results, matrix, metrics


def _state_delta_products(
    state: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
    other: Mapping[str, np.ndarray] | None = None,
) -> tuple[float, float, float]:
    """Return squared delta norm, reference norm, and optional delta dot."""
    delta_squared = 0.0
    reference_squared = 0.0
    dot = 0.0
    for name, reference_value in reference.items():
        reference_array = np.asarray(reference_value)
        if not np.issubdtype(reference_array.dtype, np.inexact):
            continue
        work_dtype = (
            np.complex128
            if np.issubdtype(reference_array.dtype, np.complexfloating)
            else np.float64
        )
        reference_float = np.asarray(reference_array, dtype=work_dtype)
        delta = np.asarray(state[name], dtype=work_dtype) - reference_float
        delta_squared += float(np.vdot(delta, delta).real)
        reference_squared += float(
            np.vdot(reference_float, reference_float).real
        )
        if other is not None:
            other_delta = np.asarray(other[name], dtype=work_dtype) - reference_float
            dot += float(np.vdot(delta, other_delta).real)
    return delta_squared, reference_squared, dot


def _output_row_update_l2(
    state: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
    *,
    num_classes: int,
) -> list[float]:
    squared = np.zeros(num_classes, dtype=np.float64)
    found = False
    for name in ("head.3.weight", "head.3.bias"):
        if name not in state or name not in reference:
            continue
        delta = np.asarray(state[name], dtype=np.float64) - np.asarray(
            reference[name], dtype=np.float64
        )
        if delta.shape[0] != num_classes:
            return []
        squared += np.square(delta).reshape(num_classes, -1).sum(axis=1)
        found = True
    return np.sqrt(squared).tolist() if found else []


def _client_update_diagnostics(
    client_ids: Sequence[str],
    local_results: Sequence[LocalTrainResult],
    before: Mapping[str, np.ndarray],
    after: Mapping[str, np.ndarray],
    *,
    num_classes: int,
) -> dict[str, dict[str, object]]:
    total_examples = float(sum(result.num_examples for result in local_results))
    aggregate_squared, _, _ = _state_delta_products(after, before)
    aggregate_norm = float(np.sqrt(aggregate_squared))
    diagnostics: dict[str, dict[str, object]] = {}
    for client_id, result in zip(client_ids, local_results, strict=True):
        delta_squared, reference_squared, dot = _state_delta_products(
            result.state, before, after
        )
        update_norm = float(np.sqrt(delta_squared))
        distance_squared, _, _ = _state_delta_products(result.state, after)
        denominator = update_norm * aggregate_norm
        diagnostics[client_id] = {
            "num_examples": result.num_examples,
            "sample_aggregation_weight": result.num_examples / total_examples,
            "update_l2": update_norm,
            "relative_update_l2": update_norm
            / max(float(np.sqrt(reference_squared)), np.finfo(float).eps),
            "distance_to_aggregate_l2": float(np.sqrt(distance_squared)),
            "cosine_to_aggregate_update": dot / denominator
            if denominator > 0
            else 0.0,
            "output_row_update_l2": _output_row_update_l2(
                result.state, before, num_classes=num_classes
            ),
        }
    return diagnostics


def run_federated_simulation(
    task: FederatedTask,
    *,
    num_rounds: int,
    train_config: LocalTrainConfig,
    client_ids: Sequence[str] | None = None,
    evaluate_split: str = "test",
    aggregate_fn: Callable[[Sequence[LocalTrainResult]], ArrayState] | None = None,
    diagnose_local_states: bool = False,
) -> FederatedRunResult:
    """Run full-participation FedAvg through the public task contract."""
    if num_rounds < 1:
        raise ValueError("num_rounds must be >= 1.")
    participants = tuple(client_ids or task.client_ids)
    if not participants:
        raise ValueError("At least one client is required.")
    unknown = sorted(set(participants) - set(task.client_ids))
    if unknown:
        raise KeyError(f"Unknown client ids: {unknown}.")

    state = task.initial_state()
    task.model_spec.validate_state(state)
    payload_bytes = state_nbytes(state)
    round_results: list[FederatedRoundResult] = []
    best_state = copy_array_state(state)
    best_round = 0
    best_macro_f1 = -1.0

    for round_number in range(1, num_rounds + 1):
        local_results = [
            task.train_local(client_id, copy_array_state(state), train_config)
            for client_id in participants
        ]
        before = state
        state = (
            aggregate_fn(local_results)
            if aggregate_fn is not None
            else weighted_fedavg(local_results, model_spec=task.model_spec)
        )
        task.model_spec.validate_state(state)
        client_diagnostics = _client_update_diagnostics(
            participants,
            local_results,
            before,
            state,
            num_classes=task.label_schema.num_classes,
        )
        if diagnose_local_states and round_number == num_rounds:
            for client_id, result in zip(
                participants, local_results, strict=True
            ):
                own = task.evaluate_local(
                    client_id, result.state, split=evaluate_split
                )
                own_metrics = classification_metrics(
                    own.confusion_matrix,
                    class_names=task.label_schema.classes,
                )
                own_metrics["loss"] = own.loss
                _, _, global_metrics = _evaluate_clients(
                    task, participants, result.state, split=evaluate_split
                )
                client_diagnostics[client_id]["local_state_own_client_metrics"] = (
                    own_metrics
                )
                client_diagnostics[client_id]["local_state_global_metrics"] = (
                    global_metrics
                )
        evaluations, matrix, metrics = _evaluate_clients(
            task, participants, state, split=evaluate_split
        )
        round_results.append(
            FederatedRoundResult(
                round_number=round_number,
                participating_clients=participants,
                train_examples=sum(
                    result.num_examples for result in local_results
                ),
                evaluation_examples=sum(
                    result.num_examples for result in evaluations
                ),
                train_metrics=_weighted_scalar_metrics(local_results),
                global_metrics=metrics,
                confusion_matrix=matrix,
                upload_bytes=payload_bytes * len(participants),
                download_bytes=payload_bytes * len(participants),
                client_diagnostics=client_diagnostics,
            )
        )
        macro_f1 = float(metrics["macro_f1"])
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_round = round_number
            best_state = copy_array_state(state)
    return FederatedRunResult(
        final_state=copy_array_state(state),
        best_state=best_state,
        best_round=best_round,
        rounds=round_results,
    )


def run_fedper_simulation(
    task: FederatedTask,
    *,
    num_rounds: int,
    train_config: LocalTrainConfig,
    personalized_prefixes: Sequence[str],
    client_ids: Sequence[str] | None = None,
    evaluate_split: str = "test",
) -> PersonalizedFederatedRunResult:
    """Run FedPer: aggregate shared parameters and retain private client heads."""
    if num_rounds < 1:
        raise ValueError("num_rounds must be >= 1.")
    participants = tuple(client_ids or task.client_ids)
    if not participants:
        raise ValueError("At least one client is required.")
    unknown = sorted(set(participants) - set(task.client_ids))
    if unknown:
        raise KeyError(f"Unknown client ids: {unknown}.")

    initial_state = task.initial_state()
    task.model_spec.validate_state(initial_state)
    shared_state, initial_personalized = split_personalized_state(
        initial_state, personalized_prefixes=personalized_prefixes
    )
    personalized_states = {
        client_id: copy_array_state(initial_personalized)
        for client_id in participants
    }
    payload_bytes = state_nbytes(shared_state)
    best_shared_state = copy_array_state(shared_state)
    best_personalized_states = _copy_personalized_states(personalized_states)
    best_round = 0
    best_macro_f1 = -1.0
    round_results: list[FederatedRoundResult] = []

    for round_number in range(1, num_rounds + 1):
        client_inputs = {
            client_id: merge_personalized_state(
                shared_state, personalized_states[client_id]
            )
            for client_id in participants
        }
        local_results = [
            task.train_local(
                client_id,
                copy_array_state(client_inputs[client_id]),
                train_config,
            )
            for client_id in participants
        ]
        local_shared_results: list[LocalTrainResult] = []
        updated_personalized: dict[str, ArrayState] = {}
        for client_id, local_result in zip(
            participants, local_results, strict=True
        ):
            task.model_spec.validate_state(local_result.state)
            local_shared, local_private = split_personalized_state(
                local_result.state,
                personalized_prefixes=personalized_prefixes,
            )
            local_shared_results.append(
                LocalTrainResult(
                    state=local_shared,
                    num_examples=local_result.num_examples,
                    metrics=local_result.metrics,
                )
            )
            updated_personalized[client_id] = local_private

        before_shared = shared_state
        shared_state = weighted_fedavg(local_shared_results)
        if set(shared_state) != set(before_shared):
            raise ValueError("Aggregated shared state changed parameter names.")
        personalized_states = updated_personalized

        client_diagnostics = _client_update_diagnostics(
            participants,
            local_shared_results,
            before_shared,
            shared_state,
            num_classes=task.label_schema.num_classes,
        )
        for client_id in participants:
            private_squared, _, _ = _state_delta_products(
                personalized_states[client_id],
                split_personalized_state(
                    client_inputs[client_id],
                    personalized_prefixes=personalized_prefixes,
                )[1],
            )
            client_diagnostics[client_id]["personalized_update_l2"] = float(
                np.sqrt(private_squared)
            )

        evaluations, matrix, metrics = _evaluate_personalized_clients(
            task,
            participants,
            shared_state,
            personalized_states,
            split=evaluate_split,
        )
        round_results.append(
            FederatedRoundResult(
                round_number=round_number,
                participating_clients=participants,
                train_examples=sum(
                    result.num_examples for result in local_results
                ),
                evaluation_examples=sum(
                    result.num_examples for result in evaluations
                ),
                train_metrics=_weighted_scalar_metrics(local_results),
                global_metrics=metrics,
                confusion_matrix=matrix,
                upload_bytes=payload_bytes * len(participants),
                download_bytes=payload_bytes * len(participants),
                client_diagnostics=client_diagnostics,
            )
        )
        macro_f1 = float(metrics["macro_f1"])
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_round = round_number
            best_shared_state = copy_array_state(shared_state)
            best_personalized_states = _copy_personalized_states(
                personalized_states
            )

    return PersonalizedFederatedRunResult(
        final_shared_state=copy_array_state(shared_state),
        final_personalized_states=_copy_personalized_states(
            personalized_states
        ),
        best_shared_state=best_shared_state,
        best_personalized_states=best_personalized_states,
        best_round=best_round,
        rounds=round_results,
    )
