"""Validation-best state tracking independent of Flower transport classes."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from src.federated.contracts.schema import ModelSpec
from src.federated.contracts.task import ArrayState
from src.federated.observability.events import Observer
from src.federated.observability.run_store import RunStore, atomic_json


class FlowerBestTracker:
    """Pair aggregated round states with validation metrics and checkpoints."""

    def __init__(
        self,
        *,
        store: RunStore,
        model_spec: ModelSpec,
        observer: Observer,
        flower_run_id: str,
    ) -> None:
        self.store = store
        self.model_spec = model_spec
        self.observer = observer
        self.flower_run_id = flower_run_id
        self._pending_states: dict[int, ArrayState] = {}
        self._best_state: ArrayState | None = None
        self.best_round = 0
        self.best_macro_f1 = -1.0

    def record_train_state(
        self, server_round: int, state: Mapping[str, np.ndarray]
    ) -> None:
        self.model_spec.validate_state(state)
        self._pending_states[server_round] = {
            name: np.asarray(value).copy() for name, value in state.items()
        }

    def record_validation(
        self, server_round: int, metrics: Mapping[str, Any]
    ) -> None:
        try:
            state = self._pending_states.pop(server_round)
        except KeyError as exc:
            raise RuntimeError(
                f"Validation round {server_round} has no aggregated train state."
            ) from exc
        if "macro-f1" not in metrics:
            raise RuntimeError("Flower validation metrics are missing 'macro-f1'.")
        macro_f1 = float(metrics["macro-f1"])
        is_best = macro_f1 > self.best_macro_f1
        self.store.checkpoint(
            state,
            round_number=server_round,
            best=is_best,
        )
        atomic_json(
            self.store.root / "metrics" / f"validation-round-{server_round:04d}.json",
            {"round": server_round, **dict(metrics)},
        )
        if is_best:
            self.best_round = server_round
            self.best_macro_f1 = macro_f1
            self._best_state = {
                name: value.copy() for name, value in state.items()
            }
        self.observer.emit(
            "flower.validation_completed",
            component="flower",
            run_id=self.flower_run_id,
            round=server_round,
            macro_f1=macro_f1,
            best=is_best,
        )

    def best_state(self) -> ArrayState:
        if self._best_state is None:
            raise RuntimeError("Flower run produced no validation-best model.")
        return {name: value.copy() for name, value in self._best_state.items()}
