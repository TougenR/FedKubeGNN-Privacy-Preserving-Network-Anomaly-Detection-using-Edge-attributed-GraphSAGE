"""Generic Flower ServerApp with benchmark-compliant model selection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import torch

from src.federated.contracts.task import FederatedTask
from src.federated.core.simulation import split_personalized_state
from src.federated.core.state import arrays_to_torch_state, torch_state_to_arrays
from src.federated.flower.config import resolve_run_config
from src.federated.flower.metrics import aggregate_evaluation_records
from src.federated.flower.tracking import FlowerBestTracker
from src.federated.observability.events import NoopObserver, Observer
from src.federated.observability.run_store import RunStore, atomic_json


TaskFactory = Callable[[Any], FederatedTask]
ObserverFactory = Callable[[Any, str], Observer]


class _TrackingStrategyMixin:
    """Capture the state associated with each aggregated validation result."""

    def __init__(self, *args: Any, tracker: FlowerBestTracker, **kwargs: Any) -> None:
        self._tracker = tracker
        self._record_as_validation = True
        super().__init__(*args, **kwargs)

    def aggregate_train(self, server_round: int, replies: Any) -> Any:
        arrays, metrics = super().aggregate_train(server_round, replies)
        if arrays is not None:
            self._tracker.record_train_state(
                server_round,
                torch_state_to_arrays(arrays.to_torch_state_dict()),
            )
        return arrays, metrics

    def aggregate_evaluate(self, server_round: int, replies: Any) -> Any:
        metrics = super().aggregate_evaluate(server_round, replies)
        if metrics is not None and self._record_as_validation:
            self._tracker.record_validation(server_round, dict(metrics))
        return metrics

    def aggregate_final_test(self, server_round: int, replies: Any) -> Any:
        self._record_as_validation = False
        try:
            return self.aggregate_evaluate(server_round, replies)
        finally:
            self._record_as_validation = True


def _resolved_config_digest(run: dict[str, Any]) -> str:
    configured = run.get("benchmark-config-digest")
    if configured:
        return str(configured)
    payload = json.dumps(run, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_server_app(
    task_factory: TaskFactory,
    *,
    observer_factory: ObserverFactory | None = None,
) -> Any:
    """Build a full-participation FedAvg/FedProx ServerApp."""
    try:
        from flwr.app import ArrayRecord, ConfigRecord
        from flwr.serverapp import ServerApp
        from flwr.serverapp.strategy import FedAvg, FedProx
    except ImportError as exc:  # pragma: no cover - dependency-specific
        raise RuntimeError(
            "Flower is not installed. Install requirements-phase2.txt."
        ) from exc

    app = ServerApp()

    class TrackingFedAvg(_TrackingStrategyMixin, FedAvg):
        pass

    class TrackingFedProx(_TrackingStrategyMixin, FedProx):
        pass

    @app.main()
    def main(grid: Any, context: Any) -> None:
        observer = (
            observer_factory(context, "server") if observer_factory else NoopObserver()
        )
        task = task_factory(context)
        run = resolve_run_config(context.run_config)
        initial_state = task.initial_state()
        task.model_spec.validate_state(initial_state)
        strategy_name = str(run["strategy"])
        personalized = strategy_name == "fedper"
        if personalized:
            prefixes = tuple(
                value.strip()
                for value in str(
                    run.get("personalized-prefixes", "head.")
                ).split(",")
                if value.strip()
            )
            initial_transport_state, _ = split_personalized_state(
                initial_state, personalized_prefixes=prefixes
            )
        else:
            prefixes = ()
            initial_transport_state = initial_state
        initial_arrays = ArrayRecord(
            torch_state_dict=arrays_to_torch_state(initial_transport_state)
        )

        metadata = dict(task.metadata())
        dataset_digest = str(
            metadata.get("dataset_digest", metadata.get("dataset_id", task.task_id))
        )
        store = RunStore.create(
            str(run["flower-output-root"]),
            strategy=strategy_name,
            config_digest=_resolved_config_digest(run),
            dataset_digest=dataset_digest,
            model_digest=task.model_spec.digest,
            config_snapshot=run,
        )
        tracker = FlowerBestTracker(
            store=store,
            observer=observer,
            flower_run_id=str(context.run_id),
            model_spec=None if personalized else task.model_spec,
            state_template=initial_transport_state if personalized else None,
        )
        try:
            observer.emit(
                "flower.server_started",
                component="flower",
                run_id=str(context.run_id),
                strategy=strategy_name,
                rounds=int(run["num-server-rounds"]),
                clients=len(task.client_ids),
                model_digest=task.model_spec.digest,
                config_digest=_resolved_config_digest(run),
                dataset_digest=dataset_digest,
                personalization="fedper_head" if personalized else "none",
            )
            common = {
                "fraction_train": 1.0,
                "fraction_evaluate": float(run["fraction-evaluate"]),
                "min_train_nodes": len(task.client_ids),
                "min_evaluate_nodes": len(task.client_ids),
                "min_available_nodes": len(task.client_ids),
                "evaluate_metrics_aggr_fn": aggregate_evaluation_records,
            }
            if strategy_name == "fedavg":
                strategy = TrackingFedAvg(**common, tracker=tracker)
            elif strategy_name == "fedprox":
                strategy = TrackingFedProx(
                    **common,
                    proximal_mu=float(run["proximal-mu"]),
                    tracker=tracker,
                )
            elif strategy_name == "fedper":
                strategy = TrackingFedAvg(**common, tracker=tracker)
            else:
                raise ValueError(
                    "strategy must be 'fedavg', 'fedprox', or 'fedper'."
                )

            result = strategy.start(
                grid=grid,
                initial_arrays=initial_arrays,
                train_config=ConfigRecord({"lr": float(run["learning-rate"])}),
                evaluate_config=ConfigRecord({"split": "val"}),
                num_rounds=int(run["num-server-rounds"]),
            )
            best_state = tracker.best_state()
            best_arrays = ArrayRecord(
                torch_state_dict=arrays_to_torch_state(best_state)
            )
            test_round = int(run["num-server-rounds"]) + 1
            test_messages = strategy.configure_evaluate(
                test_round,
                best_arrays,
                ConfigRecord({"split": str(run.get("final-split", "test"))}),
                grid,
            )
            test_replies = grid.send_and_receive(test_messages)
            test_metrics_record = strategy.aggregate_final_test(
                test_round, test_replies
            )
            if test_metrics_record is None:
                raise RuntimeError("Flower final test aggregation returned no metrics.")
            test_metrics = dict(test_metrics_record)
            summary = {
                "flower_run_id": str(context.run_id),
                "strategy": strategy_name,
                "class_names": list(task.label_schema.classes),
                "best_round": tracker.best_round,
                "validation_macro_f1": tracker.best_macro_f1,
                "test_metrics": test_metrics,
                "result_has_final_arrays": result.arrays is not None,
                "personalization": "fedper_head" if personalized else "none",
                "shared_parameter_names": list(best_state)
                if personalized
                else [],
                "personalized_parameter_prefixes": list(prefixes),
                "client_head_ownership": "edge-local" if personalized else None,
                "cold_start_policy": (
                    "initial head; inference blocked until one local round"
                    if personalized
                    else None
                ),
            }
            atomic_json(store.root / "metrics" / "summary.json", summary)
            store.complete(
                best_round=tracker.best_round,
                validation_macro_f1=tracker.best_macro_f1,
                test_macro_f1=float(test_metrics["macro-f1"]),
            )
            observer.emit(
                "flower.server_completed",
                component="flower",
                run_id=str(context.run_id),
                strategy=strategy_name,
                rounds=int(run["num-server-rounds"]),
                best_round=tracker.best_round,
                test_macro_f1=float(test_metrics["macro-f1"]),
            )
            if bool(run["save-model"]):
                output = Path(str(run["model-output"]))
                output.parent.mkdir(parents=True, exist_ok=True)
                state_key = "shared_state_dict" if personalized else "state_dict"
                torch.save(
                    {
                        state_key: arrays_to_torch_state(best_state),
                        "model_spec": task.model_spec.to_dict(),
                        "task_metadata": metadata,
                        "best_round": tracker.best_round,
                        "validation_macro_f1": tracker.best_macro_f1,
                        "test_metrics": test_metrics,
                        "personalization": "fedper_head"
                        if personalized
                        else "none",
                        "personalized_parameter_prefixes": list(prefixes),
                        "client_head_ownership": "edge-local"
                        if personalized
                        else None,
                    },
                    output,
                )
        except BaseException as error:
            store.fail(error)
            observer.emit(
                "flower.server_failed",
                level="ERROR",
                component="flower",
                run_id=str(context.run_id),
                strategy=strategy_name,
                error_type=type(error).__name__,
                error_message=str(error),
            )
            raise

    return app
