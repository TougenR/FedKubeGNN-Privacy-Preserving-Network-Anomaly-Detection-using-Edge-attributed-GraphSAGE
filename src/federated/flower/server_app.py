"""Generic Flower ServerApp selecting FedAvg or FedProx by run config."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch

from src.federated.contracts.task import FederatedTask
from src.federated.core.state import arrays_to_torch_state
from src.federated.flower.config import resolve_run_config
from src.federated.flower.metrics import aggregate_evaluation_records
from src.federated.observability.events import NoopObserver, Observer


TaskFactory = Callable[[Any], FederatedTask]
ObserverFactory = Callable[[Any, str], Observer]


def build_server_app(
    task_factory: TaskFactory,
    *,
    observer_factory: ObserverFactory | None = None,
) -> Any:
    """Build a ServerApp with strict full-participation FedAvg/FedProx."""
    try:
        from flwr.app import ArrayRecord, ConfigRecord
        from flwr.serverapp import ServerApp
        from flwr.serverapp.strategy import FedAvg, FedProx
    except ImportError as exc:  # pragma: no cover - dependency-specific
        raise RuntimeError(
            "Flower is not installed. Install requirements-phase2.txt."
        ) from exc

    app = ServerApp()

    @app.main()
    def main(grid: Any, context: Any) -> None:
        observer = (
            observer_factory(context, "server") if observer_factory else NoopObserver()
        )
        task = task_factory(context)
        run = resolve_run_config(context.run_config)
        initial_state = task.initial_state()
        task.model_spec.validate_state(initial_state)
        arrays = ArrayRecord(torch_state_dict=arrays_to_torch_state(initial_state))

        strategy_name = str(run["strategy"])
        observer.emit(
            "flower.server_started",
            component="flower",
            run_id=str(context.run_id),
            strategy=strategy_name,
            rounds=int(run["num-server-rounds"]),
            clients=len(task.client_ids),
            model_digest=task.model_spec.digest,
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
            strategy = FedAvg(**common)
        elif strategy_name == "fedprox":
            strategy = FedProx(
                **common,
                proximal_mu=float(run["proximal-mu"]),
            )
        else:
            raise ValueError("strategy must be 'fedavg' or 'fedprox'.")

        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            train_config=ConfigRecord({"lr": float(run["learning-rate"])}),
            num_rounds=int(run["num-server-rounds"]),
        )
        observer.emit(
            "flower.server_completed",
            component="flower",
            run_id=str(context.run_id),
            strategy=strategy_name,
            rounds=int(run["num-server-rounds"]),
            has_final_arrays=result.arrays is not None,
        )
        if bool(run["save-model"]):
            if result.arrays is None:
                raise RuntimeError("Flower returned no final model arrays.")
            output = Path(str(run["model-output"]))
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": result.arrays.to_torch_state_dict(),
                    "model_spec": task.model_spec.to_dict(),
                    "task_metadata": dict(task.metadata()),
                },
                output,
            )

    return app
