"""Generic Flower ClientApp backed only by the public task contract."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

from src.federated.contracts.task import FederatedTask, LocalTrainConfig
from src.federated.core.simulation import (
    merge_personalized_state,
    split_personalized_state,
)
from src.federated.core.state import (
    arrays_to_torch_state,
    torch_state_to_arrays,
    validate_array_state_like,
)
from src.federated.flower.config import resolve_run_config
from src.federated.flower.personalized_state import PersonalizedStateStore
from src.federated.observability.events import NoopObserver, Observer


TaskFactory = Callable[[Any], FederatedTask]
ObserverFactory = Callable[[Any, str], Observer]


def _client_id(task: FederatedTask, context: Any) -> str:
    node_config: Mapping[str, Any] = context.node_config
    if "client-id" in node_config:
        client_id = str(node_config["client-id"])
        if client_id not in task.client_ids:
            raise KeyError(
                f"Configured client-id '{client_id}' is not in {task.client_ids}."
            )
        return client_id
    if "partition-id" not in node_config:
        raise KeyError("Flower node_config requires 'partition-id' or 'client-id'.")
    partition_id = int(node_config["partition-id"])
    if partition_id < 0 or partition_id >= len(task.client_ids):
        raise IndexError(
            f"partition-id={partition_id} is outside task clients "
            f"[0, {len(task.client_ids)})."
        )
    return task.client_ids[partition_id]


def _train_config(msg: Any, context: Any) -> LocalTrainConfig:
    run = resolve_run_config(context.run_config)
    message_config = msg.content["config"]
    return LocalTrainConfig(
        local_epochs=int(run["local-epochs"]),
        learning_rate=float(message_config["lr"]),
        weight_decay=float(run.get("weight-decay", 0.0)),
        grad_clip=float(run.get("grad-clip", 1.0)),
        optimizer=str(run.get("optimizer", "sgd")),
        proximal_mu=float(
            message_config.get("proximal-mu", run.get("proximal-mu", 0.0))
        ),
        seed=int(run.get("seed", 42)),
    )


def _personalized_prefixes(run: Mapping[str, Any]) -> tuple[str, ...]:
    prefixes = tuple(
        value.strip()
        for value in str(run.get("personalized-prefixes", "head.")).split(",")
        if value.strip()
    )
    if not prefixes:
        raise ValueError("personalized-prefixes must contain at least one prefix.")
    return prefixes


def _personalized_store(
    task: FederatedTask,
    client_id: str,
    context: Any,
    run: Mapping[str, Any],
) -> tuple[PersonalizedStateStore, dict[str, np.ndarray], dict[str, np.ndarray]]:
    prefixes = _personalized_prefixes(run)
    shared, private = split_personalized_state(
        task.initial_state(), personalized_prefixes=prefixes
    )
    store = PersonalizedStateStore(
        str(run["personalized-state-root"]),
        client_id=client_id,
        run_id=str(context.run_id),
        model_digest=task.model_spec.digest,
        personalized_prefixes=prefixes,
        initial_state=private,
    )
    return store, shared, private


def build_client_app(
    task_factory: TaskFactory,
    *,
    log_message_sizes: bool = True,
    observer_factory: ObserverFactory | None = None,
) -> Any:
    """Build a current Flower Message API ClientApp for a task plugin."""
    try:
        from flwr.app import ArrayRecord, Message, MetricRecord, RecordDict
        from flwr.clientapp import ClientApp
        from flwr.clientapp.mod import arrays_size_mod, message_size_mod
    except ImportError as exc:  # pragma: no cover - dependency-specific
        raise RuntimeError(
            "Flower is not installed. Install requirements-phase2.txt."
        ) from exc

    mods = [message_size_mod, arrays_size_mod] if log_message_sizes else None
    app = ClientApp(mods=mods)

    @app.train()
    def train(msg: Message, context: Any) -> Message:
        observer = (
            observer_factory(context, "client") if observer_factory else NoopObserver()
        )
        task = task_factory(context)
        client_id = _client_id(task, context)
        run = resolve_run_config(context.run_config)
        state = torch_state_to_arrays(msg.content["arrays"].to_torch_state_dict())
        personalized = str(run["strategy"]) == "fedper"
        store = None
        private_metadata: Mapping[str, Any] = {}
        if personalized:
            store, shared_template, _ = _personalized_store(
                task, client_id, context, run
            )
            validate_array_state_like(
                state, shared_template, label="shared encoder state"
            )
            private_state, private_metadata = store.load(require_ready=False)
            train_state = merge_personalized_state(state, private_state)
            task.model_spec.validate_state(train_state)
        else:
            task.model_spec.validate_state(state)
            train_state = state
        observer.emit(
            "flower.client_train_received",
            component="flower",
            run_id=str(context.run_id),
            client_id=client_id,
            download_bytes=sum(value.nbytes for value in state.values()),
            personalization="fedper_head" if personalized else "none",
            cold_start=bool(private_metadata.get("cold_start", False)),
        )
        result = task.train_local(
            client_id, train_state, _train_config(msg, context)
        )
        task.model_spec.validate_state(result.state)
        if personalized:
            assert store is not None
            shared_result, private_result = split_personalized_state(
                result.state,
                personalized_prefixes=_personalized_prefixes(run),
            )
            private_metadata = store.save(private_result)
            upload_state = shared_result
        else:
            upload_state = result.state
        arrays = ArrayRecord(torch_state_dict=arrays_to_torch_state(upload_state))
        metrics: dict[str, int | float] = {
            "num-examples": int(result.num_examples),
            **{key: float(value) for key, value in result.metrics.items()},
        }
        if personalized:
            metrics.update(
                {
                    "personalized-ready": 1,
                    "personalized-rounds": int(
                        private_metadata["completed_rounds"]
                    ),
                }
            )
        content = RecordDict({"arrays": arrays, "metrics": MetricRecord(metrics)})
        observer.emit(
            "flower.client_train_completed",
            component="flower",
            run_id=str(context.run_id),
            client_id=client_id,
            examples=result.num_examples,
            upload_bytes=sum(value.nbytes for value in upload_state.values()),
            train_loss=result.metrics.get("train_loss"),
            personalization="fedper_head" if personalized else "none",
            personalized_rounds=int(
                private_metadata.get("completed_rounds", 0)
            ),
        )
        return Message(content=content, reply_to=msg)

    @app.evaluate()
    def evaluate(msg: Message, context: Any) -> Message:
        observer = (
            observer_factory(context, "client") if observer_factory else NoopObserver()
        )
        task = task_factory(context)
        client_id = _client_id(task, context)
        run = resolve_run_config(context.run_config)
        state = torch_state_to_arrays(msg.content["arrays"].to_torch_state_dict())
        personalized = str(run["strategy"]) == "fedper"
        if personalized:
            store, shared_template, _ = _personalized_store(
                task, client_id, context, run
            )
            validate_array_state_like(
                state, shared_template, label="shared encoder state"
            )
            private_state, private_metadata = store.load(require_ready=True)
            evaluate_state = merge_personalized_state(state, private_state)
            task.model_spec.validate_state(evaluate_state)
        else:
            task.model_spec.validate_state(state)
            evaluate_state = state
            private_metadata = {}
        message_config = msg.content["config"] if "config" in msg.content else {}
        split = str(message_config.get("split", run.get("evaluate-split", "val")))
        if split == "validation":
            split = "val"
        result = task.evaluate_local(client_id, evaluate_state, split=split)
        observer.emit(
            "flower.client_evaluate_completed",
            component="flower",
            run_id=str(context.run_id),
            client_id=client_id,
            split=split,
            examples=result.num_examples,
            loss=result.loss,
            personalization="fedper_head" if personalized else "none",
            personalized_rounds=int(
                private_metadata.get("completed_rounds", 0)
            ),
        )
        matrix = np.asarray(result.confusion_matrix, dtype=np.int64)
        metrics: dict[str, int | float | list[int]] = {
            "num-examples": int(result.num_examples),
            "loss": float(result.loss),
            "num-classes": int(task.label_schema.num_classes),
            "confusion-matrix": matrix.reshape(-1).tolist(),
        }
        metrics.update({key: float(value) for key, value in result.metrics.items()})
        return Message(
            content=RecordDict({"metrics": MetricRecord(metrics)}),
            reply_to=msg,
        )

    return app
