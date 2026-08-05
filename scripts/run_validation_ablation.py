#!/usr/bin/env python3
"""Run one prepared-data FL ablation without evaluating the test split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import quote

import numpy as np

from src.federated.config import load_phase2_config
from src.federated.contracts.task import LocalTrainConfig
from src.federated.core.aggregation import (
    class_balanced_client_fedavg,
    class_balanced_client_head_fedavg,
    class_balanced_client_weights,
    class_support_head_fedavg,
)
from src.federated.core.simulation import (
    run_federated_simulation,
    run_fedper_simulation,
)
from src.federated.data.manifest import PreparedDatasetManifest
from src.federated.data.storage import load_graph_arrays
from src.federated.experiments.factory import manifest_task


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--strategy", choices=("fedavg", "fedprox"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--aggregation",
        choices=(
            "sample_fedavg",
            "class_support_head",
            "class_balanced_clients",
            "class_balanced_clients_and_head",
        ),
        default="sample_fedavg",
    )
    parser.add_argument("--local-state-diagnostics", action="store_true")
    parser.add_argument(
        "--personalization",
        choices=("none", "fedper_head"),
        default="none",
    )
    parser.add_argument("--clients", nargs="+")
    args = parser.parse_args()

    if args.personalization != "none" and args.aggregation != "sample_fedavg":
        parser.error(
            "FedPer initially requires sample_fedavg so personalization is "
            "the only changed treatment."
        )
    if args.personalization != "none" and args.local_state_diagnostics:
        parser.error("FedPer does not use --local-state-diagnostics.")

    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    config = load_phase2_config(args.config)
    task = manifest_task(config, args.dataset, device=args.device)
    participants = tuple(args.clients or task.client_ids)
    unknown_clients = sorted(set(participants) - set(task.client_ids))
    if unknown_clients:
        raise ValueError(f"Unknown clients: {unknown_clients}")
    manifest = PreparedDatasetManifest.load(args.dataset, verify=True)
    train_class_support = {}
    for client_id in manifest.client_ids:
        graph = load_graph_arrays(manifest.client_path(client_id), verify=True)
        train_class_support[client_id] = np.bincount(
            graph.edge_label[graph.train_mask],
            minlength=task.label_schema.num_classes,
        ).astype(int).tolist()
    support_matrix = np.asarray(
        [train_class_support[client_id] for client_id in participants],
        dtype=np.float64,
    )
    if args.aggregation == "sample_fedavg":
        aggregate_fn = None
        aggregation_weights = support_matrix.sum(axis=1)
        aggregation_weights /= aggregation_weights.sum()
    elif args.aggregation == "class_support_head":
        def aggregate_fn(results):
            return class_support_head_fedavg(
                results,
                class_support=support_matrix,
                model_spec=task.model_spec,
            )

        aggregation_weights = support_matrix.sum(axis=1)
        aggregation_weights /= aggregation_weights.sum()
    elif args.aggregation == "class_balanced_clients":
        def aggregate_fn(results):
            return class_balanced_client_fedavg(
                results,
                class_support=support_matrix,
                model_spec=task.model_spec,
            )

        aggregation_weights = class_balanced_client_weights(support_matrix)
    else:
        def aggregate_fn(results):
            return class_balanced_client_head_fedavg(
                results,
                class_support=support_matrix,
                model_spec=task.model_spec,
            )

        aggregation_weights = class_balanced_client_weights(support_matrix)
    rounds = args.rounds or config.training.rounds
    proximal_mu = config.federation.proximal_mu if args.strategy == "fedprox" else 0.0
    train_config = LocalTrainConfig(
        local_epochs=config.training.local_epochs,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        grad_clip=config.training.grad_clip,
        optimizer=config.training.optimizer,
        proximal_mu=proximal_mu,
        seed=config.training.seed,
    )
    if args.personalization == "fedper_head":
        result = run_fedper_simulation(
            task,
            num_rounds=rounds,
            train_config=train_config,
            personalized_prefixes=("head.",),
            evaluate_split="val",
            client_ids=participants,
        )
    else:
        result = run_federated_simulation(
            task,
            num_rounds=rounds,
            train_config=train_config,
            evaluate_split="val",
            aggregate_fn=aggregate_fn,
            diagnose_local_states=args.local_state_diagnostics,
            client_ids=participants,
        )
    records = []
    for item in result.rounds:
        client_diagnostics = {}
        for client_id, values in item.client_diagnostics.items():
            row_updates = list(values.get("output_row_update_l2", []))
            sample_weight = float(values["sample_aggregation_weight"])
            client_diagnostics[client_id] = {
                **values,
                "effective_aggregation_weight": float(
                    aggregation_weights[list(participants).index(client_id)]
                ),
                "train_class_support": train_class_support[client_id],
                "sample_weighted_output_row_contribution_l2": [
                    sample_weight * float(value) for value in row_updates
                ],
            }
        records.append({
            "round": item.round_number,
            "train_examples": item.train_examples,
            "validation_examples": item.evaluation_examples,
            "train_metrics": item.train_metrics,
            "validation_metrics": item.global_metrics,
            "confusion_matrix": item.confusion_matrix.tolist(),
            "upload_bytes": item.upload_bytes,
            "download_bytes": item.download_bytes,
            "client_diagnostics": client_diagnostics,
        })
    best = max(records, key=lambda item: item["validation_metrics"]["macro_f1"])
    if result.best_round != best["round"]:
        raise RuntimeError("Runner best-state round differs from recorded validation.")
    if args.personalization == "fedper_head":
        np.savez(output / "final_shared_state.npz", **result.final_shared_state)
        np.savez(output / "best_shared_state.npz", **result.best_shared_state)
        for checkpoint_name, states in (
            ("final_personalized_heads", result.final_personalized_states),
            ("best_personalized_heads", result.best_personalized_states),
        ):
            checkpoint_dir = output / checkpoint_name
            checkpoint_dir.mkdir()
            for client_id, state in states.items():
                filename = f"{quote(client_id, safe='')}.npz"
                np.savez(checkpoint_dir / filename, **state)
        selection_checkpoint = {
            "shared": "best_shared_state.npz",
            "personalized_heads": "best_personalized_heads/",
        }
    else:
        np.savez(output / "final_state.npz", **result.final_state)
        np.savez(output / "best_state.npz", **result.best_state)
        selection_checkpoint = "best_state.npz"
    _write_json(output / "rounds.json", records)
    metadata = dict(task.metadata())
    summary = {
        "kind": "validation_only_ablation",
        "strategy": args.strategy,
        "aggregation": args.aggregation,
        "personalization": args.personalization,
        "personalized_parameter_prefixes": ["head."]
        if args.personalization == "fedper_head"
        else [],
        "local_state_diagnostics": args.local_state_diagnostics,
        "rounds": rounds,
        "best_round": best["round"],
        "selection_checkpoint": selection_checkpoint,
        "best_validation_metrics": best["validation_metrics"],
        "test_evaluations": 0,
        "config_digest": config.digest,
        "dataset_digest": metadata["dataset_digest"],
        "model_digest": task.model_spec.digest,
        "class_weight_scope": metadata["class_weight_scope"],
        "global_class_weights": metadata["global_class_weights"],
        "train_class_support": train_class_support,
        "aggregation_weights": {
            client_id: float(aggregation_weights[index])
            for index, client_id in enumerate(participants)
        },
        "participants": list(participants),
        "diagnostics": {
            "fields": [
                "update_l2",
                "relative_update_l2",
                "distance_to_aggregate_l2",
                "cosine_to_aggregate_update",
                "output_row_update_l2",
                "sample_weighted_output_row_contribution_l2",
            ],
            "first_round": records[0]["client_diagnostics"],
            "final_round": records[-1]["client_diagnostics"],
        },
        "config": config.to_dict(),
    }
    summary["experiment_digest"] = hashlib.sha256(
        json.dumps(
            {
                "config_digest": config.digest,
                "strategy": args.strategy,
                "aggregation": args.aggregation,
                "personalization": args.personalization,
                "rounds": rounds,
                "local_state_diagnostics": args.local_state_diagnostics,
                "participants": participants,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    _write_json(output / "summary.json", summary)
    files = sorted(
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != "evidence_manifest.json"
    )
    _write_json(
        output / "evidence_manifest.json",
        {
            "files": {name: _sha256(output / name) for name in files},
            "test_evaluations": 0,
        },
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
