#!/usr/bin/env python3
"""Run one prepared-data FL ablation without evaluating the test split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.federated.config import load_phase2_config
from src.federated.contracts.task import LocalTrainConfig
from src.federated.core.simulation import run_federated_simulation
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
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    config = load_phase2_config(args.config)
    task = manifest_task(config, args.dataset, device=args.device)
    rounds = args.rounds or config.training.rounds
    proximal_mu = config.federation.proximal_mu if args.strategy == "fedprox" else 0.0
    result = run_federated_simulation(
        task,
        num_rounds=rounds,
        train_config=LocalTrainConfig(
            local_epochs=config.training.local_epochs,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            grad_clip=config.training.grad_clip,
            optimizer=config.training.optimizer,
            proximal_mu=proximal_mu,
            seed=config.training.seed,
        ),
        evaluate_split="val",
    )
    records = [
        {
            "round": item.round_number,
            "train_examples": item.train_examples,
            "validation_examples": item.evaluation_examples,
            "train_metrics": item.train_metrics,
            "validation_metrics": item.global_metrics,
            "confusion_matrix": item.confusion_matrix.tolist(),
            "upload_bytes": item.upload_bytes,
            "download_bytes": item.download_bytes,
        }
        for item in result.rounds
    ]
    best = max(records, key=lambda item: item["validation_metrics"]["macro_f1"])
    np.savez(output / "final_state.npz", **result.final_state)
    _write_json(output / "rounds.json", records)
    metadata = dict(task.metadata())
    summary = {
        "kind": "validation_only_ablation",
        "strategy": args.strategy,
        "rounds": rounds,
        "best_round": best["round"],
        "best_validation_metrics": best["validation_metrics"],
        "test_evaluations": 0,
        "config_digest": config.digest,
        "dataset_digest": metadata["dataset_digest"],
        "model_digest": task.model_spec.digest,
        "class_weight_scope": metadata["class_weight_scope"],
        "global_class_weights": metadata["global_class_weights"],
        "config": config.to_dict(),
    }
    _write_json(output / "summary.json", summary)
    files = ("final_state.npz", "rounds.json", "summary.json")
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
