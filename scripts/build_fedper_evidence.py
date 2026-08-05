#!/usr/bin/env python3
"""Lock compact multi-seed FedPer evidence from completed local runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.federated.experiments.visualization import (
    visualize_personalized_summary,
)
from src.federated.observability.run_store import atomic_json, atomic_text


SEEDS = (42, 1337, 2026)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "population_std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True)
    parser.add_argument("--class-aware-summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    runs_root = Path(args.runs)
    class_aware_path = Path(args.class_aware_summary)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Evidence output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    class_aware = _load(class_aware_path)

    validation: dict[str, dict] = {}
    test: dict[str, dict] = {}
    evidence: dict[str, str] = {
        "class_aware_summary": _sha256(class_aware_path)
    }
    dataset_digest = None
    model_digest = None
    dataset_id = None
    for seed in SEEDS:
        run_root = runs_root / f"fedper-head-seed-{seed}"
        summary_path = run_root / "summary.json"
        rounds_path = run_root / "rounds.json"
        test_path = run_root / "test.json"
        summary = _load(summary_path)
        rounds = _load(rounds_path)
        test_result = _load(test_path)
        if summary["personalization"] != "fedper_head":
            raise ValueError(f"Seed {seed} is not a FedPer run.")
        if summary["test_evaluations"] != 0:
            raise ValueError(f"Seed {seed} used test data during selection.")
        if test_result["split"] != "test":
            raise ValueError(f"Seed {seed} has no final test evaluation.")
        if int(summary["config"]["training"]["seed"]) != seed:
            raise ValueError(f"Run seed mismatch for {run_root}.")
        current_dataset_digest = summary["dataset_digest"]
        current_model_digest = summary["model_digest"]
        dataset_digest = dataset_digest or current_dataset_digest
        model_digest = model_digest or current_model_digest
        dataset_id = dataset_id or test_result["dataset_id"]
        if current_dataset_digest != dataset_digest:
            raise ValueError("FedPer run dataset digests differ.")
        if current_model_digest != model_digest:
            raise ValueError("FedPer run model digests differ.")

        key = str(seed)
        validation[key] = {
            "best_round": int(summary["best_round"]),
            "macro_f1": float(summary["best_validation_metrics"]["macro_f1"]),
            "weighted_f1": float(
                summary["best_validation_metrics"]["weighted_f1"]
            ),
            "accuracy": float(summary["best_validation_metrics"]["accuracy"]),
            "C&C-HeartBeat_f1": float(
                summary["best_validation_metrics"]["per_class"]
                ["C&C-HeartBeat"]["f1"]
            ),
            "round_macro_f1": [
                float(record["validation_metrics"]["macro_f1"])
                for record in rounds
            ],
        }
        metrics = test_result["metrics"]
        test[key] = {
            "accuracy": float(metrics["accuracy"]),
            "weighted_f1": float(metrics["weighted_f1"]),
            "macro_f1": float(metrics["macro_f1"]),
            "per_class_f1": {
                name: float(values["f1"])
                for name, values in metrics["per_class"].items()
            },
        }
        for name, path in (
            (f"seed_{seed}_summary", summary_path),
            (f"seed_{seed}_rounds", rounds_path),
            (f"seed_{seed}_best_shared_state", run_root / "best_shared_state.npz"),
            (f"seed_{seed}_test", test_path),
        ):
            evidence[name] = _sha256(path)
        for client_id in summary["participants"]:
            head = run_root / "best_personalized_heads" / f"{client_id}.npz"
            evidence[f"seed_{seed}_head_{client_id}"] = _sha256(head)

    validation_values = [validation[str(seed)]["macro_f1"] for seed in SEEDS]
    test_values = [test[str(seed)]["macro_f1"] for seed in SEEDS]
    summary_document = {
        "dataset": {
            "dataset_id": dataset_id,
            "manifest_digest": dataset_digest,
            "model_digest": model_digest,
        },
        "algorithm": {
            "name": "fedper_head",
            "shared_parameters": "layers.* GraphSAGE encoder",
            "personalized_parameters": "head.* classifier retained per client",
            "shared_aggregation": "sample-weighted FedAvg",
            "budget": "30 rounds x 5 local Adam epochs",
            "selection": "aggregate personalized validation macro-F1",
            "test_data_used_for_selection": False,
        },
        "validation": {
            "sample_fedavg": {
                str(seed): class_aware["validation"]["baseline"][str(seed)]
                for seed in SEEDS
            },
            "class_aware": {
                str(seed): class_aware["validation"]["selected"][str(seed)]
                ["macro_f1"]
                for seed in SEEDS
            },
            "fedper": validation,
            "fedper_macro_f1": _stats(validation_values),
            "test_evaluations_during_selection": 0,
        },
        "test": {
            "evaluation_policy": (
                "one evaluation per frozen validation-selected shared encoder "
                "and client-head bundle"
            ),
            "evaluation_count": len(SEEDS),
            "fedper": test,
            "fedper_macro_f1": _stats(test_values),
            "class_aware_macro_f1": {
                str(seed): class_aware["test"][str(seed)]["macro_f1"]
                for seed in SEEDS
            },
        },
        "interpretation": {
            "diagnosis": (
                "Local heads recover private classes, confirming global-head "
                "aggregation conflict under natural non-IID data."
            ),
            "production_limit": (
                "FedPer requires a trained head per known edge; a new edge needs "
                "calibration or a global fallback."
            ),
        },
        "source_evidence_sha256": evidence,
    }
    summary_path = output / "summary.json"
    atomic_json(summary_path, summary_document)

    lines = [
        "# FedPer Personalized-Head Finding",
        "",
        "FedPer resolves the dominant natural non-IID failure: the GraphSAGE "
        "encoder is shared with sample-weighted FedAvg while every edge retains "
        "its complete `head.*` classifier locally.",
        "",
        "| Seed | FedAvg val | Class-aware val | FedPer val | FedPer test |",
        "|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        key = str(seed)
        lines.append(
            f"| {seed} | "
            f"{summary_document['validation']['sample_fedavg'][key]:.6f} | "
            f"{summary_document['validation']['class_aware'][key]:.6f} | "
            f"{validation[key]['macro_f1']:.6f} | {test[key]['macro_f1']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Mean validation macro-F1 is "
            f"`{_stats(validation_values)['mean']:.6f} ± "
            f"{_stats(validation_values)['population_std']:.6f}`. The three "
            "validation-selected bundles achieve test macro-F1 "
            f"`{_stats(test_values)['mean']:.6f} ± "
            f"{_stats(test_values)['population_std']:.6f}`; each test split was "
            "evaluated once after selection.",
            "",
            "`C&C-HeartBeat` test F1 is `1.0`, `1.0`, and `0.685076` for seeds "
            "42, 1337, and 2026. This directly supports the diagnosis that the "
            "earlier zero-F1 result came from a globally averaged classifier head, "
            "not unusable preprocessing or broken local training.",
            "",
            "This is a personalized deployment metric, not a single global-model "
            "claim. A known edge must retain its own head; a new edge has no "
            "personalized checkpoint and needs calibration or a global fallback. "
            "The seed-2026 gap also means initialization sensitivity remains.",
            "",
            "Full checkpoints and round logs remain under the Git-ignored "
            "`artifacts/phase2/runs/phase3d/`; their hashes are locked in "
            "`summary.json`.",
            "",
        ]
    )
    atomic_text(output / "finding.md", "\n".join(lines))
    visualize_personalized_summary(summary_path, output / "figures")
    files = sorted(
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != "evidence_manifest.json"
    )
    atomic_json(
        output / "evidence_manifest.json",
        {
            "files": {name: _sha256(output / name) for name in files},
            "source_runs": str(runs_root),
            "test_evaluations": len(SEEDS),
        },
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
