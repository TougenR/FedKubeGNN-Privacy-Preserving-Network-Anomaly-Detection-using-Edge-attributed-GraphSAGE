"""Phase 2 command line entrypoint with structured operational evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from src.federated.config import load_phase2_config
from src.federated.analysis import analyze_prepared_dataset
from src.federated.contracts.task import LocalTrainConfig
from src.federated.data.manifest import PreparedDatasetManifest
from src.federated.data.preparation import doctor, prepare_iot23
from src.federated.data.repartition import derive_seven_class_datasets
from src.federated.experiments.comparison import compare_runs
from src.federated.experiments.factory import task_from_name
from src.federated.experiments.visualization import visualize_runs
from src.federated.observability import (
    CompositeObserver,
    ConsoleObserver,
    JsonlObserver,
    NoopObserver,
)
from src.federated.registry import builtin_registry


DEFAULT_CONFIG = "configs/phase2/iot23-federated.yaml"


def _observer(config, *, repository_root: Path):
    sinks = []
    if config.observability.console:
        sinks.append(ConsoleObserver())
    if config.observability.jsonl:
        sinks.append(
            JsonlObserver(
                repository_root
                / config.observability.output_root
                / "orchestrator.jsonl"
            )
        )
    return CompositeObserver(*sinks) if sinks else NoopObserver()


def _train_config(config) -> LocalTrainConfig:
    return LocalTrainConfig(
        local_epochs=config.training.local_epochs,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        grad_clip=config.training.grad_clip,
        optimizer=config.training.optimizer,
        seed=config.training.seed,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Observable Phase 2 IoT-23 federation")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--repository-root", default=".")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="read-only dependency/data/disk preflight")
    commands.add_parser("prepare", help="prepare immutable six-client artifacts")
    repartition = commands.add_parser(
        "repartition-seven",
        help="derive natural and stratified-IID seven-class datasets",
    )
    repartition.add_argument("--dataset", required=True)
    repartition.add_argument("--output-root", required=True)
    repartition.add_argument("--seed", type=int, default=42)
    validate = commands.add_parser("validate", help="verify manifest and all checksums")
    validate.add_argument("--dataset", required=True)
    analyze_data = commands.add_parser(
        "analyze-data",
        help="write immutable balance, feature, and graph-topology evidence",
    )
    analyze_data.add_argument("--dataset", required=True)
    analyze_data.add_argument("--output", required=True)
    analyze_data.add_argument("--no-figures", action="store_true")
    run = commands.add_parser(
        "run", help="run FedAvg or FedProx using validation per round"
    )
    run.add_argument("--dataset")
    run.add_argument("--task", choices=("toy", "iot23_manifest"))
    run.add_argument("--strategy", choices=("fedavg", "fedprox"), required=True)
    run.add_argument("--rounds", type=int)
    run.add_argument(
        "--resume", help="failed run directory to resume after digest validation"
    )
    centralized = commands.add_parser(
        "centralized", help="rerun the clean centralized reference"
    )
    centralized.add_argument("--dataset", required=True)
    evaluate = commands.add_parser("evaluate", help="evaluate one portable checkpoint")
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--split", choices=("validation", "test"), default="test")
    evaluate.add_argument("--output")
    compare = commands.add_parser(
        "compare", help="write one CSV from completed run summaries"
    )
    compare.add_argument("--runs", nargs="+", required=True)
    compare.add_argument("--output", required=True)
    visualize = commands.add_parser(
        "visualize",
        help="render completed federated run metrics without re-evaluating data",
    )
    visualize.add_argument("--runs", nargs="+", required=True)
    visualize.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository_root = Path(args.repository_root).resolve()
    config = load_phase2_config(repository_root / args.config)
    components = builtin_registry()
    # Resolve every configured extension point before any command mutates data.
    for kind, name in config.components.__dict__.items():
        components.resolve(kind, name)
    observer = _observer(config, repository_root=repository_root)
    if args.command == "doctor":
        result = doctor(config, repository_root=repository_root)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["ready_for_prepare"] else 2
    if args.command == "prepare":
        path = prepare_iot23(config, repository_root=repository_root, observer=observer)
        print(path)
        return 0
    if args.command == "repartition-seven":
        paths = derive_seven_class_datasets(
            args.dataset, args.output_root, seed=args.seed
        )
        print(
            json.dumps(
                {kind: str(path) for kind, path in paths.items()},
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "validate":
        manifest = PreparedDatasetManifest.load(args.dataset, verify=True)
        result = {
            "valid": True,
            "dataset_id": manifest.dataset_id,
            "dataset_digest": manifest.digest,
            "clients": list(manifest.client_ids),
        }
        observer.emit("dataset.validated", component="validation", **result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "analyze-data":
        path = analyze_prepared_dataset(
            args.dataset,
            args.output,
            render_figures=not args.no_figures,
        )
        print(path)
        return 0
    if args.command == "run":
        task_name = args.task or config.components.task
        task = task_from_name(
            task_name,
            config=config,
            dataset_root=args.dataset,
            observer=observer,
            components=components,
        )
        policy_factory = components.resolve("strategy", args.strategy)
        policy = (
            policy_factory()
            if args.strategy == "fedavg"
            else policy_factory(config.federation.proximal_mu)
        )
        runtime = components.resolve("runtime", "inprocess")
        result = runtime(
            task,
            policy=policy,
            num_rounds=args.rounds or config.training.rounds,
            train_config=_train_config(config),
            output_root=repository_root / config.observability.output_root,
            config_digest=config.digest,
            config_snapshot=config.to_dict(),
            observer=observer,
            resume_root=args.resume,
        )
        print(result.run_root)
        return 0
    if args.command == "centralized":
        from src.federated.experiments.centralized import run_centralized_reference

        path = run_centralized_reference(
            config,
            args.dataset,
            output_root=repository_root / config.observability.output_root,
            observer=observer,
        )
        print(path)
        return 0
    if args.command == "evaluate":
        from src.federated.experiments.evaluation import evaluate_checkpoint

        result = evaluate_checkpoint(
            config,
            args.dataset,
            args.checkpoint,
            split=args.split,
            output=args.output,
            observer=observer,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "compare":
        print(compare_runs(args.runs, args.output))
        return 0
    if args.command == "visualize":
        print(visualize_runs(args.runs, args.output))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    sys.exit(main())
