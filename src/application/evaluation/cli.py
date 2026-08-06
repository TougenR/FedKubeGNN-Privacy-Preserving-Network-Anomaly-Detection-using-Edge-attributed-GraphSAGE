"""Command line entrypoint for Phase 4 scientific evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.application.evaluation.fedper import (
    evaluate_correctly_routed,
    evaluate_cross_head,
    evaluate_oracle_once,
    select_oracle_mapping,
    write_report,
)
from src.application.inference.bundle_loader import load_inference_bundle


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("correctly-routed", "cross-head"):
        child = subparsers.add_parser(command)
        child.add_argument("--split", choices=("validation", "test"), required=True)
    select = subparsers.add_parser("select-oracle")
    select.add_argument("--validation-report", type=Path, required=True)
    oracle = subparsers.add_parser("evaluate-oracle")
    oracle.add_argument("--mapping", type=Path, required=True)
    args = parser.parse_args()

    bundle = load_inference_bundle(args.bundle, device=args.device)
    if args.command == "correctly-routed":
        document = evaluate_correctly_routed(
            bundle, args.dataset, split=args.split
        )
    elif args.command == "cross-head":
        document = evaluate_cross_head(bundle, args.dataset, split=args.split)
    elif args.command == "select-oracle":
        document = {
            "selection_split": "validation",
            "class_head_mapping": select_oracle_mapping(
                _read_json(args.validation_report)
            ),
        }
    else:
        mapping_document = _read_json(args.mapping)
        document = evaluate_oracle_once(
            bundle,
            args.dataset,
            class_head_mapping=mapping_document["class_head_mapping"],
        )
    write_report(args.output, document)
    print(args.output)


if __name__ == "__main__":
    main()
