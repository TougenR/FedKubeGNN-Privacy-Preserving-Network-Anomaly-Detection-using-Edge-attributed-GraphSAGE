"""Render the machine-readable Phase 4 evaluation into one reviewable report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _number(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.6f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    output.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return output


def _confusion(title: str, report: Mapping[str, Any], classes: Sequence[str]) -> list[str]:
    rows = [
        [truth, *matrix_row]
        for truth, matrix_row in zip(classes, report["confusion_matrix"])
    ]
    return [f"### {title}", "", *_table(["truth / predicted", *classes], rows)]


def render_evaluation_report(
    directory: str | Path, *, bundle_manifest: str | Path | None = None
) -> str:
    root = Path(directory)
    routed_validation = _read(root / "correctly-routed-validation.json")
    routed_test = _read(root / "correctly-routed-test.json")
    cross_validation = _read(root / "cross-head-validation.json")
    cross_test = _read(root / "cross-head-test.json")
    oracle_mapping = _read(root / "oracle-mapping-validation.json")
    oracle_test = _read(root / "oracle-test.json")
    label_mapping = routed_test.get("label_mapping")
    if label_mapping is None and bundle_manifest is not None:
        label_mapping = _read(Path(bundle_manifest))["label_mapping"]
    if not isinstance(label_mapping, dict):
        raise ValueError(
            "Evaluation has no label_mapping; provide the source bundle manifest."
        )
    classes = tuple(
        name for name, _ in sorted(label_mapping.items(), key=lambda item: item[1])
    )

    lines = [
        "# Phase 4 exact FedPer scientific evaluation",
        "",
        f"Bundle: `{routed_test['bundle_id']}`  ",
        f"Model digest: `{routed_test['model_digest']}`",
        "",
        "## Correctly routed summary",
        "",
        *_table(
            ["split", "samples", "accuracy", "fixed-7 macro-F1", "weighted-F1"],
            [
                [
                    "validation",
                    routed_validation["metrics"]["num_examples"],
                    _number(routed_validation["metrics"]["accuracy"]),
                    _number(routed_validation["metrics"]["macro_f1_fixed"]),
                    _number(routed_validation["metrics"]["weighted_f1"]),
                ],
                [
                    "test",
                    routed_test["metrics"]["num_examples"],
                    _number(routed_test["metrics"]["accuracy"]),
                    _number(routed_test["metrics"]["macro_f1_fixed"]),
                    _number(routed_test["metrics"]["weighted_f1"]),
                ],
            ],
        ),
        "",
        "Validation-to-test gaps are descriptive evidence; they are not by themselves a proof of overfitting:",
        "",
        f"- Accuracy gap: `{routed_validation['metrics']['accuracy'] - routed_test['metrics']['accuracy']:.6f}`",
        f"- Fixed-7 macro-F1 gap: `{routed_validation['metrics']['macro_f1_fixed'] - routed_test['metrics']['macro_f1_fixed']:.6f}`",
        f"- Weighted-F1 gap: `{routed_validation['metrics']['weighted_f1'] - routed_test['metrics']['weighted_f1']:.6f}`",
        "",
        "## Correctly routed test per-class metrics",
        "",
        *_table(
            ["class", "precision", "recall", "F1", "support"],
            [
                [
                    name,
                    _number(routed_test["metrics"]["per_class"][name]["precision"]),
                    _number(routed_test["metrics"]["per_class"][name]["recall"]),
                    _number(routed_test["metrics"]["per_class"][name]["f1"]),
                    routed_test["metrics"]["per_class"][name]["support"],
                ]
                for name in classes
            ],
        ),
        "",
        "## Local owner-head 6 x 7 test matrix",
        "",
        "`N/A` means zero support for that source client; it is not interpreted as F1=0.",
        "",
        *_table(
            ["head / client", *classes],
            [
                [
                    client,
                    *[
                        _number(
                            routed_test["per_client"][client]["metrics"]["per_class"][
                                name
                            ]["f1"]
                        )
                        for name in classes
                    ],
                ]
                for client in routed_test["per_client"]
            ],
        ),
        "",
        "## Cross-head aggregate 6 x 7 test matrix",
        "",
        *_table(
            ["head", *classes],
            [
                [head, *[_number(values[name]) for name in classes]]
                for head, values in cross_test["head_by_class_f1"].items()
            ],
        ),
        "",
        "## Validation-selected oracle/class-aware upper bound",
        "",
        "The mapping below was selected from validation cross-head results and locked before test. It uses the true class for routing, so it is not a production policy.",
        "",
        *_table(
            ["class", "selected head", "validation F1"],
            [
                [
                    name,
                    oracle_mapping["class_head_mapping"][name],
                    _number(
                        cross_validation["head_by_class_f1"][
                            oracle_mapping["class_head_mapping"][name]
                        ][name]
                    ),
                ]
                for name in classes
            ],
        ),
        "",
        f"Oracle test accuracy: `{oracle_test['metrics']['accuracy']:.6f}`  ",
        f"Oracle test fixed-7 macro-F1: `{oracle_test['metrics']['macro_f1_fixed']:.6f}`  ",
        f"Correct routing test fixed-7 macro-F1: `{routed_test['metrics']['macro_f1_fixed']:.6f}`",
        "",
        *_confusion("Correctly routed validation confusion matrix", routed_validation, classes),
        "",
        *_confusion("Correctly routed test confusion matrix", routed_test, classes),
        "",
        "## Interpretation boundary",
        "",
        "This report evaluates known seven-class IoT-23 behavior. It does not establish zero-day detection, production readiness, live-window equivalence, or an automatic blocking policy.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--bundle-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_evaluation_report(
            args.evaluation_dir, bundle_manifest=args.bundle_manifest
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
