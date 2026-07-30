#!/usr/bin/env python3
"""Analyze clean Phase 1 bundles across seeds without retraining.

The analyzer never reconstructs unavailable per-sample information from
aggregate metrics. Missing inputs are rendered as ``NOT_AVAILABLE`` with the
exact artifact fields needed to enable the analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.phase1_clean import FIXED_LABELS


NOT_AVAILABLE = "NOT_AVAILABLE"
PROBABILITY_COLUMNS = tuple(
    f"probability::{label}" for label in FIXED_LABELS
)
LOGIT_COLUMNS = tuple(f"logit::{label}" for label in FIXED_LABELS)
PREDICTION_IDENTITY_COLUMNS = (
    "true_label",
    "predicted_label",
    "scenario",
    "split",
    "seed",
)


def fixed_class_macro_f1(
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
) -> float:
    return float(
        f1_score(
            true_labels,
            predicted_labels,
            labels=list(FIXED_LABELS),
            average="macro",
            zero_division=0,
        )
    )


def seen_class_macro_f1(
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
    train_support: Mapping[str, int],
) -> float | None:
    """Macro-F1 over evaluated true classes that had positive train support."""

    present_true = set(str(label) for label in true_labels)
    seen = [
        label
        for label in FIXED_LABELS
        if int(train_support.get(label, 0)) > 0 and label in present_true
    ]
    if not seen:
        return None
    return float(
        f1_score(
            true_labels,
            predicted_labels,
            labels=seen,
            average="macro",
            zero_division=0,
        )
    )


def collapse_binary(labels: Sequence[str]) -> np.ndarray:
    return np.asarray(
        ["Benign" if str(label) == "Benign" else "Malicious" for label in labels],
        dtype=object,
    )


def binary_metrics(
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
) -> dict[str, float]:
    true_binary = collapse_binary(true_labels)
    predicted_binary = collapse_binary(predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_binary,
        predicted_binary,
        labels=["Malicious"],
        average=None,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(true_binary, predicted_binary)),
        "malicious_precision": float(precision[0]),
        "malicious_recall": float(recall[0]),
        "malicious_f1": float(f1[0]),
    }


def probabilities_and_entropy(
    predictions: pd.DataFrame,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    """Return probabilities/entropy from probabilities first, then logits."""

    if set(PROBABILITY_COLUMNS).issubset(predictions.columns):
        probabilities = predictions.loc[:, PROBABILITY_COLUMNS].to_numpy(
            dtype=float
        )
        source = "probabilities"
        if (
            not np.isfinite(probabilities).all()
            or np.any(probabilities < 0)
            or np.any(probabilities > 1)
            or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
        ):
            return None, None, (
                "Invalid probability columns: require finite [0,1] values "
                "whose rows sum to 1."
            )
    elif set(LOGIT_COLUMNS).issubset(predictions.columns):
        logits = predictions.loc[:, LOGIT_COLUMNS].to_numpy(dtype=float)
        source = "logits"
        if not np.isfinite(logits).all():
            return None, None, "Invalid logit columns: values must be finite."
        shifted = logits - logits.max(axis=1, keepdims=True)
        exponentiated = np.exp(shifted)
        probabilities = exponentiated / exponentiated.sum(
            axis=1, keepdims=True
        )
    else:
        return None, None, (
            "predictions.csv requires all probability::<fixed-label> columns "
            "or all logit::<fixed-label> columns."
        )
    entropy = -np.sum(
        probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)),
        axis=1,
    )
    return probabilities, entropy, source


def entropy_analysis(
    predictions: pd.DataFrame,
    train_support: Mapping[str, int],
) -> tuple[pd.DataFrame, float | None, str | None]:
    _, entropy, source_or_error = probabilities_and_entropy(predictions)
    if entropy is None:
        return pd.DataFrame(), None, source_or_error
    frame = predictions.copy()
    frame["_entropy"] = entropy
    absent = frame["true_label"].map(
        lambda label: int(train_support.get(str(label), 0)) == 0
    )
    correct = frame["true_label"].astype(str) == frame[
        "predicted_label"
    ].astype(str)
    groups = {
        "known-correct": (~absent) & correct,
        "known-incorrect": (~absent) & (~correct),
        "class-absent-from-train": absent,
    }
    rows: list[dict[str, Any]] = []
    for group_name, mask in groups.items():
        values = frame.loc[mask, "_entropy"].to_numpy(dtype=float)
        row: dict[str, Any] = {
            "group": group_name,
            "count": int(len(values)),
            "mean": NOT_AVAILABLE,
            "median": NOT_AVAILABLE,
            "q05": NOT_AVAILABLE,
            "q25": NOT_AVAILABLE,
            "q75": NOT_AVAILABLE,
            "q95": NOT_AVAILABLE,
            "entropy_source": source_or_error,
        }
        if len(values):
            row.update(
                {
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "q05": float(np.quantile(values, 0.05)),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                    "q95": float(np.quantile(values, 0.95)),
                }
            )
        rows.append(row)

    auc: float | None = None
    auc_error: str | None = None
    eligible = groups["known-correct"] | groups["class-absent-from-train"]
    targets = absent[eligible].astype(int).to_numpy()
    scores = frame.loc[eligible, "_entropy"].to_numpy(dtype=float)
    if len(np.unique(targets)) == 2:
        auc = float(roc_auc_score(targets, scores))
    else:
        auc_error = (
            "AUROC requires at least one known-correct row and one row whose "
            "true class has train support 0."
        )
    return pd.DataFrame(rows), auc, auc_error


def aggregate_seed_metrics(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    metric_columns: Sequence[str],
) -> pd.DataFrame:
    """Aggregate numeric run metrics with population std (ddof=0)."""

    if frame.empty:
        columns = list(group_columns) + ["seed_count"]
        for metric in metric_columns:
            columns.extend(
                [f"{metric}_mean", f"{metric}_std", f"{metric}_mean_std"]
            )
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(
        list(group_columns), dropna=False, sort=True
    )
    for group_key, group in grouped:
        keys = group_key if isinstance(group_key, tuple) else (group_key,)
        row = dict(zip(group_columns, keys))
        row["seed_count"] = int(group["seed"].nunique())
        for metric in metric_columns:
            numeric = pd.to_numeric(group[metric], errors="coerce").dropna()
            if numeric.empty:
                row[f"{metric}_mean"] = NOT_AVAILABLE
                row[f"{metric}_std"] = NOT_AVAILABLE
                row[f"{metric}_mean_std"] = NOT_AVAILABLE
            else:
                mean = float(numeric.mean())
                std = float(numeric.std(ddof=0))
                row[f"{metric}_mean"] = mean
                row[f"{metric}_std"] = std
                row[f"{metric}_mean_std"] = f"{mean:.6f} ± {std:.6f}"
        rows.append(row)
    return pd.DataFrame(rows)


def discover_bundles(inputs: Iterable[Path], output_dir: Path) -> list[Path]:
    bundles: set[Path] = set()
    output_resolved = output_dir.resolve()
    for input_path in inputs:
        if not input_path.exists():
            continue
        candidates = (
            [input_path]
            if input_path.is_dir()
            and (
                (input_path / "metadata.json").is_file()
                or (input_path / "metrics.json").is_file()
            )
            else [
                path.parent
                for path in input_path.rglob("metadata.json")
            ]
        )
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved == output_resolved or output_resolved in resolved.parents:
                continue
            bundles.add(resolved)
    return sorted(bundles)


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, f"Missing {path.name}."
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"Cannot read {path.name}: {exc}"
    if not isinstance(value, dict):
        return None, f"{path.name} must contain a JSON object."
    return value, None


def analyze_bundle(bundle: Path) -> dict[str, Any]:
    metadata, metadata_error = _read_json(bundle / "metadata.json")
    metrics, metrics_error = _read_json(bundle / "metrics.json")
    notes = [
        note for note in (metadata_error, metrics_error) if note is not None
    ]
    metadata = metadata or {}
    metrics = metrics or {}
    final = metrics.get("final", {})
    protocol = metadata.get("protocol", NOT_AVAILABLE)
    seed = metadata.get("seed", NOT_AVAILABLE)
    held_out = metadata.get("held_out")
    train_support = metadata.get("class_support", {}).get("train")
    if not isinstance(train_support, dict):
        train_support = {}
        notes.append(
            "Seen-class/absent-class analysis requires "
            "metadata.json:class_support.train."
        )

    run: dict[str, Any] = {
        "bundle": str(bundle),
        "protocol": protocol,
        "seed": seed,
        "held_out": held_out if held_out is not None else "ALL",
        "accuracy": final.get("accuracy", NOT_AVAILABLE),
        "weighted_f1": final.get("weighted_f1", NOT_AVAILABLE),
        "fixed_8_macro_f1": final.get("macro_f1", NOT_AVAILABLE),
        "seen_class_macro_f1": NOT_AVAILABLE,
        "prediction_status": NOT_AVAILABLE,
        "notes": notes,
        "class_support": metadata.get("class_support", {}),
        "zero_train_support_classes": sorted(
            label
            for label in FIXED_LABELS
            if int(train_support.get(label, 0)) == 0
        ),
        "binary": None,
        "entropy": pd.DataFrame(),
        "entropy_auroc": None,
        "entropy_auroc_note": None,
        "confusion_matrix": None,
    }
    prediction_path = bundle / "predictions.csv"
    if not prediction_path.is_file():
        run["notes"].append(
            "Per-sample analysis requires predictions.csv with true_label, "
            "predicted_label, scenario, split, seed, and all fixed-label "
            "probability::* or logit::* columns."
        )
        return run
    try:
        predictions = pd.read_csv(prediction_path)
    except (OSError, pd.errors.ParserError) as exc:
        run["notes"].append(f"Cannot read predictions.csv: {exc}")
        return run
    missing_identity = sorted(
        set(PREDICTION_IDENTITY_COLUMNS) - set(predictions.columns)
    )
    if missing_identity or predictions.empty:
        run["notes"].append(
            "predictions.csv missing required identity fields "
            f"{missing_identity} or contains no rows."
        )
        return run

    true_labels = predictions["true_label"].astype(str).tolist()
    predicted_labels = predictions["predicted_label"].astype(str).tolist()
    unknown_labels = sorted(
        (set(true_labels) | set(predicted_labels)) - set(FIXED_LABELS)
    )
    if unknown_labels:
        run["notes"].append(
            "predictions.csv contains labels outside the fixed taxonomy: "
            + ", ".join(unknown_labels)
        )
        return run
    run["prediction_status"] = "AVAILABLE"
    run["accuracy"] = float(accuracy_score(true_labels, predicted_labels))
    run["weighted_f1"] = float(
        f1_score(
            true_labels,
            predicted_labels,
            average="weighted",
            zero_division=0,
        )
    )
    run["fixed_8_macro_f1"] = fixed_class_macro_f1(
        true_labels, predicted_labels
    )
    seen = seen_class_macro_f1(
        true_labels, predicted_labels, train_support
    )
    run["seen_class_macro_f1"] = (
        seen if seen is not None else NOT_AVAILABLE
    )
    run["binary"] = binary_metrics(true_labels, predicted_labels)
    run["confusion_matrix"] = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(FIXED_LABELS),
    )
    entropy, auc, auc_note = entropy_analysis(
        predictions, train_support
    )
    run["entropy"] = entropy
    run["entropy_auroc"] = auc
    run["entropy_auroc_note"] = auc_note
    if entropy.empty:
        run["notes"].append(
            "Entropy NOT_AVAILABLE: "
            + str(auc_note)
        )
    elif auc_note:
        run["notes"].append("Entropy AUROC NOT_AVAILABLE: " + auc_note)
    return run


def _run_frame(runs: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    columns = [
        "bundle",
        "protocol",
        "seed",
        "held_out",
        "accuracy",
        "weighted_f1",
        "fixed_8_macro_f1",
        "seen_class_macro_f1",
        "zero_train_support_classes",
        "prediction_status",
        "notes",
    ]
    return pd.DataFrame(
        [
            {
                "bundle": run["bundle"],
                "protocol": run["protocol"],
                "seed": run["seed"],
                "held_out": run["held_out"],
                "accuracy": run["accuracy"],
                "weighted_f1": run["weighted_f1"],
                "fixed_8_macro_f1": run["fixed_8_macro_f1"],
                "seen_class_macro_f1": run["seen_class_macro_f1"],
                "zero_train_support_classes": "|".join(
                    run["zero_train_support_classes"]
                ),
                "prediction_status": run["prediction_status"],
                "notes": " | ".join(run["notes"]),
            }
            for run in runs
        ],
        columns=columns,
    )


def _class_support_frame(
    runs: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in runs:
        support = run["class_support"]
        for split in ("train", "validation", "test"):
            split_support = support.get(split, {})
            for label in FIXED_LABELS:
                value = split_support.get(label, NOT_AVAILABLE)
                rows.append(
                    {
                        "bundle": run["bundle"],
                        "protocol": run["protocol"],
                        "seed": run["seed"],
                        "held_out": run["held_out"],
                        "split": split,
                        "class": label,
                        "support": value,
                        "train_support_zero": (
                            int(support.get("train", {}).get(label, 0)) == 0
                            if support.get("train")
                            else NOT_AVAILABLE
                        ),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "bundle",
            "protocol",
            "seed",
            "held_out",
            "split",
            "class",
            "support",
            "train_support_zero",
        ],
    )


def _binary_frame(runs: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for run in runs:
        row = {
            "bundle": run["bundle"],
            "protocol": run["protocol"],
            "seed": run["seed"],
            "held_out": run["held_out"],
        }
        if run["binary"] is None:
            row.update(
                {
                    "accuracy": NOT_AVAILABLE,
                    "malicious_precision": NOT_AVAILABLE,
                    "malicious_recall": NOT_AVAILABLE,
                    "malicious_f1": NOT_AVAILABLE,
                }
            )
        else:
            row.update(run["binary"])
        rows.append(row)
    return pd.DataFrame(
        rows,
        columns=[
            "bundle",
            "protocol",
            "seed",
            "held_out",
            "accuracy",
            "malicious_precision",
            "malicious_recall",
            "malicious_f1",
        ],
    )


def _entropy_frame(runs: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    frames = []
    for run in runs:
        entropy = run["entropy"]
        if entropy.empty:
            continue
        entropy = entropy.copy()
        entropy.insert(0, "held_out", run["held_out"])
        entropy.insert(0, "seed", run["seed"])
        entropy.insert(0, "protocol", run["protocol"])
        entropy.insert(0, "bundle", run["bundle"])
        entropy["absent_vs_known_correct_auroc"] = (
            run["entropy_auroc"]
            if run["entropy_auroc"] is not None
            else NOT_AVAILABLE
        )
        entropy["auroc_note"] = run["entropy_auroc_note"] or ""
        frames.append(entropy)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _save_confusion_figures(
    runs: Sequence[Mapping[str, Any]],
    figures_dir: Path,
) -> list[str]:
    available = [run for run in runs if run["confusion_matrix"] is not None]
    if not available:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for run in available:
        matrix = np.asarray(run["confusion_matrix"], dtype=int)
        figure, axis = plt.subplots(figsize=(10, 8))
        image = axis.imshow(matrix, cmap="Blues")
        figure.colorbar(image, ax=axis)
        axis.set_xticks(
            range(len(FIXED_LABELS)),
            FIXED_LABELS,
            rotation=45,
            ha="right",
        )
        axis.set_yticks(range(len(FIXED_LABELS)), FIXED_LABELS)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.set_title(
            f"{run['protocol']} seed={run['seed']} held-out={run['held_out']}"
        )
        figure.tight_layout()
        safe_held = str(run["held_out"]).replace("/", "_")
        path = figures_dir / (
            f"confusion_{run['protocol']}_seed-{run['seed']}_"
            f"held-{safe_held}.png"
        )
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths.append(str(path))
    return paths


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return NOT_AVAILABLE
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for _, row in frame.iterrows():
        values = [
            str(row[column]).replace("|", "\\|").replace("\n", " ")
            for column in columns
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_analysis(
    runs: Sequence[Mapping[str, Any]],
    output_dir: Path,
    input_paths: Sequence[Path],
) -> dict[str, Any]:
    historical = (REPOSITORY_ROOT / "artifacts/phase1_results").resolve()
    resolved_output = output_dir.resolve()
    if resolved_output == historical or historical in resolved_output.parents:
        raise ValueError("Analysis output cannot be historical Phase 1 artifacts.")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    run_frame = _run_frame(runs)
    pooled = run_frame[run_frame["protocol"] == "pooled"].copy()
    loso = run_frame[run_frame["protocol"] == "loso"].copy()
    class_support = _class_support_frame(runs)
    binary = _binary_frame(runs)
    entropy = _entropy_frame(runs)
    metrics = (
        "accuracy",
        "weighted_f1",
        "fixed_8_macro_f1",
        "seen_class_macro_f1",
    )
    summary = aggregate_seed_metrics(
        run_frame,
        group_columns=("protocol", "held_out"),
        metric_columns=metrics,
    )

    summary.to_csv(output_dir / "summary.csv", index=False)
    pooled.to_csv(output_dir / "pooled_summary.csv", index=False)
    loso.to_csv(output_dir / "loso_summary.csv", index=False)
    class_support.to_csv(output_dir / "class_support.csv", index=False)
    binary.to_csv(output_dir / "binary_metrics.csv", index=False)
    if not entropy.empty:
        entropy.to_csv(output_dir / "entropy_summary.csv", index=False)
    entropy_path = output_dir / "entropy_summary.csv"
    if entropy.empty and entropy_path.exists():
        entropy_path.unlink()
    figures = _save_confusion_figures(runs, figures_dir)

    unavailable = [
        {
            "bundle": run["bundle"],
            "notes": run["notes"],
        }
        for run in runs
        if run["notes"]
    ]
    report_lines = [
        "# Phase 1 Clean Analysis",
        "",
        "This report aggregates existing clean artifacts only. It does not "
        "train a model and does not infer unavailable per-sample data.",
        "",
        "## Inputs",
        "",
        *[f"- `{path}`" for path in input_paths],
        "",
        f"Discovered bundles: **{len(runs)}**",
        "",
        "## Mean ± standard deviation across seeds",
        "",
        _markdown_table(summary),
        "",
        "Population standard deviation (`ddof=0`) is used across the supplied "
        "seed runs.",
        "",
        "## Per-run pooled results",
        "",
        _markdown_table(pooled),
        "",
        "## Per-fold LOSO results",
        "",
        _markdown_table(loso),
        "",
        "## Classes with train support 0",
        "",
        _markdown_table(
            run_frame[
                [
                    "protocol",
                    "seed",
                    "held_out",
                    "zero_train_support_classes",
                ]
            ]
        ),
        "",
        "Full train/validation/test support for every fixed class is in "
        "`class_support.csv`.",
        "",
        "## Binary Benign/Malicious metrics",
        "",
        _markdown_table(binary),
        "",
        "All non-Benign fixed-taxonomy labels are collapsed to the positive "
        "`Malicious` class.",
        "",
        "## Entropy groups",
        "",
        _markdown_table(entropy),
        "",
        "## Availability and limitations",
        "",
    ]
    if unavailable:
        for item in unavailable:
            report_lines.append(f"- `{item['bundle']}`")
            for note in item["notes"]:
                report_lines.append(f"  - {note}")
    else:
        report_lines.append("- All discovered runs include usable predictions.csv.")
    report_lines.extend(
        [
            "",
            "Entropy is an uncertainty analysis only. AUROC here compares rows "
            "whose true class has training support 0 against known-correct rows; "
            "it is not evidence of general zero-day detection.",
            "",
            "## Generated figures",
            "",
        ]
    )
    report_lines.extend(
        [f"- `{path}`" for path in figures]
        if figures
        else [
            f"- {NOT_AVAILABLE}: confusion figures require predictions.csv "
            "with true_label and predicted_label."
        ]
    )
    if entropy.empty:
        report_lines.extend(
            [
                "",
                "## Entropy analysis",
                "",
                f"{NOT_AVAILABLE}: add `predictions.csv` with either all "
                "`probability::<fixed-label>` columns or all "
                "`logit::<fixed-label>` columns.",
            ]
        )
    (output_dir / "report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    return {
        "bundles": len(runs),
        "pooled_runs": len(pooled),
        "loso_runs": len(loso),
        "entropy_available": not entropy.empty,
        "figures": len(figures),
        "output_dir": str(output_dir.resolve()),
    }


def _default_inputs() -> list[Path]:
    return sorted(Path("artifacts/phase1_clean").glob("seed-*"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate one or more clean Phase 1 seed directories."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Seed roots or individual clean bundle directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/phase1_clean/analysis"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    inputs = args.inputs or _default_inputs()
    bundles = discover_bundles(inputs, args.output_dir)
    runs = [analyze_bundle(bundle) for bundle in bundles]
    result = write_analysis(runs, args.output_dir, inputs)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not bundles:
        print(
            "No clean bundles found. Outputs contain headers/NOT_AVAILABLE only.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
