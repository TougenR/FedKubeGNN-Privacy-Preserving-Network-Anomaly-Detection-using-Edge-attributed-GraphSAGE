"""Headless, provenance-checked visual diagnostics for completed FL runs."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.federated.observability.run_store import atomic_json, atomic_text


@dataclass(frozen=True)
class VisualizedRun:
    root: Path
    run_id: str
    strategy: str
    provenance: Mapping[str, str]
    rounds: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any]
    class_names: tuple[str, ...]
    confusion_matrix: np.ndarray


def _number(value: Any, *, where: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where} must be numeric; got {value!r}.") from exc
    if not np.isfinite(number):
        raise ValueError(f"{where} must be finite; got {value!r}.")
    return number


def _metric(
    values: Mapping[str, Any], *names: str, where: str
) -> float:
    for name in names:
        if name in values:
            return _number(values[name], where=f"{where}.{name}")
    raise ValueError(f"{where} is missing metric aliases {names}.")


def _load_rounds(root: Path) -> tuple[Mapping[str, Any], ...]:
    flower = sorted((root / "metrics").glob("validation-round-*.json"))
    inprocess = sorted((root / "metrics" / "rounds").glob("round-*.json"))
    paths = flower or inprocess
    if not paths:
        raise ValueError(f"Run has no validation-round metrics: {root}")
    rounds = tuple(json.loads(path.read_text(encoding="utf-8")) for path in paths)
    numbers = [int(record.get("round", -1)) for record in rounds]
    if numbers != list(range(1, len(rounds) + 1)):
        raise ValueError(f"Run rounds are not contiguous from 1: {root}")
    for number, record in zip(numbers, rounds):
        where = f"{root}:round-{number}"
        _metric(record, "loss", "validation_loss", where=where)
        _metric(record, "accuracy", where=where)
        _metric(record, "macro-f1", "macro_f1", where=where)
        _metric(record, "weighted-f1", "weighted_f1", where=where)
    return rounds


def _test_metrics(summary: Mapping[str, Any], *, root: Path) -> Mapping[str, Any]:
    metrics = summary.get("test_metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(f"Run summary has no test_metrics mapping: {root}")
    return metrics


def _confusion_matrix(
    summary: Mapping[str, Any], test: Mapping[str, Any], *, root: Path
) -> np.ndarray:
    raw = test.get("confusion-matrix", summary.get("test_confusion_matrix"))
    if raw is None:
        raise ValueError(f"Run summary has no final confusion matrix: {root}")
    values = np.asarray(raw, dtype=np.int64)
    if values.ndim == 1:
        configured = test.get("num-classes")
        size = int(configured) if configured is not None else int(np.sqrt(values.size))
        if size * size != values.size:
            raise ValueError(f"Final confusion matrix is not square: {root}")
        values = values.reshape(size, size)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"Final confusion matrix is not square: {root}")
    if np.any(values < 0):
        raise ValueError(f"Final confusion matrix contains negative counts: {root}")
    return values


def _load_run(root_value: str | Path) -> VisualizedRun:
    root = Path(root_value)
    status = json.loads((root / "run.json").read_text(encoding="utf-8"))
    if status.get("status") != "completed":
        raise ValueError(f"Run is not completed: {root}")
    provenance = {
        key: str(status.get(key, ""))
        for key in ("config_digest", "dataset_digest", "model_digest")
    }
    missing = [key for key, value in provenance.items() if not value]
    if missing:
        raise ValueError(f"Run {root} is missing provenance fields: {missing}.")
    summary = json.loads(
        (root / "metrics" / "summary.json").read_text(encoding="utf-8")
    )
    test = _test_metrics(summary, root=root)
    for aliases in (
        ("loss",),
        ("accuracy",),
        ("macro-f1", "macro_f1"),
        ("weighted-f1", "weighted_f1"),
    ):
        _metric(test, *aliases, where=f"{root}:test_metrics")
    matrix = _confusion_matrix(summary, test, root=root)
    configured_names = summary.get("class_names", ())
    class_names = tuple(str(name) for name in configured_names)
    if not class_names:
        class_names = tuple(f"class_{index}" for index in range(matrix.shape[0]))
    if len(class_names) != matrix.shape[0]:
        raise ValueError(
            f"Run class_names length differs from confusion matrix size: {root}"
        )
    strategy = str(summary.get("strategy", status.get("strategy", ""))).lower()
    if not strategy:
        raise ValueError(f"Run has no strategy: {root}")
    return VisualizedRun(
        root=root,
        run_id=str(status["run_id"]),
        strategy=strategy,
        provenance=provenance,
        rounds=_load_rounds(root),
        summary=summary,
        class_names=class_names,
        confusion_matrix=matrix,
    )


def _validate_comparability(runs: Sequence[VisualizedRun]) -> None:
    if not runs:
        raise ValueError("At least one completed run is required.")
    expected = dict(runs[0].provenance)
    for run in runs[1:]:
        if dict(run.provenance) != expected:
            mismatches = {
                key: (expected[key], run.provenance[key])
                for key in expected
                if run.provenance[key] != expected[key]
            }
            raise ValueError(
                f"Run {run.root} is not comparable with the first run: {mismatches}."
            )
        if run.class_names != runs[0].class_names:
            raise ValueError(
                f"Run {run.root} uses a different ordered class vocabulary."
            )
    strategies = [run.strategy for run in runs]
    if len(strategies) != len(set(strategies)):
        raise ValueError(f"Visualization requires one run per strategy: {strategies}.")


def _csv_text(fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    return buffer.getvalue()


def _plot_runtime() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_figure(figure: Any, output: Path, stem: str) -> list[str]:
    names = []
    for extension in ("png", "pdf"):
        destination = output / f"{stem}.{extension}"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{stem}.", suffix=f".{extension}", dir=output
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            figure.savefig(
                temporary,
                format=extension,
                dpi=300 if extension == "png" else None,
                bbox_inches="tight",
            )
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        names.append(destination.name)
    return names


def _round_value(record: Mapping[str, Any], metric: str) -> float:
    aliases = {
        "validation_loss": ("loss", "validation_loss"),
        "accuracy": ("accuracy",),
        "macro_f1": ("macro-f1", "macro_f1"),
        "weighted_f1": ("weighted-f1", "weighted_f1"),
    }
    return _metric(record, *aliases[metric], where=f"round-{record['round']}")


def _final_value(run: VisualizedRun, metric: str) -> float:
    aliases = {
        "loss": ("loss",),
        "accuracy": ("accuracy",),
        "macro_f1": ("macro-f1", "macro_f1"),
        "weighted_f1": ("weighted-f1", "weighted_f1"),
    }
    return _metric(
        _test_metrics(run.summary, root=run.root),
        *aliases[metric],
        where=f"{run.root}:test_metrics",
    )


def _per_class_f1(
    test: Mapping[str, Any], class_name: str, class_index: int, *, root: Path
) -> float:
    flower_key = f"f1-class-{class_index}"
    if flower_key in test:
        return _number(test[flower_key], where=f"{root}:test_metrics.{flower_key}")
    per_class = test.get("per_class")
    if not isinstance(per_class, Mapping):
        raise ValueError(
            f"{root}:test_metrics is missing both {flower_key!r} and per_class."
        )
    class_metrics = per_class.get(class_name)
    if not isinstance(class_metrics, Mapping) or "f1" not in class_metrics:
        raise ValueError(
            f"{root}:test_metrics.per_class.{class_name} is missing numeric f1."
        )
    return _number(
        class_metrics["f1"],
        where=f"{root}:test_metrics.per_class.{class_name}.f1",
    )


def _plot_learning_curves(runs: Sequence[VisualizedRun], output: Path) -> list[str]:
    plt = _plot_runtime()
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for run in runs:
        rounds = [int(record["round"]) for record in run.rounds]
        axes[0].plot(
            rounds,
            [_round_value(record, "validation_loss") for record in run.rounds],
            marker="o",
            markersize=3,
            label=run.strategy,
        )
        axes[1].plot(
            rounds,
            [_round_value(record, "macro_f1") for record in run.rounds],
            marker="o",
            markersize=3,
            label=f"{run.strategy} macro-F1",
        )
        axes[1].plot(
            rounds,
            [_round_value(record, "accuracy") for record in run.rounds],
            linestyle="--",
            alpha=0.8,
            label=f"{run.strategy} accuracy",
        )
        best_round = int(run.summary["best_round"])
        axes[1].axvline(best_round, color="gray", alpha=0.2)
    axes[0].set(title="Validation loss by federated round", xlabel="Round", ylabel="Loss")
    axes[1].set(
        title="Validation quality by federated round",
        xlabel="Round",
        ylabel="Score",
        ylim=(0, 1),
    )
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    names = _save_figure(figure, output, "federated_learning_curves")
    plt.close(figure)
    return names


def _plot_final_metrics(runs: Sequence[VisualizedRun], output: Path) -> list[str]:
    plt = _plot_runtime()
    metrics = ("accuracy", "macro_f1", "weighted_f1")
    labels = ("Accuracy", "Macro-F1", "Weighted-F1")
    x = np.arange(len(metrics))
    width = 0.8 / len(runs)
    figure, axis = plt.subplots(figsize=(9, 5))
    for index, run in enumerate(runs):
        offset = (index - (len(runs) - 1) / 2) * width
        values = [_final_value(run, metric) for metric in metrics]
        bars = axis.bar(x + offset, values, width, label=run.strategy)
        axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Final test score")
    axis.set_title("Federated strategy comparison on the one final test")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    names = _save_figure(figure, output, "federated_final_metrics")
    plt.close(figure)
    return names


def _plot_per_class_f1(runs: Sequence[VisualizedRun], output: Path) -> list[str]:
    plt = _plot_runtime()
    class_names = runs[0].class_names
    x = np.arange(len(class_names))
    width = 0.8 / len(runs)
    figure, axis = plt.subplots(figsize=(max(10, len(class_names) * 1.2), 5))
    for run_index, run in enumerate(runs):
        test = _test_metrics(run.summary, root=run.root)
        values = [
            _per_class_f1(
                test,
                class_name,
                class_index,
                root=run.root,
            )
            for class_index, class_name in enumerate(class_names)
        ]
        offset = (run_index - (len(runs) - 1) / 2) * width
        axis.bar(x + offset, values, width, label=run.strategy)
    axis.set_xticks(x, class_names, rotation=45, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Final test F1")
    axis.set_title("Per-class F1 after federated training")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    names = _save_figure(figure, output, "federated_per_class_f1")
    plt.close(figure)
    return names


def _plot_confusions(runs: Sequence[VisualizedRun], output: Path) -> list[str]:
    plt = _plot_runtime()
    figure, axes = plt.subplots(
        len(runs),
        2,
        figsize=(14, max(5.5, 5.5 * len(runs))),
        squeeze=False,
        constrained_layout=True,
    )
    for row, run in enumerate(runs):
        counts = run.confusion_matrix
        denominators = counts.sum(axis=1, keepdims=True)
        normalized = np.divide(
            counts,
            denominators,
            out=np.zeros_like(counts, dtype=float),
            where=denominators != 0,
        )
        for column, (values, title) in enumerate(
            ((counts, "counts"), (normalized, "row-normalized"))
        ):
            axis = axes[row][column]
            image = axis.imshow(
                values,
                cmap="Blues",
                vmin=0,
                vmax=1 if column == 1 else None,
                aspect="auto",
            )
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            axis.set_xticks(
                range(len(run.class_names)), run.class_names, rotation=45, ha="right"
            )
            axis.set_yticks(range(len(run.class_names)), run.class_names)
            axis.set_xlabel("Predicted label")
            axis.set_ylabel("True label")
            axis.set_title(f"{run.strategy} final test — {title}")
            peak = float(values.max()) if values.size else 0.0
            threshold = peak * 0.6
            for true_index in range(values.shape[0]):
                for predicted_index in range(values.shape[1]):
                    value = values[true_index, predicted_index]
                    label = f"{int(value)}" if column == 0 else f"{value:.2f}"
                    axis.text(
                        predicted_index,
                        true_index,
                        label,
                        ha="center",
                        va="center",
                        color="white" if peak and value >= threshold else "black",
                        fontsize=7 if len(run.class_names) > 6 else 9,
                    )
    names = _save_figure(figure, output, "federated_confusion_matrices")
    plt.close(figure)
    return names


def visualize_runs(
    run_roots: Iterable[str | Path], output: str | Path
) -> Path:
    """Create Phase-1-style diagnostics without re-evaluating test data."""
    runs = tuple(_load_run(root) for root in run_roots)
    _validate_comparability(runs)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)

    round_fields = (
        "run_id",
        "strategy",
        "round",
        "validation_loss",
        "accuracy",
        "macro_f1",
        "weighted_f1",
    )
    round_rows = [
        {
            "run_id": run.run_id,
            "strategy": run.strategy,
            "round": int(record["round"]),
            "validation_loss": _round_value(record, "validation_loss"),
            "accuracy": _round_value(record, "accuracy"),
            "macro_f1": _round_value(record, "macro_f1"),
            "weighted_f1": _round_value(record, "weighted_f1"),
        }
        for run in runs
        for record in run.rounds
    ]
    atomic_text(
        destination / "round_metrics.csv", _csv_text(round_fields, round_rows)
    )

    final_fields = (
        "run_id",
        "strategy",
        "best_round",
        "validation_macro_f1",
        "test_loss",
        "test_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
    )
    final_rows = [
        {
            "run_id": run.run_id,
            "strategy": run.strategy,
            "best_round": int(run.summary["best_round"]),
            "validation_macro_f1": _number(
                run.summary["validation_macro_f1"],
                where=f"{run.root}:validation_macro_f1",
            ),
            "test_loss": _final_value(run, "loss"),
            "test_accuracy": _final_value(run, "accuracy"),
            "test_macro_f1": _final_value(run, "macro_f1"),
            "test_weighted_f1": _final_value(run, "weighted_f1"),
        }
        for run in runs
    ]
    atomic_text(
        destination / "final_metrics.csv", _csv_text(final_fields, final_rows)
    )

    figures = [
        *_plot_learning_curves(runs, destination),
        *_plot_final_metrics(runs, destination),
        *_plot_per_class_f1(runs, destination),
        *_plot_confusions(runs, destination),
    ]
    manifest = {
        "visualization_version": 1,
        "provenance": dict(runs[0].provenance),
        "class_names": list(runs[0].class_names),
        "runs": [
            {
                "run_id": run.run_id,
                "strategy": run.strategy,
                "rounds": len(run.rounds),
                "root": str(run.root),
            }
            for run in runs
        ],
        "tables": ["round_metrics.csv", "final_metrics.csv"],
        "figures": figures,
        "test_evaluation_reused": True,
    }
    manifest_path = destination / "visualization_manifest.json"
    atomic_json(manifest_path, manifest)
    return manifest_path


def visualize_class_aware_summary(
    summary_path: str | Path, output: str | Path
) -> Path:
    """Render a locked multi-seed class-aware summary without re-evaluation."""
    source = Path(summary_path)
    summary = json.loads(source.read_text(encoding="utf-8"))
    validation = summary["validation"]
    baseline = validation["baseline"]
    selected = validation["selected"]
    test = summary["test"]
    seeds = tuple(
        sorted(
            (key for key in selected if str(key).isdigit()),
            key=lambda value: int(value),
        )
    )
    if not seeds or any(seed not in baseline or seed not in test for seed in seeds):
        raise ValueError("Class-aware summary has incomplete seed evidence.")
    destination = Path(output)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Visualization output is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    plt = _plot_runtime()
    figure_names: list[str] = []

    x = np.arange(len(seeds))
    width = 0.36
    figure, axis = plt.subplots(figsize=(9, 5))
    baseline_values = [
        _number(baseline[seed], where=f"validation.baseline.{seed}")
        for seed in seeds
    ]
    selected_values = [
        _number(
            selected[seed]["macro_f1"],
            where=f"validation.selected.{seed}.macro_f1",
        )
        for seed in seeds
    ]
    axis.bar(x - width / 2, baseline_values, width, label="Sample FedAvg")
    axis.bar(x + width / 2, selected_values, width, label="Class-aware selected")
    axis.set_xticks(x, [f"Seed {seed}" for seed in seeds])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Validation macro-F1")
    axis.set_title("Natural non-IID validation improvement")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    figure_names.extend(
        _save_figure(figure, destination, "class_aware_validation_by_seed")
    )
    plt.close(figure)

    metrics = ("accuracy", "weighted_f1", "macro_f1")
    figure, axis = plt.subplots(figsize=(10, 5))
    metric_x = np.arange(len(metrics))
    metric_width = 0.8 / len(seeds)
    for seed_index, seed in enumerate(seeds):
        values = [
            _number(test[seed][metric], where=f"test.{seed}.{metric}")
            for metric in metrics
        ]
        offset = (seed_index - (len(seeds) - 1) / 2) * metric_width
        axis.bar(metric_x + offset, values, metric_width, label=f"Seed {seed}")
    axis.set_xticks(metric_x, ("Accuracy", "Weighted-F1", "Macro-F1"))
    axis.set_ylim(0, 1)
    axis.set_ylabel("Test score")
    axis.set_title("Validation-selected class-aware test metrics")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    figure_names.extend(
        _save_figure(figure, destination, "class_aware_test_metrics")
    )
    plt.close(figure)

    class_names = tuple(test[seeds[0]]["per_class_f1"])
    if not class_names:
        raise ValueError("Class-aware summary has no per-class test metrics.")
    class_x = np.arange(len(class_names))
    class_width = 0.8 / len(seeds)
    figure, axis = plt.subplots(figsize=(max(11, len(class_names) * 1.3), 5))
    for seed_index, seed in enumerate(seeds):
        per_class = test[seed]["per_class_f1"]
        if tuple(per_class) != class_names:
            raise ValueError("Per-class metric order differs between seeds.")
        values = [
            _number(per_class[name], where=f"test.{seed}.per_class_f1.{name}")
            for name in class_names
        ]
        offset = (seed_index - (len(seeds) - 1) / 2) * class_width
        axis.bar(class_x + offset, values, class_width, label=f"Seed {seed}")
    axis.set_xticks(class_x, class_names, rotation=45, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Test F1")
    axis.set_title("Selected class-aware per-class F1")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    figure_names.extend(
        _save_figure(figure, destination, "class_aware_per_class_f1")
    )
    plt.close(figure)

    seed_rows = [
        {
            "seed": seed,
            "baseline_validation_macro_f1": baseline_values[index],
            "selected_validation_macro_f1": selected_values[index],
            "test_accuracy": test[seed]["accuracy"],
            "test_weighted_f1": test[seed]["weighted_f1"],
            "test_macro_f1": test[seed]["macro_f1"],
        }
        for index, seed in enumerate(seeds)
    ]
    atomic_text(
        destination / "class_aware_seed_metrics.csv",
        _csv_text(tuple(seed_rows[0]), seed_rows),
    )
    manifest = {
        "visualization_version": 1,
        "source_summary": str(source),
        "source_summary_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "seeds": list(seeds),
        "figures": figure_names,
        "tables": ["class_aware_seed_metrics.csv"],
        "test_evaluation_reused": True,
    }
    manifest_path = destination / "visualization_manifest.json"
    atomic_json(manifest_path, manifest)
    return manifest_path


def visualize_personalized_summary(
    summary_path: str | Path, output: str | Path
) -> Path:
    """Render locked FedPer multi-seed evidence without evaluating data."""
    source = Path(summary_path)
    summary = json.loads(source.read_text(encoding="utf-8"))
    validation = summary["validation"]
    sample = validation["sample_fedavg"]
    class_aware = validation["class_aware"]
    fedper = validation["fedper"]
    test = summary["test"]["fedper"]
    seeds = tuple(
        sorted(
            (key for key in fedper if str(key).isdigit()),
            key=lambda value: int(value),
        )
    )
    if not seeds or any(
        seed not in sample or seed not in class_aware or seed not in test
        for seed in seeds
    ):
        raise ValueError("Personalized summary has incomplete seed evidence.")
    destination = Path(output)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Visualization output is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    plt = _plot_runtime()
    figure_names: list[str] = []

    x = np.arange(len(seeds))
    width = 0.25
    figure, axis = plt.subplots(figsize=(10, 5))
    sample_values = [
        _number(sample[seed], where=f"validation.sample_fedavg.{seed}")
        for seed in seeds
    ]
    class_aware_values = [
        _number(class_aware[seed], where=f"validation.class_aware.{seed}")
        for seed in seeds
    ]
    fedper_values = [
        _number(
            fedper[seed]["macro_f1"],
            where=f"validation.fedper.{seed}.macro_f1",
        )
        for seed in seeds
    ]
    axis.bar(x - width, sample_values, width, label="Sample FedAvg")
    axis.bar(x, class_aware_values, width, label="Class-aware global")
    axis.bar(x + width, fedper_values, width, label="FedPer")
    axis.set_xticks(x, [f"Seed {seed}" for seed in seeds])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Validation macro-F1")
    axis.set_title("Natural non-IID validation treatments")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    figure_names.extend(
        _save_figure(figure, destination, "fedper_validation_by_seed")
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5))
    for seed in seeds:
        values = fedper[seed]["round_macro_f1"]
        if not values:
            raise ValueError(f"FedPer seed {seed} has no validation rounds.")
        axis.plot(
            np.arange(1, len(values) + 1),
            [_number(value, where=f"validation.fedper.{seed}.round") for value in values],
            label=f"Seed {seed}",
        )
    axis.set(xlabel="Round", ylabel="Validation macro-F1", ylim=(0, 1))
    axis.set_title("FedPer validation learning curves")
    axis.grid(alpha=0.2)
    axis.legend()
    figure_names.extend(
        _save_figure(figure, destination, "fedper_learning_curves")
    )
    plt.close(figure)

    metrics = ("accuracy", "weighted_f1", "macro_f1")
    metric_x = np.arange(len(metrics))
    metric_width = 0.8 / len(seeds)
    figure, axis = plt.subplots(figsize=(10, 5))
    for seed_index, seed in enumerate(seeds):
        values = [
            _number(test[seed][metric], where=f"test.fedper.{seed}.{metric}")
            for metric in metrics
        ]
        offset = (seed_index - (len(seeds) - 1) / 2) * metric_width
        axis.bar(metric_x + offset, values, metric_width, label=f"Seed {seed}")
    axis.set_xticks(metric_x, ("Accuracy", "Weighted-F1", "Macro-F1"))
    axis.set_ylim(0, 1)
    axis.set_ylabel("Test score")
    axis.set_title("Validation-selected FedPer test metrics")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    figure_names.extend(_save_figure(figure, destination, "fedper_test_metrics"))
    plt.close(figure)

    class_names = tuple(test[seeds[0]]["per_class_f1"])
    class_x = np.arange(len(class_names))
    class_width = 0.8 / len(seeds)
    figure, axis = plt.subplots(figsize=(max(11, len(class_names) * 1.3), 5))
    for seed_index, seed in enumerate(seeds):
        per_class = test[seed]["per_class_f1"]
        if tuple(per_class) != class_names:
            raise ValueError("Per-class metric order differs between seeds.")
        values = [
            _number(per_class[name], where=f"test.fedper.{seed}.{name}")
            for name in class_names
        ]
        offset = (seed_index - (len(seeds) - 1) / 2) * class_width
        axis.bar(class_x + offset, values, class_width, label=f"Seed {seed}")
    axis.set_xticks(class_x, class_names, rotation=45, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Test F1")
    axis.set_title("Personalized-head per-class F1")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    figure_names.extend(
        _save_figure(figure, destination, "fedper_per_class_f1")
    )
    plt.close(figure)

    seed_rows = [
        {
            "seed": seed,
            "sample_validation_macro_f1": sample_values[index],
            "class_aware_validation_macro_f1": class_aware_values[index],
            "fedper_validation_macro_f1": fedper_values[index],
            "fedper_test_accuracy": test[seed]["accuracy"],
            "fedper_test_weighted_f1": test[seed]["weighted_f1"],
            "fedper_test_macro_f1": test[seed]["macro_f1"],
        }
        for index, seed in enumerate(seeds)
    ]
    atomic_text(
        destination / "fedper_seed_metrics.csv",
        _csv_text(tuple(seed_rows[0]), seed_rows),
    )
    manifest = {
        "visualization_version": 1,
        "source_summary": str(source),
        "source_summary_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "seeds": list(seeds),
        "figures": figure_names,
        "tables": ["fedper_seed_metrics.csv"],
        "test_evaluation_reused": True,
    }
    manifest_path = destination / "visualization_manifest.json"
    atomic_json(manifest_path, manifest)
    return manifest_path
