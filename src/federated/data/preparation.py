"""Prepare six train-only-fitted IoT-23 clients transactionally."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.federated.adapters.phase1_iot23 import (
    Phase1IoT23Task,
    make_phase1_model_factory,
)
from src.federated.config.schema import Phase2Config
from src.federated.data.manifest import MANIFEST_VERSION, PreparedDatasetManifest
from src.federated.data.partitioners.scenario import deterministic_edge_masks
from src.federated.data.sources.iot23 import read_clean_priority_sample
from src.federated.data.storage import (
    GraphArrays,
    checksum_index_digest,
    sha256_file,
    write_graph_arrays,
)
from src.federated.observability.events import NoopObserver, Observer


def source_paths(
    config: Phase2Config, *, repository_root: str | Path = "."
) -> dict[str, Path]:
    base = Path(repository_root) / config.data.raw_root
    return {scenario.id: base / scenario.path for scenario in config.data.scenarios}


def doctor(
    config: Phase2Config, *, repository_root: str | Path = "."
) -> dict[str, Any]:
    """Read-only preflight: dependencies, source files, GPU, and free disk."""
    import importlib.util
    import shutil as disk

    paths = source_paths(config, repository_root=repository_root)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    dependency = {
        name: importlib.util.find_spec(module) is not None
        for name, module in {
            "numpy": "numpy",
            "pandas": "pandas",
            "sklearn": "sklearn",
            "yaml": "yaml",
            "torch": "torch",
            "torch_geometric": "torch_geometric",
            "flwr": "flwr",
        }.items()
    }
    free_bytes = disk.disk_usage(Path(repository_root)).free
    source_bytes = sum(path.stat().st_size for path in paths.values() if path.is_file())
    # Sources already occupy disk and are streamed in place.  Conservatively
    # reserve 2 GiB because capped graph arrays plus the transactional temporary
    # directory need additional workspace,
    # not a second copy of every raw log.
    required_free_bytes = 2 * 1024**3
    cuda = {"available": False, "device_count": 0}
    if dependency["torch"]:
        import torch

        cuda = {
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
        }
    return {
        "ready_for_prepare": not missing
        and all(
            dependency[name]
            for name in (
                "numpy",
                "pandas",
                "sklearn",
                "yaml",
                "torch",
                "torch_geometric",
            )
        )
        and free_bytes >= required_free_bytes,
        "ready_for_flower": dependency["flwr"],
        "missing_sources": missing,
        "dependencies": dependency,
        "cuda": cuda,
        "free_bytes": free_bytes,
        "source_bytes": source_bytes,
        "required_free_bytes": required_free_bytes,
        "graph_protocol": config.data.graph_protocol,
    }


def _preprocessor_contract(
    task: Phase1IoT23Task,
    preprocessor: Any,
    *,
    config: Phase2Config,
    raw_sources: Mapping[str, Any],
):
    return task.contract_bundle(
        preprocessor=preprocessor,
        metadata={
            "preprocessing": "train_only_global",
            "graph_protocol": config.data.graph_protocol,
            "config_digest": config.digest,
            "raw_sources": dict(raw_sources),
        },
    )


def prepare_iot23(
    config: Phase2Config,
    *,
    repository_root: str | Path = ".",
    observer: Observer | None = None,
) -> Path:
    """Build one immutable prepared dataset; never overwrite an existing ID."""
    observer = observer or NoopObserver()
    checks = doctor(config, repository_root=repository_root)
    if not checks["ready_for_prepare"]:
        raise RuntimeError(f"Phase 2 preparation preflight failed: {checks}")
    started = time.perf_counter()
    paths = source_paths(config, repository_root=repository_root)
    raw_sources = {}
    for client_id, path in paths.items():
        hash_started = time.perf_counter()
        observer.emit(
            "prepare.source_hash_started",
            component="data_source",
            client_id=client_id,
            source_bytes=path.stat().st_size,
        )
        raw_sources[client_id] = {
            "path": str(path),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        observer.emit(
            "prepare.source_hash_completed",
            component="data_source",
            client_id=client_id,
            source_bytes=path.stat().st_size,
            duration_seconds=time.perf_counter() - hash_started,
        )
    identity_payload = json.dumps(
        {"config": config.digest, "sources": raw_sources}, sort_keys=True
    )
    dataset_id = "iot23-" + hashlib.sha256(identity_payload.encode()).hexdigest()[:16]
    prepared_root = Path(repository_root) / config.data.prepared_root
    final = prepared_root / dataset_id
    if final.exists():
        raise FileExistsError(f"Prepared dataset already exists: {final}")
    prepared_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{dataset_id}.", dir=prepared_root))
    try:
        cleaned: dict[str, Any] = {}
        masks: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for index, (client_id, path) in enumerate(paths.items()):
            client_started = time.perf_counter()
            frame = read_clean_priority_sample(
                path,
                cap_per_class=config.data.cap_per_class,
                chunk_size=config.data.chunk_size,
                seed=config.training.seed + index,
            )
            split = config.data.split
            mask_tuple = deterministic_edge_masks(
                frame[config.data.target_column].astype(str).to_numpy(),
                train_ratio=split.train,
                validation_ratio=split.validation,
                test_ratio=split.test,
                seed=config.training.seed,
            )
            cleaned[client_id], masks[client_id] = frame, mask_tuple
            observer.emit(
                "prepare.client_sampled",
                component="data_source",
                client_id=client_id,
                rows=len(frame),
                duration_seconds=time.perf_counter() - client_started,
            )

        import pandas as pd
        import torch
        from src.core.graph import build_graph
        from src.core.preprocess import fit_preprocessor, transform

        train_rows = pd.concat(
            [frame.loc[masks[client_id][0]] for client_id, frame in cleaned.items()],
            ignore_index=True,
        )
        preprocessor = fit_preprocessor(train_rows)
        all_classes = {
            str(label)
            for frame in cleaned.values()
            for label in frame[config.data.target_column].unique()
        }
        classes = (("Benign",) if "Benign" in all_classes else ()) + tuple(
            sorted(all_classes - {"Benign"})
        )
        class_to_idx = {label: index for index, label in enumerate(classes)}
        graphs: dict[str, Any] = {}
        clients_document: list[dict[str, Any]] = []
        for client_id, frame in cleaned.items():
            graph = build_graph(
                transform(frame, preprocessor),
                class_to_idx,
                preprocessor.feature_columns,
            )
            graph.train_mask, graph.val_mask, graph.test_mask = [
                torch.from_numpy(mask) for mask in masks[client_id]
            ]
            graphs[client_id] = graph
            relative = Path("clients") / client_id
            arrays = GraphArrays.from_graph(graph)
            write_graph_arrays(
                temporary / relative,
                arrays,
                metadata={
                    "client_id": client_id,
                    "graph_protocol": config.data.graph_protocol,
                },
            )
            clients_document.append(
                {
                    "client_id": client_id,
                    "path": str(relative),
                    "num_edges": arrays.num_edges,
                    "artifact_digest": checksum_index_digest(temporary / relative),
                }
            )

        model_cfg = {"model": config.model.__dict__}
        torch.manual_seed(config.training.seed)
        task = Phase1IoT23Task(
            client_graphs=graphs,
            feature_columns=preprocessor.feature_columns,
            class_to_idx=class_to_idx,
            model_factory=make_phase1_model_factory(
                model_name=config.components.model, cfg=model_cfg
            ),
            model_family=config.components.model,
            model_hyperparameters=config.model.__dict__,
            imbalance_mode=config.training.imbalance,
            device="cpu",
            source_metadata={
                "dataset_id": dataset_id,
                "graph_protocol": config.data.graph_protocol,
            },
        )
        contract_root = _preprocessor_contract(
            task, preprocessor, config=config, raw_sources=raw_sources
        ).write(temporary / "contract")
        initial_path = temporary / "initial_state.npz"
        np.savez(initial_path, **task.initial_state())
        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "dataset_id": dataset_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config_digest": config.digest,
            "graph_protocol": config.data.graph_protocol,
            "preprocessing": config.data.preprocessing,
            "contract_path": "contract",
            "contract_digest": checksum_index_digest(contract_root),
            "initial_state_path": "initial_state.npz",
            "initial_state_sha256": sha256_file(initial_path),
            "raw_sources": raw_sources,
            "clients": clients_document,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        PreparedDatasetManifest.load(temporary, verify=True)
        os.replace(temporary, final)
        observer.emit(
            "prepare.completed",
            component="preparation",
            dataset_id=dataset_id,
            clients=len(clients_document),
            duration_seconds=time.perf_counter() - started,
        )
        return final
    except BaseException as error:
        observer.emit(
            "prepare.failed",
            level="ERROR",
            component="preparation",
            error_type=type(error).__name__,
            error_message=str(error),
            duration_seconds=time.perf_counter() - started,
        )
        shutil.rmtree(temporary, ignore_errors=True)
        raise
