"""
multi_scenario.py — Tầng dữ liệu ĐA-SCENARIO + harness LOSO
                (Leave-One-Scenario-Out) inductive.

Mục đích
--------
Giai đoạn 1 mới có kết quả trên 1 scenario (34-1). Để chứng minh mô hình
TỔNG QUÁT HÓA giữa các scenario (mỗi scenario có tập service/conn_state/
nhãn khác nhau), ta cần:

    1. Tầng dữ liệu ĐA-SCENARIO dùng chung:
       •  MỘT preprocessor chung (fit trên train).
       •  MỘT class_to_idx chung = hợp mọi nhãn (train + held-out).
       •  Mọi Data phải cùng ``feature_dim`` và ``num_classes``.

    2. Harness LOSO inductive:
       •  Mỗi scenario lần lượt làm held-out.
       •  Train trên N-1 scenario, test trên scenario thứ N **chưa thấy**.
       •  Đánh giá bằng macro-F1 (chỉ số chính do mất cân bằng lớp),
          per-class F1, confusion matrix.

Quy ước dùng lại (KHÔNG sửa single-scenario logic)
---------------------------------------------------
- ``src.data_io.load_scenario``        — đọc conn.log.labeled.
- ``src.preprocess.{clean_flows, fit_preprocessor, transform, Preprocessor}``
- ``src.imbalance.{compute_class_weights, undersample_majority}``
- ``src.graph_build.build_graph``      — dựng PyG Data.
- ``src.model.build_model``            — 'egraphsage' | 'gcn' | 'graphsage'
                                          | 'sage_edge_concat' | 'gat'.
- ``src.train.{set_seed, get_device, make_criterion}``
- ``src.evaluate.plot_confusion_matrix``

Components
----------
1.  ``load_all_scenarios(paths, cap_per_class, chunksize)``
2.  ``fit_shared_preprocessor(train_dfs)``
3.  ``build_shared_class_to_idx(all_dfs)``
4.  ``build_scenario_graphs(dfs, preprocessor, class_to_idx)``
5.  ``run_loso(scenario_paths, config_path, model_name, imbalance_mode,
              cap_per_class, chunksize, epochs_override, seed, out_dir,
              verbose)``

Cảnh báo giới hạn inductive
----------------------------
Nếu held-out scenario có lớp KHÔNG xuất hiện trong train (lớp "private" của
held-out), F1 cho lớp đó = 0 — đây là bản chất của LOSO inductive, KHÔNG
phải lỗi. Số lớp private được ghi vào cột ``n_unseen_in_train`` để báo cáo
trong suốt.

Lưu ý thiết bị & reproducibility
--------------------------------
- Device-agnostic: ``device = 'cuda' if torch.cuda.is_available() else 'cpu'``.
- Seed numpy + torch + random ngay đầu hàm.
- Cùng ``seed`` → cùng ``val_mask``, cùng init, cùng split.
- Smoke test trên CPU (Mac M2 Pro); train thật trên vast.ai GPU.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch_geometric.data import Data
from src.train import safe_stratified_split

# ---- sys.path setup ------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ---- Matplotlib: Agg backend (CI / CPU safe, evaluate.py dùng cùng) ------
import matplotlib

matplotlib.use("Agg")  # noqa: E402


__all__ = [
    "load_all_scenarios",
    "fit_shared_preprocessor",
    "build_shared_class_to_idx",
    "build_scenario_graphs",
    "run_loso",
]


logger = logging.getLogger(__name__)


# ============================================================================
# 1. load_all_scenarios
# ============================================================================

def _read_chunked_with_cap(
    path: str,
    cap_per_class: int,
    chunksize: int,
) -> pd.DataFrame:
    """
    Đọc file ``conn.log.labeled`` THEO CHUNK + cap per-class ngay khi đọc để
    không bao giờ giữ cả file trong RAM (vd 39-1 10GB).

    Tái sử dụng:
        •  ``src.data_io.split_label_column``  — tách cột label.
        •  ``src.preprocess.clean_flows``      — pipeline làm sạch.

    Tham số
    -------
    path : str
        Đường dẫn file.
    cap_per_class : int
        Số dòng TỐI ĐA giữ lại cho mỗi ``detailed-label``.
    chunksize : int
        Số dòng mỗi chunk (``pd.read_csv(chunksize=...)``).
    """
    # Local import để tránh vòng nếu multi_scenario được import sớm.
    from src.data_io import split_label_column
    from src.preprocess import clean_flows

    # ---- Parse tên cột từ '#fields' (giống data_io.read_conn_log) ----
    canonical_map = {
        "det_label": "detailed-label",
        "detailed_label": "detailed-label",
        "label_val": "label",
    }
    field_names: Optional[List[str]] = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#fields"):
                parts = line.rstrip("\n").split("\t")
                field_names = [canonical_map.get(p, p) for p in parts[1:]]
                break
    if field_names is None:
        raise ValueError(
            f"_read_chunked_with_cap: không tìm thấy '#fields' trong {path}."
        )

    # ---- Stream chunks + cap per-class ----
    per_class_buffers: List[pd.DataFrame] = []
    per_class_count: Dict[str, int] = {}

    reader = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        header=None,
        names=field_names,
        na_values=[],
        keep_default_na=False,
        skip_blank_lines=True,
        dtype=str,
        engine="python",
        on_bad_lines="skip",
        chunksize=chunksize,
    )

    for chunk_df in reader:
        chunk_df = split_label_column(chunk_df)
        chunk_df = clean_flows(chunk_df)
        for label, group in chunk_df.groupby("detailed-label"):
            already = per_class_count.get(label, 0)
            remaining = cap_per_class - already
            if remaining <= 0:
                continue
            if len(group) > remaining:
                group = group.sample(n=remaining, random_state=42)
            per_class_buffers.append(group)
            per_class_count[label] = already + len(group)

    if not per_class_buffers:
        logger.warning(
            "_read_chunked_with_cap: %s rỗng sau khi cap (cap=%d).",
            path, cap_per_class,
        )
        return pd.DataFrame(columns=field_names)
    out = pd.concat(per_class_buffers, axis=0, ignore_index=True)
    logger.info(
        "_read_chunked_with_cap: %s → %d dòng × %d cột (cap=%d).",
        path, out.shape[0], out.shape[1], cap_per_class,
    )
    return out


def load_all_scenarios(
    paths: Dict[str, str],
    cap_per_class: Optional[int] = None,
    chunksize: int = 200_000,
    *,
    cache_enabled: Optional[bool] = None,
    cache_dir: str = "artifacts/data_cache",
    rebuild_cache: bool = False,
    cache_format: str = "parquet",
    load_reports: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Đọc + clean MỌI scenario; trả ``dict { name → df_clean }``.

    Hai chế độ
    ----------
    - ``cap_per_class is None``: file nhỏ → đọc nguyên bằng ``load_scenario``
      + ``clean_flows`` (nhanh, đơn giản).
    - ``cap_per_class`` được đặt: file lớn → đọc THEO CHUNK và cap per-class
      NGAY KHI ĐỌC, không bao giờ giữ cả file trong RAM.

    Parameters
    ----------
    paths : dict
        ``{scenario_name: path_to_conn.log.labeled}``.
    cap_per_class : int | None
        Số flow TỐI ĐA mỗi ``detailed-label`` (áp cho MỌI scenario).
        ``None`` = đọc nguyên.
    chunksize : int
        Số dòng mỗi chunk khi cap.
    cache_enabled : bool | None
        ``None`` giữ nguyên historical loader. ``True`` dùng canonical
        scenario cache; ``False`` dùng cùng canonical parser nhưng không lưu.
        Clean Phase 1 truyền cờ này tường minh, các caller cũ không đổi.
    cache_dir, rebuild_cache, cache_format
        Điều khiển cache canonical pre-split. Cache không chứa fitted state.
    load_reports : list | None
        Nếu được truyền, append báo cáo HIT/MISS, tốc độ và raw-open count.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mỗi df đã qua ``clean_flows``.
    """
    # Local import — tránh vòng.
    from src.data_io import load_scenario
    from src.preprocess import clean_flows

    if not paths:
        raise ValueError("load_all_scenarios: paths rỗng.")
    out: Dict[str, pd.DataFrame] = {}
    for name, path in paths.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"load_all_scenarios: scenario '{name}' thiếu file {path}."
            )
        if cache_enabled is not None:
            from src.phase1_data_cache import load_canonical_scenario

            df_clean, report = load_canonical_scenario(
                path,
                name,
                cache_dir=cache_dir,
                cache_enabled=cache_enabled,
                rebuild_cache=rebuild_cache,
                cache_format=cache_format,
                cap_per_class=cap_per_class,
                chunksize=chunksize,
            )
            if load_reports is not None:
                load_reports.append(report.to_dict())
        elif cap_per_class is None:
            df_clean = clean_flows(load_scenario(path))
            logger.info(
                "load_all_scenarios: %s (whole-file) → %d dòng.",
                name, df_clean.shape[0],
            )
        else:
            df_clean = _read_chunked_with_cap(
                path, cap_per_class=cap_per_class, chunksize=chunksize,
            )
        out[name] = df_clean
    return out


# ============================================================================
# 2. fit_shared_preprocessor
# ============================================================================

def fit_shared_preprocessor(
    train_dfs: List[pd.DataFrame],
) -> Any:  # returns src.preprocess.Preprocessor
    """
    GỘP các df train (chỉ dùng cho FIT), rồi gọi ``fit_preprocessor``.

    CHỈ fit trên các scenario dùng để train (KHÔNG chạm held-out).
    Mọi scenario dùng chung preprocessor này → cùng ``feature_dim``.

    Returns
    -------
    Preprocessor (đối tượng từ ``src.preprocess``).
    """
    from src.preprocess import fit_preprocessor

    if not train_dfs:
        raise ValueError("fit_shared_preprocessor: train_dfs rỗng.")
    df_g = pd.concat(train_dfs, axis=0, ignore_index=True)
    logger.info(
        "fit_shared_preprocessor: concat %d train dfs → %d dòng; fitting...",
        len(train_dfs), df_g.shape[0],
    )
    return fit_preprocessor(df_g)


# ============================================================================
# 3. build_shared_class_to_idx
# ============================================================================

def build_shared_class_to_idx(
    all_dfs: Dict[str, pd.DataFrame],
) -> Dict[Any, int]:
    """
    ``class_to_idx`` = hợp mọi giá trị ``detailed-label`` trên TẤT CẢ
    scenario (kể cả held-out), sort theo tên → index ổn định.

    Cũng in ma trận hiện diện ``class × scenario`` để thấy lớp hiếm /
    lớp chỉ xuất hiện ở 1 scenario.
    """
    class_to_scenario: Dict[str, set] = {}
    for name, df in all_dfs.items():
        if "detailed-label" not in df.columns:
            raise KeyError(
                f"build_shared_class_to_idx: df '{name}' thiếu "
                f"cột 'detailed-label'."
            )
        labels = sorted(set(df["detailed-label"].astype(str).unique()))
        for lbl in labels:
            class_to_scenario.setdefault(lbl, set()).add(name)

    classes = sorted(class_to_scenario.keys())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # ---- In ma trận hiện diện ----
    scenario_names = sorted(all_dfs.keys())
    n_sc = len(scenario_names)
    name_w = max((len(n) for n in scenario_names), default=8)
    print("=" * 70)
    print(" build_shared_class_to_idx — MA TRẬN HIỆN DIỆN LỚP × SCENARIO")
    print("=" * 70)
    print(f"  Tổng số lớp union: {len(classes)}  (K = {len(classes)})")
    print(f"  Số scenario: {n_sc}")
    print()
    header_cells = [f"{n[:name_w]:>{name_w}s}" for n in scenario_names]
    print(f"  {'class':<35s}  " + "  ".join(header_cells))
    print("  " + "-" * (35 + 2 + (name_w + 2) * n_sc))
    for c in classes:
        cells = [
            f"{'✓':>{name_w}s}" if n in class_to_scenario[c] else f"{'.':>{name_w}s}"
            for n in scenario_names
        ]
        print(f"  {c:<35s}  " + "  ".join(cells))
    print("=" * 70)
    print(f"  class_to_idx = {class_to_idx}")
    print()

    return class_to_idx


# ============================================================================
# 4. build_scenario_graphs
# ============================================================================

def build_scenario_graphs(
    dfs: Dict[str, pd.DataFrame],
    preprocessor: Any,
    class_to_idx: Dict[Any, int],
) -> Dict[str, Data]:
    """
    Transform MỌI df bằng preprocessor CHUNG → ``build_graph`` với
    ``class_to_idx`` CHUNG → Data. Assert mọi graph cùng ``feature_dim``
    và ``num_classes``.
    """
    from src.graph_build import build_graph
    from src.preprocess import transform

    out: Dict[str, Data] = {}
    for name, df in dfs.items():
        df_feat = transform(df, preprocessor)
        data = build_graph(
            df_feat,
            class_to_idx=class_to_idx,
            feature_columns=preprocessor.feature_columns,
        )
        out[name] = data
        logger.info(
            "build_scenario_graphs: %s → N=%d, E=%d, F=%d, K=%d.",
            name, int(data.num_nodes), int(data.edge_index.shape[1]),
            int(data.feature_dim), int(data.num_classes),
        )

    fd = {int(d.feature_dim) for d in out.values()}
    nc = {int(d.num_classes) for d in out.values()}
    if len(fd) != 1:
        raise ValueError(
            f"build_scenario_graphs: feature_dim lệch giữa các scenario: {fd}."
        )
    if len(nc) != 1:
        raise ValueError(
            f"build_scenario_graphs: num_classes lệch giữa các scenario: {nc}."
        )
    return out


# ============================================================================
# 5. run_loso — harness chính
# ============================================================================

def _compute_val_mask(
    edge_label: torch.Tensor,
    val_ratio: float,
    seed: int,
) -> torch.Tensor:
    """
    10% val mask [E] dùng cho early stopping (val).

    Tách 90/10 stratified theo ``edge_label``. Chịu được lớp cực hiếm
    (vd Okiru) — nếu có lớp < 2 mẫu, fallback random split + in cảnh báo
    (xem ``safe_stratified_split`` trong ``src.train``).
    """
    y = edge_label.detach().cpu().numpy()
    idx_all = np.arange(len(y))
    E = len(y)

    # Tìm singleton trong edge_label (count == 1) → ÉP vào phần TRAIN
    # (chỉ có 1 chỗ để đi, không thể vào val 10%).
    unique_all, counts_all = np.unique(y, return_counts=True)
    singleton_classes = unique_all[counts_all == 1]
    singleton_indices = np.where(np.isin(y, singleton_classes))[0]
    pool_mask = ~np.isin(idx_all, singleton_indices)
    idx_pool = idx_all[pool_mask]
    y_pool = y[pool_mask]

    # Tách 90/10 trên pool. idx_first (train) bị DISCARD; idx_second (val)
    # là cái ta cần.
    _, idx_val = safe_stratified_split(
        idx_pool,
        y_pool,
        test_size=val_ratio,
        seed=seed,
        context=f"_compute_val_mask (E={E})",
        force_into_first=singleton_indices,
    )

    val_mask = torch.zeros(E, dtype=torch.bool)
    val_mask[idx_val] = True
    return val_mask


def _attach_val_masks(
    graphs: Dict[str, Data],
    val_ratio: float,
    seed: int,
) -> None:
    """Gắn ``g.val_mask`` (10% stratified) cho MỖI graph trong dict (in-place)."""
    for g in graphs.values():
        g.val_mask = _compute_val_mask(g.edge_label, val_ratio=val_ratio, seed=seed)


def run_loso(
    scenario_paths: Dict[str, str],
    config_path: str,
    model_name: str = "egraphsage",
    imbalance_mode: str = "class_weight",
    cap_per_class: Optional[int] = None,
    chunksize: int = 200_000,
    epochs_override: Optional[int] = None,
    patience_override: Optional[int] = None,
    val_ratio: float = 0.10,
    seed: Optional[int] = None,
    out_dir: str = "artifacts/loso",
    verbose: bool = True,
    preloaded_clean_dfs: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """
    Leave-One-Scenario-Out inductive: train trên N-1 scenario, test trên
    scenario thứ N (chưa thấy).

    Mỗi scenario làm held-out đúng 1 lần.

    Pipeline cho mỗi held-out ``s``
    --------------------------------
    1.  Load + clean tất cả scenarios (train + held-out).
    2.  ``class_to_idx`` = hợp mọi nhãn (kể cả held-out). Warn nếu held-out
        có lớp không có trong train → F1 lớp đó = 0 (giới hạn inductive).
    3.  ``Preprocessor`` fit trên union TRAIN (KHÔNG chạm held-out).
    4.  Transform mọi df; build full Data cho MỖI scenario.
        - mode='undersample'  → build thêm 1 graph đã undersample per scenario
                                  cho TRAIN (val_mask vẫn trên FULL graph).
        - mode='class_weight' → weight tensor build từ union TRAIN labels
                                  (lớp unseen trong held-out → weight=0).
    5.  Build model từ MỘT graph train bất kỳ (chúng cùng schema nhờ tiền xử lý
        chung).
    6.  Train N epoch (mặc định đọc từ ``config.yaml``):
        - Mỗi epoch: ``zero_grad`` → forward từng train graph → loss trên
          TOÀN BỘ cạnh gốc → ``backward`` (gradient CỘNG dồn) → ``step``.
        - Sau train: eval trên union val_mask của train FULL graphs (10% mỗi
          graph, stratified); chọn checkpoint best theo val macro-F1.
    7.  Eval trên held-out: forward TOÀN BỘ cạnh của graph held-out (inference
        trên scenario CHƯA THẤY trong train).
    8.  Ghi 1 dòng kết quả: held_out, macro_F1, per-class F1, n_unseen.

    Cuối cùng
    ---------
    - Trả ``DataFrame`` (mỗi dòng = 1 held-out) + dòng ``MEAN``.
    - Lưu CSV ``loso_<model>_<mode>.csv`` vào ``out_dir``.
    - Vẽ confusion matrix của held-out KHÓ NHẤT (macro-F1 thấp nhất)
      → ``confusion_matrix_loso_<model>_<mode>_hardest_<s>.png``.

    Parameters
    ----------
    scenario_paths : dict
        ``{name: path_to_conn.log.labeled}``.
    config_path : str
        Đường dẫn ``config.yaml``.
    model_name : str
        Một trong: ``'egraphsage'|'gcn'|'graphsage'|'sage_edge_concat'|'gat'``.
    imbalance_mode : str
        ``'none' | 'class_weight' | 'undersample'``.
    cap_per_class : int | None
        Cap mỗi lớp khi load (giúp RAM ổn với file lớn như 39-1).
    chunksize : int
        Chunk size khi cap_per_class != None.
    epochs_override : int | None
        Override số epoch (mặc định từ ``config.yaml``).
    patience_override : int | None
        Override early-stopping patience (mặc định từ ``config.yaml``
        ``training.early_stop_patience``, fallback 10).
    val_ratio : float
        Tỉ lệ VAL mask per train graph (mặc định 0.10).
    seed : int | None
        Seed; mặc định lấy từ ``config.yaml``.
    out_dir : str
        Thư mục lưu CSV + confusion matrix PNG.
    verbose : bool
    preloaded_clean_dfs : dict | None
        Nếu được cung cấp (từ ``DataCache``), BỎ QUA ``load_all_scenarios``
        và dùng trực tiếp dict này. Tránh đọc + clean lặp lại khi chạy nhiều
        model trên cùng bộ scenarios.

    Returns
    -------
    pd.DataFrame
        DataFrame với dòng cuối là ``MEAN``.
    """
    from src.evaluate import plot_confusion_matrix
    from src.graph_build import build_graph
    from src.imbalance import (
        compute_class_weights,
        undersample_majority,
    )
    from src.model import build_model
    from src.preprocess import transform
    from src.train import get_device, make_criterion, set_seed

    # ---- Validate ----
    valid_models = {"egraphsage", "gcn", "graphsage", "sage_edge_concat", "gat"}
    if model_name not in valid_models:
        raise ValueError(
            f"run_loso: model_name='{model_name}' không hỗ trợ. "
            f"Chọn một trong: {sorted(valid_models)}."
        )
    valid_modes = {"none", "class_weight", "undersample"}
    if imbalance_mode not in valid_modes:
        raise ValueError(
            f"run_loso: imbalance_mode='{imbalance_mode}' không hỗ trợ. "
            f"Chỉ chấp nhận: {sorted(valid_modes)}."
        )
    if not scenario_paths:
        raise ValueError("run_loso: scenario_paths rỗng.")
    if len(scenario_paths) < 2:
        raise ValueError(
            "run_loso: LOSO cần ≥ 2 scenarios (held-out + train mới có ý nghĩa)."
        )

    # ---- cfg + seed ----
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if seed is None:
        seed = int(cfg.get("reproducibility", {}).get("seed", 42))

    set_seed(seed)
    device = get_device()

    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    scenario_names = sorted(scenario_paths.keys())

    if verbose:
        print("=" * 70)
        print(
            f" run_loso  ·  {len(scenario_names)} scenarios  ·  "
            f"model={model_name}  ·  mode={imbalance_mode}"
        )
        print("=" * 70)
        print(f"  seed         : {seed}")
        print(f"  device       : {device}")
        print(f"  cap_per_class: {cap_per_class}")
        print(f"  val_ratio    : {val_ratio}")
        print(f"  scenarios    : {scenario_names}")
        print(f"  out_dir      : {out_dir}")
        if epochs_override is not None:
            print(f"  epochs_override: {epochs_override}")
        print()

    # ---- Load + clean tất cả scenario MỘT LẦN ----
    if preloaded_clean_dfs is not None:
        all_dfs = preloaded_clean_dfs
        if verbose:
            print(
                f"[LOSO] Dùng preloaded clean dfs ({len(all_dfs)} scenarios, "
                f"KHÔNG load lại từ đĩa)."
            )
            for n in scenario_names:
                df = all_dfs[n]
                print(
                    f"  {n:<35s}  {df.shape[0]:>8,} dòng   "
                    f"({df['detailed-label'].nunique()} lớp)"
                )
            print()
    else:
        t_load0 = time.perf_counter()
        all_dfs = load_all_scenarios(
            scenario_paths,
            cap_per_class=cap_per_class,
            chunksize=chunksize,
        )
        if verbose:
            print(
                f"[load_all_scenarios] {len(all_dfs)} scenarios trong "
                f"{time.perf_counter() - t_load0:.2f}s."
            )
            for n in scenario_names:
                df = all_dfs[n]
                print(
                    f"  {n:<35s}  {df.shape[0]:>8,} dòng   "
                    f"({df['detailed-label'].nunique()} lớp)"
                )
            print()

    # ---- class_to_idx CHUNG (một lần duy nhất) ----
    shared_class_to_idx = build_shared_class_to_idx(all_dfs)
    K = len(shared_class_to_idx)
    if verbose:
        print(f"  K = {K} (cố định cho MỌI held-out round)")

    # ---- 'class_weight': compute weights từ UNION TRAIN LABELS ----
    # (Phải tính trước vòng held-out; vẫn chỉ dùng labels của TRAIN scenarios
    # — KHÔNG peek held-out.)
    union_train_label_list: Optional[List[str]] = None
    if imbalance_mode == "class_weight":
        union_train_label_list = []
        # NOTE: tổng hợp labels ở đây sẽ dùng cho MỌI held-out round (vì
        # thứ tự train thay đổi nhưng union_train_labels về bản chất không
        # phụ thuộc held-out cụ thể). Tuy nhiên để an toàn, ta tính lại
        # theo từng held-out bên dưới — đảm bảo không leak held-out nào.

    # ---- Vòng LOSO ----
    records: List[Dict[str, Any]] = []
    n_params_total: Optional[int] = None

    for held_out in scenario_names:
        t_round0 = time.perf_counter()
        train_names = [n for n in scenario_names if n != held_out]

        if verbose:
            print()
            print("#" * 70)
            print(f"# LOSO round  ·  held_out = {held_out}")
            print(f"#              train     = {train_names}")
            print("#" * 70)

        # ---- 1) Fit shared preprocessor trên TRAIN ----
        train_dfs = {n: all_dfs[n] for n in train_names}
        shared_pre = fit_shared_preprocessor([train_dfs[n] for n in train_names])

        # ---- 2) Cảnh báo lớp private (chỉ có ở held-out) ----
        train_classes = set()
        for n in train_names:
            train_classes.update(
                all_dfs[n]["detailed-label"].astype(str).unique()
            )
        held_classes = set(all_dfs[held_out]["detailed-label"].astype(str).unique())
        unseen_in_train = sorted(held_classes - train_classes)
        if unseen_in_train:
            logger.warning(
                "LOSO [%s]: held-out có %d lớp KHÔNG có trong train: %s. "
                "F1 những lớp này = 0 (giới hạn inductive, KHÔNG phải lỗi).",
                held_out, len(unseen_in_train), unseen_in_train,
            )
            if verbose:
                print(
                    f"  [CẢNH BÁO] held-out có {len(unseen_in_train)} lớp "
                    f"'private' (chỉ xuất hiện ở held-out): {unseen_in_train}"
                )

        # ---- 3) Transform + build FULL graph cho mỗi scenario ----
        train_graphs_full: Dict[str, Data] = {}
        for n in train_names:
            df_feat = transform(train_dfs[n], shared_pre)
            g_full = build_graph(
                df_feat,
                class_to_idx=shared_class_to_idx,
                feature_columns=shared_pre.feature_columns,
            )
            train_graphs_full[n] = g_full

        # Held-out graph.
        df_held_feat = transform(all_dfs[held_out], shared_pre)
        held_graph = build_graph(
            df_held_feat,
            class_to_idx=shared_class_to_idx,
            feature_columns=shared_pre.feature_columns,
        )

        # ---- 4) Val mask per FULL train graph (10% stratified) ----
        _attach_val_masks(train_graphs_full, val_ratio=val_ratio, seed=seed)

        # ---- 5) mode='undersample': build thêm UNDERSAMPLED graph cho TRAIN ----
        train_graphs: Dict[str, Data] = {}
        if imbalance_mode == "undersample":
            for n in train_names:
                df_feat = transform(train_dfs[n], shared_pre)
                df_under = undersample_majority(
                    df_feat,
                    strategy="to_second_largest",
                    random_state=seed,
                    verbose=False,
                )
                g_under = build_graph(
                    df_under,
                    class_to_idx=shared_class_to_idx,
                    feature_columns=shared_pre.feature_columns,
                )
                # val_mask KHÔNG dùng được trên g_under (khác edge indices).
                train_graphs[n] = g_under
        else:
            for n in train_names:
                train_graphs[n] = train_graphs_full[n]

        # ---- 6) mode='class_weight': weight tensor từ UNION TRAIN LABELS ----
        weight_tensor: Optional[torch.Tensor] = None
        if imbalance_mode == "class_weight":
            # QUAN TRỌNG: ``compute_class_weights`` yêu cầu TÊN LỚP (str)
            # làm key, không phải chỉ số đã encode. ``edge_label`` của PyG
            # Data là int; phải map NGƯỢC qua ``shared_class_to_idx`` để lấy
            # tên lớp gốc trước khi tính weight.
            inv_cti: Dict[int, str] = {
                int(v): str(k) for k, v in shared_class_to_idx.items()
            }
            train_labels: List[str] = []
            for n in train_names:
                # Dùng labels của FULL train graph (không undersample).
                idx_arr = train_graphs_full[n].edge_label.cpu().numpy()
                train_labels.extend(inv_cti[int(i)] for i in idx_arr)
            weights_dict, _, _ = compute_class_weights(
                train_labels, scheme="balanced",
            )
            weight_tensor = torch.zeros(K, dtype=torch.float32)
            for c, idx in shared_class_to_idx.items():
                if c in weights_dict:
                    weight_tensor[int(idx)] = float(weights_dict[c])
                else:
                    # Lớp có trong shared_class_to_idx nhưng không có trong
                    # train (chỉ xuất hiện ở held-out) → weight=0.
                    weight_tensor[int(idx)] = 0.0
            if verbose:
                nonzero = int((weight_tensor > 0).sum().item())
                print(
                    f"  [class_weight] {nonzero}/{K} lớp có weight > 0; "
                    f"weight_tensor = {[round(float(w), 3) for w in weight_tensor]}"
                )

        # ---- 7) Build model (từ 1 graph bất kỳ — chúng cùng schema) ----
        set_seed(seed)  # đảm bảo init trọng số ổn định giữa các held-out
        any_train_g = next(iter(train_graphs_full.values()))
        model = build_model(model_name, any_train_g, cfg).to(device)
        n_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        if n_params_total is None:
            n_params_total = n_params
            if verbose:
                print(
                    f"\n[model] {model_name} — {n_params:,} tham số "
                    f"(dùng chung cho MỌI held-out round)."
                )

        # ---- 8) Optimizer + criterion ----
        tr_cfg = cfg.get("training", {}) if isinstance(cfg, dict) else {}
        lr = float(tr_cfg.get("learning_rate", 1e-3))
        wd = float(tr_cfg.get("weight_decay", 0.0))
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=wd,
        )
        grad_clip = float(tr_cfg.get("grad_clip", 1.0))
        epochs = int(
            epochs_override
            if epochs_override is not None
            else tr_cfg.get("epochs", 50)
        )
        patience = int(
            patience_override
            if patience_override is not None
            else tr_cfg.get("early_stop_patience", 10)
        )

        criterion = make_criterion(
            imbalance_mode, weight_tensor=weight_tensor, device=device,
        )

        # ---- 9) Đưa graph lên device MỘT LẦN ----
        train_graphs_dev = {n: g.to(device) for n, g in train_graphs.items()}
        train_graphs_full_dev = {
            n: g.to(device) for n, g in train_graphs_full.items()
        }
        held_dev = held_graph.to(device)

        # ---- 10) Training loop ----
        best_val_f1 = -1.0
        best_epoch = -1
        bad_epochs = 0
        best_state: Optional[Dict[str, torch.Tensor]] = None
        history = {
            "epoch": [],
            "train_loss": [],
            "val_macro_f1": [],
        }

        for epoch in range(1, epochs + 1):
            # ---- Train 1 epoch: sum loss qua các train graph; 1 optimizer step ----
            model.train()
            optimizer.zero_grad()
            total_loss = 0.0
            for g in train_graphs_dev.values():
                logits = model(g)
                loss = criterion(logits, g.edge_label)
                loss.backward()
                total_loss += float(loss.item())
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            avg_loss = total_loss / max(len(train_graphs_dev), 1)

            # ---- Val: union val_mask của FULL train graphs ----
            model.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for g in train_graphs_full_dev.values():
                    mask = g.val_mask
                    logits = model(g)
                    val_preds.append(logits[mask].argmax(dim=-1).cpu())
                    val_labels.append(g.edge_label[mask].cpu())
            yt_val = torch.cat(val_labels).numpy()
            yp_val = torch.cat(val_preds).numpy()
            val_macro_f1 = float(
                f1_score(
                    yt_val, yp_val,
                    labels=list(range(K)), average="macro", zero_division=0,
                )
            )

            history["epoch"].append(epoch)
            history["train_loss"].append(avg_loss)
            history["val_macro_f1"].append(val_macro_f1)

            if verbose and (
                epoch == 1 or epoch % 5 == 0 or epoch == epochs
            ):
                tag = " *" if val_macro_f1 > best_val_f1 else ""
                print(
                    f"  epoch {epoch:>3d}/{epochs}  "
                    f"train_loss={avg_loss:.4f}  "
                    f"val_macroF1={val_macro_f1:.4f}{tag}"
                )

            if val_macro_f1 > best_val_f1:
                best_val_f1 = val_macro_f1
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                bad_epochs = 0
            else:
                bad_epochs += 1

            if bad_epochs >= patience:
                if verbose:
                    print(
                        f"  [LOSO|{held_out}|{model_name}|{imbalance_mode}]  "
                        f"early-stop tại epoch {epoch} "
                        f"(best_val={best_val_f1:.4f} @ epoch {best_epoch}, "
                        f"patience={patience})."
                    )
                break

        # ---- Restore best ----
        if best_state is not None:
            model.load_state_dict(
                {k: v.to(device) for k, v in best_state.items()}
            )

        # ---- 11) Test trên HELD-OUT (toàn bộ cạnh, unseen) ----
        model.eval()
        with torch.no_grad():
            logits_held = model(held_dev)
            preds_held = logits_held.argmax(dim=-1).cpu().numpy()
            labels_held = held_dev.edge_label.cpu().numpy()

        test_macro = float(
            f1_score(
                labels_held, preds_held,
                labels=list(range(K)), average="macro", zero_division=0,
            )
        )
        test_weighted = float(
            f1_score(
                labels_held, preds_held,
                labels=list(range(K)), average="weighted", zero_division=0,
            )
        )
        test_acc = float((preds_held == labels_held).mean())
        per_class_arr = f1_score(
            labels_held, preds_held,
            labels=list(range(K)), average=None, zero_division=0,
        )
        cm = confusion_matrix(
            labels_held, preds_held, labels=list(range(K)),
        )

        # Tên lớp theo shared_class_to_idx (đã sort).
        target_names: List[Optional[str]] = [None] * K
        for name, idx in shared_class_to_idx.items():
            target_names[int(idx)] = str(name)
        target_names_filled = [
            n if n is not None else f"class_{i}"
            for i, n in enumerate(target_names)
        ]

        record: Dict[str, Any] = {
            "held_out": held_out,
            "macro_f1": test_macro,
            "weighted_f1": test_weighted,
            "accuracy": test_acc,
            "best_epoch": best_epoch,
            "best_val_f1": best_val_f1,
            "n_unseen_in_train": len(unseen_in_train),
            "_cm": cm,
            "_target_names": target_names_filled,
        }
        for cn in target_names_filled:
            idx = shared_class_to_idx[cn]
            record[f"f1_{cn}"] = float(per_class_arr[int(idx)])
            record[f"support_{cn}"] = int((labels_held == int(idx)).sum())
        records.append(record)

        if verbose:
            print(
                f"\n[EVAL on '{held_out}' (held-out, unseen)]\n"
                f"  macro_F1   = {test_macro:.4f}      ← chỉ số chính\n"
                f"  weighted_F1= {test_weighted:.4f}\n"
                f"  accuracy   = {test_acc:.4f}      ← THAM KHẢO (lệch lớp)\n"
                f"  best @ epoch {best_epoch} (val_macro_F1={best_val_f1:.4f})\n"
                f"  unseen_in_train = {len(unseen_in_train)} lớp: "
                f"{unseen_in_train if unseen_in_train else '∅'}"
            )
            print(
                f"\n[round {held_out}] {time.perf_counter() - t_round0:.1f}s"
            )

    # ---- Tổng hợp DataFrame + dòng MEAN ----
    public_rows = [
        {k: v for k, v in r.items() if not k.startswith("_")}
        for r in records
    ]
    df = pd.DataFrame(public_rows)

    mean_row: Dict[str, Any] = {"held_out": "MEAN"}
    for col in df.columns:
        if col == "held_out":
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            mean_row[col] = float(df[col].mean())
    df_with_mean = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    # ---- Save CSV ----
    csv_path = os.path.join(
        out_dir, f"loso_{model_name}_{imbalance_mode}.csv"
    )
    df_with_mean.to_csv(csv_path, index=False)
    logger.info("Đã lưu LOSO CSV: %s", csv_path)

    # ---- Confusion matrix PNG cho held-out KHÓ NHẤT (macro-F1 thấp nhất) ----
    hardest = min(records, key=lambda r: r["macro_f1"])
    png_path = os.path.join(
        out_dir,
        f"confusion_matrix_loso_{model_name}_{imbalance_mode}"
        f"_hardest_{hardest['held_out']}.png",
    )
    plot_confusion_matrix(
        hardest["_cm"],
        class_names=hardest["_target_names"],
        save_path=png_path,
        title=(
            f"LOSO hardest — held_out={hardest['held_out']}  "
            f"macro_F1={hardest['macro_f1']:.4f}  "
            f"(model={model_name}, mode={imbalance_mode})"
        ),
    )

    if verbose:
        print()
        print("=" * 70)
        print(" LOSO — BẢNG TỔNG HỢP (sort theo thứ tự held_out)")
        print("=" * 70)
        with pd.option_context(
            "display.max_columns", None,
            "display.width", 220,
            "display.float_format", "{:.4f}".format,
        ):
            print(df_with_mean.to_string(index=False))
        print()
        print("=" * 70)
        print(" ARTIFACTS")
        print("=" * 70)
        print(f"  CSV : {csv_path}")
        print(f"  PNG : {png_path}")
        print(f"  dir : {out_dir}/")

    return df_with_mean


# ============================================================================
# CLI
# ============================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Leave-One-Scenario-Out (LOSO) inductive: train trên N-1 "
            "scenario IoT-23, test trên scenario thứ N chưa thấy."
        ),
    )
    p.add_argument(
        "--scenarios", type=str, nargs="+", required=True,
        help=(
            "Danh sách scenario theo cặp name=PATH, vd "
            "--scenarios 34-1=data/CTU-IoT-Malware-Capture-34-1/conn.log.labeled "
            "3-1=data/CTU-IoT-Malware-Capture-3-1/conn.log.labeled"
        ),
    )
    p.add_argument(
        "--config", type=str, default="config.yaml",
        help="Đường dẫn config.yaml (mặc định: config.yaml).",
    )
    p.add_argument(
        "--model", type=str, default="egraphsage",
        choices=["egraphsage", "gcn", "graphsage", "sage_edge_concat", "gat"],
        help="Loại model (mặc định: egraphsage).",
    )
    p.add_argument(
        "--imbalance", type=str, default="class_weight",
        choices=["none", "class_weight", "undersample"],
        help="Cách xử lý mất cân bằng (mặc định: class_weight).",
    )
    p.add_argument(
        "--cap-per-class", type=int, default=None,
        help=(
            "Cap số flow mỗi lớp khi load (giúp RAM ổn với file lớn như 39-1). "
            "Mặc định None = đọc nguyên."
        ),
    )
    p.add_argument(
        "--chunksize", type=int, default=200_000,
        help="Số dòng mỗi chunk khi --cap-per-class != None (mặc định 200,000).",
    )
    p.add_argument(
        "--val-ratio", type=float, default=0.10,
        help="Tỉ lệ VAL mask per train graph (mặc định 0.10).",
    )
    p.add_argument(
        "--epochs", type=int, default=None,
        help="Override số epoch (mặc định: từ config.yaml).",
    )
    p.add_argument(
        "--patience", type=int, default=None,
        help=(
            "Override early-stopping patience "
            "(mặc định: training.early_stop_patience trong config.yaml)."
        ),
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Override seed (mặc định: từ config.yaml).",
    )
    p.add_argument(
        "--out-dir", type=str, default="artifacts/loso",
        help="Thư mục lưu CSV + confusion matrix PNG (mặc định artifacts/loso/).",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Tắt log per-epoch (mặc định: in mỗi epoch).",
    )
    return p


def _parse_scenario_arg(items: List[str]) -> Dict[str, str]:
    """Parse ['name=PATH', ...] thành dict {name: PATH}."""
    out: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"sai định dạng --scenarios item: '{item}' (cần name=PATH)."
            )
        name, path = item.split("=", 1)
        out[name.strip()] = path.strip()
    return out


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    scenario_paths = _parse_scenario_arg(args.scenarios)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("multi_scenario")

    run_loso(
        scenario_paths=scenario_paths,
        config_path=args.config,
        model_name=args.model,
        imbalance_mode=args.imbalance,
        cap_per_class=args.cap_per_class,
        chunksize=args.chunksize,
        val_ratio=args.val_ratio,
        epochs_override=args.epochs,
        patience_override=args.patience,
        seed=args.seed,
        out_dir=args.out_dir,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
