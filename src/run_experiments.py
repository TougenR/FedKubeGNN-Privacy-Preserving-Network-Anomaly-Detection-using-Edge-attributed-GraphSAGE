"""
run_experiments.py — Orchestrator thí nghiệm hoàn chỉnh cho Giai đoạn 1 (Task 1.13+).

Mục đích
--------
GLUE nối các thành phần ĐÃ CÓ:
    • src.model.build_model
    • src.train.{train_model, split_edge_masks, set_seed, get_device, make_criterion,
                  save_checkpoint}
    • src.evaluate.{evaluate_model, plot_confusion_matrix}
    • src.multi_scenario.{run_loso, load_all_scenarios, build_shared_class_to_idx,
                            fit_shared_preprocessor}
    • src.imbalance.{compute_class_weights, undersample_majority}
    • src.preprocess.transform
    • src.graph_build.build_graph

KHÔNG viết thuật toán/model mới, KHÔNG sửa logic cũ. Chỉ điều phối +
tổng hợp bảng kết quả cho báo cáo.

CHIẾN LƯỢC 2 PHA (để tiết kiệm GPU, vẫn công bằng)
---------------------------------------------------
* Phase A — CHỌN mode xử lý mất cân bằng: chỉ model ``egraphsage``, chạy cả
  3 mode ``['none', 'class_weight', 'undersample']``. Chọn mode có
  **mean macro-F1** cao nhất làm "mode thắng". In bảng Phase A.
* Phase B — SO SÁNH kiến trúc: cố định ``winning_mode`` từ Phase A, chạy cả
  5 model ``['egraphsage', 'gat', 'sage_edge_concat', 'graphsage', 'gcn']``.
  In bảng Phase B.

CÔNG BẰNG
---------
Mọi lần train dùng CÙNG ``max_epochs`` + CÙNG ``early-stopping patience``
+ CÙNG ``seed`` + CÙNG ``split``. Model đơn giản tự hội tụ sớm & tự dừng,
model phức tạp chạy lâu hơn — ai cũng được train tới hội tụ. KHÔNG cố định
epoch thấp cho baseline. Số epoch thực tế mỗi model đã chạy ghi vào cột
``epochs_ran`` để minh bạch.

3 PROTOCOL ĐÁNH GIÁ (chạy cả 3, mỗi protocol ra 1 bộ bảng Phase A + B)
---------------------------------------------------------------------
1. ``'per_scenario'``: với mỗi scenario, dùng edge-mask transductive
   (``split_edge_masks`` của train.py), train+eval riêng từng scenario,
   báo cáo bảng per-scenario + dòng MEAN.
2. ``'pooled'``: gộp cạnh của mọi scenario (mỗi epoch lặp qua các graph,
   loss CỘNG DỒN rồi 1 backward + 1 step); chia edge-mask train/val/test
   trong mỗi graph; eval trên union test_mask. Đây là "mô hình tập trung"
   để sau này so với FL ở Giai đoạn 2.
3. ``'loso'``: dùng ``run_loso`` của multi_scenario.py (inductive,
   held-out từng scenario).

OUTPUT
------
* Mỗi (protocol, phase) ghi CSV riêng vào ``out_dir``:
  - ``phase_a_<protocol>_egraphsage_3modes.csv``
  - ``phase_b_<protocol>_mode-<mode>_5models.csv``
* Một file tổng hợp ``results_summary.csv`` gộp tất cả (cột:
  protocol, phase, scenario, model, imbalance_mode, macro_F1, weighted_F1,
  accuracy, per-class F1…, epochs_ran).
* Confusion matrix PNG cho cấu hình tốt nhất mỗi protocol.
* Best checkpoint mỗi cấu hình lưu vào ``out_dir/checkpoints/``.

DEVICE & REPRODUCIBILITY
------------------------
* Device-agnostic (``cuda`` nếu có, ``cpu`` nếu không).
* Set seed numpy + torch + random trước MỖI config (công bằng).
* Log rõ mỗi cấu hình bắt đầu/kết thúc + thời gian + epochs_ran để ước
  lượng chi phí GPU trên vast.ai ở bước vận hành.

LỚP PRIVATE (LOSO inductive)
----------------------------
Chịu được scenario có lớp private (xuất hiện ở held-out, không có trong
train): KHÔNG crash, ghi F1=0 cho lớp đó, log rõ. Số lớp private được
ghi vào cột ``n_unseen_in_train`` (chỉ xuất hiện trong protocol LOSO).

Lưu ý thiết bị & reproducibility
--------------------------------
Khác với CLAUDE.md mục 2: để smoke test NHANH trên CPU, file này có 1 vòng
train tuỳ biến (cho protocol ``pooled``) tương tự phần gradient-accumulation
trong ``multi_scenario.run_loso``. Mọi thứ khác đều dùng lại các hàm CÓ SẴN.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import confusion_matrix, f1_score

# ---- Setup sys.path + matplotlib Agg backend -------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib

matplotlib.use("Agg")  # noqa: E402


__all__ = [
    "run_all",
    "run_phase_a",
    "run_phase_b",
    "PROTOCOLS",
    "IMBALANCE_MODES",
    "MODEL_POOL",
]


logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

PROTOCOLS: List[str] = ["per_scenario", "pooled", "loso"]
IMBALANCE_MODES: List[str] = ["none", "class_weight", "undersample"]
MODEL_POOL: List[str] = [
    "egraphsage",
    "gat",
    "sage_edge_concat",
    "graphsage",
    "gcn",
]


# ============================================================================
# DataCache — tránh lặp lại load/clean/build khi cùng khóa dữ liệu
# ============================================================================

class DataCache:
    """
    Cache in-memory cho DataFrames sạch và PyG Data đã build.

    **Bài toán**: ``run_per_scenario`` / ``run_pooled`` / ``run_loso`` đều
    gọi ``load_all_scenarios`` (đọc conn.log.labeled + clean) MỖI LẦN chạy
    một config (model, imbalance_mode). Trong thực tế Phase A chạy 3 modes,
    Phase B chạy 5 models trên CÙNG mode → đọc sạch lặp lại 8 lần.
    Đặc biệt scenario 39-1 (10GB) mất hàng chục phút mỗi lần đọc → đây
    là bottleneck chính, không phải training.

    **Cache 2 tầng**:

    1. ``all_dfs`` (clean DataFrames — kết quả ``load_all_scenarios``):
       Key: ``(frozenset(scenario_paths), cap_per_class, chunksize)``.
       Vì tất cả protocol/phases dùng cùng bộ scenarios + cùng cap.

    2. ``graph`` (PyG Data — kết quả transform + undersample + build_graph):
       Key: ``(scenario_name, imbalance_mode, cap_per_class)``.
       Phase B (5 model × cùng winner_mode) dùng cùng graph per scenario.

    KHÔNG lưu disk (in-memory only) — cache mất khi kết thúc tiến trình.
    Không cần gitignore thêm vì không tạo file.
    """

    def __init__(self) -> None:
        self._clean_dfs: Dict[Tuple, Dict[str, Any]] = {}
        self._graphs: Dict[Tuple, Any] = {}
        self._stats: Dict[str, int] = {
            "clean_hit": 0, "clean_miss": 0,
            "graph_hit": 0, "graph_miss": 0,
        }

    # ---- Tier 1: clean DataFrames ----

    def get_clean_dfs(
        self,
        scenario_paths: Dict[str, str],
        cap_per_class: Optional[int],
        chunksize: int,
    ) -> Dict[str, Any]:
        """
        Trả ``Dict[str, pd.DataFrame]`` — clean DataFrames per scenario.

        Nếu đã cache (cùng ``scenario_paths`` + ``cap_per_class`` +
        ``chunksize``) → trả ngay, KHÔNG đọc lại từ đĩa.
        """
        key: Tuple = (
            frozenset(scenario_paths.items()),
            cap_per_class,
            chunksize,
        )
        if key in self._clean_dfs:
            self._stats["clean_hit"] += 1
            print(f"    [CACHE HIT] clean_dfs (cap={cap_per_class}) — "
                  f"bỏ qua load+clean ({len(self._clean_dfs[key])} scenarios).")
            return self._clean_dfs[key]

        self._stats["clean_miss"] += 1
        print(f"    [CACHE MISS] clean_dfs (cap={cap_per_class}) — "
              f"đang load+clean {len(scenario_paths)} scenarios ...")
        t0 = time.perf_counter()
        from src.multi_scenario import load_all_scenarios
        dfs = load_all_scenarios(
            scenario_paths, cap_per_class=cap_per_class, chunksize=chunksize,
        )
        dt = time.perf_counter() - t0
        print(f"    [CACHE MISS] clean_dfs xong trong {dt:.1f}s.")
        self._clean_dfs[key] = dfs
        return dfs

    # ---- Tier 2: PyG Data (graph) ----

    def get_graph(
        self,
        scenario_name: str,
        imbalance_mode: str,
        cap_per_class: Optional[int],
        build_fn: Any,
    ) -> Any:
        """
        Trả ``torch_geometric.data.Data`` đã build.

        ``build_fn`` là callable ``() -> Data`` sẽ được gọi CHỈ KHI cache miss
        (để tránh import phụ thuộc trong class).

        Key = ``(scenario_name, imbalance_mode, cap_per_class)``.
        """
        key: Tuple = (scenario_name, imbalance_mode, cap_per_class)
        if key in self._graphs:
            self._stats["graph_hit"] += 1
            g = self._graphs[key]
            E = int(g.edge_index.shape[1]) if hasattr(g, "edge_index") else 0
            print(f"      [CACHE HIT] graph({scenario_name}, {imbalance_mode}) "
                  f"— E={E}.")
            return g

        self._stats["graph_miss"] += 1
        data = build_fn()
        E = int(data.edge_index.shape[1]) if hasattr(data, "edge_index") else 0
        print(f"      [CACHE MISS] graph({scenario_name}, {imbalance_mode}) "
              f"— E={E} (đã build mới).")
        self._graphs[key] = data
        return data

    # ---- Stats ----

    def print_stats(self) -> None:
        """In thống kê cache hit/miss cuối run."""
        s = self._stats
        total_clean = s["clean_hit"] + s["clean_miss"]
        total_graph = s["graph_hit"] + s["graph_miss"]
        print(
            f"\n  [CACHE STATS] clean_dfs: {s['clean_hit']}/{total_clean} hits "
            f"({s['clean_miss']} misses)  |  "
            f"graph: {s['graph_hit']}/{total_graph} hits "
            f"({s['graph_miss']} misses)"
        )
        if s["clean_miss"] > 0:
            print(
                f"           → {s['clean_miss']} lần load+clean từ đĩa (lần đầu)"
            )
        if s["clean_hit"] > 0:
            print(
                f"           → {s['clean_hit']} lần skip load+clean (cache hit) ✓"
            )

    def clear_graphs(self) -> None:
        """Xóa tier 2 (graph cache) giữ nguyên tier 1. Dùng khi muốn rebuild graph."""
        n = len(self._graphs)
        self._graphs.clear()
        self._stats["graph_hit"] = 0
        self._stats["graph_miss"] = 0
        if n > 0:
            print(f"  [CACHE] Đã xóa {n} graph cache (tier 2).")


# ============================================================================
# Resume helpers — đọc / ghi results_summary.csv incremental
# ============================================================================

def _summary_csv_path(out_dir: str) -> str:
    return os.path.join(out_dir, "results_summary.csv")


def _save_summary_csv(
    summary_records: List[Dict[str, Any]],
    out_dir: str,
) -> str:
    """Ghi ``results_summary.csv`` từ list of dict (overwrite)."""
    df = pd.DataFrame(summary_records)
    path = _summary_csv_path(out_dir)
    df.to_csv(path, index=False)
    return path


def _load_existing_summary(out_dir: str) -> pd.DataFrame:
    """Load ``results_summary.csv`` hiện có; trả DataFrame rỗng nếu chưa có."""
    path = _summary_csv_path(out_dir)
    if not os.path.isfile(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:  # noqa: BLE001
        print(f"  [WARN] Không đọc được {path}: {e}. Bỏ qua, chạy từ đầu.")
        return pd.DataFrame()


def _compute_resume_state(
    existing: pd.DataFrame,
    protocols: List[str],
) -> Tuple[Set[Tuple[str, str, str, str]], Dict[str, str]]:
    """
    Từ ``results_summary.csv`` hiện có, tính:

    - ``skip_keys``: set ``(protocol, phase, model, mode)`` đã có kết quả
      → bỏ qua, không train lại.
    - ``winners_per_protocol``: ``{protocol: winning_mode}`` cho protocol
      đã chạy XONG Phase A (đủ 3 mode) → không cần train Phase A lại.

    Quy tắc chọn winner giống hệt logic trong ``run_phase_a``:
    bỏ dòng ``MEAN``, groupby ``imbalance_mode`` lấy mean ``macro_f1``,
    sort giảm dần → mode đầu tiên là winner.
    """
    skip_keys: Set[Tuple[str, str, str, str]] = set()
    winners: Dict[str, str] = {}

    if existing.empty:
        return skip_keys, winners

    # Đảm bảo các cột cần thiết tồn tại.
    needed = {"protocol", "phase", "model", "imbalance_mode", "scenario", "macro_f1"}
    if not needed.issubset(set(existing.columns)):
        return skip_keys, winners

    for proto in protocols:
        # ---- Phase A: egraphsage × {none, class_weight, undersample} ----
        sub_a = existing[
            (existing["protocol"].astype(str) == str(proto))
            & (existing["phase"].astype(str) == "A")
            & (existing["model"].astype(str) == "egraphsage")
        ]
        modes_done = set(sub_a["imbalance_mode"].astype(str).unique())
        for m in modes_done:
            skip_keys.add((str(proto), "A", "egraphsage", str(m)))

        # Nếu đủ 3 mode → derive winner, dùng cho Phase B.
        if set(IMBALANCE_MODES).issubset(modes_done):
            non_mean = sub_a[~sub_a["scenario"].astype(str).isin(["MEAN"])]
            if len(non_mean) > 0:
                try:
                    winner = (
                        non_mean.groupby("imbalance_mode")["macro_f1"]
                               .mean()
                               .sort_values(ascending=False)
                               .index[0]
                    )
                    winners[str(proto)] = str(winner)
                except Exception:  # noqa: BLE001
                    pass

        # ---- Phase B: skip (model, winner) configs đã chạy ----
        if str(proto) in winners:
            winner_mode = winners[str(proto)]
            sub_b = existing[
                (existing["protocol"].astype(str) == str(proto))
                & (existing["phase"].astype(str) == "B")
                & (existing["imbalance_mode"].astype(str) == winner_mode)
            ]
            models_done = set(sub_b["model"].astype(str).unique())
            for m_name in models_done:
                skip_keys.add((str(proto), "B", str(m_name), winner_mode))

    return skip_keys, winners


# ============================================================================
# Helpers — seed, idx→names, deep-copy cfg
# ============================================================================

def _set_seed(seed: int) -> None:
    """Seed toàn bộ (random + numpy + torch + cuda + PYTHONHASHSEED)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _idx_to_names(class_to_idx: Dict[Any, int], K: int) -> List[str]:
    name_at: List[Optional[str]] = [None] * K
    for name, idx in class_to_idx.items():
        if 0 <= int(idx) < K:
            name_at[int(idx)] = str(name)
    return [n if n is not None else f"class_{i}" for i, n in enumerate(name_at)]


def _deep_copy_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(cfg)


def _print_header(s: str, ch: str = "=") -> None:
    line = ch * 70
    print()
    print(line)
    print(f" {s}")
    print(line)


# ============================================================================
# Building blocks: chuẩn bị weight_tensor + graph theo imbalance_mode
# ============================================================================

def _build_weight_tensor(
    labels: List[str],
    class_to_idx: Dict[Any, int],
    K: int,
) -> torch.Tensor:
    """
    Tính weight_tensor cho ``class_weight`` mode.

    QUAN TRỌNG: ``compute_class_weights`` yêu cầu TÊN LỚP làm key.
    Trả về 1-D float32 tensor length K (đúng thứ tự ``class_to_idx``).
    """
    from src.imbalance import compute_class_weights
    wd, _, _ = compute_class_weights(labels, scheme="balanced")
    w = torch.zeros(K, dtype=torch.float32)
    for c, idx in class_to_idx.items():
        w[int(idx)] = float(wd.get(c, 0.0))
    return w


def _maybe_undersample(
    df_feat: pd.DataFrame,
    imbalance_mode: str,
    seed: int,
) -> pd.DataFrame:
    """Nếu mode='undersample' trả về df đã undersample; ngược lại trả df gốc."""
    if imbalance_mode == "undersample":
        from src.imbalance import undersample_majority
        return undersample_majority(
            df_feat,
            strategy="to_second_largest",
            random_state=seed,
            verbose=False,
        )
    return df_feat


# ============================================================================
# Protocol: per_scenario — train+eval riêng từng scenario (transductive)
# ============================================================================

def run_per_scenario(
    model_name: str,
    imbalance_mode: str,
    scenario_paths: Dict[str, str],
    cfg: Dict[str, Any],
    preprocessor,
    class_to_idx: Dict[Any, int],
    seed: int,
    save_dir: str,
    epochs_override: Optional[int] = None,
    cap_per_class: Optional[int] = None,
    chunksize: int = 100_000,
    data_cache: Optional[DataCache] = None,
) -> pd.DataFrame:
    """
    Với MỖI scenario:
      - Load + transform (dùng SHARED preprocessor + class_to_idx).
      - Build graph (undersample nếu mode='undersample').
      - ``train_model`` (transductive split — 70/10/20 với seed=42).
      - ``evaluate_model`` trên test_mask.

    Trả về DataFrame: 1 dòng / scenario + 1 dòng MEAN.
    """
    from src.multi_scenario import load_all_scenarios
    from src.preprocess import clean_flows, transform
    from src.data_io import load_scenario
    from src.graph_build import build_graph
    from src.train import train_model, split_edge_masks, get_device
    from src.evaluate import evaluate_model

    # ---- Load + clean (CHUNG cho cả phase) ----
    if data_cache is not None:
        all_dfs = data_cache.get_clean_dfs(
            scenario_paths, cap_per_class=cap_per_class, chunksize=chunksize,
        )
    elif cap_per_class is not None:
        all_dfs = load_all_scenarios(
            scenario_paths, cap_per_class=cap_per_class, chunksize=chunksize,
        )
    else:
        all_dfs = {n: clean_flows(load_scenario(p)) for n, p in scenario_paths.items()}
    K = len(class_to_idx)

    # Cfg copy + override
    cfg_eff = _deep_copy_cfg(cfg)
    tr_cfg = cfg_eff.setdefault("training", {})
    if epochs_override is not None:
        tr_cfg["epochs"] = int(epochs_override)
    train_ratio = float(tr_cfg.get("train_ratio", 0.70))
    val_ratio = float(tr_cfg.get("val_ratio", 0.10))
    test_ratio = float(tr_cfg.get("test_ratio", 0.20))

    target_names = _idx_to_names(class_to_idx, K)
    records: List[Dict[str, Any]] = []

    for sname in sorted(scenario_paths.keys()):
        t0 = time.perf_counter()
        _set_seed(seed)

        # --- Graph cache (tier 2): transform + undersample + build_graph ---
        def _build_one_graph(_sname=sname, _mode=imbalance_mode):
            df_feat = transform(all_dfs[_sname], preprocessor)
            df_for_graph = _maybe_undersample(df_feat, _mode, seed)
            return build_graph(
                df_for_graph,
                class_to_idx=class_to_idx,
                feature_columns=preprocessor.feature_columns,
            )

        if data_cache is not None:
            data = data_cache.get_graph(
                sname, imbalance_mode, cap_per_class,
                build_fn=_build_one_graph,
            )
        else:
            data = _build_one_graph()

        # weight_tensor nếu class_weight — lấy labels từ transform()
        wt: Optional[torch.Tensor] = None
        if imbalance_mode == "class_weight":
            _df_feat = transform(all_dfs[sname], preprocessor)
            wt = _build_weight_tensor(
                _df_feat["detailed-label"].astype(str).tolist(),
                class_to_idx, K,
            )

        # Train (tự split_edge_masks bên trong với cùng seed).
        model, history, ckpt = train_model(
            model_name, data, cfg_eff,
            imbalance_mode=imbalance_mode, weight_tensor=wt,
            seed=seed, save_dir=save_dir, verbose=False,
        )

        # TÁI DỰNG test_mask bằng cách gọi lại split_edge_masks — deterministic
        # với cùng seed → đảm bảo khớp với mask lúc train (KHÔNG trộn mask
        # giữa các graph).
        _, _, test_mask = split_edge_masks(
            data.edge_label,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        device = get_device()
        data = data.to(device)
        result = evaluate_model(
            model, data, test_mask, class_to_idx, device, verbose=False,
        )

        per_class = result["per_class"]
        rec = {
            "scenario": sname,
            "model": model_name,
            "imbalance_mode": imbalance_mode,
            "macro_f1": result["macro_f1"],
            "weighted_f1": result["weighted_f1"],
            "accuracy": result["accuracy"],
            "best_epoch": history["best_epoch"],
            "epochs_ran": history["final_epoch"],
            "best_val_f1": history["best_val_f1"],
        }
        for cn in target_names:
            entry = per_class.get(cn, {})
            rec[f"f1_{cn}"] = float(entry.get("f1-score", 0.0))
            rec[f"support_{cn}"] = int(entry.get("support", 0))
        records.append(rec)

        dt = time.perf_counter() - t0
        print(
            f"    [{model_name}|{imbalance_mode}|{sname}]  "
            f"macro_F1={result['macro_f1']:.4f}  "
            f"epochs_ran={history['final_epoch']}/{cfg_eff['training']['epochs']}  "
            f"({dt:.1f}s)"
        )

    df = pd.DataFrame(records)
    # Dòng MEAN
    mean_row: Dict[str, Any] = {
        "scenario": "MEAN",
        "model": model_name,
        "imbalance_mode": imbalance_mode,
    }
    for col in df.columns:
        if col in ("scenario", "model", "imbalance_mode"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            mean_row[col] = float(df[col].mean())
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    return df


# ============================================================================
# Protocol: pooled — train 1 model trên union, eval trên union test_mask
# ============================================================================

def run_pooled(
    model_name: str,
    imbalance_mode: str,
    scenario_paths: Dict[str, str],
    cfg: Dict[str, Any],
    preprocessor,
    class_to_idx: Dict[Any, int],
    seed: int,
    save_dir: str,
    epochs_override: Optional[int] = None,
    cap_per_class: Optional[int] = None,
    chunksize: int = 100_000,
    data_cache: Optional[DataCache] = None,
) -> pd.DataFrame:
    """
    Train 1 model trên UNION của NHIỀU graph; eval trên union test_mask.

    Vòng train (1 epoch):
        zero_grad → for g in graphs: loss = CE(logits, edge_label); loss.backward()
        → clip_grad → step(). ⇒ loss CỘNG DỒN, 1 optimizer step.

    Eval: union test_mask của MỌI graph → metric global.
    """
    from src.multi_scenario import load_all_scenarios
    from src.preprocess import clean_flows, transform
    from src.data_io import load_scenario
    from src.graph_build import build_graph
    from src.train import (
        split_edge_masks, get_device, make_criterion, save_checkpoint,
    )
    from src.model import build_model
    from src.evaluate import plot_confusion_matrix

    _set_seed(seed)
    device = get_device()

    # ---- Load ----
    if data_cache is not None:
        all_dfs = data_cache.get_clean_dfs(
            scenario_paths, cap_per_class=cap_per_class, chunksize=chunksize,
        )
    elif cap_per_class is not None:
        all_dfs = load_all_scenarios(
            scenario_paths, cap_per_class=cap_per_class, chunksize=chunksize,
        )
    else:
        all_dfs = {n: clean_flows(load_scenario(p)) for n, p in scenario_paths.items()}
    K = len(class_to_idx)
    target_names = _idx_to_names(class_to_idx, K)

    # ---- Cfg override ----
    cfg_eff = _deep_copy_cfg(cfg)
    tr_cfg = cfg_eff.setdefault("training", {})
    if epochs_override is not None:
        tr_cfg["epochs"] = int(epochs_override)
    train_ratio = float(tr_cfg.get("train_ratio", 0.70))
    val_ratio = float(tr_cfg.get("val_ratio", 0.10))
    test_ratio = float(tr_cfg.get("test_ratio", 0.20))

    # ---- Build graphs (with cache) ----
    graphs: Dict[str, Any] = {}
    weight_tensor: Optional[torch.Tensor] = None

    if imbalance_mode == "class_weight":
        labels_union: List[str] = []
        for n in sorted(scenario_paths.keys()):
            df_feat = transform(all_dfs[n], preprocessor)
            labels_union.extend(df_feat["detailed-label"].astype(str).tolist())
        weight_tensor = _build_weight_tensor(labels_union, class_to_idx, K)

    for n in sorted(scenario_paths.keys()):
        # --- Graph cache (tier 2) ---
        def _build_pooled_graph(_n=n):
            df_feat = transform(all_dfs[_n], preprocessor)
            df_for_graph = _maybe_undersample(df_feat, imbalance_mode, seed)
            return build_graph(
                df_for_graph,
                class_to_idx=class_to_idx,
                feature_columns=preprocessor.feature_columns,
            )

        if data_cache is not None:
            graphs[n] = data_cache.get_graph(
                n, imbalance_mode, cap_per_class, build_fn=_build_pooled_graph,
            )
        else:
            graphs[n] = _build_pooled_graph()

    # ---- Edge masks per graph ----
    for g in graphs.values():
        tr_m, va_m, te_m = split_edge_masks(
            g.edge_label,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        g.train_mask = tr_m
        g.val_mask = va_m
        g.test_mask = te_m

    # ---- Move to device ----
    graphs_dev = {n: g.to(device) for n, g in graphs.items()}

    # ---- Build model (1 graph bất kỳ — cùng schema) ----
    any_g = next(iter(graphs.values()))
    model = build_model(model_name, any_g, cfg_eff).to(device)

    # ---- Loss + Optim ----
    criterion = make_criterion(
        imbalance_mode, weight_tensor=weight_tensor, device=device,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(tr_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(tr_cfg.get("weight_decay", 0.0)),
    )
    grad_clip = float(tr_cfg.get("grad_clip", 1.0))
    epochs = int(tr_cfg.get("epochs", 50))
    patience = int(tr_cfg.get("early_stop_patience", 10))

    # ---- Train loop (gradient sum across graphs) ----
    best_val_f1 = -1.0
    best_epoch = -1
    bad_epochs = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    final_epoch = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        # Train
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        for g in graphs_dev.values():
            logits = model(g)
            mask = g.train_mask.to(device)
            logits_m = logits[mask]
            labels = g.edge_label.to(device)[mask]
            loss = criterion(logits_m, labels)
            loss.backward()
            total_loss += float(loss.item())
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        avg_loss = total_loss / len(graphs_dev)

        # Val (union val_mask)
        model.eval()
        with torch.no_grad():
            vp_list, vl_list = [], []
            for g in graphs_dev.values():
                mask = g.val_mask.to(device)
                logits = model(g)
                vp_list.append(logits[mask].argmax(dim=-1).cpu())
                vl_list.append(g.edge_label.to(device)[mask].cpu())
        vp = torch.cat(vp_list).numpy()
        vl = torch.cat(vl_list).numpy()
        val_macro = float(
            f1_score(
                vl, vp, labels=list(range(K)),
                average="macro", zero_division=0,
            )
        )
        history.append({
            "epoch": epoch, "train_loss": avg_loss, "val_macro_f1": val_macro,
        })
        final_epoch = epoch

        if val_macro > best_val_f1:
            best_val_f1 = val_macro
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print(
                f"    [pooled|{model_name}|{imbalance_mode}]  "
                f"early-stop tại epoch {epoch} "
                f"(best_val={best_val_f1:.4f} @ epoch {best_epoch})."
            )
            break

    # ---- Restore best ----
    if best_state is not None:
        model.load_state_dict(
            {k: v.to(device) for k, v in best_state.items()}
        )

    # ---- Test (union test_mask) ----
    model.eval()
    with torch.no_grad():
        tp_list, tl_list = [], []
        for g in graphs_dev.values():
            mask = g.test_mask.to(device)
            logits = model(g)
            tp_list.append(logits[mask].argmax(dim=-1).cpu())
            tl_list.append(g.edge_label.to(device)[mask].cpu())
    tp = torch.cat(tp_list).numpy()
    tl = torch.cat(tl_list).numpy()

    macro = float(
        f1_score(tl, tp, labels=list(range(K)), average="macro", zero_division=0)
    )
    weighted = float(
        f1_score(tl, tp, labels=list(range(K)), average="weighted", zero_division=0)
    )
    accuracy = float((tp == tl).mean())
    per_class_arr = f1_score(
        tl, tp, labels=list(range(K)), average=None, zero_division=0,
    )
    cm = confusion_matrix(tl, tp, labels=list(range(K)))

    epochs_ran = final_epoch

    # ---- Lưu checkpoint ----
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(
        save_dir, f"pooled_{model_name}_{imbalance_mode}_seed{seed}.pt"
    )
    save_checkpoint(
        model, ckpt_path,
        class_to_idx=class_to_idx, cfg=cfg_eff,
        feature_dim=int(any_g.feature_dim), num_classes=K,
        imbalance_mode=imbalance_mode, val_macro_f1=best_val_f1,
        history_meta={
            "best_epoch": int(best_epoch),
            "best_val_f1": float(best_val_f1),
            "final_epoch": int(final_epoch),
            "seed": int(seed),
            "protocol": "pooled",
            "n_scenarios": len(graphs),
        },
        feature_columns=preprocessor.feature_columns,
    )

    # ---- CM PNG ----
    cm_path = os.path.join(
        save_dir, f"cm_pooled_{model_name}_{imbalance_mode}_seed{seed}.png"
    )
    plot_confusion_matrix(
        cm, class_names=target_names, save_path=cm_path,
        title=(
            f"POOLED  model={model_name}  mode={imbalance_mode}  "
            f"seed={seed}  macro_F1={macro:.4f}"
        ),
    )

    rec: Dict[str, Any] = {
        "scenario": "POOLED",
        "model": model_name,
        "imbalance_mode": imbalance_mode,
        "macro_f1": macro,
        "weighted_f1": weighted,
        "accuracy": accuracy,
        "best_epoch": int(best_epoch),
        "epochs_ran": int(epochs_ran),
        "best_val_f1": float(best_val_f1),
        "n_scenarios": len(graphs),
    }
    for cn in target_names:
        idx = class_to_idx[cn]
        rec[f"f1_{cn}"] = float(per_class_arr[int(idx)])
        rec[f"support_{cn}"] = int((tl == int(idx)).sum())

    print(
        f"    [pooled|{model_name}|{imbalance_mode}]  "
        f"macro_F1={macro:.4f}  "
        f"epochs_ran={epochs_ran}/{epochs}  "
        f"n_scenarios={len(graphs)}"
    )
    return pd.DataFrame([rec])


# ============================================================================
# Protocol: loso — dùng multi_scenario.run_loso
# ============================================================================

def run_loso_protocol(
    model_name: str,
    imbalance_mode: str,
    scenario_paths: Dict[str, str],
    cfg: Dict[str, Any],
    seed: int,
    save_dir: str,
    config_path: str,
    epochs_override: Optional[int] = None,
    patience_override: Optional[int] = None,
    cap_per_class: Optional[int] = None,
    chunksize: int = 100_000,
    data_cache: Optional[DataCache] = None,
) -> pd.DataFrame:
    """Wrap ``multi_scenario.run_loso``; augment + reorder DataFrame."""
    from src.multi_scenario import run_loso as ms_run_loso

    # Nếu cache có preloaded clean dfs → truyền vào LOSO để skip load_all_scenarios.
    preloaded_dfs = None
    if data_cache is not None:
        preloaded_dfs = data_cache.get_clean_dfs(
            scenario_paths, cap_per_class=cap_per_class, chunksize=chunksize,
        )

    df = ms_run_loso(
        scenario_paths=scenario_paths,
        config_path=config_path,
        model_name=model_name,
        imbalance_mode=imbalance_mode,
        cap_per_class=cap_per_class,
        chunksize=chunksize,
        epochs_override=epochs_override,
        patience_override=patience_override,
        val_ratio=cfg.get("training", {}).get("val_ratio", 0.10),
        seed=seed,
        out_dir=save_dir,
        verbose=False,
        preloaded_clean_dfs=preloaded_dfs,
    )

    # run_loso trả DataFrame với các cột: held_out, macro_f1, weighted_f1,
    # accuracy, best_epoch, best_val_f1, n_unseen_in_train, f1_<class>,
    # support_<class>. Thêm model/imbalance_mode và đổi tên cột scenario.
    df = df.copy()
    df["model"] = model_name
    df["imbalance_mode"] = imbalance_mode
    df = df.rename(columns={"held_out": "scenario"})

    # Tính epochs_ran nếu chưa có
    if "epochs_ran" not in df.columns:
        max_e = int(epochs_override) if epochs_override is not None else int(
            cfg.get("training", {}).get("epochs", 50)
        )
        df["epochs_ran"] = max_e

    # Reorder cột
    front_cols = [
        "scenario", "model", "imbalance_mode", "macro_f1", "weighted_f1",
        "accuracy", "best_epoch", "epochs_ran", "best_val_f1",
        "n_unseen_in_train",
    ]
    cols = [c for c in front_cols if c in df.columns] + [
        c for c in df.columns if c not in front_cols
    ]
    df = df[cols]

    # Print tóm tắt
    non_mean = df[df["scenario"] != "MEAN"]
    if len(non_mean) > 0:
        mean_macro = float(non_mean["macro_f1"].mean())
        print(
            f"    [loso|{model_name}|{imbalance_mode}]  "
            f"mean_macro_F1={mean_macro:.4f}  "
            f"n_rounds={len(non_mean)}"
        )
    return df


# ============================================================================
# Phase runners
# ============================================================================

def run_phase_a(
    protocol: str,
    scenario_paths: Dict[str, str],
    cfg: Dict[str, Any],
    preprocessor,
    class_to_idx: Dict[Any, int],
    seed: int,
    save_dir: str,
    config_path: str,
    epochs_override: Optional[int],
    cap_per_class: Optional[int],
    chunksize: int,
    patience_override: Optional[int] = None,
    skip_keys: Optional[Set[Tuple[str, str, str, str]]] = None,
    data_cache: Optional[DataCache] = None,
) -> Tuple[pd.DataFrame, str]:
    """
    Phase A: egraphsage × 3 imbalance_mode.

    Trả (df_all, winning_mode).
    ``winning_mode`` = mode có mean macro_f1 cao nhất (loại trừ dòng MEAN/POOLED).

    ``skip_keys`` : nếu (protocol, "A", "egraphsage", mode) có trong set →
    KHÔNG train lại config đó, trả về DataFrame rỗng cho mode đó. Phase
    A vẫn chạy các mode còn lại; winner được tính trên TẤT CẢ mode đã
    có (kết quả rỗng + kết quả từ skip_keys được merge bên ngoài).
    """
    _print_header(
        f"PHASE A · protocol={protocol} · model=egraphsage · 3 imbalance_mode"
    )

    all_dfs: List[pd.DataFrame] = []
    for mode in IMBALANCE_MODES:
        if (
            skip_keys is not None
            and (protocol, "A", "egraphsage", mode) in skip_keys
        ):
            print(
                f"\n>>> {protocol} · egraphsage · {mode}  "
                f"[SKIP — đã có trong results_summary.csv]"
            )
            continue
        t0 = time.perf_counter()
        print(f"\n>>> {protocol} · egraphsage · {mode}")
        if protocol == "per_scenario":
            df = run_per_scenario(
                "egraphsage", mode, scenario_paths, cfg, preprocessor,
                class_to_idx, seed=seed, save_dir=save_dir,
                epochs_override=epochs_override,
                cap_per_class=cap_per_class, chunksize=chunksize,
                data_cache=data_cache,
            )
        elif protocol == "pooled":
            df = run_pooled(
                "egraphsage", mode, scenario_paths, cfg, preprocessor,
                class_to_idx, seed=seed, save_dir=save_dir,
                epochs_override=epochs_override,
                cap_per_class=cap_per_class, chunksize=chunksize,
                data_cache=data_cache,
            )
        elif protocol == "loso":
            df = run_loso_protocol(
                "egraphsage", mode, scenario_paths, cfg,
                seed=seed, save_dir=save_dir, config_path=config_path,
                epochs_override=epochs_override,
                patience_override=patience_override,
                cap_per_class=cap_per_class, chunksize=chunksize,
                data_cache=data_cache,
            )
        else:
            raise ValueError(f"protocol không hỗ trợ: {protocol}")
        dt = time.perf_counter() - t0
        # Tính mean macro_f1 excluding MEAN (giữ POOLED — nó là kết quả thật)
        sub = df[~df["scenario"].isin(["MEAN"])]
        mean_macro = float(sub["macro_f1"].mean()) if len(sub) else float("nan")
        print(
            f"    ===> egraphsage|{mode}  mean_macro_F1={mean_macro:.4f}  "
            f"({dt:.1f}s)"
        )
        all_dfs.append(df)

    df_all = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    if df_all.empty:
        # Tất cả Phase A configs đã skip (resume) → caller đã có winner.
        # Trả về DataFrame rỗng + winning_mode rỗng.
        print()
        print(">>> PHASE A — TẤT CẢ config đã được skip (resume case).")
        return df_all, ""

    # Tổng hợp theo imbalance_mode.
    # - per_scenario: mỗi config có nhiều scenario + 1 dòng "MEAN" → bỏ "MEAN"
    # - pooled:      mỗi config có ĐÚNG 1 dòng "POOLED" (là kết quả)
    # - loso:        mỗi config có nhiều held-out + 1 dòng "MEAN" → bỏ "MEAN"
    # ⇒ lọc chỉ bỏ "MEAN" (KHÔNG bỏ "POOLED" — nó là kết quả thật).
    sub = df_all[~df_all["scenario"].isin(["MEAN"])]
    summary = (
        sub.groupby("imbalance_mode")["macro_f1"]
           .mean()
           .sort_values(ascending=False)
    )
    print()
    print(">>> PHASE A — bảng tổng hợp (sort theo mean macro_F1):")
    print(summary.to_string())
    winning_mode = str(summary.index[0])
    print(
        f"\n>>> PHASE A winner: imbalance_mode = {winning_mode!r}  "
        f"(mean macro_F1 = {summary.iloc[0]:.4f})"
    )

    return df_all, winning_mode


def run_phase_b(
    protocol: str,
    scenario_paths: Dict[str, str],
    cfg: Dict[str, Any],
    preprocessor,
    class_to_idx: Dict[Any, int],
    seed: int,
    save_dir: str,
    config_path: str,
    epochs_override: Optional[int],
    cap_per_class: Optional[int],
    chunksize: int,
    winning_mode: str,
    patience_override: Optional[int] = None,
    skip_keys: Optional[Set[Tuple[str, str, str, str]]] = None,
    data_cache: Optional[DataCache] = None,
) -> pd.DataFrame:
    """
    Phase B: cố định winning_mode × 5 model.

    ``skip_keys`` : nếu (protocol, "B", model, winning_mode) có trong set →
    KHÔNG train lại config đó.
    """
    _print_header(
        f"PHASE B · protocol={protocol} · mode={winning_mode} · 5 model"
    )

    all_dfs: List[pd.DataFrame] = []
    for m in MODEL_POOL:
        if (
            skip_keys is not None
            and (protocol, "B", m, winning_mode) in skip_keys
        ):
            print(
                f"\n>>> {protocol} · {m} · {winning_mode}  "
                f"[SKIP — đã có trong results_summary.csv]"
            )
            continue
        t0 = time.perf_counter()
        print(f"\n>>> {protocol} · {m} · {winning_mode}")
        if protocol == "per_scenario":
            df = run_per_scenario(
                m, winning_mode, scenario_paths, cfg, preprocessor,
                class_to_idx, seed=seed, save_dir=save_dir,
                epochs_override=epochs_override,
                cap_per_class=cap_per_class, chunksize=chunksize,
                data_cache=data_cache,
            )
        elif protocol == "pooled":
            df = run_pooled(
                m, winning_mode, scenario_paths, cfg, preprocessor,
                class_to_idx, seed=seed, save_dir=save_dir,
                epochs_override=epochs_override,
                cap_per_class=cap_per_class, chunksize=chunksize,
                data_cache=data_cache,
            )
        elif protocol == "loso":
            df = run_loso_protocol(
                m, winning_mode, scenario_paths, cfg,
                seed=seed, save_dir=save_dir, config_path=config_path,
                epochs_override=epochs_override,
                patience_override=patience_override,
                cap_per_class=cap_per_class, chunksize=chunksize,
                data_cache=data_cache,
            )
        else:
            raise ValueError(f"protocol không hỗ trợ: {protocol}")
        dt = time.perf_counter() - t0
        # Bỏ MEAN, giữ POOLED — pooled row là kết quả thật duy nhất.
        sub = df[~df["scenario"].isin(["MEAN"])]
        mean_macro = float(sub["macro_f1"].mean()) if len(sub) else float("nan")
        print(
            f"    ===> {m}|{winning_mode}  mean_macro_F1={mean_macro:.4f}  "
            f"({dt:.1f}s)"
        )
        all_dfs.append(df)

    df_all = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    if df_all.empty:
        print()
        print(">>> PHASE B — TẤT CẢ config đã được skip (đã có trong summary).")
        print(">>>           Trả về DataFrame rỗng — caller chịu trách nhiệm merge.")
        return df_all

    # Tổng hợp theo model — giống Phase A: bỏ "MEAN", GIỮ "POOLED".
    sub = df_all[~df_all["scenario"].isin(["MEAN"])]
    summary = (
        sub.groupby("model")["macro_f1"]
           .mean()
           .sort_values(ascending=False)
    )
    print()
    print(">>> PHASE B — bảng tổng hợp (sort theo mean macro_F1):")
    print(summary.to_string())
    return df_all


# ============================================================================
# Orchestrator chính
# ============================================================================

def run_all(
    scenario_paths: Dict[str, str],
    config_path: str = "config.yaml",
    protocols: Optional[List[str]] = None,
    cap_per_class: Optional[int] = None,
    chunksize: int = 100_000,
    epochs_override: Optional[int] = None,
    patience_override: Optional[int] = None,
    seed: Optional[int] = None,
    out_dir: str = "artifacts/phase1_results",
    verbose: bool = True,
    resume_from_summary: bool = False,
) -> pd.DataFrame:
    """
    Chạy ��ầy đủ Phase A + Phase B cho mọi protocol.

    Parameters
    ----------
    resume_from_summary : bool
        Nếu ``True``: đọc ``results_summary.csv`` hiện có trong ``out_dir``,
        tự động build ``skip_keys`` cho các config đã chạy + derive Phase A
        winner cho protocol đã xong. Dùng để chạy TIẾP sau khi instance
        vast.ai chết giữa chừng.

    Returns
    -------
    pd.DataFrame
        ``results_summary.csv`` đã được lưu; trả về cùng DataFrame.
    """
    from src.multi_scenario import (
        load_all_scenarios,
        build_shared_class_to_idx,
        fit_shared_preprocessor,
    )

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if seed is None:
        seed = int(cfg.get("reproducibility", {}).get("seed", 42))
    if protocols is None:
        protocols = list(
            cfg.get("experiments", {}).get("protocols") or PROTOCOLS
        )

    out_dir = os.path.abspath(out_dir)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    if verbose:
        _print_header("RUN_EXPERIMENTS · Orchestrator")
        print(f"  config         : {os.path.abspath(config_path)}")
        print(f"  protocols      : {protocols}")
        print(f"  scenarios      : {list(scenario_paths.keys())}")
        print(f"  seed           : {seed}")
        print(f"  cap_per_class  : {cap_per_class}")
        print(f"  epochs_override: {epochs_override}")
        print(f"  patience       : {patience_override}")
        print(f"  out_dir        : {out_dir}")
        print(f"  ckpt_dir       : {ckpt_dir}")

    # ---- 1) Load + build shared assets (class_to_idx + preprocessor) ----
    _print_header("Build SHARED assets (class_to_idx + preprocessor)", "-")
    t0 = time.perf_counter()
    all_dfs = load_all_scenarios(
        scenario_paths, cap_per_class=cap_per_class, chunksize=chunksize,
    )
    # Khởi tạo DataCache: pre-populate với all_dfs vừa load để các call site
    # sau (per_scenario, pooled, loso) đều thấy cache HIT, không load lại.
    data_cache = DataCache()
    # Tự ghi vào tier 1 (bypass miss path) — dùng cùng key như get_clean_dfs.
    _cache_key: Tuple = (
        frozenset(scenario_paths.items()), cap_per_class, chunksize,
    )
    data_cache._clean_dfs[_cache_key] = all_dfs
    data_cache._stats["clean_hit"] = 0
    data_cache._stats["clean_miss"] = 0
    class_to_idx = build_shared_class_to_idx(all_dfs)
    K = len(class_to_idx)
    if verbose:
        for n in sorted(scenario_paths.keys()):
            print(f"  {n:<10s}  shape={all_dfs[n].shape}  "
                  f"#class={all_dfs[n]['detailed-label'].nunique()}")
        print(f"  → K = {K} (class union)")
    pre = fit_shared_preprocessor(list(all_dfs.values()))
    pre.save(os.path.join(ckpt_dir, "preprocessor.pkl"))
    if verbose:
        print(
            f"  → shared preprocessor fit trên union "
            f"({sum(len(d) for d in all_dfs.values()):,} rows)."
        )
        print(f"  → shared assets xong trong {time.perf_counter() - t0:.1f}s.")

    # ---- 2) Phase A + Phase B cho MỖI protocol ----
    summary_records: List[Dict[str, Any]] = []
    n_configs_total = 0
    t_orch = time.perf_counter()

    # ---- 2a) Resume: nạp summary cũ + build skip_keys ----
    skip_keys: Set[Tuple[str, str, str, str]] = set()
    winners_from_summary: Dict[str, str] = {}
    if resume_from_summary:
        existing = _load_existing_summary(out_dir)
        if not existing.empty:
            skip_keys, winners_from_summary = _compute_resume_state(
                existing, protocols,
            )
            # Pre-populate summary_records với rows cũ để cuối run ghi đủ.
            for _, row in existing.iterrows():
                rec = row.to_dict()
                summary_records.append(rec)
            if verbose:
                print(
                    f"  [RESUME] Loaded {len(existing)} dòng cũ từ "
                    f"{_summary_csv_path(out_dir)}."
                )
                print(
                    f"  [RESUME] skip_keys    = {len(skip_keys)} "
                    f"config đã chạy → sẽ BỎ QUA."
                )
                print(
                    f"  [RESUME] winners_cũ   = {winners_from_summary}"
                )
                # Save ngay bản sao để có checkpoint.
                _save_summary_csv(summary_records, out_dir)
                print(
                    f"  [RESUME] Đã ghi lại summary CSV hiện có "
                    f"({len(summary_records)} dòng)."
                )
        else:
            if verbose:
                print(
                    "  [RESUME] Không thấy results_summary.csv → chạy từ đầu."
                )

    for protocol in protocols:
        _print_header(f"PROTOCOL · {protocol}", "#")
        t_proto = time.perf_counter()

        # Phase A — derive winner từ summary nếu đã có.
        if protocol in winners_from_summary:
            winning_mode = winners_from_summary[protocol]
            print(
                f"\n>>> [SKIP Phase A] winner đã có từ summary: "
                f"{winning_mode!r}."
            )
            # Tải rows cũ Phase A cho CSV per-phase.
            csv_a = os.path.join(
                out_dir,
                f"phase_a_{protocol}_egraphsage_3modes.csv",
            )
            existing = _load_existing_summary(out_dir)
            df_a = existing[
                (existing["protocol"].astype(str) == str(protocol))
                & (existing["phase"].astype(str) == "A")
            ].reset_index(drop=True)
            if not df_a.empty:
                df_a.to_csv(csv_a, index=False)
        else:
            # Phase A — nhưng skip từng (mode) config đã có.
            df_a, winning_mode = run_phase_a(
                protocol, scenario_paths, cfg, pre, class_to_idx,
                seed=seed, save_dir=ckpt_dir, config_path=config_path,
                epochs_override=epochs_override, cap_per_class=cap_per_class,
                chunksize=chunksize, patience_override=patience_override,
                skip_keys=skip_keys,
                data_cache=data_cache,
            )
            # Save Phase A CSV
            csv_a = os.path.join(
                out_dir,
                f"phase_a_{protocol}_egraphsage_3modes.csv",
            )
            df_a.to_csv(csv_a, index=False)
            # Lưu cả summary có nhãn protocol/phase
            for _, row in df_a.iterrows():
                rec = row.to_dict()
                rec["protocol"] = protocol
                rec["phase"] = "A"
                summary_records.append(rec)
                n_configs_total += 1
            # Save summary CSV ngay sau Phase A — checkpoint.
            _save_summary_csv(summary_records, out_dir)

        # Phase B
        df_b = run_phase_b(
            protocol, scenario_paths, cfg, pre, class_to_idx,
            seed=seed, save_dir=ckpt_dir, config_path=config_path,
            epochs_override=epochs_override, cap_per_class=cap_per_class,
            chunksize=chunksize, winning_mode=winning_mode,
            patience_override=patience_override,
            skip_keys=skip_keys,
            data_cache=data_cache,
        )
        # Save Phase B CSV
        csv_b = os.path.join(
            out_dir,
            f"phase_b_{protocol}_mode-{winning_mode}_5models.csv",
        )
        df_b.to_csv(csv_b, index=False)
        for _, row in df_b.iterrows():
            rec = row.to_dict()
            rec["protocol"] = protocol
            rec["phase"] = "B"
            summary_records.append(rec)
            n_configs_total += 1

        # Save summary CSV ngay sau Phase B — checkpoint cuối protocol.
        _save_summary_csv(summary_records, out_dir)

        if verbose:
            print(
                f"\n>>> Protocol {protocol!r} xong trong "
                f"{time.perf_counter() - t_proto:.1f}s."
            )
            print(
                f"    [CHECKPOINT] Đã ghi {_summary_csv_path(out_dir)} "
                f"({len(summary_records)} dòng tổng cộng)."
            )

    # ---- 3) Cache stats ----
    if verbose:
        data_cache.print_stats()

    # ---- 4) results_summary.csv ----
    summary_path = _save_summary_csv(summary_records, out_dir)
    df_summary = pd.DataFrame(summary_records)

    if verbose:
        _print_header("TỔNG KẾT · Orchestrator")
        dt_orch = time.perf_counter() - t_orch
        print(f"  Tổng số cấu hình đã chạy: {n_configs_total}")
        print(f"  Thời gian tổng           : {dt_orch:.1f}s ({dt_orch/60:.1f} phút)")
        print(f"  summary CSV              : {summary_path}")
        print(f"  checkpoints dir          : {ckpt_dir}/")
        print()
        # Bảng tóm tắt: trung bình macro_f1 theo (protocol, phase)
        if (
            "macro_f1" in df_summary.columns
            and "scenario" in df_summary.columns
        ):
            agg = (
                df_summary[~df_summary["scenario"].isin(["MEAN"])]
                .groupby(["protocol", "phase", "model"])["macro_f1"]
                .mean()
                .reset_index()
                .sort_values(["protocol", "phase", "macro_f1"],
                             ascending=[True, True, False])
            )
            print(">>> Bảng tóm tắt (mean macro_F1 per (protocol, phase, model)):")
            with pd.option_context(
                "display.max_rows", None,
                "display.width", 200,
                "display.float_format", "{:.4f}".format,
            ):
                print(agg.to_string(index=False))

    return df_summary


# ============================================================================
# CLI
# ============================================================================

def _parse_scenarios_arg(items: List[str]) -> Dict[str, str]:
    """Parse ['name=PATH', ...]."""
    out: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"sai định dạng --scenarios item: '{item}' (cần name=PATH)."
            )
        name, path = item.split("=", 1)
        out[name.strip()] = path.strip()
    return out


def _resolve_scenarios(
    cfg: Dict[str, Any], cli_scenarios: Optional[List[str]],
) -> Dict[str, str]:
    """Ưu tiên CLI, fallback ``config['experiments']['scenarios']``."""
    if cli_scenarios:
        return _parse_scenarios_arg(cli_scenarios)
    exp = cfg.get("experiments", {}) or {}
    out: Dict[str, str] = {}
    for sc in exp.get("scenarios", []) or []:
        name = sc.get("name")
        path = sc.get("path")
        if name and path:
            out[str(name)] = str(path)
    return out


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Orchestrator thí nghiệm Giai đoạn 1 — Phase A (mode) + "
            "Phase B (model), cả 3 protocol (per_scenario, pooled, loso)."
        ),
    )
    p.add_argument(
        "--config", default="config.yaml",
        help="Đường dẫn config.yaml (mặc định: config.yaml).",
    )
    p.add_argument(
        "--protocols", nargs="+", default=None,
        choices=PROTOCOLS,
        help="Protocol chạy (mặc định từ config['experiments']['protocols']).",
    )
    p.add_argument(
        "--scenarios", nargs="+", default=None,
        help="Danh sách name=PATH (mặc định từ config['experiments']['scenarios']).",
    )
    p.add_argument(
        "--cap-per-class", type=int, default=None,
        help="Cap flow mỗi lớp khi load (mặc định từ config).",
    )
    p.add_argument(
        "--chunksize", type=int, default=100_000,
        help="Số dòng mỗi chunk khi cap_per_class != None.",
    )
    p.add_argument(
        "--epochs", type=int, default=None,
        help="Override số epoch (mặc định từ config['experiments']['max_epochs']).",
    )
    p.add_argument(
        "--patience", type=int, default=None,
        help=(
            "Override early-stopping patience "
            "(mặc định từ config['training']['early_stop_patience'], fallback 10)."
        ),
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Override seed (mặc định từ config).",
    )
    p.add_argument(
        "--out-dir", default=None,
        help="Thư mục output (mặc định từ config).",
    )
    p.add_argument(
        "--auto-resume", action="store_true",
        help=(
            "Đọc results_summary.csv hiện có trong out_dir, tự BỎ QUA các "
            "config (protocol, phase, model, mode) đã chạy và TỰ derive "
            "Phase A winner từ summary cũ. Dùng để chạy tiếp sau khi "
            "instance vast.ai bị chết giữa chừng."
        ),
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Resolve parameters với precedence: CLI > config > default
    scenario_paths = _resolve_scenarios(cfg, args.scenarios)
    if not scenario_paths:
        raise FileNotFoundError(
            "Không có scenario nào. Truyền --scenarios hoặc "
            "thêm block experiments.scenarios vào config.yaml."
        )

    exp_cfg = cfg.get("experiments", {}) or {}

    cap_per_class = (
        args.cap_per_class
        if args.cap_per_class is not None
        else exp_cfg.get("cap_per_class")
    )
    epochs_override = (
        args.epochs
        if args.epochs is not None
        else exp_cfg.get("max_epochs")
    )
    if epochs_override is None:
        epochs_override = int(cfg.get("training", {}).get("epochs", 50))
    # Resolve patience: ưu tiên CLI > training.early_stop_patience (10)
    # > experiments.patience (legacy alias).
    patience_override = (
        args.patience
        if args.patience is not None
        else cfg.get("training", {}).get("early_stop_patience")
    )
    if patience_override is None:
        # Fallback an toàn: legacy config key.
        patience_override = exp_cfg.get("patience")
    seed = (
        args.seed
        if args.seed is not None
        else int(cfg.get("reproducibility", {}).get("seed", 42))
    )
    out_dir = args.out_dir or exp_cfg.get("out_dir") or "artifacts/phase1_results"
    protocols = args.protocols or list(exp_cfg.get("protocols") or PROTOCOLS)

    run_all(
        scenario_paths=scenario_paths,
        config_path=args.config,
        protocols=protocols,
        cap_per_class=cap_per_class,
        chunksize=args.chunksize,
        epochs_override=int(epochs_override),
        patience_override=(
            int(patience_override) if patience_override is not None else None
        ),
        seed=seed,
        out_dir=out_dir,
        resume_from_summary=args.auto_resume,
    )


if __name__ == "__main__":
    main()
