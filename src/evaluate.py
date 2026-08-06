"""
evaluate.py — Đánh giá & so sánh nhiều model cho edge classification (Task 1.12).

Mục đích
--------
Đóng gói 3 bước cuối của Giai đoạn 1 (baseline GNN tập trung) theo đúng
tinh thần CLAUDE.md mục 8:

    •  Chỉ số chính = **macro-F1** và **per-class F1** — KHÔNG đánh giá
       qua accuracy tổng thể (vô nghĩa với phân bố lệch cực đoan của
       IoT-23). Accuracy được in kèm nhưng ghi rõ CHÚ Ý là tham khảo.
    •  Mọi đánh giá chạy trên **TEST MASK** của đồ thị gốc (cùng seed, cùng
       split với lúc train) để so sánh công bằng giữa các model.
    •  Confusion matrix **bắt buộc**, có 2 dạng cạnh nhau:
       (i) số đếm tuyệt đối, (ii) chuẩn hoá theo hàng (= recall mỗi lớp).
    •  Precision/Recall per-class và support: in qua ``classification_report``.

Thiết kế
--------
evaluate là khâu độc lập với train:
    •  ``evaluate_model(...)`` — chạy 1 model trên test_mask, trả dict
       metric + in classification_report dạng bảng.
    •  ``plot_confusion_matrix(...)`` — vẽ heatmap 2 panel, lưu PNG.
    •  ``run_comparison(...)`` — Cartesian product (model × imbalance_mode),
       ghi bảng DataFrame + CSV + CM cho model tốt nhất.
    •  CLI: chạy được ``python -m src.evaluate --scenario <path> ...``.

Reproducibility & không leak
---------------------------
``run_comparison`` dùng LẠI y hệt các hàm split/train/checkpoint của
``src.train``. Mỗi config có graph + mask riêng (graph khác nhau giữa
``none``/``class_weight``/``undersample``), nhưng trong MỖI config: cùng
seed → cùng split → ``test_mask`` khớp hệt với ``train_model`` lúc so
sánh. Không trộn mask giữa các graph khác nhau.

Lưu ý thiết bị
--------------
Device-agnostic. Vẽ bằng matplotlib dùng backend ``Agg`` (không cần
display server, an toàn trên Mac/CPU).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Optional

# ---- Matplotlib: ép backend Agg để vẽ headless (CI / CPU server an toàn) ----
import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    classification_report,
    confusion_matrix,
    f1_score,
)

# ---- sys.path setup ------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


__all__ = [
    "evaluate_model",
    "plot_confusion_matrix",
    "run_comparison",
]


logger = logging.getLogger(__name__)


# ============================================================================
# 1. evaluate_model — chạy 1 model trên test_mask
# ============================================================================

def _idx_to_names(class_to_idx: Dict[Any, int], K: int) -> List[str]:
    """
    Đảo mapping ``class_to_idx: {name → idx}`` thành ``[name_at_0, name_at_1, ...]``.

    Nếu class_to_idx thiếu vài index (lỗi khó xảy ra với K khớp len(class_to_idx))
    thì điền ``f'class_{i}'`` cho chỗ thiếu.
    """
    name_at: List[Optional[str]] = [None] * K
    for name, idx in class_to_idx.items():
        if 0 <= int(idx) < K:
            name_at[int(idx)] = str(name)
    return [n if n is not None else f"class_{i}" for i, n in enumerate(name_at)]


def evaluate_model(
    model: torch.nn.Module,
    data,
    test_mask: torch.Tensor,
    class_to_idx: Dict[Any, int],
    device: torch.device,
    *,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Forward + đánh giá trên ``test_mask``. Trả về dict gọn các metric.

    Parameters
    ----------
    model : nn.Module
        Model đã ở trên ``device``, đã load best weights (vd từ
        ``train_model`` trả về, hoặc sau ``load_checkpoint``).
    data : torch_geometric.data.Data
        Đồ thị chứa ``edge_label``, ``edge_index``, ``edge_attr``, …
    test_mask : torch.Tensor [E] bool
        Boolean mask tách cạnh TEST. Mask cùng seed/split với lúc train.
    class_to_idx : dict
        ``{class_name: idx}`` — KHỚP với ``data.edge_label`` (data class_to_idx
        hoặc ckpt class_to_idx đều OK vì cùng graph).
    device : torch.device
    verbose : bool, mặc định True
        In classification_report + macro/wieghted_F1/accuracy.

    Returns
    -------
    dict
        {
            'macro_f1'       : float,
            'weighted_f1'    : float,
            'accuracy'       : float,        # CHÚ Ý: tham khảo, lệch lớp
            'per_class'      : dict,         # sklearn classification_report (dict)
            'confusion_matrix': np.ndarray [K, K],
            'target_names'   : list[str],
            'preds'          : Tensor,
            'labels'         : Tensor,
        }
    """
    model.eval()
    with torch.no_grad():
        logits = model(data)

    mask = test_mask.to(logits.device)
    logits_m = logits[mask]
    labels = data.edge_label.to(logits.device)[mask]
    preds = logits_m.argmax(dim=-1)

    K = int(data.num_classes)
    target_names = _idx_to_names(class_to_idx, K)
    labels_range = list(range(K))

    yt = labels.detach().cpu().numpy()
    yp = preds.detach().cpu().numpy()

    macro = float(
        f1_score(
            yt, yp,
            average="macro",
            labels=labels_range,
            zero_division=0,
        )
    )
    weighted = float(
        f1_score(
            yt, yp,
            average="weighted",
            labels=labels_range,
            zero_division=0,
        )
    )
    # Accuracy tổng thể: CHỈ để tham khảo (mất cân bằng lớp cực đoan →
    # model "đoán trúng lớp đa số" vẫn có accuracy cao nhưng F1 thảm hại).
    accuracy = float((preds == labels).float().mean().item())

    report_dict = classification_report(
        yt, yp,
        labels=labels_range,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(yt, yp, labels=labels_range)

    if verbose:
        print()
        print("=" * 70)
        print(" CLASSIFICATION REPORT (test_mask)")
        print("=" * 70)
        print(
            classification_report(
                yt, yp,
                labels=labels_range,
                target_names=target_names,
                zero_division=0,
            )
        )
        print(
            f"  macro_F1   = {macro:.4f}     ← chỉ số chính\n"
            f"  weighted_F1= {weighted:.4f}\n"
            f"  accuracy   = {accuracy:.4f}     ← THAM KHẢO (mất cân bằng lớp)\n"
            f"  K = {K}, target_names = {target_names}"
        )

    return {
        "macro_f1": macro,
        "weighted_f1": weighted,
        "accuracy": accuracy,
        "per_class": report_dict,
        "confusion_matrix": cm,
        "target_names": target_names,
        "preds": preds.detach().cpu(),
        "labels": labels.detach().cpu(),
    }


# ============================================================================
# 2. plot_confusion_matrix — heatmap 2-panel (counts + row-normalized)
# ============================================================================

def _annotate_heatmap(ax, M: np.ndarray, as_int: bool) -> None:
    """Ghi số trong từng ô heatmap; chọn màu chữ theo cường độ nền.

    ``as_int=True`` cho ô counts (ép int). ``as_int=False`` cho ô đã chuẩn
    hoá (``{:.2f}``). KHÔNG dùng ``format(float, 'd')`` vì Python's 'd' chỉ
    nhận int — gây ``ValueError: Unknown format code 'd' for object of type 'float'``.
    """
    peak = float(M.max()) if M.size > 0 else 0.0
    thresh = peak * 0.6 if peak > 0 else 0.0
    nrows, ncols = M.shape
    for i in range(nrows):
        for j in range(ncols):
            val = float(M[i, j])
            if as_int:
                txt = f"{int(round(val))}"
            else:
                txt = f"{val:.2f}"
            color = "white" if (thresh and val >= thresh) else "black"
            ax.text(
                j, i, txt,
                ha="center", va="center",
                color=color, fontsize=9,
            )


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str = "Confusion Matrix",
) -> None:
    """
    Vẽ 2 heatmap cạnh nhau trên 1 file PNG:

        •  Panel trái  : số đếm tuyệt đối (``fmt='d'``).
        •  Panel phải   : chuẩn hoá theo HÀNG (= recall mỗi lớp thật, ``fmt='.2f'``).
                          Hàng có tổng = 0 (lớp không có mẫu test) được giữ 0, không NaN.

    PNG lưu ở ``save_path``. Thư mục cha được tạo nếu thiếu.
    """
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(
            f"plot_confusion_matrix: cm phải vuông [K,K]; got shape {cm.shape}."
        )
    K = cm.shape[0]

    # Chuẩn hoá theo hàng (recall). Hàng 0 (lớp không xuất hiện trong test):
    # thay bằng 0 để tránh cảnh báo RuntimeWarning divide-by-zero.
    row_sums = cm.sum(axis=1, keepdims=True)
    safe_sums = np.where(row_sums == 0, 1, row_sums)
    cm_norm = cm.astype("float") / safe_sums.astype("float")

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.5))

    # ---- Panel trái: counts ----
    im0 = axes[0].imshow(cm, cmap="Blues", aspect="auto", vmin=0)
    axes[0].set_xticks(range(K))
    axes[0].set_xticklabels(class_names, rotation=45, ha="right")
    axes[0].set_yticks(range(K))
    axes[0].set_yticklabels(class_names)
    axes[0].set_xlabel("Predicted label")
    axes[0].set_ylabel("True label")
    axes[0].set_title(f"{title}\n(số đếm tuyệt đối)")
    _annotate_heatmap(axes[0], cm, as_int=True)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # ---- Panel phải: normalized (recall) ----
    im1 = axes[1].imshow(cm_norm, cmap="Blues", aspect="auto", vmin=0.0, vmax=1.0)
    axes[1].set_xticks(range(K))
    axes[1].set_xticklabels(class_names, rotation=45, ha="right")
    axes[1].set_yticks(range(K))
    axes[1].set_yticklabels(class_names)
    axes[1].set_xlabel("Predicted label")
    axes[1].set_ylabel("True label")
    axes[1].set_title(f"{title}\n(row-normalized = recall mỗi lớp)")
    _annotate_heatmap(axes[1], cm_norm, as_int=False)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Đã lưu confusion matrix PNG: %s", save_path)


# ============================================================================
# 3. run_comparison — train + eval Cartesian product
# ============================================================================

def run_comparison(
    scenario_path: str,
    config_path: str,
    models: Optional[List[str]] = None,
    imbalance_modes: Optional[List[str]] = None,
    seed: Optional[int] = None,
    out_dir: str = "artifacts",
    epochs_override: Optional[int] = None,
    save_dir_ckpts: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Train + evaluate Cartesian product của ``models × imbalance_modes``.

    Quy trình từng cell ``(model, mode)``
    -------------------------------------
    1.  Build ``df_for_graph`` + ``class_to_idx`` + ``weight_tensor`` theo mode
        (CHIA SẺ ``df_feat`` đã preprocess 1 lần — chỉ fit preprocessor 1 lần).
    2.  ``build_graph`` → ``Data``.
    3.  ``train_model`` (verbose=False để log đỡ rối; train_model vẫn trả model
        với best weights qua val_F1).
    4.  ``split_edge_masks`` lại với CÙNG seed → cùng test_mask.
    5.  ``evaluate_model`` trên test_mask (đã có model, không phải reload ckpt).
    6.  Ghi dict vào ``records`` (gồm cm + class_names để lát vẽ best CM).

    Sau vòng lặp:
        •  DataFrame sort theo macro_f1 giảm dần.
        •  Save CSV ``comparison_<scenario>.csv``.
        •  Vẽ confusion matrix cho CẤU HÌNH CÓ MACRO_F1 CAO NHẤT.

    Parameters
    ----------
    scenario_path : str
        Đường dẫn ``conn.log.labeled`` (vd CTU-IoT-Malware-Capture-34-1).
    config_path : str
        Đường dẫn ``config.yaml``.
    models : list[str] | None
        Mặc định ``['egraphsage', 'sage_edge_concat', 'graphsage', 'gcn']``.
    imbalance_modes : list[str] | None
        Mặc định ``['none', 'class_weight', 'undersample']`` (đủ 3 theo spec).
    seed : int | None
        Mặc định lấy từ ``config.yaml['reproducibility']['seed']``.
    out_dir : str
        Thư mục lưu CSV + PNG + checkpoint (mặc định ``artifacts/``).
    epochs_override : int | None
        Override số epoch (mặc định giữ từ config).
    save_dir_ckpts : str | None
        Thư mục lưu checkpoint của MỖI config; mặc định ``<out_dir>/checkpoints``.

    Returns
    -------
    pd.DataFrame
        Có cột: ``model, imbalance, macro_f1, weighted_f1, accuracy,
        best_epoch, best_val_f1, f1_<class>, support_<class>``
        (per-class columns đặt động theo class_to_idx).
        Sort theo macro_f1 giảm dần.
    """
    # ---- Defaults ----
    if models is None:
        models = ["egraphsage", "gat", "sage_edge_concat", "graphsage", "gcn"]
    if imbalance_modes is None:
        imbalance_modes = ["none", "class_weight", "undersample"]

    valid_models = {"egraphsage", "gcn", "graphsage", "sage_edge_concat", "gat"}
    bad_m = [m for m in models if m not in valid_models]
    if bad_m:
        raise ValueError(
            f"run_comparison: models không hợp lệ: {bad_m}. "
            f"Chỉ chấp nhận: {sorted(valid_models)}."
        )
    valid_modes = {"none", "class_weight", "undersample"}
    bad_md = [m for m in imbalance_modes if m not in valid_modes]
    if bad_md:
        raise ValueError(
            f"run_comparison: imbalance_modes không hợp lệ: {bad_md}. "
            f"Chỉ chấp nhận: {sorted(valid_modes)}."
        )

    # ---- Load config + seed ----
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if seed is None:
        seed = int(cfg.get("reproducibility", {}).get("seed", 42))

    # Tên scenario ưu tiên THƯ MỤC CHA (vd "CTU-IoT-Malware-Capture-34-1") —
    # đẹp hơn nhiều so với file stem ("conn.log" khi tên file là
    # conn.log.labeled, vì os.path.splitext chỉ strip 1 extension cuối).
    _abs_path = os.path.abspath(scenario_path)
    _parent_name = os.path.basename(os.path.dirname(_abs_path))
    _file_stem = os.path.splitext(os.path.basename(_abs_path))[0]
    scenario_name = (
        _parent_name
        if _parent_name and _parent_name not in (".", "")
        else _file_stem
    )

    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    if save_dir_ckpts is None:
        save_dir_ckpts = os.path.join(out_dir, "checkpoints")
    os.makedirs(save_dir_ckpts, exist_ok=True)

    # ---- Lazy imports (tránh vòng nếu evaluate import train trước) ----
    from src.data_io import load_scenario
    from src.core.graph import build_graph
    from src.imbalance import compute_class_weights, prepare_imbalance_variants
    from src.core.model import build_model  # noqa: F401 (giữ cho mở rộng)
    from src.core.preprocess import clean_flows, fit_preprocessor, transform
    from src.train import (
        get_device,
        split_edge_masks,
        train_model,
    )

    device = get_device()

    if verbose:
        print("=" * 70)
        print(f" run_comparison  ·  scenario: {scenario_name}")
        print("=" * 70)
        print(f"  seed        : {seed}")
        print(f"  device      : {device}")
        print(f"  models      : {models}")
        print(f"  modes       : {imbalance_modes}")
        print(f"  out_dir     : {out_dir}")
        if epochs_override is not None:
            print(f"  epochs_override: {epochs_override}")
        print()

    # ---- Preprocess MỘT LẦN ----
    if verbose:
        print("[run_comparison] load + preprocess (1 lần, chia sẻ giữa các config)...")
    df_clean = clean_flows(load_scenario(scenario_path))
    pre = fit_preprocessor(df_clean)
    df_feat = transform(df_clean, pre)
    if verbose:
        print(f"  df_feat.shape = {df_feat.shape}\n")

    # ---- Vòng Cartesian ----
    records: List[Dict[str, Any]] = []

    for model_name in models:
        for mode in imbalance_modes:
            print("=" * 70)
            print(f" CONFIG: model={model_name}  imbalance={mode}")
            print("=" * 70)

            # ---- Chuẩn bị df + weight theo imbalance_mode ----
            if mode == "class_weight":
                variants = prepare_imbalance_variants(df_feat, random_state=seed)
                weight_tensor = variants["weight_tensor"]
                class_to_idx = variants["class_to_idx"]
                df_for_graph = df_feat
            elif mode == "undersample":
                variants = prepare_imbalance_variants(df_feat, random_state=seed)
                weight_tensor = None
                class_to_idx = variants["class_to_idx"]
                df_for_graph = variants["undersampled"]
            elif mode == "none":
                _, class_to_idx, _ = compute_class_weights(
                    df_feat["detailed-label"].tolist(), scheme="balanced",
                )
                weight_tensor = None
                df_for_graph = df_feat
            else:
                # Không tới đây nhờ check ở trên.
                raise ValueError(f"imbalance_mode='{mode}' không hỗ trợ.")

            data = build_graph(
                df_for_graph,
                class_to_idx=class_to_idx,
                feature_columns=pre.feature_columns,
            )

            # ---- cfg_eff để override epochs nếu cần ----
            cfg_eff: Dict[str, Any] = dict(cfg)
            cfg_eff["training"] = dict(cfg.get("training", {}))
            if epochs_override is not None:
                cfg_eff["training"]["epochs"] = int(epochs_override)

            # ---- Train ----
            model, history, ckpt_path = train_model(
                model_name, data, cfg_eff,
                imbalance_mode=mode,
                weight_tensor=weight_tensor,
                seed=seed,
                save_dir=save_dir_ckpts,
                verbose=False,  # tránh log đè giữa 12 config
            )

            # ---- Tái tạo test_mask với cùng seed/split ----
            tr = cfg_eff["training"]
            train_ratio = float(tr.get("train_ratio", 0.70))
            val_ratio = float(tr.get("val_ratio", 0.10))
            test_ratio = float(tr.get("test_ratio", 0.20))
            _, _, test_mask = split_edge_masks(
                data.edge_label,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=seed,
            )
            test_mask = test_mask.to(device)
            data = data.to(device)

            # ---- Evaluate ----
            metrics = evaluate_model(
                model, data, test_mask, class_to_idx, device, verbose=True,
            )

            # ---- Ghi record ----
            row: Dict[str, Any] = {
                "model": model_name,
                "imbalance": mode,
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "accuracy": metrics["accuracy"],
                "best_epoch": history.get("best_epoch", -1),
                "best_val_f1": history.get("best_val_f1", float("nan")),
                "checkpoint": ckpt_path,
            }
            for cn in metrics["target_names"]:
                rec = metrics["per_class"].get(cn, {})
                row[f"f1_{cn}"] = float(rec.get("f1-score", 0.0))
                row[f"support_{cn}"] = int(rec.get("support", 0))
            # Hidden fields: confusion matrix + class mapping để lát vẽ best.
            row["_cm"] = metrics["confusion_matrix"]
            row["_class_names"] = metrics["target_names"]
            row["_class_to_idx"] = class_to_idx
            records.append(row)
            print()  # tách dòng cho dễ đọc log

    # ---- Tổng hợp DataFrame ----
    public_rows = [{k: v for k, v in r.items() if not k.startswith("_")}
                   for r in records]
    df = pd.DataFrame(public_rows)
    df_sorted = (
        df.sort_values("macro_f1", ascending=False)
          .reset_index(drop=True)
    )

    # ---- Save CSV ----
    csv_path = os.path.join(out_dir, f"comparison_{scenario_name}.csv")
    df_sorted.to_csv(csv_path, index=False)
    logger.info("Đã lưu CSV so sánh: %s", csv_path)

    # ---- Vẽ confusion matrix cho cấu hình TỐT NHẤT (macro_f1 cao nhất) ----
    best_idx = int(df_sorted.index[0])
    best_model = df_sorted.loc[best_idx, "model"]
    best_mode = df_sorted.loc[best_idx, "imbalance"]
    # Tìm record tương ứng trong records (có _cm/_class_names).
    best_record = next(
        r for r in records
        if r["model"] == best_model and r["imbalance"] == best_mode
    )
    cm_png = os.path.join(
        out_dir, f"confusion_matrix_{scenario_name}_best.png"
    )
    plot_confusion_matrix(
        best_record["_cm"],
        class_names=best_record["_class_names"],
        save_path=cm_png,
        title=(
            f"Confusion Matrix — best config: {best_model} + "
            f"{best_mode}  (macro_F1={best_record['macro_f1']:.4f})"
        ),
    )

    # ---- In bảng tóm tắt ----
    if verbose:
        print()
        print("=" * 70)
        print(" BẢNG SO SÁNH (sort theo macro_F1 giảm dần)")
        print("=" * 70)
        with pd.option_context(
            "display.max_columns", None,
            "display.width", 200,
            "display.float_format", "{:.4f}".format,
        ):
            print(df_sorted.to_string(index=False))
        print()
        print("=" * 70)
        print(" ARTIFACTS")
        print("=" * 70)
        print(f"  CSV  : {csv_path}")
        print(f"  PNG  : {cm_png}")
        print(f"  CKPTS: {save_dir_ckpts}/")

    return df_sorted


# ============================================================================
# CLI
# ============================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Đánh giá edge-classification GNN trên 1 scenario IoT-23. "
            "Train + eval Cartesian product(models × imbalance_modes); "
            "ghi CSV + confusion-matrix PNG cho model tốt nhất."
        ),
    )
    p.add_argument(
        "--scenario", type=str, required=True,
        help="Đường dẫn tới file conn.log.labeled (vd "
             "data/CTU-IoT-Malware-Capture-34-1/conn.log.labeled).",
    )
    p.add_argument(
        "--config", type=str, default="config.yaml",
        help="Đường dẫn config.yaml (mặc định: config.yaml).",
    )
    p.add_argument(
        "--models", nargs="+",
        default=["egraphsage", "gat", "sage_edge_concat", "graphsage", "gcn"],
        choices=["egraphsage", "gcn", "graphsage", "sage_edge_concat", "gat"],
        help="Danh sách model cần so sánh.",
    )
    p.add_argument(
        "--modes", nargs="+",
        default=["none", "class_weight", "undersample"],
        choices=["none", "class_weight", "undersample"],
        help="Danh sách imbalance mode cần so sánh.",
    )
    p.add_argument(
        "--epochs", type=int, default=None,
        help="Override số epoch (mặc định: lấy từ config.yaml).",
    )
    p.add_argument(
        "--out-dir", type=str, default="artifacts",
        help="Thư mục lưu CSV + PNG + checkpoint (mặc định: artifacts/).",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Tắt log per-config (chỉ in bảng tổng kết cuối).",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    df = run_comparison(
        scenario_path=args.scenario,
        config_path=args.config,
        models=args.models,
        imbalance_modes=args.modes,
        out_dir=args.out_dir,
        epochs_override=args.epochs,
        verbose=not args.quiet,
    )

    print()
    print("=" * 70)
    print(" TOP 3 (theo macro_F1)")
    print("=" * 70)
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 200,
        "display.float_format", "{:.4f}".format,
    ):
        print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
