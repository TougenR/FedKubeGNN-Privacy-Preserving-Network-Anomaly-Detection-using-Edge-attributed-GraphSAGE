"""
train.py — Vòng train device-agnostic cho edge classification (Task 1.10).

Thiết kế (đã chốt trong CLAUDE.md mục 2 & 5 & 8):

1.  Transductive edge classification.
    Dựng MỘT đồ thị đầy đủ từ toàn bộ flow của scenario. MESSAGE PASSING
    chạy trên TOÀN BỘ ``edge_index_mp`` (mọi cạnh, 2 chiều) — embedding
    node được dùng đầy đủ cấu trúc đồ thị. NHƯNG loss/eval tách theo 3
    boolean mask trên cạnh GỐC:

        • ``train_mask`` — tính loss & backprop.
        • ``val_mask``   — chọn checkpoint theo macro-F1.
        • ``test_mask``  — báo cáo cuối.

    ⇒ Tránh rò rỉ nhãn (mask-derived split) nhưng tận dụng được cấu trúc
    đồ thị cho message passing.

2.  3 chế độ mất cân bằng (``imbalance_mode``):
    • ``'none'``         : CrossEntropy thường.
    • ``'class_weight'`` : CrossEntropy(weight=...) — weight tính trên
      cạnh train, thứ tự lớp khớp ``class_to_idx``.
    • ``'undersample'``  : graph dựng từ df đã undersample lớp đa số
      (DDoS trên 34-1).

3.  Vòng train: Adam (lr/weight_decay từ cfg) + grad clip + early stop.
    Checkpoint tốt nhất (theo macro-F1 val) được lưu cùng metadata
    (class_to_idx, cfg, feature_dim, imbalance_mode, …) để evaluate.py
    dựng lại model đúng kiến trúc và dùng đúng chuẩn hoá.

4.  Reproducible: seed numpy + torch + random ngay đầu hàm.

CLI:
    python -m src.train --scenario <file> --model egraphsage \\
        --imbalance class_weight [--config config.yaml] [--epochs 30] \\
        [--save-dir checkpoints]

Lưu ý thiết bị:
    ``device = 'cuda' if torch.cuda.is_available() else 'cpu'``. KHÔNG
    hardcode ``.cuda()``. local (Mac M2 Pro) test ở CPU; train thật trên
    vast.ai GPU.
"""


from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

# ---- Đảm bảo 'src' luôn import được dù chạy script hay module ----
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


__all__ = [
    "set_seed",
    "get_device",
    "split_edge_masks",
    "safe_stratified_split",
    "macro_f1",
    "make_criterion",
    "train_one_epoch",
    "evaluate",
    "save_checkpoint",
    "load_checkpoint",
    "train_model",
    "run_scenario",
]


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Repro / device
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """
    Set seed toàn bộ (numpy + torch + random + PYTHONHASHSEED).

    KHÔNG đảm bảo full deterministic trên GPU với scatter ops của PyG;
    vẫn đảm bảo reproducibility trên CPU đủ cho baseline. Nếu cần
    strict reproducibility trên GPU: dùng ``torch.use_deterministic_algorithms``.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """Trả về cuda nếu có, ngược lại cpu. KHÔNG hardcode."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Edge mask (transductive split)
# ---------------------------------------------------------------------------

def safe_stratified_split(
    idx_pool: np.ndarray,
    y_pool: np.ndarray,
    test_size: float,
    seed: int,
    *,
    context: str = "",
    force_into_first: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wrapper quanh ``sklearn.train_test_split`` chịu được lớp CỰC HIẾM.

    Vấn đề
    -------
    Một số scenario IoT-23 (vd Okiru) có lớp chỉ 1-3 flow toàn dataset.
    ``train_test_split(..., stratify=y)`` raise ``ValueError: The least
    populated classes in y have only 1 member`` nếu BẤT KỲ lớp nào có
    < 2 mẫu. Đây là giới hạn của dữ liệu, KHÔNG phải lỗi code.

    Hành vi
    -------
    1. Đếm số mẫu mỗi lớp. Nếu MỌI lớp đều có >= 2 mẫu → stratified
       (giống cũ).
    2. Nếu có lớp < 2 mẫu → fallback ``stratify=None`` (random split)
       VÀ in cảnh báo liệt kê lớp nào quá hiếm + số mẫu (để ghi báo cáo).
    3. Nếu ``force_into_first`` được cung cấp (vd các mẫu singleton),
       ép CHÚNG vào phần "first" (train) SAU khi random split — đảm bảo
       mỗi lớp có ≥ 1 mẫu ở train "nếu có thể".
    4. Nếu cả stratified lẫn random đều fail (rất hiếm), raise rõ ràng.

    Args
    ----
    idx_pool : np.ndarray
        Index gốc sẽ được tách.
    y_pool : np.ndarray
        Nhãn tương ứng (cùng độ dài ``idx_pool``).
    test_size : float
        Tỉ lệ phần "second" (vd 0.30 cho train vs rest).
    seed : int
        Seed cho sklearn.
    context : str
        Nhãn ngữ cảnh (vd "split_edge_masks: step1 train vs rest") — chỉ
        để in warning cho dễ debug.
    force_into_first : np.ndarray | None
        Index PHẢI nằm trong phần "first" (train). Hữu ích cho singleton
        (count=1): chỉ có 1 chỗ để đi là train.

    Returns
    -------
    (idx_first, idx_second)
    """
    n = len(idx_pool)
    if n == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    unique, counts = np.unique(y_pool, return_counts=True)
    n_classes = len(unique)
    min_count = int(counts.min()) if n_classes > 0 else 0

    # Quyết định stratify được không
    use_stratify = (n_classes > 0) and (min_count >= 2)

    if not use_stratify and n_classes > 0:
        # In cảnh báo 1 LẦN — liệt kê TẤT CẢ lớp hiếm
        rare_pairs = [
            (int(c), int(cnt))
            for c, cnt in zip(unique, counts)
            if cnt < 2
        ]
        rare_str = ", ".join(f"class {c} (n={n_})" for c, n_ in rare_pairs)
        ctx = f" [{context}]" if context else ""
        n_singleton = sum(n_ for _, n_ in rare_pairs)
        force_msg = (
            f" {len(force_into_first)} mẫu singleton sẽ bị ÉP vào phần FIRST."
            if force_into_first is not None and len(force_into_first) > 0
            else ""
        )
        print(
            f"[safe_stratified_split]{ctx} ⚠ Có {len(rare_pairs)} lớp < 2 mẫu → "
            f"fallback sang random split (stratify=None). "
            f"Lớp hiếm: [{rare_str}]. "
            f"Tổng {n_singleton} mẫu singleton.{force_msg} "
            f"Đây là giới hạn dữ liệu, KHÔNG phải lỗi — sẽ ghi vào báo cáo."
        )

    # Thử stratified trước
    if use_stratify:
        try:
            idx_first, idx_second = train_test_split(
                idx_pool,
                test_size=test_size,
                stratify=y_pool,
                random_state=seed,
            )
            if force_into_first is not None and len(force_into_first) > 0:
                # Nếu forced index đã lọt vào idx_second, dời nó sang first.
                force_set = set(int(x) for x in force_into_first)
                idx_second = np.array(
                    [int(x) for x in idx_second if int(x) not in force_set],
                    dtype=np.int64,
                )
                idx_first = np.unique(
                    np.concatenate([idx_first, force_into_first])
                )
            return idx_first, idx_second
        except ValueError:
            # Defensive — không nên xảy ra nếu min_count >= 2
            print(
                f"[safe_stratified_split]{(' ['+context+']') if context else ''} "
                f"⚠ Stratify bất ngờ fail dù min_count={min_count}>=2, "
                f"fallback random."
            )

    # Fallback: random split (stratify=None)
    idx_first, idx_second = train_test_split(
        idx_pool,
        test_size=test_size,
        random_state=seed,
    )
    if force_into_first is not None and len(force_into_first) > 0:
        # Nếu forced index đã lọt vào idx_second, dời nó sang first.
        force_set = set(int(x) for x in force_into_first)
        idx_second = np.array(
            [int(x) for x in idx_second if int(x) not in force_set],
            dtype=np.int64,
        )
        idx_first = np.unique(
            np.concatenate([idx_first, force_into_first])
        )
    return idx_first, idx_second


def split_edge_masks(
    edge_label: torch.Tensor,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Stratified split theo ``edge_label`` thành 3 boolean mask [E].

    Dùng 2 lần ``safe_stratified_split``:
        1) train vs (val+test)    — tỉ lệ (1 - train_ratio).
        2) val vs test (trong rest) — tỉ lệ val/(val+test).

    Chịu được lớp cực hiếm
    -----------------------
    Nếu BẤT KỲ lớp nào có < 2 mẫu (vd Okiru-Attack chỉ 1-3 flow toàn
    dataset), hàm KHÔNG crash — fallback sang random split (stratify=None)
    VÀ in cảnh báo liệt kê lớp hiếm + số mẫu (để ghi vào báo cáo là
    giới hạn dữ liệu, không phải lỗi). Mọi singleton (count=1) được ÉP
    vào tập train để model còn thấy lớp đó khi học.

    Args
    ----
    edge_label : [E] long tensor.
    train_ratio, val_ratio, test_ratio : tỉ lệ, phải cộng = 1.0.
    seed : seed cho sklearn.

    Returns
    -------
    (train_mask, val_mask, test_mask) : 3 boolean tensor [E].
    """
    s = train_ratio + val_ratio + test_ratio
    if abs(s - 1.0) > 1e-6:
        raise ValueError(
            f"split_edge_masks: train+val+test = {s} ≠ 1.0."
        )
    y = edge_label.detach().cpu().numpy()
    idx_all = np.arange(len(y))
    E = len(y)

    # Tìm singleton (count == 1) trong edge_label.
    # Singleton BUỘC vào train vì chỉ có 1 chỗ để đi.
    unique_all, counts_all = np.unique(y, return_counts=True)
    singleton_classes = unique_all[counts_all == 1]
    singleton_indices = np.where(np.isin(y, singleton_classes))[0]
    pool_mask = ~np.isin(idx_all, singleton_indices)
    idx_pool = idx_all[pool_mask]
    y_pool = y[pool_mask]

    # Bước 1: train vs (val+test) trên pool (đã loại singleton).
    idx_train_pool, idx_rest = safe_stratified_split(
        idx_pool,
        y_pool,
        test_size=(1.0 - train_ratio),
        seed=seed,
        context="step1 train vs rest",
        force_into_first=singleton_indices,
    )
    # Gộp singleton vào train (an toàn: singleton không có trong idx_rest).
    idx_train = np.unique(
        np.concatenate([idx_train_pool, singleton_indices])
    )

    # Bước 2: val vs test trên phần còn lại.
    val_frac_of_rest = val_ratio / (val_ratio + test_ratio)
    y_rest = y[idx_rest]
    idx_val, idx_test = safe_stratified_split(
        idx_rest,
        y_rest,
        test_size=(1.0 - val_frac_of_rest),
        seed=seed,
        context="step2 val vs test",
        force_into_first=None,   # KHÔNG ép thêm — singleton đã vào train
    )

    # Sanity: 3 tập rời nhau, hợp = idx_all
    assert len(idx_train) + len(idx_val) + len(idx_test) == E, (
        f"split_edge_masks: tổng split ({len(idx_train)+len(idx_val)+len(idx_test)}) "
        f"!= E ({E})."
    )

    train_mask = torch.zeros(E, dtype=torch.bool)
    val_mask = torch.zeros(E, dtype=torch.bool)
    test_mask = torch.zeros(E, dtype=torch.bool)
    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True
    return train_mask, val_mask, test_mask


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def macro_f1(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    num_classes: int,
) -> float:
    """
    Macro-F1 (trung bình F1 qua các lớp 0..num_classes-1). Lớp không có
    mẫu nào trong ``y_true`` được tính F1 = 0 (không đóng góp vào mean).
    """
    yt = y_true.detach().cpu().numpy()
    yp = y_pred.detach().cpu().numpy()
    return float(
        f1_score(
            yt, yp,
            average='macro',
            labels=list(range(num_classes)),
            zero_division=0,
        )
    )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def make_criterion(
    imbalance_mode: str,
    weight_tensor: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Khởi tạo loss theo imbalance_mode.

    • 'none'         → CrossEntropyLoss() thường.
    • 'class_weight' → CrossEntropyLoss(weight=weight_tensor). Lỗi nếu
      weight_tensor is None.
    • 'undersample'  → CrossEntropyLoss() thường (graph đã được build từ
      df undersample, lớp đa số đã giảm — không cần weight nữa).
    """
    if imbalance_mode == 'none':
        return nn.CrossEntropyLoss()
    if imbalance_mode == 'class_weight':
        if weight_tensor is None:
            raise ValueError(
                "make_criterion: imbalance_mode='class_weight' cần "
                "weight_tensor khác None."
            )
        w = weight_tensor.float()
        if device is not None:
            w = w.to(device)
        return nn.CrossEntropyLoss(weight=w)
    if imbalance_mode == 'undersample':
        # Đồ thị đã build từ df undersample → phân bố lớp đã cân — không weight.
        return nn.CrossEntropyLoss()
    raise ValueError(
        f"make_criterion: imbalance_mode='{imbalance_mode}' không hỗ trợ. "
        f"Chọn 'none' | 'class_weight' | 'undersample'."
    )


# ---------------------------------------------------------------------------
# Forward / backward / eval (per-epoch)
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    data: Data,
    train_mask: torch.Tensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    """
    Train 1 epoch FULL-BATCH trên TOÀN ĐỒ THỊ (message passing dùng hết
    2E cạnh MP) nhưng loss/backprop CHỈ trên ``train_mask``.

    Trả về ``train_loss`` (float).
    """
    model.train()
    logits = model(data)                                # [E, K] trên device
    mask = train_mask.to(logits.device)
    logits_m = logits[mask]
    labels = data.edge_label.to(logits.device)[mask]

    loss = criterion(logits_m, labels)                  # CHỈ trên train_mask
    optimizer.zero_grad()
    loss.backward()
    if grad_clip and grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
    optimizer.step()
    return float(loss.item())


def evaluate(
    model: nn.Module,
    data: Data,
    mask: torch.Tensor,
    num_classes: int,
    device: torch.device,
) -> Tuple[float, float, torch.Tensor, torch.Tensor]:
    """
    Predict trên tập ``mask``. Trả về ``(loss, macro_f1, preds, labels)``.
    Loss ở đây CHỈ để in/so sánh — KHÔNG backprop.
    """
    model.eval()
    with torch.no_grad():
        logits = model(data)
    mask = mask.to(logits.device)
    logits_m = logits[mask]
    labels = data.edge_label.to(logits.device)[mask]
    loss = F.cross_entropy(logits_m, labels).item()
    preds = logits_m.argmax(dim=-1)
    f1 = macro_f1(labels, preds, num_classes)
    return loss, f1, preds, labels


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    path: str,
    *,
    class_to_idx: Dict[Any, int],
    cfg: Dict[str, Any],
    feature_dim: int,
    num_classes: int,
    imbalance_mode: str,
    val_macro_f1: float,
    history_meta: Dict[str, Any],
    feature_columns: Optional[Sequence[str]] = None,
) -> None:
    """Lưu state_dict + metadata cần để reconstruct model sau này."""
    ckpt = {
        'state_dict': model.state_dict(),
        'class_to_idx': dict(class_to_idx),
        'cfg': dict(cfg),
        'feature_dim': int(feature_dim),
        'num_classes': int(num_classes),
        'imbalance_mode': str(imbalance_mode),
        'val_macro_f1': float(val_macro_f1),
        'history_meta': dict(history_meta),
    }
    if feature_columns is not None:
        columns = [str(column) for column in feature_columns]
        if len(columns) != int(feature_dim):
            raise ValueError(
                "save_checkpoint: feature_columns length "
                f"{len(columns)} != feature_dim {feature_dim}."
            )
        ckpt['feature_columns'] = columns
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    torch.save(ckpt, path)
    logger.info("Đã lưu checkpoint: %s", path)


def load_checkpoint(path: str, model_template: nn.Module,
                    device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load state_dict vào ``model_template``; trả về (model trên device, ckpt dict)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if 'state_dict' not in ckpt:
        raise KeyError(f"load_checkpoint: file {path} không có 'state_dict'.")
    model_template.load_state_dict(ckpt['state_dict'])
    model_template = model_template.to(device)
    return model_template, ckpt


# ---------------------------------------------------------------------------
# Vòng train chính
# ---------------------------------------------------------------------------

def train_model(
    model_name: str,
    data: Data,
    cfg: Dict[str, Any],
    imbalance_mode: str = 'none',
    weight_tensor: Optional[torch.Tensor] = None,
    seed: int = 42,
    save_dir: str = 'checkpoints',
    verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, Any], str]:
    """
    Vòng train đầy đủ (Task 1.10).

    Parameters
    ----------
    model_name : str
        Tên model trong ``build_model``: 'egraphsage' | 'gcn' | 'graphsage'
        | 'sage_edge_concat'.
    data : torch_geometric.data.Data
        Đồ thị đã build bằng ``src.graph_build.build_graph``.
    cfg : dict
        Parse từ ``config.yaml``. Dùng:
            ``model.{hidden_dim,num_layers,dropout}``
            ``training.{learning_rate,weight_decay,epochs,grad_clip,
                        early_stop_patience,train_ratio,val_ratio,test_ratio}``
            ``logging.log_every_n_epochs``.
    imbalance_mode : str
        'none' | 'class_weight' | 'undersample'.
    weight_tensor : torch.Tensor hoặc None
        Vector [K] float, **chỉ cần** khi ``imbalance_mode='class_weight'``.
    seed : int
    save_dir : str
        Thư mục lưu checkpoint.

    Returns
    -------
    (best_model trên device, history dict, checkpoint_path).

    Lưu ý rò rỉ dữ liệu
    --------------------
    Mọi split/mask/eval đều dựa trên ``edge_label`` gốc; message passing
    thấy cả 2E cạnh MP. Vì nhãn trong test/val KHÔNG đi vào loss của
    train, không có rò rỉ nhãn.
    """
    set_seed(seed)
    device = get_device()

    if verbose:
        print(
            f"\n[train_model] model={model_name}  "
            f"imbalance={imbalance_mode}  seed={seed}  device={device}"
        )

    # ---- Di chuyển dữ liệu sang device ----
    # Giữ nguyên các thuộc tính custom (class_to_idx, feature_dim, ...) trong
    # Data; PyG's ``.to()`` di chuyển mọi tensor trong store.
    data = data.to(device)

    # ---- Model ----
    from src.core.model import build_model  # import trong hàm để tránh vòng lặp
    model = build_model(model_name, data, cfg)
    model = model.to(device)

    # ---- Loss & Optimizer ----
    criterion = make_criterion(
        imbalance_mode,
        weight_tensor=weight_tensor,
        device=device,
    )

    tr = cfg.get('training', {}) if isinstance(cfg, dict) else {}
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(tr.get('learning_rate', 1e-3)),
        weight_decay=float(tr.get('weight_decay', 0.0)),
    )

    # ---- Edge masks (transductive split) ----
    train_ratio = float(tr.get('train_ratio', 0.70))
    val_ratio = float(tr.get('val_ratio', 0.10))
    test_ratio = float(tr.get('test_ratio', 0.20))
    train_mask, val_mask, test_mask = split_edge_masks(
        data.edge_label,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    E = int(data.edge_index.shape[1])
    K = int(data.num_classes)
    if verbose:
        print(
            f"[split] E={E}  train={int(train_mask.sum())}  "
            f"val={int(val_mask.sum())}  test={int(test_mask.sum())}  "
            f"K={K}"
        )

    # ---- Vòng lặp train ----
    epochs = int(tr.get('epochs', 50))
    grad_clip = float(tr.get('grad_clip', 1.0))
    patience = int(tr.get('early_stop_patience', 10))
    log_every = int(cfg.get('logging', {}).get('log_every_n_epochs', 1))

    history: Dict[str, list] = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_macro_f1': [],
    }
    best_val_f1 = -1.0
    best_epoch = -1
    bad_epochs = 0
    final_epoch = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None

    if verbose:
        print(
            f"[train] epochs={epochs}  patience={patience}  "
            f"grad_clip={grad_clip}  lr={optimizer.param_groups[0]['lr']:.2e}"
        )

    for epoch in range(1, epochs + 1):
        final_epoch = epoch
        # ---- Train 1 epoch ----
        train_loss = train_one_epoch(
            model, data, train_mask, criterion, optimizer, device, grad_clip,
        )

        # ---- Eval trên val ----
        val_loss, val_f1, _, _ = evaluate(
            model, data, val_mask, K, device,
        )

        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_macro_f1'].append(val_f1)

        # ---- Logging (gọn) ----
        if verbose and (
            epoch == 1
            or epoch % log_every == 0
            or epoch == epochs
        ):
            tag = "  *" if val_f1 > best_val_f1 else ""
            print(
                f"  epoch {epoch:>3d}/{epochs}  "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"val_macroF1={val_f1:.4f}{tag}"
            )

        # ---- Best checkpoint trên val ----
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        # ---- Early stopping ----
        if bad_epochs >= patience:
            if verbose:
                print(
                    f"  [early-stop] dừng tại epoch {epoch}: "
                    f"val_macroF1 không cải thiện sau {patience} epoch."
                )
            break

    # ---- Khôi phục trọng số tốt nhất ----
    if best_state is not None:
        model.load_state_dict(
            {k: v.to(device) for k, v in best_state.items()}
        )

    # ---- Đánh giá cuối trên test_mask (chỉ để thông báo) ----
    test_loss, test_f1, _, _ = evaluate(
        model, data, test_mask, K, device,
    )
    if verbose:
        print(
            f"\n[best @ epoch {best_epoch}] "
            f"best_val_f1={best_val_f1:.4f} | "
            f"test_loss={test_loss:.4f}  test_macroF1={test_f1:.4f}"
        )

    # ---- Lưu checkpoint ----
    ckpt_path = os.path.join(
        save_dir, f"{model_name}_{imbalance_mode}_seed{seed}.pt"
    )
    save_checkpoint(
        model, ckpt_path,
        class_to_idx=dict(data.class_to_idx),
        cfg=cfg,
        feature_dim=int(data.feature_dim),
        num_classes=K,
        imbalance_mode=imbalance_mode,
        val_macro_f1=best_val_f1,
        history_meta={
            'best_epoch': int(best_epoch),
            'best_val_f1': float(best_val_f1),
            'test_f1': float(test_f1),
            'final_epoch': int(final_epoch),
            'seed': int(seed),
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
        },
        feature_columns=getattr(data, 'feature_columns', None),
    )

    history_out: Dict[str, Any] = dict(history)
    history_out['best_val_f1'] = float(best_val_f1)
    history_out['best_epoch'] = int(best_epoch)
    history_out['test_f1'] = float(test_f1)
    history_out['final_epoch'] = int(final_epoch)

    return model, history_out, ckpt_path


# ---------------------------------------------------------------------------
# Pipeline end-to-end cho CLI
# ---------------------------------------------------------------------------

def run_scenario(
    log_path: str,
    model_name: str = 'egraphsage',
    imbalance_mode: str = 'class_weight',
    config_path: str = 'config.yaml',
    epochs_override: Optional[int] = None,
    save_dir: str = 'checkpoints',
    early_stop_patience_override: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Pipeline standalone: load 1 ``conn.log.labeled`` → preprocess →
    build_graph → ``train_model``. In ra history tốt nhất.

    Dùng cho CLI và các test smoke.
    """
    from src.data_io import load_scenario
    from src.core.preprocess import clean_flows, fit_preprocessor, transform
    from src.imbalance import compute_class_weights, prepare_imbalance_variants
    from src.core.graph import build_graph

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    seed = int(cfg.get('reproducibility', {}).get('seed', 42))

    if verbose:
        print(f"[run_scenario] config seed = {seed}")

    # ---- Load + preprocess (CHẠY 1 LẦN, CPU) ----
    if verbose:
        print(f"[run_scenario] load + preprocess ...")
    df_clean = clean_flows(load_scenario(log_path))
    pre = fit_preprocessor(df_clean)
    df_feat = transform(df_clean, pre)

    # ---- Build graph theo imbalance_mode ----
    if imbalance_mode == 'class_weight':
        variants = prepare_imbalance_variants(df_feat, random_state=seed)
        weight_tensor = variants['weight_tensor']
        class_to_idx = variants['class_to_idx']
        df_for_graph = df_feat
        if verbose:
            print(f"[run_scenario] class_weight: weights = "
                  f"{variants['class_weights']}")
    elif imbalance_mode == 'undersample':
        variants = prepare_imbalance_variants(df_feat, random_state=seed)
        weight_tensor = None
        class_to_idx = variants['class_to_idx']
        df_for_graph = variants['undersampled']
        if verbose:
            print(f"[run_scenario] undersample: từ {len(df_feat)} → "
                  f"{len(df_for_graph)} dòng.")
    elif imbalance_mode == 'none':
        _, class_to_idx, _ = compute_class_weights(
            df_feat['detailed-label'].tolist(), scheme='balanced',
        )
        weight_tensor = None
        df_for_graph = df_feat
    else:
        raise ValueError(
            f"run_scenario: imbalance_mode='{imbalance_mode}' không hỗ trợ."
        )

    data = build_graph(
        df_for_graph, class_to_idx=class_to_idx,
        feature_columns=pre.feature_columns,
    )
    if verbose:
        print(
            f"[run_scenario] graph: N={int(data.num_nodes)}  "
            f"E={int(data.edge_index.shape[1])}  "
            f"F={int(data.feature_dim)}  K={int(data.num_classes)}"
        )

    # ---- Override epochs/patience nếu được yêu cầu (test) ----
    cfg_eff: Dict[str, Any] = dict(cfg)
    cfg_eff['training'] = dict(cfg.get('training', {}))
    if epochs_override is not None:
        cfg_eff['training']['epochs'] = int(epochs_override)
    if early_stop_patience_override is not None:
        cfg_eff['training']['early_stop_patience'] = int(
            early_stop_patience_override,
        )

    model, history, ckpt_path = train_model(
        model_name, data, cfg_eff,
        imbalance_mode=imbalance_mode,
        weight_tensor=weight_tensor,
        seed=seed,
        save_dir=save_dir,
        verbose=verbose,
    )

    return {
        'history': history,
        'checkpoint': ckpt_path,
        'cfg': cfg_eff,
        'model_name': model_name,
        'imbalance_mode': imbalance_mode,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Train edge-classification GNN trên 1 scenario IoT-23. "
            "Hyperparam đọc từ config.yaml; --epochs và "
            "--early-stop-patience có thể override."
        ),
    )
    p.add_argument(
        '--scenario', type=str, required=True,
        help='Đường dẫn tới file conn.log.labeled (vd '
             'data/CTU-IoT-Malware-Capture-34-1/conn.log.labeled).',
    )
    p.add_argument(
        '--model', type=str, default='egraphsage',
        choices=['egraphsage', 'gcn', 'graphsage', 'sage_edge_concat', 'gat'],
        help='Loại model (mặc định: egraphsage).',
    )
    p.add_argument(
        '--imbalance', type=str, default='class_weight',
        choices=['none', 'class_weight', 'undersample'],
        help='Cách xử lý mất cân bằng (mặc định: class_weight).',
    )
    p.add_argument(
        '--config', type=str, default='config.yaml',
        help='Đường dẫn config.yaml (mặc định: config.yaml).',
    )
    p.add_argument(
        '--epochs', type=int, default=None,
        help='Override số epoch (mặc định: lấy từ config.yaml).',
    )
    p.add_argument(
        '--early-stop-patience', type=int, default=None,
        help='Override patience cho early stopping.',
    )
    p.add_argument(
        '--save-dir', type=str, default='checkpoints',
        help='Thư mục lưu checkpoint (mặc định: checkpoints/).',
    )
    p.add_argument(
        '--quiet', action='store_true',
        help='Tắt log per-epoch (mặc định: in mỗi epoch).',
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(name)s: %(message)s',
    )

    result = run_scenario(
        log_path=args.scenario,
        model_name=args.model,
        imbalance_mode=args.imbalance,
        config_path=args.config,
        epochs_override=args.epochs,
        early_stop_patience_override=args.early_stop_patience,
        save_dir=args.save_dir,
        verbose=not args.quiet,
    )

    # In kết quả val/test cuối — theo spec.
    h = result['history']
    print(
        f"\n[RESULT] model={result['model_name']}  "
        f"imbalance={result['imbalance_mode']}\n"
        f"  best_val_macro_f1 = {h['best_val_f1']:.4f}  "
        f"@ epoch {h['best_epoch']}\n"
        f"  test_macro_f1      = {h['test_f1']:.4f}\n"
        f"  checkpoint         = {result['checkpoint']}"
    )


if __name__ == '__main__':
    main()
