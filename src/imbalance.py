"""
imbalance.py — Xử lý mất cân bằng lớp cho IoT-23.

IoT-23 lệch lớp tới hàng trăm triệu lần, NHƯNG lớp đa số THAY ĐỔI THEO
TỪNG SCENARIO (đã chốt trong CLAUDE.md mục 5). Do đó:
    • KHÔNG hardcode bất kỳ tên lớp nào trong code.
    • Mọi quyết định "lớp nào đa số", "lớp nào hiếm" đều tự tính từ
      value_counts của dữ liệu thật.

Chiến lược đã chốt (đã thống nhất 3 cấu hình train ở GĐ1):
    • NO-OP         : giữ nguyên, dùng macro-F1 để chịu.
    • CLASS-WEIGHT  : compute_class_weights → truyền vào CrossEntropyLoss.
    • UNDERSAMPLE   : undersample_majority trước khi dựng đồ thị (vì nó
                      bỏ bớt cạnh).
    Module này CHỈ lo phần dữ liệu (tính weight + undersample). Phần
    dùng weight trong loss để dành cho train.py.

QUY TẮC CHỐNG RÒ RỈ DỮ LIỆU:
    • compute_class_weights / undersample_majority CHỈ nhận df_train.
    • KHÔNG BAO GIỜ truyền df_test vào đây.
    • Test phải giữ nguyên phân bố thật để đánh giá công bằng.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cấu hình
# ---------------------------------------------------------------------------

LABEL_COLUMN = "detailed-label"   # cột nhãn multi-class

# Cột IP dùng để tính số node (đồ thị). KHÔNG tính trên label/feature.
IP_COLUMNS = ("id.orig_h", "id.resp_h")

# Scheme hợp lệ cho compute_class_weights.
VALID_SCHEMES = ("balanced",)


# ---------------------------------------------------------------------------
# 1. compute_class_weights
# ---------------------------------------------------------------------------

def compute_class_weights(
    y_train: Sequence[Any],
    scheme: str = "balanced",
) -> Tuple[Dict[Any, float], Dict[Any, int], "Any"]:
    """
    Tính trọng số lớp cho ``torch.nn.CrossEntropyLoss(weight=...)``.

    LỚP ĐA SỐ KHÔNG HARDCODE — tự tính từ phân bố của ``y_train``.

    Parameters
    ----------
    y_train : array-like
        Nhãn của tập TRAIN (cột 'detailed-label'). KHÔNG truyền test.
    scheme : str
        Chỉ hỗ trợ ``'balanced'`` (công thức sklearn):
            weight_c = n_samples / (n_classes * count_c)
        Lớp hiếm (count nhỏ) → weight cao; lớp đa số (count lớn) → weight thấp.

    Returns
    -------
    weights_dict : dict
        {class_name: weight} — đầy đủ mọi lớp có trong train.
    class_to_idx : dict
        {class_name: index} — index sắp theo TÊN LỚP tăng dần (alphabetical)
        để thứ tự weight LUÔN khớp với thứ tự lớp model dùng (deterministic).
    weight_tensor : torch.Tensor
        Vector 1-D, dtype float32, length = n_classes. Phần tử i ứng với
        lớp có ``class_to_idx[c] == i``. Truyền thẳng vào
        ``CrossEntropyLoss(weight=...)``.

    Raises
    ------
    ValueError
        Nếu ``scheme`` không hợp lệ hoặc ``y_train`` rỗng / 1 lớp.
    """
    if scheme not in VALID_SCHEMES:
        raise ValueError(
            f"compute_class_weights: scheme không hợp lệ '{scheme}'. "
            f"Chỉ hỗ trợ {VALID_SCHEMES}."
        )
    y_list = list(y_train)
    if len(y_list) == 0:
        raise ValueError("compute_class_weights: y_train rỗng.")

    counts: Dict[Any, int] = {}
    for v in y_list:
        counts[v] = counts.get(v, 0) + 1
    classes = sorted(counts.keys())  # sort TÊN → index ổn định giữa các lần chạy
    n = len(y_list)
    k = len(classes)

    weights_dict: Dict[Any, float] = {}
    for c in classes:
        c_count = counts[c]
        if c_count <= 0:
            raise ValueError(
                f"compute_class_weights: lớp '{c}' có count={c_count}."
            )
        if scheme == "balanced":
            w = n / (k * c_count)
        else:
            raise ValueError(f"scheme không hợp lệ: {scheme}")  # unreachable
        weights_dict[c] = float(w)

    class_to_idx = {c: i for i, c in enumerate(classes)}

    try:
        import torch
        weight_tensor = torch.tensor(
            [weights_dict[c] for c in classes],
            dtype=torch.float32,
        )
    except ImportError as e:
        raise ImportError(
            "compute_class_weights: cần cài torch để tạo weight_tensor. "
            "Chạy `pip install torch` hoặc dùng weights_dict/class_to_idx "
            "trực tiếp cho loss khác."
        ) from e

    logger.info(
        "compute_class_weights: scheme=%s, n=%d, n_classes=%d. "
        "Majority tự phát hiện: '%s' (%d mẫu, weight=%.4f).",
        scheme, n, k, max(counts, key=counts.get),
        max(counts.values()), weights_dict[max(counts, key=counts.get)],
    )
    return weights_dict, class_to_idx, weight_tensor


def print_class_weight_table(
    weights_dict: Dict[Any, float],
    counts: Optional[Dict[Any, int]] = None,
) -> None:
    """In bảng weight từng lớp (sort theo weight giảm dần — lớp hiếm nhất trước)."""
    rows = sorted(
        weights_dict.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )
    print("\nBảng class weights (sort theo weight giảm dần):")
    print(f"  {'class':<35s}  {'count':>10s}  {'weight':>10s}  {'count*weight':>12s}")
    print(f"  {'-'*35}  {'-'*10}  {'-'*10}  {'-'*12}")
    for c, w in rows:
        cnt = counts[c] if counts else "?"
        cw = (cnt * w) if counts else "?"
        print(f"  {str(c):<35s}  {str(cnt):>10s}  {w:>10.4f}  {str(cw):>12s}")


# ---------------------------------------------------------------------------
# 2. undersample_majority
# ---------------------------------------------------------------------------

def _target_count(
    counts: pd.Series,
    strategy: str,
    min_keep: Optional[int],
) -> int:
    """Quyết định cỡ mục tiêu sau undersample (dùng nội bộ)."""
    sorted_desc = counts.sort_values(ascending=False)
    if len(sorted_desc) == 0:
        raise ValueError("undersample_majority: counts rỗng.")
    if strategy == "to_second_largest":
        if len(sorted_desc) < 2:
            # Chỉ 1 lớp → giữ nguyên.
            target = int(sorted_desc.iloc[0])
        else:
            target = int(sorted_desc.iloc[1])
    else:
        raise ValueError(
            f"undersample_majority: strategy không hợp lệ '{strategy}'. "
            f"Hiện hỗ trợ 'to_second_largest'."
        )

    if min_keep is not None:
        target = min(target, int(min_keep))
    return target


def _graph_stats(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    ip_cols: Sequence[str] = IP_COLUMNS,
) -> Dict[str, int]:
    """Thống kê tác động lên cấu trúc đồ thị (chỉ dùng nội bộ)."""
    edges_before = len(df_before)
    edges_after = len(df_after)

    def _nodes(df: pd.DataFrame) -> set:
        s: set = set()
        for col in ip_cols:
            s.update(df[col].astype(str).tolist())
        return s

    nodes_before = _nodes(df_before)
    nodes_after = _nodes(df_after)
    # Node "cô lập do undersample" = node có mặt trước nhưng không còn
    # edge nào trong after (ước lượng bằng: biến mất khỏi df_after).
    isolated = nodes_before - nodes_after

    return {
        "edges_before": edges_before,
        "edges_after": edges_after,
        "nodes_before": len(nodes_before),
        "nodes_after": len(nodes_after),
        "isolated_nodes": len(isolated),
    }


def undersample_majority(
    df_train: pd.DataFrame,
    strategy: str = "to_second_largest",
    random_state: int = 42,
    min_keep: Optional[int] = None,
    label_column: str = LABEL_COLUMN,
    ip_columns: Sequence[str] = IP_COLUMNS,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Undersample lớp đa số để giảm mất cân bằng. **CHỈ áp dụng trên train**.

    LỚP ĐA SỐ TỰ TÍNH từ value_counts (KHÔNG hardcode tên lớp).

    Parameters
    ----------
    df_train : pd.DataFrame
        DataFrame đã transform (P3), có cột ``label_column``. KHÔNG truyền test.
    strategy : str
        Cách chọn cỡ mục tiêu:
            - ``'to_second_largest'`` (mặc định): hạ lớp đa số xuống bằng
              cỡ lớp lớn-thứ-hai. Tránh cào bằng về lớp hiếm nhất.
    random_state : int
        Seed cho random sampling (mặc định 42).
    min_keep : int, optional
        Cap tuyệt đối (max_per_class): số mẫu TỐI ĐA giữ lại cho mỗi lớp
        đa số. ``target_eff = min(target_from_strategy, min_keep)``.
        Dùng cho scenario lớn (vd 39-1 10GB) để chặn memory.
        ``None`` → không cap.
    label_column : str
        Tên cột nhãn (mặc định ``'detailed-label'``).
    ip_columns : sequence of str
        2 cột IP dùng để tính số node cho thống kê đồ thị.
    verbose : bool
        In thống kê tác động đồ thị (mặc định True).

    Returns
    -------
    pd.DataFrame
        Bản sao df_train đã undersample. Index được reset về 0..N-1.
        Mỗi lớp có count <= target_eff. KHÔNG đụng lớp có count <= target.
    """
    if label_column not in df_train.columns:
        raise KeyError(
            f"undersample_majority: thiếu cột nhãn '{label_column}'."
        )

    counts = df_train[label_column].value_counts()
    target = _target_count(counts, strategy, min_keep)

    rng = np.random.default_rng(random_state)
    parts = []
    for cls, cnt in counts.items():
        class_df = df_train[df_train[label_column] == cls]
        if cnt > target:
            keep_n = target
            pick_idx = rng.choice(len(class_df), size=keep_n, replace=False)
            class_df = class_df.iloc[pick_idx]
            logger.info(
                "undersample_majority: lớp '%s' có %d → giữ %d (target=%d).",
                cls, cnt, keep_n, target,
            )
        else:
            logger.info(
                "undersample_majority: lớp '%s' có %d ≤ target=%d → giữ nguyên.",
                cls, cnt, target,
            )
        parts.append(class_df)

    out = pd.concat(parts, axis=0)
    # Shuffle để tránh lớp nào cũng dồn cụm (đã có random_state ở rng
    # trên, nhưng shuffle cuối đảm bảo thứ tự không lộ class).
    out = out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    if verbose:
        stats = _graph_stats(df_train, out, ip_cols=ip_columns)
        print("\n" + "=" * 70)
        print("TÁC ĐỘNG CỦA UNDERSAMPLE LÊN CẤU TRÚC ĐỒ THỊ")
        print("=" * 70)
        print(f"  Edges (flow/cạnh): {stats['edges_before']:,} → "
              f"{stats['edges_after']:,}  "
              f"(giảm {stats['edges_before'] - stats['edges_after']:,}, "
              f"{(1 - stats['edges_after'] / stats['edges_before']) * 100:.1f}%)")
        print(f"  Nodes (IP duy nhất): {stats['nodes_before']:,} → "
              f"{stats['nodes_after']:,}  "
              f"(giảm {stats['nodes_before'] - stats['nodes_after']:,})")
        print(f"  Nodes cô lập (mất hết cạnh): {stats['isolated_nodes']:,} "
              f"({stats['isolated_nodes'] / max(stats['nodes_before'], 1) * 100:.2f}% "
              f"tổng node ban đầu)")
        print(f"\n  Phân bố '{label_column}' TRƯỚC undersample:")
        for c, n in counts.items():
            print(f"    {str(c):<35s} {n:>10,}")
        print(f"\n  Phân bố '{label_column}' SAU undersample (target={target:,}):")
        new_counts = out[label_column].value_counts()
        for c, n in new_counts.items():
            print(f"    {str(c):<35s} {n:>10,}")
        print("=" * 70)

    return out


# ---------------------------------------------------------------------------
# 3. prepare_imbalance_variants
# ---------------------------------------------------------------------------

def prepare_imbalance_variants(
    df_train: pd.DataFrame,
    strategy: str = "to_second_largest",
    random_state: int = 42,
    min_keep: Optional[int] = None,
    label_column: str = LABEL_COLUMN,
) -> Dict[str, Any]:
    """
    Chuẩn bị dữ liệu cho 3 cấu hình imbalance ở GĐ1.

    Trả về dict để train.py dễ chọn:
        {
          'full':           df_train (không đổi),
          'undersampled':   df_train sau undersample_majority,
          'class_weights':  {class: weight},       # tính trên 'full'
          'class_to_idx':   {class: idx},          # sort theo tên lớp
          'weight_tensor':  torch.Tensor (float32),
        }

    Class weights LUÔN tính trên 'full' (không phải sau undersample) — đây
    là điểm quan trọng: dù dùng cấu hình nào, nếu không undersample thì
    weight là của phân bố thật; còn nếu dùng 'undersampled' thì weight
    sẽ là của phân bố đã cân bằng — caller tự quyết. Ở đây cung cấp weight
    của 'full' làm mặc định cho cấu hình 'class_weight'.

    Parameters
    ----------
    df_train : pd.DataFrame
        DataFrame TRAIN đã transform (P3). KHÔNG truyền test.
    strategy, random_state, min_keep, label_column : xem ``undersample_majority``.

    Returns
    -------
    dict
        Xem mô tả ở trên.
    """
    if label_column not in df_train.columns:
        raise KeyError(
            f"prepare_imbalance_variants: thiếu cột nhãn '{label_column}'."
        )

    weights_dict, class_to_idx, weight_tensor = compute_class_weights(
        df_train[label_column].tolist(), scheme="balanced",
    )

    df_under = undersample_majority(
        df_train,
        strategy=strategy,
        random_state=random_state,
        min_keep=min_keep,
        label_column=label_column,
        verbose=False,  # caller tự in nếu muốn
    )

    return {
        "full": df_train.reset_index(drop=True),
        "undersampled": df_under,
        "class_weights": weights_dict,
        "class_to_idx": class_to_idx,
        "weight_tensor": weight_tensor,
    }


# ---------------------------------------------------------------------------
# Mock test
# ---------------------------------------------------------------------------

# Mock data NHỎ sau clean_flows + transform (chỉ cần các cột liên quan để test
# imbalance: detailed-label + id.orig_h + id.resp_h).
_MOCK_TRANSFORM_DF = pd.DataFrame(
    [
        # DDoS (10 — đa số)
        ("10.0.0.1", "8.8.8.8", "DDoS"),
        ("10.0.0.1", "8.8.4.4", "DDoS"),
        ("10.0.0.2", "8.8.8.8", "DDoS"),
        ("10.0.0.3", "1.1.1.1", "DDoS"),
        ("10.0.0.1", "1.1.1.1", "DDoS"),
        ("10.0.0.4", "8.8.8.8", "DDoS"),
        ("10.0.0.5", "8.8.4.4", "DDoS"),
        ("10.0.0.6", "1.1.1.1", "DDoS"),
        ("10.0.0.7", "8.8.8.8", "DDoS"),
        ("10.0.0.8", "8.8.4.4", "DDoS"),
        # C&C (5 — lớn thứ hai)
        ("10.0.0.1", "9.9.9.9", "C&C"),
        ("10.0.0.2", "9.9.9.9", "C&C"),
        ("10.0.0.9", "9.9.9.9", "C&C"),
        ("10.0.0.10", "9.9.9.9", "C&C"),
        ("10.0.0.11", "9.9.9.9", "C&C"),
        # Benign (3)
        ("10.0.0.1", "10.0.0.1", "Benign"),  # self-loop (cũng tính)
        ("10.0.0.2", "10.0.0.2", "Benign"),
        ("10.0.0.12", "10.0.0.12", "Benign"),
        # PortScan (2 — hiếm nhất)
        ("10.0.0.1", "10.0.0.99", "PartOfAHorizontalPortScan"),
        ("10.0.0.2", "10.0.0.99", "PartOfAHorizontalPortScan"),
    ],
    columns=["id.orig_h", "id.resp_h", "detailed-label"],
)


def _run_mock_test() -> None:
    """Mock test cho compute_class_weights + undersample_majority + variants."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    df = _MOCK_TRANSFORM_DF.copy()
    print(f"\n>>> [MOCK] Mock df shape: {df.shape}")
    print(f">>> [MOCK] Phân bố detailed-label:\n"
          f"{df['detailed-label'].value_counts().to_string()}")

    # ---- compute_class_weights ----
    weights_dict, class_to_idx, weight_tensor = compute_class_weights(
        df["detailed-label"].tolist(), scheme="balanced",
    )
    print_class_weight_table(
        weights_dict, counts=dict(df["detailed-label"].value_counts()),
    )

    # Assert: weight tỉ lệ nghịch count (PortScan hiếm nhất → weight cao nhất).
    counts = df["detailed-label"].value_counts()
    max_cls = counts.idxmax()
    min_cls = counts.idxmin()
    assert weights_dict[min_cls] > weights_dict[max_cls], (
        f"[MOCK] weight của lớp hiếm '{min_cls}' phải > weight lớp đa số "
        f"'{max_cls}'."
    )
    # Assert: class_to_idx sort theo tên lớp (alphabetical).
    assert list(class_to_idx.keys()) == sorted(class_to_idx.keys()), (
        "[MOCK] class_to_idx phải sort theo tên lớp."
    )
    # Assert: weight_tensor khớp với class_to_idx.
    for c, idx in class_to_idx.items():
        assert abs(float(weight_tensor[idx]) - weights_dict[c]) < 1e-6, (
            f"[MOCK] weight_tensor[{idx}]={float(weight_tensor[idx])} != "
            f"weights_dict['{c}']={weights_dict[c]}"
        )
    # Assert: tổng count * weight = n (đặc trưng 'balanced').
    n = len(df)
    k = len(counts)
    s = sum(counts[c] * weights_dict[c] for c in counts.index)
    assert abs(s - n) < 1e-6, (
        f"[MOCK] Σ count*weight phải = n={n}, got {s}"
    )
    # Không hardcode — lớp đa số tự phát hiện đúng.
    assert max_cls == "DDoS", (
        f"[MOCK] lớp đa số phải tự phát hiện là DDoS, got {max_cls}"
    )

    # ---- undersample_majority ----
    df_under = undersample_majority(
        df, strategy="to_second_largest", random_state=42,
    )
    # Target = second largest = 5 (C&C). DDoS 10 → 5, các lớp khác giữ nguyên.
    assert int((df_under["detailed-label"] == "DDoS").sum()) == 5, (
        f"[MOCK] DDoS phải còn 5, got "
        f"{(df_under['detailed-label'] == 'DDoS').sum()}"
    )
    assert int((df_under["detailed-label"] == "C&C").sum()) == 5
    assert int((df_under["detailed-label"] == "Benign").sum()) == 3
    assert int((df_under["detailed-label"] ==
                "PartOfAHorizontalPortScan").sum()) == 2
    # Lớp hiếm KHÔNG bị đụng.
    assert df_under.shape[0] == 5 + 5 + 3 + 2, (
        f"[MOCK] tổng sau undersample = 15, got {df_under.shape[0]}"
    )

    # ---- min_keep (cap tuyệt đối) ----
    df_under_capped = undersample_majority(
        df, strategy="to_second_largest", random_state=42, min_keep=3,
    )
    # Target_eff = min(5, 3) = 3. DDoS 10 → 3; C&C 5 → 3 (cũng bị cap).
    assert int((df_under_capped["detailed-label"] == "DDoS").sum()) == 3, (
        f"[MOCK] với min_keep=3, DDoS phải còn 3, got "
        f"{(df_under_capped['detailed-label'] == 'DDoS').sum()}"
    )
    assert int((df_under_capped["detailed-label"] == "C&C").sum()) == 3, (
        f"[MOCK] với min_keep=3, C&C cũng bị cap 5→3, got "
        f"{(df_under_capped['detailed-label'] == 'C&C').sum()}"
    )
    # Lớp có count <= min_keep giữ nguyên.
    assert int((df_under_capped["detailed-label"] == "Benign").sum()) == 3
    assert int((df_under_capped["detailed-label"] ==
                "PartOfAHorizontalPortScan").sum()) == 2

    # ---- prepare_imbalance_variants ----
    variants = prepare_imbalance_variants(df, random_state=42)
    assert "full" in variants and "undersampled" in variants
    assert "class_weights" in variants and "class_to_idx" in variants
    assert "weight_tensor" in variants
    # 'full' phải chính là df_train (shape, value_counts).
    assert variants["full"].shape == df.shape
    assert (variants["full"]["detailed-label"].value_counts()
            == df["detailed-label"].value_counts()).all()

    print("\n[MOCK TEST imbalance] Tất cả assertions đều PASS.")


# ---------------------------------------------------------------------------
# Real test (file thật 34-1, sau clean + transform + 80/20 split)
# ---------------------------------------------------------------------------

def _run_real_test(path: str) -> None:
    """Test imbalance trên file thật 34-1."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    from sklearn.model_selection import train_test_split
    from src.data_io import load_scenario
    from src.core.preprocess import clean_flows, fit_preprocessor, transform

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    print(f"\n>>> [REAL] Real file: {path}")

    df_clean = clean_flows(load_scenario(path))
    pre = fit_preprocessor(df_clean)
    df_feat = transform(df_clean, pre)

    # 80/20 stratified theo detailed-label (chỉ để test).
    df_train, df_test = train_test_split(
        df_feat,
        test_size=0.2,
        stratify=df_feat["detailed-label"],
        random_state=42,
    )
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    print(f">>> [REAL] Train: {df_train.shape}, Test: {df_test.shape}")

    # Lưu lại phân bố test GỐC để xác nhận KHÔNG bị đụng.
    test_dist_before = df_test["detailed-label"].value_counts().to_dict()
    test_shape_before = df_test.shape
    test_columns_before = list(df_test.columns)

    # ---- compute_class_weights ----
    print("\n>>> [REAL] compute_class_weights trên TRAIN:")
    weights_dict, class_to_idx, weight_tensor = compute_class_weights(
        df_train["detailed-label"].tolist(), scheme="balanced",
    )
    train_counts = df_train["detailed-label"].value_counts().to_dict()
    print_class_weight_table(weights_dict, counts=train_counts)

    # Kỳ vọng: PortScan (chỉ ~98 mẫu train) có weight rất cao.
    portscan_weight = weights_dict["PartOfAHorizontalPortScan"]
    ddos_weight = weights_dict["DDoS"]
    print(f"\n>>> [REAL] PortScan weight = {portscan_weight:.4f}, "
          f"DDoS weight = {ddos_weight:.4f} "
          f"(tỉ lệ ≈ {portscan_weight / ddos_weight:.1f}×).")
    assert portscan_weight > ddos_weight, (
        "[REAL] PortScan (hiếm) phải có weight > DDoS (đa số)."
    )
    # Tỉ lệ PortScan/DDoS trên 34-1 ≈ 11515/98 ≈ 117× (đúng imbalance gốc).
    ratio = portscan_weight / ddos_weight
    assert 80 < ratio < 200, (
        f"[REAL] tỉ lệ weight PortScan/DDoS ≈ {ratio:.1f}×, kỳ vọng ~117×."
    )

    # ---- undersample_majority ----
    print("\n>>> [REAL] Chạy undersample_majority (to_second_largest):")
    df_under = undersample_majority(
        df_train, strategy="to_second_largest", random_state=42,
    )

    # Sanity: target = second_largest = ~5365 (C&C), DDoS ~11515 → ~5365.
    new_ddos = int((df_under["detailed-label"] == "DDoS").sum())
    second_largest = sorted(train_counts.values(), reverse=True)[1]
    assert new_ddos == second_largest, (
        f"[REAL] DDoS sau undersample phải = second_largest={second_largest}, "
        f"got {new_ddos}."
    )
    # Lớp hiếm KHÔNG bị đụng.
    for rare_cls in ("PartOfAHorizontalPortScan", "Benign"):
        before = train_counts.get(rare_cls, 0)
        after = int((df_under["detailed-label"] == rare_cls).sum())
        assert after == before, (
            f"[REAL] lớp hiếm '{rare_cls}' bị đụng ({before} → {after})."
        )

    # ---- Xác nhận TEST KHÔNG bị đụng ----
    test_dist_after = df_test["detailed-label"].value_counts().to_dict()
    assert test_dist_after == test_dist_before, (
        f"[REAL] TEST bị thay đổi phân bố! before={test_dist_before}, "
        f"after={test_dist_after}"
    )
    assert df_test.shape == test_shape_before, (
        f"[REAL] TEST bị thay đổi shape! before={test_shape_before}, "
        f"after={df_test.shape}"
    )
    assert list(df_test.columns) == test_columns_before, (
        "[REAL] TEST bị thay đổi schema cột!"
    )
    print("\n>>> [REAL] OK — TEST KHÔNG bị đụng (phân bố, shape, schema "
          "đều nguyên vẹn).")

    # ---- prepare_imbalance_variants ----
    variants = prepare_imbalance_variants(df_train, random_state=42)
    assert variants["full"].shape == df_train.shape
    assert variants["undersampled"].shape[0] == df_under.shape[0]
    assert variants["class_weights"] == weights_dict

    print("\n[REAL TEST imbalance] Tất cả assertions đều PASS.")


if __name__ == "__main__":
    # Chạy:
    #   python -m src.imbalance                     → mock test.
    #   python -m src.imbalance <path/to/log>       → real test.
    if len(sys.argv) >= 2:
        _run_real_test(sys.argv[1])
    else:
        _run_mock_test()
