"""
graph_build.py — Dựng đồ thị PyG từ DataFrame đã qua preprocess.

Quy tắc cốt lõi (đã chốt trong CLAUDE.md mục 6):

    • node = IP, cạnh = flow.
    • Đặc trưng hành vi gắn lên CẠNH (`edge_attr`).
    • Đây là bài toán EDGE classification (phân loại từng flow = từng cạnh),
      KHÔNG phải node classification. Nhãn `detailed-label` gắn trên CẠNH.
    • E-GraphSAGE: đặc trưng node khởi tạo là vector HẰNG (all-ones) — thông
      tin phân biệt nằm ở cạnh, không ở node.
    • Map IP → chỉ số node (index) liên tục 0..N-1; `edge_index` phải là
      tensor `[2, num_edges]` kiểu long. Đây là chỗ dễ sai âm thầm nhất.
    • GĐ1: gộp toàn bộ flow của một scenario thành MỘT đồ thị tĩnh duy nhất.

Quy ước tensor trong Data trả về:
    • x                      : [N, node_in_dim] float32 — all-ones.
    • edge_index             : [2, E]   long   — cạnh GỐC (orig → resp),
                                                   dùng cho loss & đánh giá.
    • edge_attr              : [E, F]   float32 — đặc trưng cạnh GỐC.
    • edge_label             : [E]      long    — nhãn đa lớp (multi-class).
    • edge_label_binary      : [E]      long    — 0=Benign, 1=Malicious.
    • edge_index_mp          : [2, 2E]  long    — gốc + đảo (concat), dùng cho
                                                   message passing.
    • edge_attr_mp           : [2E, F]  float32 — đặc trưng cho cạnh message
                                                   passing (gốc + đảo, dùng lại
                                                   attr gốc).
    • num_nodes              : N
    • ip_to_idx              : dict {ip_str: idx} (truy vết ngược, KHÔNG phải
                                                   input của model).
    • feature_dim, num_classes, class_to_idx : metadata.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from torch_geometric.data import Data


logger = logging.getLogger(__name__)


IP_COLUMNS: Tuple[str, str] = ("id.orig_h", "id.resp_h")
LABEL_COLUMN: str = "detailed-label"


# ---------------------------------------------------------------------------
# Hàm chính
# ---------------------------------------------------------------------------

def build_graph(
    df: pd.DataFrame,
    class_to_idx: Dict[Any, int],
    feature_columns: Sequence[str],
    node_in_dim: int = 1,
    label_column: str = LABEL_COLUMN,
    ip_columns: Sequence[str] = IP_COLUMNS,
) -> Data:
    """
    Dựng đồ thị PyG từ DataFrame đã qua transform ở P3.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame đã transform. Cần chứa:
            - 2 cột IP trong ``ip_columns`` (mặc định id.orig_h, id.resp_h).
            - Cột ``label_column`` (mặc định 'detailed-label').
            - Mọi cột trong ``feature_columns`` (đúng thứ tự).
    class_to_idx : dict
        Ánh xạ ``{class_name: index}`` từ
        ``imbalance.compute_class_weights`` (sort tên lớp).
        Đảm bảo thứ tự index ổn định giữa train/test/GĐ2.
    feature_columns : sequence of str
        Danh sách tên cột feature THEO THỨ TỰ từ
        ``Preprocessor.feature_columns``. KHÔNG tự sắp lại.
    node_in_dim : int
        Số chiều vector hằng của node feature (mặc định 1, theo tinh thần
        E-GraphSAGE: thông tin phân biệt nằm ở cạnh).
    label_column : str
        Tên cột nhãn đa lớp (mặc định 'detailed-label').
    ip_columns : sequence of str
        2 cột IP dùng làm node id.

    Returns
    -------
    torch_geometric.data.Data
        Xem danh sách tensor ở docstring đầu file.

    Raises
    ------
    KeyError
        Thiếu cột bắt buộc.
    ValueError
        Có nhãn trong df không có trong ``class_to_idx``.
    AssertionError
        Sai bất biến shape / range index (chỉ ra lỗi lập trình).
    """
    # ---- Validate inputs ----
    ip_cols = list(ip_columns)
    for col in ip_cols + [label_column]:
        if col not in df.columns:
            raise KeyError(
                f"build_graph: thiếu cột '{col}' trong DataFrame."
            )
    feat_cols = list(feature_columns)
    if not feat_cols:
        raise ValueError("build_graph: feature_columns rỗng.")
    for col in feat_cols:
        if col not in df.columns:
            raise KeyError(
                f"build_graph: thiếu cột feature '{col}' trong DataFrame."
            )
    if len(df) == 0:
        raise ValueError("build_graph: DataFrame rỗng (0 dòng).")

    if node_in_dim < 1:
        raise ValueError(
            f"build_graph: node_in_dim={node_in_dim} < 1."
        )

    # ---- Bước 1: Map IP -> node index (sort → deterministic) ----
    all_ips: set = set()
    for col in ip_cols:
        # Ép str để IP nhất quán (tránh 192.168.1.1 vs '192.168.1.1').
        all_ips.update(df[col].astype(str).tolist())
    sorted_ips = sorted(all_ips)
    ip_to_idx: Dict[str, int] = {ip: i for i, ip in enumerate(sorted_ips)}
    num_nodes: int = len(ip_to_idx)

    # ---- Bước 2: edge_index (cạnh GỐC, orig → resp) ----
    src = df[ip_cols[0]].astype(str).map(ip_to_idx).to_numpy()
    dst = df[ip_cols[1]].astype(str).map(ip_to_idx).to_numpy()
    # np.stack theo axis=0 → shape [2, E].
    edge_index_np = np.stack([src, dst], axis=0)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    num_edges: int = edge_index.shape[1]

    # ---- Bước 3: edge_attr (đặc trưng cạnh) ----
    # Thứ tự cột LẤY ĐÚNG từ feature_columns (KHÔNG sắp lại).
    feat_vals = (
        df[feat_cols].astype("float32").to_numpy()
    )
    edge_attr = torch.tensor(feat_vals, dtype=torch.float32)

    # ---- Bước 4: edge_label (đa lớp) & edge_label_binary ----
    label_series = df[label_column].astype(str)
    unknown_labels = sorted(
        set(label_series.unique()) - set(class_to_idx.keys())
    )
    if unknown_labels:
        raise ValueError(
            f"build_graph: nhãn không có trong class_to_idx: {unknown_labels}. "
            f"Cần fit preprocessor/imbalance trên train trước."
        )
    edge_label_np = label_series.map(class_to_idx).to_numpy()
    edge_label = torch.tensor(edge_label_np, dtype=torch.long)
    # Benign = 0, mọi nhãn khác (đã biết là Malicious) = 1.
    edge_label_binary = torch.tensor(
        (label_series != "Benign").astype(np.int64).to_numpy(),
        dtype=torch.long,
    )

    # ---- Bước 5: Node features x = all-ones [N, node_in_dim] ----
    x = torch.ones((num_nodes, node_in_dim), dtype=torch.float32)

    # ---- Bước 6: Cạnh đảo cho message passing ----
    # Gốc: (src, dst). Đảo: (dst, src). Concat theo cột (dim=1) → [2, 2E].
    edge_index_rev = edge_index.flip(0)
    edge_index_mp = torch.cat([edge_index, edge_index_rev], dim=1)
    # Đặc trưng cạnh đảo dùng LẠI attr gốc (thông tin hành vi không đổi khi
    # đảo chiều, vì cạnh vô hướng về mặt nội dung).
    edge_attr_mp = torch.cat([edge_attr, edge_attr], dim=0)

    # ---- Bước 7: Đóng gói Data ----
    num_classes = len(class_to_idx)
    feature_dim = len(feat_cols)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_label=edge_label,
        edge_label_binary=edge_label_binary,
        edge_index_mp=edge_index_mp,
        edge_attr_mp=edge_attr_mp,
        num_nodes=num_nodes,
    )
    # Metadata (không phải standard PyG field → gán sau).
    data.ip_to_idx = ip_to_idx
    data.feature_dim = feature_dim
    data.feature_columns = list(feat_cols)
    data.num_classes = num_classes
    data.class_to_idx = dict(class_to_idx)  # copy để an toàn

    # ---- Bất biến bắt buộc (assert) ----
    assert data.edge_index.shape == (2, num_edges), (
        f"edge_index.shape={tuple(data.edge_index.shape)} != (2, {num_edges})"
    )
    assert data.edge_attr.shape == (num_edges, feature_dim), (
        f"edge_attr.shape={tuple(data.edge_attr.shape)} != "
        f"({num_edges}, {feature_dim})"
    )
    assert data.edge_label.shape == (num_edges,), (
        f"edge_label.shape={tuple(data.edge_label.shape)} != ({num_edges},)"
    )
    assert data.edge_label_binary.shape == (num_edges,), (
        f"edge_label_binary.shape={tuple(data.edge_label_binary.shape)} != "
        f"({num_edges},)"
    )
    assert data.edge_index_mp.shape == (2, 2 * num_edges), (
        f"edge_index_mp.shape={tuple(data.edge_index_mp.shape)} != "
        f"(2, {2 * num_edges})"
    )
    assert data.edge_attr_mp.shape == (2 * num_edges, feature_dim), (
        f"edge_attr_mp.shape={tuple(data.edge_attr_mp.shape)} != "
        f"({2 * num_edges}, {feature_dim})"
    )
    # Mọi index node nằm trong [0, N).
    assert int(data.edge_index.min()) >= 0, (
        f"edge_index có giá trị âm: {int(data.edge_index.min())}"
    )
    assert int(data.edge_index.max()) < num_nodes, (
        f"edge_index.max()={int(data.edge_index.max())} >= num_nodes={num_nodes}"
    )
    # Số lớp xuất hiện trong edge_label <= num_classes.
    unique_labels = torch.unique(data.edge_label).tolist()
    assert max(unique_labels) < num_classes, (
        f"edge_label có giá trị {max(unique_labels)} >= num_classes={num_classes}"
    )

    logger.info(
        "build_graph: N=%d node, E=%d edge, F=%d feature_dim, K=%d classes. "
        "(%d cạnh message passing.)",
        num_nodes, num_edges, feature_dim, num_classes, 2 * num_edges,
    )
    return data


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------

def save_graph(data: Data, path: str) -> None:
    """Lưu Data ra file .pt để tái dùng (artifacts/<graph_file>.pt)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    # weights_only=False vì Data có chứa dict (ip_to_idx, class_to_idx)
    # — torch >= 2.5 default weights_only=True sẽ từ chối.
    torch.save(data, path)
    logger.info("save_graph: đã ghi %s.", path)


def load_graph(path: str) -> Data:
    """Load Data từ file .pt."""
    data = torch.load(path, weights_only=False)
    if not isinstance(data, Data):
        raise TypeError(
            f"load_graph: file {path} không chứa PyG Data "
            f"(got {type(data).__name__})."
        )
    logger.info("load_graph: đã load %s.", path)
    return data


# ---------------------------------------------------------------------------
# Thống kê đồ thị
# ---------------------------------------------------------------------------

def graph_stats(data: Data) -> Dict[str, Any]:
    """
    Tính & in thống kê đồ thị. Trả về dict để gọi từ test.

    In:
        - N (num_nodes)
        - E (cạnh GỐC)
        - E_mp (cạnh message passing = 2E)
        - feature_dim, num_classes
        - Số node cô lập (degree 0 tính trên đồ thị VÔ HƯỚNG từ cạnh gốc)
        - Phân bố edge_label (đa lớp)
        - Phân bố edge_label_binary (Benign/Malicious)
    """
    n = int(data.num_nodes)
    e = int(data.edge_index.shape[1])
    e_mp = int(data.edge_index_mp.shape[1])
    fd = int(data.feature_dim)
    nc = int(data.num_classes)

    # Đếm degree vô hướng (cộng cả in + out).
    deg = torch.zeros(n, dtype=torch.long)
    src = data.edge_index[0]
    dst = data.edge_index[1]
    ones = torch.ones_like(src, dtype=torch.long)
    deg.scatter_add_(0, src, ones)
    deg.scatter_add_(0, dst, ones)
    n_isolated = int((deg == 0).sum())

    # Phân bố edge_label (đa lớp).
    label_counts = torch.bincount(data.edge_label, minlength=nc).tolist()

    # Phân bố edge_label_binary.
    bin_counts = torch.bincount(data.edge_label_binary, minlength=2).tolist()

    # In.
    print("=" * 70)
    print("GRAPH STATS")
    print("=" * 70)
    print(f"  N (num_nodes):           {n}")
    print(f"  E (cạnh GỐC):            {e:,}")
    print(f"  E_mp (message passing):  {e_mp:,}  (= 2E cho cạnh đảo)")
    print(f"  feature_dim:             {fd}")
    print(f"  num_classes:             {nc}")
    print(f"  Node cô lập (deg=0):     {n_isolated}  "
          f"({n_isolated / max(n, 1) * 100:.2f}% tổng node)")
    print()
    print(f"  Phân bố edge_label (đa lớp):")
    cls_to_idx = dict(data.class_to_idx)
    idx_to_cls = {v: k for k, v in cls_to_idx.items()}
    for idx in range(nc):
        name = idx_to_cls.get(idx, f"class_{idx}")
        print(f"    [{idx:>2d}] {str(name):<35s} {label_counts[idx]:>10,}")
    print()
    print(f"  Phân bố edge_label_binary (Benign/Malicious):")
    print(f"    [0] Benign                              "
          f"{bin_counts[0]:>10,}  ({bin_counts[0] / max(e, 1) * 100:.2f}%)")
    print(f"    [1] Malicious                           "
          f"{bin_counts[1]:>10,}  ({bin_counts[1] / max(e, 1) * 100:.2f}%)")
    print()
    print(f"  ip_to_idx size:          {len(data.ip_to_idx)}")
    print("=" * 70)

    return {
        "num_nodes": n,
        "num_edges": e,
        "num_edges_mp": e_mp,
        "feature_dim": fd,
        "num_classes": nc,
        "num_isolated": n_isolated,
        "label_counts": label_counts,
        "binary_counts": bin_counts,
    }


# ---------------------------------------------------------------------------
# Mock test (chạy được khi không có file thật / torch_geometric)
# ---------------------------------------------------------------------------

_MOCK_DF = pd.DataFrame(
    {
        "id.orig_h":  ["10.0.0.1", "10.0.0.1", "10.0.0.2", "10.0.0.3"],
        "id.resp_h":  ["10.0.0.2", "10.0.0.3", "10.0.0.3", "10.0.0.4"],
        "ts":         [1.0, 2.0, 3.0, 4.0],
        "feat_a":     [1.0, 2.0, 3.0, 4.0],
        "feat_b":     [10.0, 20.0, 30.0, 40.0],
        "feat_c":     [0.1, 0.2, 0.3, 0.4],
        "label":      ["Benign", "Malicious", "Malicious", "Malicious"],
        "detailed-label": ["Benign", "DDoS", "C&C", "DDoS"],
    },
)

_MOCK_FEATURES = ["feat_a", "feat_b", "feat_c"]
_MOCK_CLASS_TO_IDX = {"Benign": 0, "C&C": 1, "DDoS": 2}


def _run_mock_test() -> None:
    """Mock test cho build_graph — kiểm tra IP->idx mapping bằng tay."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    print(">>> [MOCK] DataFrame đầu vào:")
    print(_MOCK_DF.to_string(index=False))

    data = build_graph(
        _MOCK_DF,
        class_to_idx=_MOCK_CLASS_TO_IDX,
        feature_columns=_MOCK_FEATURES,
        node_in_dim=1,
    )

    # ---- IP -> idx đúng thủ công ----
    expected_ips_sorted = sorted(
        set(_MOCK_DF["id.orig_h"]) | set(_MOCK_DF["id.resp_h"])
    )
    # expected: ['10.0.0.1', '10.0.0.2', '10.0.0.3', '10.0.0.4']
    assert data.ip_to_idx == {
        ip: i for i, ip in enumerate(expected_ips_sorted)
    }, (
        f"[MOCK] ip_to_idx sai.\nKỳ vọng: "
        f"{ {ip: i for i, ip in enumerate(expected_ips_sorted)} }\n"
        f"Thực tế: {data.ip_to_idx}"
    )
    assert data.num_nodes == 4

    # ---- edge_index shape & giá trị đúng ----
    # Flow 1: 10.0.0.1 (idx=0) -> 10.0.0.2 (idx=1)
    # Flow 2: 10.0.0.1 (idx=0) -> 10.0.0.3 (idx=2)
    # Flow 3: 10.0.0.2 (idx=1) -> 10.0.0.3 (idx=2)
    # Flow 4: 10.0.0.3 (idx=2) -> 10.0.0.4 (idx=3)
    expected_ei = torch.tensor(
        [[0, 0, 1, 2], [1, 2, 2, 3]], dtype=torch.long,
    )
    assert torch.equal(data.edge_index, expected_ei), (
        f"[MOCK] edge_index sai.\nKỳ vọng:\n{expected_ei}\n"
        f"Thực tế:\n{data.edge_index}"
    )

    # ---- edge_attr shape & giá trị đúng (thứ tự feat_cols) ----
    expected_ea = torch.tensor(
        [[1.0, 10.0, 0.1], [2.0, 20.0, 0.2], [3.0, 30.0, 0.3],
         [4.0, 40.0, 0.4]],
        dtype=torch.float32,
    )
    assert torch.equal(data.edge_attr, expected_ea), "[MOCK] edge_attr sai."

    # ---- edge_label đúng theo class_to_idx ----
    expected_el = torch.tensor([0, 2, 1, 2], dtype=torch.long)
    assert torch.equal(data.edge_label, expected_el), "[MOCK] edge_label sai."

    # ---- edge_label_binary: Benign=0, còn lại=1 ----
    expected_bin = torch.tensor([0, 1, 1, 1], dtype=torch.long)
    assert torch.equal(data.edge_label_binary, expected_bin), (
        "[MOCK] edge_label_binary sai."
    )

    # ---- Node features: all-ones [4, 1] ----
    assert data.x.shape == (4, 1)
    assert torch.equal(data.x, torch.ones((4, 1), dtype=torch.float32))

    # ---- Message passing: [2, 8], [8, 3] ----
    assert data.edge_index_mp.shape == (2, 8)
    assert data.edge_attr_mp.shape == (8, 3)
    # 4 cạnh gốc + 4 cạnh đảo.
    expected_ei_mp = torch.tensor(
        [[0, 0, 1, 2, 1, 2, 2, 3], [1, 2, 2, 3, 0, 0, 1, 2]],
        dtype=torch.long,
    )
    assert torch.equal(data.edge_index_mp, expected_ei_mp), (
        "[MOCK] edge_index_mp sai."
    )
    # edge_attr_mp: 4 gốc + 4 lặp lại (cùng attr).
    assert torch.equal(
        data.edge_attr_mp,
        torch.cat([expected_ea, expected_ea], dim=0),
    ), "[MOCK] edge_attr_mp sai."

    # ---- node_in_dim=3 (linh hoạt) ----
    data3 = build_graph(
        _MOCK_DF, class_to_idx=_MOCK_CLASS_TO_IDX,
        feature_columns=_MOCK_FEATURES, node_in_dim=3,
    )
    assert data3.x.shape == (4, 3)
    assert torch.equal(data3.x, torch.ones((4, 3), dtype=torch.float32))

    # ---- graph_stats ----
    stats = graph_stats(data)
    assert stats["num_nodes"] == 4
    assert stats["num_edges"] == 4
    assert stats["num_edges_mp"] == 8
    assert stats["feature_dim"] == 3
    assert stats["num_classes"] == 3
    # Node 0 (10.0.0.1) có degree 2 (out: 2), node 1 degree 3, node 2 degree 4,
    # node 3 degree 1. Không có node cô lập.
    assert stats["num_isolated"] == 0
    assert stats["label_counts"] == [1, 1, 2]  # Benign=1, C&C=1, DDoS=2
    assert stats["binary_counts"] == [1, 3]    # Benign=1, Malicious=3

    # ---- save / load round-trip ----
    tmp_dir = tempfile.mkdtemp(prefix="iot23_graph_mock_")
    save_path = os.path.join(tmp_dir, "mock_graph.pt")
    save_graph(data, save_path)
    data_loaded = load_graph(save_path)
    assert torch.equal(data_loaded.edge_index, data.edge_index)
    assert torch.equal(data_loaded.edge_attr, data.edge_attr)
    assert torch.equal(data_loaded.edge_label, data.edge_label)
    assert torch.equal(data_loaded.edge_index_mp, data.edge_index_mp)
    assert torch.equal(data_loaded.edge_attr_mp, data.edge_attr_mp)
    assert data_loaded.ip_to_idx == data.ip_to_idx
    assert data_loaded.feature_dim == data.feature_dim
    assert data_loaded.num_classes == data.num_classes
    assert data_loaded.class_to_idx == data.class_to_idx

    print("\n[MOCK TEST graph_build] Tất cả assertions đều PASS.")


# ---------------------------------------------------------------------------
# Real test (file thật 34-1)
# ---------------------------------------------------------------------------

def _run_real_test(path: str) -> None:
    """Test build_graph trên file thật 34-1 (full + undersampled)."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    from sklearn.model_selection import train_test_split
    from src.data_io import load_scenario
    from src.preprocess import clean_flows, fit_preprocessor, transform
    from src.imbalance import prepare_imbalance_variants

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    print(f"\n>>> [REAL] Real file: {path}")

    # Pipeline: clean -> transform.
    df_clean = clean_flows(load_scenario(path))
    pre = fit_preprocessor(df_clean)
    df_feat = transform(df_clean, pre)
    print(f">>> [REAL] Sau transform: shape={df_feat.shape}")

    # 80/20 stratified (chỉ để test).
    df_train, df_test = train_test_split(
        df_feat,
        test_size=0.2,
        stratify=df_feat["detailed-label"],
        random_state=42,
    )
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    print(f">>> [REAL] Train: {df_train.shape}, Test: {df_test.shape}")

    # class_to_idx từ imbalance.
    from src.imbalance import compute_class_weights
    _, class_to_idx, _ = compute_class_weights(
        df_train["detailed-label"].tolist(), scheme="balanced",
    )
    print(f">>> [REAL] class_to_idx: {class_to_idx}")

    # ---- Build đồ thị trên FULL train ----
    print("\n>>> [REAL] Build đồ thị trên TRAIN (full):")
    data_full = build_graph(
        df_train,
        class_to_idx=class_to_idx,
        feature_columns=pre.feature_columns,
    )
    stats_full = graph_stats(data_full)

    # ---- Build đồ thị trên UNDERSAMPLED train ----
    print("\n>>> [REAL] Build đồ thị trên TRAIN (undersampled):")
    variants = prepare_imbalance_variants(df_train, random_state=42)
    df_under = variants["undersampled"]
    data_under = build_graph(
        df_under,
        class_to_idx=class_to_idx,
        feature_columns=pre.feature_columns,
    )
    stats_under = graph_stats(data_under)

    # ---- Bất biến / sanity ----
    # feature_dim đúng.
    assert stats_full["feature_dim"] == len(pre.feature_columns)
    assert stats_under["feature_dim"] == len(pre.feature_columns)
    # num_classes đúng = 4.
    assert stats_full["num_classes"] == 4
    assert stats_under["num_classes"] == 4
    # Phân bố edge_label của full phải giữ imbalance (PortScan rất ít).
    counts_full = stats_full["label_counts"]
    counts_under = stats_under["label_counts"]
    # Cộng phải bằng tổng edge.
    assert sum(counts_full) == stats_full["num_edges"]
    assert sum(counts_under) == stats_under["num_edges"]
    # Undersample làm giảm DDoS về bằng C&C.
    # Số cạnh undersampled < số cạnh full.
    assert stats_under["num_edges"] < stats_full["num_edges"], (
        f"[REAL] undersample phải giảm số cạnh: "
        f"full={stats_full['num_edges']}, under={stats_under['num_edges']}"
    )
    # Số node undersampled <= số node full (có thể giảm vì node DDoS bị cô lập).
    assert stats_under["num_nodes"] <= stats_full["num_nodes"]
    # Trên full: 4 lớp đều xuất hiện.
    assert all(c > 0 for c in counts_full), (
        f"[REAL] trên full phải có đủ 4 lớp, got {counts_full}"
    )

    # ---- Save / load round-trip ----
    tmp_dir = tempfile.mkdtemp(prefix="iot23_graph_real_")
    save_path = os.path.join(tmp_dir, "graph_full_34-1.pt")
    save_graph(data_full, save_path)
    data_loaded = load_graph(save_path)
    assert torch.equal(data_loaded.edge_index, data_full.edge_index)
    assert torch.equal(data_loaded.edge_attr, data_full.edge_attr)
    assert torch.equal(data_loaded.edge_label, data_full.edge_label)
    assert torch.equal(data_loaded.edge_index_mp, data_full.edge_index_mp)
    assert torch.equal(data_loaded.edge_attr_mp, data_full.edge_attr_mp)
    assert data_loaded.ip_to_idx == data_full.ip_to_idx
    assert data_loaded.feature_dim == data_full.feature_dim
    assert data_loaded.num_classes == data_full.num_classes
    print(">>> [REAL] Save/load round-trip OK.")

    print("\n[REAL TEST graph_build] Tất cả assertions đều PASS.")


if __name__ == "__main__":
    # Chạy:
    #   python -m src.graph_build                     → mock test.
    #   python -m src.graph_build <path/to/log>       → real test.
    if len(sys.argv) >= 2:
        _run_real_test(sys.argv[1])
    else:
        _run_mock_test()
