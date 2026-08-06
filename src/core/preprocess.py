"""
preprocess.py — Làm sạch, encode, scale dữ liệu IoT-23.

Viết dạng hàm/module TÁI SỬ DỤNG (không code kiểu notebook), vì:
    • GĐ3 sẽ dùng lại để xử lý luồng sự kiện Falco real-time.
    • Cách xử lý mất cân bằng & thiếu giá trị phải NHẤT QUÁN giữa baseline
      và các client FL ở GĐ2.

Các quy tắc đã chốt trong CLAUDE.md (KHÔNG tự ý thay đổi):

    Missing:
        • '-' của Zeek → NaN thật, ép kiểu TRƯỚC.
        • Cột categorical (`service`): NaN → nhãn riêng "unknown"
          (bản thân "không nhận diện được service" cũng là tín hiệu).
        • Cột numeric: NaN → 0 + thêm 1 cột cờ nhị phân "bị thiếu".

    Loại bỏ hoàn toàn khỏi feature:
        • `uid` (định danh, high-cardinality, gây học vẹt).
        • `tunnel_parents` (gần như luôn '-').
        • `local_orig`, `local_resp` (gần như không đổi trong IoT-23).
        • `id.orig_h`, `id.resp_h` (địa chỉ IP — TUYỆT ĐỐI KHÔNG đưa vào feature,
          chỉ dùng để định danh node khi dựng đồ thị).

    Cổng:
        • `id.orig_p`: ephemeral → bỏ, hoặc chỉ giữ 1 cờ "well-known hay không".
        • `id.resp_p`: QUAN TRỌNG → bucket 3 nhóm
          (well-known 0–1023 / registered 1024–49151 / dynamic 49152+).
          KHÔNG one-hot toàn bộ 65536 giá trị, KHÔNG coi là số liên tục.

    Categorical (`proto`, `service`, `conn_state`, `history`):
        • `conn_state` và `history` là 2 cột tín hiệu hành vi mạnh nhất
          (mô tả hình dạng bắt tay TCP) → ưu tiên làm đặc trưng cạnh.

    Numeric (`duration`, `orig_bytes`, `resp_bytes`, `orig_pkts`, `resp_pkts`,
    `orig_ip_bytes`, `resp_ip_bytes`, `missed_bytes`):
        • Heavy-tailed → `log1p(x)` TRƯỚC, rồi mới standard-scale.
        • Không min-max / z-score trực tiếp lên giá trị thô.

    `ts`: KHÔNG dùng làm feature, nhưng GIỮ LẠI trong dữ liệu đã xử lý để
    GĐ3 có thể cần khi chia đồ thị theo cửa sổ thời gian.

Quy tắc fit/transform:
    • Fit scaler & encoder trên TẬP TRAIN; chỉ transform trên tập test.
    • Lưu lại scaler/encoder để tái dùng giữa các lần chạy & giữa các client.

Thứ tự bắt buộc trong clean_flows (Task 1.5):
    1. detailed-label: '-' / '(empty)' -> 'Benign'   (TRƯỚC khi đổi '-' -> NaN)
    2. Loại 4 cột: uid, tunnel_parents, local_orig, local_resp.
    3. Các cột còn lại: '-' / '(empty)' -> NaN thật.
    4. Ép kiểu: float cho ts + 8 numeric; int cho id.orig_p, id.resp_p;
       giữ string cho id.orig_h, id.resp_h, proto, service, conn_state,
       history, label, detailed-label.
    5. Xử lý missing: service NaN -> 'unknown';
       mỗi cột numeric có NaN -> fill 0 + thêm cờ <col>_missing.

KHÔNG encode, KHÔNG scale, KHÔNG dựng đồ thị trong clean_flows.
"""

from __future__ import annotations

import logging
import os
import pickle
import re
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, List, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cấu hình cột (tham chiếu cho clean_flows & các bước sau)
# ---------------------------------------------------------------------------

# Cột numeric — ép float.
FLOAT_COLUMNS: List[str] = [
    "ts",
    "duration",
    "orig_bytes",
    "resp_bytes",
    "missed_bytes",
    "orig_pkts",
    "orig_ip_bytes",
    "resp_pkts",
    "resp_ip_bytes",
]

# Cột port — ép int (sẽ được bucket ở bước sau).
INT_COLUMNS: List[str] = [
    "id.orig_p",
    "id.resp_p",
]

# Cột loại bỏ hoàn toàn khỏi df sau bước xử lý nhãn.
DROP_COLUMNS: List[str] = [
    "uid",
    "tunnel_parents",
    "local_orig",
    "local_resp",
]

# Cột giữ nguyên dạng chuỗi (không ép kiểu numeric).
STRING_COLUMNS: List[str] = [
    "id.orig_h",
    "id.resp_h",
    "proto",
    "service",
    "conn_state",
    "history",
    "label",            # Benign / Malicious — đối chiếu nhanh
    "detailed-label",   # multi-class target (đã xử lý '-' -> 'Benign')
]

# Token Zeek biểu thị "không có giá trị".
MISSING_TOKENS = ("-", "(empty)")

# ---------------------------------------------------------------------------
# Cấu hình cho feature engineering (Task 1.6)
# ---------------------------------------------------------------------------

# Bucket cho id.resp_p (3 hạng, fix cứng để train/test cùng schema).
RESP_PORT_CATEGORIES: List[str] = ["well_known", "registered", "dynamic"]

# Ký tự cờ Zeek phổ biến trong cột `history`.
# Theo tài liệu Zeek: s/S=SYN, h/H=SYN-ACK, a/A=ACK, d/D=data, f/F=FIN,
# r/R=RST, c/C=close, g/G=gap, t/T=timeout, w/W=unidirectional-data.
# Thường = phía originator, hoa = phía responder → GIỮ CẢ HAI để giữ ngữ nghĩa.
HISTORY_FLAG_CHARS: List[str] = list("sShHaAdDfFrRcCgGtTwW")

# Ngưỡng tần suất tối thiểu để giữ một giá trị service làm feature one-hot
# riêng. Mọi giá trị dưới ngưỡng (kể cả "unknown") gom vào "service_other".
SERVICE_FREQ_THRESHOLD: float = 0.01

# Cột numeric đưa vào FEATURE (đã trừ `ts` — chỉ giữ để dựng đồ thị).
NUMERIC_FEATURE_COLUMNS: List[str] = [
    "duration",
    "orig_bytes",
    "resp_bytes",
    "missed_bytes",
    "orig_pkts",
    "orig_ip_bytes",
    "resp_pkts",
    "resp_ip_bytes",
]


# ---------------------------------------------------------------------------
# Hàm chính (Task 1.5)
# ---------------------------------------------------------------------------

def clean_flows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Làm sạch DataFrame thô từ ``load_scenario()`` → DataFrame sẵn sàng cho
    bước encode/scale tiếp theo.

    Pipeline (ĐÃ CHỐT trong CLAUDE.md — KHÔNG tự ý đổi thứ tự):

      1. ``detailed-label``: ``"-"`` và ``"(empty)"`` → ``"Benign"``
         (LÀM ĐẦU TIÊN — vì sau đó sẽ đổi mọi ``"-"`` thành NaN ở bước 3,
         nếu không làm bước này trước thì cả lớp Benign sẽ biến mất).
      2. Loại bỏ 4 cột: ``uid``, ``tunnel_parents``, ``local_orig``,
         ``local_resp``.
      3. Các cột CÒN LẠI: ``"-"`` và ``"(empty)"`` → ``NaN`` thật (``pd.NA``).
      4. Ép kiểu:
           - float: ``ts`` + 8 cột numeric (``duration``, ``orig_bytes``,
             ``resp_bytes``, ``missed_bytes``, ``orig_pkts``,
             ``orig_ip_bytes``, ``resp_pkts``, ``resp_ip_bytes``).
           - int: ``id.orig_p``, ``id.resp_p`` (sẽ bucket ở bước sau).
           - giữ string: ``id.orig_h``, ``id.resp_h``, ``proto``, ``service``,
             ``conn_state``, ``history``, ``label``, ``detailed-label``.
      5. Xử lý missing:
           - ``service`` NaN → ``"unknown"``.
           - Mỗi cột numeric có NaN: điền ``0`` + thêm cột cờ nhị phân
             ``<col>_missing`` (``1`` nếu vốn thiếu, ``0`` nếu không).

    Hàm này KHÔNG encode, KHÔNG scale, KHÔNG dựng đồ thị.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame thô từ ``load_scenario()`` (đã tách cột nhãn cuối).
        Phải chứa cột ``detailed-label`` (đã được canonical hoá).

    Returns
    -------
    pd.DataFrame
        Bản sao đã sạch. Hàm thuần — không sửa ``df`` đầu vào.

    Raises
    ------
    TypeError
        Nếu ``df`` không phải DataFrame.
    ValueError
        Nếu ``df`` rỗng.
    KeyError
        Nếu thiếu cột ``detailed-label``.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"clean_flows: expected DataFrame, got {type(df).__name__}."
        )
    if df.empty:
        raise ValueError("clean_flows: DataFrame rỗng (0 dòng).")

    out = df.copy()

    # ---- Bước 1: detailed-label "-" / "(empty)" -> "Benign" (ĐẦU TIÊN) ----
    if "detailed-label" not in out.columns:
        raise KeyError(
            "clean_flows: DataFrame thiếu cột 'detailed-label'. "
            "Đã chạy load_scenario() trước chưa?"
        )
    label_series = out["detailed-label"].astype("string").str.strip()
    mask_missing = label_series.isin(MISSING_TOKENS)
    n_benign_from_missing = int(mask_missing.sum())
    out["detailed-label"] = label_series.where(~mask_missing, "Benign")
    logger.info(
        "clean_flows [1/5]: đã đổi %d dòng detailed-label từ %s -> 'Benign'.",
        n_benign_from_missing, list(MISSING_TOKENS),
    )

    # ---- Bước 2: loại 4 cột ----
    present_drops = [c for c in DROP_COLUMNS if c in out.columns]
    absent_drops = [c for c in DROP_COLUMNS if c not in out.columns]
    if absent_drops:
        logger.warning(
            "clean_flows [2/5]: cột DROP không có sẵn trong df: %s.",
            absent_drops,
        )
    if present_drops:
        out = out.drop(columns=present_drops)
        logger.info(
            "clean_flows [2/5]: đã loại %d cột: %s.",
            len(present_drops), present_drops,
        )

    # ---- Bước 3: "-" / "(empty)" -> NaN ở các cột còn lại ----
    # Sau bước 2, các cột numeric vẫn đang là object (string). Bước này chuyển
    # token thiếu thành pd.NA để bước 4 ép kiểu không bị lỗi.
    obj_cols = out.select_dtypes(include=["object", "string"]).columns.tolist()
    n_total_replaced = 0
    for col in obj_cols:
        col_str = out[col].astype("string")
        before = int(col_str.isin(MISSING_TOKENS).sum())
        if before > 0:
            out[col] = col_str.replace(list(MISSING_TOKENS), pd.NA)
            n_total_replaced += before
    logger.info(
        "clean_flows [3/5]: đã đổi %d giá trị '-' / '(empty)' -> NaN "
        "trên %d cột object/string.",
        n_total_replaced, len(obj_cols),
    )

    # ---- Bước 4: ép kiểu ----
    # 4a. Float columns: dùng pd.to_numeric(errors='coerce') để bắt mọi giá
    #     trị lạ còn sót (NaN giữ nguyên ở dạng float NaN).
    for col in FLOAT_COLUMNS:
        if col not in out.columns:
            logger.warning(
                "clean_flows [4a]: cột float '%s' không tồn tại, bỏ qua.", col,
            )
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

    # 4b. Int columns (ports): ép qua Int64 (pandas nullable int) để GIỮ NaN.
    #     Bước 5 sẽ fill 0 + thêm cờ, rồi cast về int64 cuối cùng.
    for col in INT_COLUMNS:
        if col not in out.columns:
            logger.warning(
                "clean_flows [4b]: cột int '%s' không tồn tại, bỏ qua.", col,
            )
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

    # 4c. String columns (ép về string để các thao tác NA nhất quán).
    for col in STRING_COLUMNS:
        if col not in out.columns:
            continue
        out[col] = out[col].astype("string")

    # ---- Bước 5: xử lý missing ----
    # 5a. service NaN -> "unknown".
    if "service" in out.columns:
        n_svc_na = int(out["service"].isna().sum())
        if n_svc_na > 0:
            out["service"] = out["service"].fillna("unknown")
            logger.info(
                "clean_flows [5a]: đã điền %d NaN ở 'service' -> 'unknown'.",
                n_svc_na,
            )

    # 5b. Mỗi cột numeric có NaN: fill 0 + thêm cờ <col>_missing.
    numeric_cols_all = FLOAT_COLUMNS + INT_COLUMNS
    n_flag_added = 0
    for col in numeric_cols_all:
        if col not in out.columns:
            continue
        n_na = int(out[col].isna().sum())
        if n_na == 0:
            continue
        flag_col = f"{col}_missing"
        out[flag_col] = out[col].isna().astype("int8")
        if col in INT_COLUMNS:
            # Int64 + fillna(0) -> cast int64 (loại bỏ nullable, an toàn cho
            # bucket/bincount sau này).
            out[col] = out[col].fillna(0).astype("int64")
        else:
            out[col] = out[col].fillna(0).astype("float64")
        logger.info(
            "clean_flows [5b]: cột '%s' có %d NaN -> fill 0, thêm cờ '%s'.",
            col, n_na, flag_col,
        )
        n_flag_added += 1

    if n_flag_added == 0:
        logger.info(
            "clean_flows [5b]: không có cột numeric nào còn NaN sau bước 4."
        )

    # 5c. Đảm bảo các cột port LUÔN ra dtype int64 (không nullable), kể cả khi
    #     không có NaN ban đầu (lúc đó vẫn đang ở Int64 do bước 4b).
    for col in INT_COLUMNS:
        if col in out.columns and str(out[col].dtype) != "int64":
            out[col] = out[col].astype("int64")

    # Ép các cột string về object để tương thích tốt hơn với IO & downstream
    # (pd.NA -> NaN khi xuất parquet; pandas thường xử lý object thuận tiện hơn).
    for col in out.select_dtypes(include=["string"]).columns:
        out[col] = out[col].astype(object)

    logger.info(
        "clean_flows: xong — shape=%s, %d cột.", out.shape, out.shape[1],
    )
    return out


# ---------------------------------------------------------------------------
# Feature engineering (Task 1.6) — fit / transform tách rời
# ---------------------------------------------------------------------------

def _bucket_resp_port(port: Any) -> str:
    """Phân nhóm id.resp_p thành 3 hạng (well_known / registered / dynamic)."""
    try:
        p = int(port)
    except (TypeError, ValueError):
        return "well_known"  # port NaN sau clean_flows đã fill 0 → well_known
    if p < 1024:
        return "well_known"
    if p < 49152:
        return "registered"
    return "dynamic"


@dataclass
class FrozenStandardScaler:
    """Inference-only StandardScaler state without sklearn pickle coupling."""

    mean_: np.ndarray
    scale_: np.ndarray
    var_: np.ndarray

    def __post_init__(self) -> None:
        self.mean_ = np.asarray(self.mean_, dtype=np.float64).copy()
        self.scale_ = np.asarray(self.scale_, dtype=np.float64).copy()
        self.var_ = np.asarray(self.var_, dtype=np.float64).copy()
        if not (
            self.mean_.shape == self.scale_.shape == self.var_.shape
            and self.mean_.ndim == 1
        ):
            raise ValueError("FrozenStandardScaler arrays must be aligned vectors.")
        if np.any(self.scale_ <= 0) or not np.all(np.isfinite(self.scale_)):
            raise ValueError("FrozenStandardScaler scale must be finite and positive.")
        self.n_features_in_ = int(self.mean_.shape[0])

    def transform(self, values: Any) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != self.n_features_in_:
            raise ValueError(
                "FrozenStandardScaler input feature dimension does not match."
            )
        return (array - self.mean_) / self.scale_


@dataclass
class Preprocessor:
    """
    Chứa MỌI tham số đã học từ tập train cho feature engineering.

    Tách fit / transform để chống rò rỉ dữ liệu:
      - fit_preprocessor() → Preprocessor (học categories, scaler, …).
      - transform(df, preprocessor) → áp đúng tham số đã học lên df bất kỳ.
    Save / load qua pickle hoặc joblib để tái dùng giữa các lần chạy & GĐ3.
    """

    # Danh mục categorical đã học / fix cứng.
    resp_port_categories: List[str]
    proto_categories: List[str]
    service_categories: List[str]
    conn_state_categories: List[str]
    history_flag_chars: List[str]

    # Cột numeric để scale & thứ tự cột numeric (fix cứng).
    numeric_columns: List[str]

    # Cột cờ *_missing phát hiện được trong train (passthrough).
    missing_flag_columns: List[str]

    # StandardScaler đã fit trên log1p(train[numeric_columns]).
    scaler: Any

    # Thứ tự TÊN CỘT feature cuối cùng (deterministic). Đảm bảo train và
    # test luôn cùng schema.
    feature_columns: List[str] = field(default_factory=list)

    # -------- save / load (joblib — kèm fallback pickle) --------

    def save(self, path: str) -> None:
        """Lưu preprocessor ra file để tái dùng (GĐ3 dùng lại)."""
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Preprocessor.save: đã ghi ra %s.", path)

    @staticmethod
    def load(path: str) -> "Preprocessor":
        """Load preprocessor từ file (pickle)."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, Preprocessor):
            raise TypeError(
                f"Preprocessor.load: file {path} không chứa Preprocessor "
                f"mà là {type(obj).__name__}."
            )
        logger.info("Preprocessor.load: đã load từ %s.", path)
        return obj

    @property
    def feature_dim(self) -> int:
        """Số chiều feature (= len(feature_columns))."""
        return len(self.feature_columns)


# ---------------------------------------------------------------------------
# fit_preprocessor / transform
# ---------------------------------------------------------------------------

def _require_columns(df: pd.DataFrame, cols: Sequence[str], ctx: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"{ctx}: DataFrame thiếu các cột bắt buộc: {missing}. "
            f"Đã chạy clean_flows() trước chưa?"
        )


def fit_preprocessor(df_train: pd.DataFrame) -> Preprocessor:
    """
    Học MỌI tham số feature engineering từ tập train.

    Bước học (CHỈ dùng train — chống rò rỉ dữ liệu):
      1. resp_port_categories: cố định 3 hạng.
      2. proto_categories: các giá trị duy nhất trong train (sắp xếp).
      3. service_categories: giá trị xuất hiện >= SERVICE_FREQ_THRESHOLD
         tổng dòng train, TRỪ 'unknown' (luôn gom vào 'service_other').
      4. conn_state_categories: các giá trị duy nhất trong train (sắp xếp).
      5. history_flag_chars: cố định (20 ký tự cờ Zeek phổ biến).
      6. numeric_columns: 8 cột traffic stats (đã fix cứng trong config).
      7. missing_flag_columns: các cột *_missing có trong train.
      8. StandardScaler: fit trên log1p(train[numeric_columns]).

    Parameters
    ----------
    df_train : pd.DataFrame
        DataFrame ĐÃ QUA clean_flows() (numeric = float64, port = int64,
        *_missing = int8). KHÔNG gọi trên test hoặc toàn bộ dữ liệu.

    Returns
    -------
    Preprocessor
        Đối tượng chứa mọi tham số đã học + danh sách feature_columns
        theo thứ tự cố định.
    """
    _require_columns(
        df_train,
        ["id.orig_h", "id.resp_h", "ts",
         "id.orig_p", "id.resp_p",
         "proto", "service", "conn_state", "history",
         "label", "detailed-label"],
        ctx="fit_preprocessor",
    )
    _require_columns(df_train, NUMERIC_FEATURE_COLUMNS,
                     ctx="fit_preprocessor (numeric)")

    # 1. resp_port: cố định 3 hạng.
    resp_port_cats = list(RESP_PORT_CATEGORIES)

    # 2. proto: từ train.
    proto_cats = sorted(
        {str(v) for v in df_train["proto"].dropna().unique()}
    )

    # 3. service: ngưỡng tần suất, loại trừ 'unknown' (luôn → service_other).
    svc_counts = df_train["service"].astype(str).value_counts(normalize=True)
    service_cats = sorted(
        c for c, p in svc_counts.items()
        if p >= SERVICE_FREQ_THRESHOLD and c != "unknown" and c != ""
    )

    # 4. conn_state: từ train.
    conn_state_cats = sorted(
        {str(v) for v in df_train["conn_state"].dropna().unique()}
    )

    # 5. history flag chars: cố định 20 ký tự Zeek phổ biến.
    history_chars = list(HISTORY_FLAG_CHARS)

    # 6. numeric columns: cố định.
    numeric_cols = [c for c in NUMERIC_FEATURE_COLUMNS if c in df_train.columns]

    # 7. missing flag columns: tự phát hiện *_missing trong train.
    missing_cols = sorted(
        c for c in df_train.columns if c.endswith("_missing")
    )

    # 8. Fit StandardScaler trên log1p(train[numeric]).
    log_train = np.log1p(df_train[numeric_cols].astype("float64").values)
    scaler = StandardScaler()
    scaler.fit(log_train)
    logger.info(
        "fit_preprocessor: scaler fitted trên log1p(%d numeric cols), "
        "mean_=%s, scale_=%s.",
        len(numeric_cols),
        np.round(scaler.mean_, 4).tolist(),
        np.round(scaler.scale_, 4).tolist(),
    )

    # ---- Tính feature_columns theo thứ tự CỐ ĐỊNH ----
    feat_cols: List[str] = []
    # 1) resp_port one-hot
    for c in resp_port_cats:
        feat_cols.append(f"resp_port_{c}")
    # 2) orig_port binary flag
    feat_cols.append("orig_port_is_wellknown")
    # 3) proto one-hot
    for c in proto_cats:
        feat_cols.append(f"proto_{c}")
    # 4) service one-hot + service_other (luôn có)
    for c in service_cats:
        feat_cols.append(f"service_{c}")
    feat_cols.append("service_other")
    # 5) conn_state one-hot
    for c in conn_state_cats:
        feat_cols.append(f"conn_state_{c}")
    # 6) history flag counts
    for ch in history_chars:
        feat_cols.append(f"history_n_{ch}")
    # 7) numeric scaled
    for c in numeric_cols:
        feat_cols.append(f"{c}_scaled")
    # 8) missing flags passthrough
    feat_cols.extend(missing_cols)

    logger.info(
        "fit_preprocessor: proto=%d, service_kept=%d (+other), "
        "conn_state=%d, history_chars=%d, numeric=%d, missing=%d → "
        "feature_dim=%d.",
        len(proto_cats), len(service_cats), len(conn_state_cats),
        len(history_chars), len(numeric_cols), len(missing_cols),
        len(feat_cols),
    )

    return Preprocessor(
        resp_port_categories=resp_port_cats,
        proto_categories=proto_cats,
        service_categories=service_cats,
        conn_state_categories=conn_state_cats,
        history_flag_chars=history_chars,
        numeric_columns=numeric_cols,
        missing_flag_columns=missing_cols,
        scaler=scaler,
        feature_columns=feat_cols,
    )


def transform(df: pd.DataFrame, preprocessor: Preprocessor) -> pd.DataFrame:
    """
    Áp feature engineering lên df dùng tham số đã học trong ``preprocessor``.

    KHÔNG học gì thêm từ df — đảm bảo test chỉ dùng tham số của train.

    Thứ tự cột trả về:
        [id.orig_h, id.resp_h, ts, <feature_columns...>, label, detailed-label]

    Cột feature (đúng thứ tự trong ``preprocessor.feature_columns``) là
    MA TRẬN ĐẶC TRƯNG CẠNH (edge features) cho bước dựng đồ thị.

    Với giá trị test KHÔNG xuất hiện trong train (proto/conn_state lạ,
    service hiếm), one-hot cột đó được set = 0; service lạ → service_other = 1.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame ĐÃ QUA clean_flows(). Có thể là train hoặc test.
    preprocessor : Preprocessor
        Preprocessor đã fit trên train.

    Returns
    -------
    pd.DataFrame
        DataFrame mới chứa các cột ở trên.
    """
    _require_columns(
        df,
        ["id.orig_h", "id.resp_h", "ts",
         "id.orig_p", "id.resp_p",
         "proto", "service", "conn_state", "history",
         "label", "detailed-label"],
        ctx="transform",
    )

    n = len(df)
    out = pd.DataFrame(index=df.index)

    # ---- Cột giữ lại (không phải feature, dùng cho graph / nhãn / GĐ3) ----
    out["id.orig_h"] = df["id.orig_h"].astype(object).values
    out["id.resp_h"] = df["id.resp_h"].astype(object).values
    out["ts"] = df["ts"].astype("float64").values

    # ---- 1) resp_port one-hot (3 hạng fix cứng) ----
    resp_bucket = df["id.resp_p"].apply(_bucket_resp_port)
    for cat in preprocessor.resp_port_categories:
        out[f"resp_port_{cat}"] = (resp_bucket == cat).astype("int8").values

    # ---- 2) orig_port binary flag is_wellknown ----
    out["orig_port_is_wellknown"] = (
        (df["id.orig_p"] >= 0) & (df["id.orig_p"] < 1024)
    ).astype("int8").values

    # ---- 3) proto one-hot ----
    proto_series = df["proto"].astype(object)
    for cat in preprocessor.proto_categories:
        out[f"proto_{cat}"] = (proto_series == cat).astype("int8").values

    # ---- 4) service one-hot + service_other ----
    svc_series = df["service"].astype(object).fillna("unknown")
    svc_known = svc_series.isin(preprocessor.service_categories)
    for cat in preprocessor.service_categories:
        out[f"service_{cat}"] = (svc_series == cat).astype("int8").values
    out["service_other"] = (~svc_known).astype("int8").values

    # ---- 5) conn_state one-hot ----
    cs_series = df["conn_state"].astype(object)
    for cat in preprocessor.conn_state_categories:
        out[f"conn_state_{cat}"] = (cs_series == cat).astype("int8").values

    # ---- 6) history flag counts ----
    hist_series = df["history"].astype(object).fillna("").astype(str)
    for ch in preprocessor.history_flag_chars:
        # str.count chỉ đếm non-overlapping occurrences.
        out[f"history_n_{ch}"] = (
            hist_series.str.count(re.escape(ch)).astype("int8").values
        )

    # ---- 7) numeric scaled: log1p trước, rồi scaler.transform ----
    num_vals = df[preprocessor.numeric_columns].astype("float64").values
    log_vals = np.log1p(num_vals)
    scaled = preprocessor.scaler.transform(log_vals).astype("float32")
    for i, col in enumerate(preprocessor.numeric_columns):
        out[f"{col}_scaled"] = scaled[:, i]

    # ---- 8) missing flags passthrough (0/1) ----
    for col in preprocessor.missing_flag_columns:
        if col in df.columns:
            out[col] = df[col].astype("int8").values
        else:
            # Test thiếu cờ này (rất hiếm) → mặc định 0.
            out[col] = np.zeros(n, dtype="int8")

    # ---- Nhãn & nhãn phụ (giữ nguyên ở cuối) ----
    out["label"] = df["label"].astype(object).values
    out["detailed-label"] = df["detailed-label"].astype(object).values

    # ---- Đảm bảo schema feature_columns đúng thứ tự ----
    kept_cols = ["id.orig_h", "id.resp_h", "ts"]
    final_cols = kept_cols + list(preprocessor.feature_columns) + \
        ["label", "detailed-label"]
    # Không thêm cột lạ, không mất cột.
    missing_final = [c for c in final_cols if c not in out.columns]
    extra_final = [c for c in out.columns if c not in final_cols]
    if missing_final:
        raise RuntimeError(f"transform: thiếu cột {missing_final}.")
    if extra_final:
        raise RuntimeError(f"transform: cột lạ {extra_final}.")

    out = out[final_cols]
    logger.info(
        "transform: %d dòng × %d cột (feature_dim=%d).",
        out.shape[0], out.shape[1], preprocessor.feature_dim,
    )
    return out


def fit_transform(df_train: pd.DataFrame) -> tuple[pd.DataFrame, Preprocessor]:
    """Tiện ích: fit trên train rồi transform luôn train (KHÔNG dùng cho test)."""
    pp = fit_preprocessor(df_train)
    df_feat = transform(df_train, pp)
    return df_feat, pp


# ---------------------------------------------------------------------------
# Placeholder cho các task sau (giữ nguyên signature đã có)
# ---------------------------------------------------------------------------

def clean_missing(df):
    """Chuyển '-' của Zeek thành NaN, điền NaN theo từng nhóm cột (placeholder)."""
    raise NotImplementedError("preprocess.clean_missing sẽ triển khai ở task sau.")


def drop_identifier_columns(df):
    """Loại bỏ uid, tunnel_parents, local_orig, local_resp (placeholder)."""
    raise NotImplementedError("preprocess.drop_identifier_columns sẽ triển khai ở task sau.")


def bucket_resp_port(df):
    """Bucket id.resp_p thành 3 nhóm (well-known/registered/dynamic) (placeholder)."""
    raise NotImplementedError("preprocess.bucket_resp_port sẽ triển khai ở task sau.")


def encode_categoricals(df, encoder=None):
    """Encode proto / service / conn_state / history (placeholder)."""
    raise NotImplementedError("preprocess.encode_categoricals sẽ triển khai ở task sau.")


def scale_numeric(df, scaler=None):
    """log1p rồi standard-scale các cột numeric (placeholder)."""
    raise NotImplementedError("preprocess.scale_numeric sẽ triển khai ở task sau.")


def run_preprocess(config_path: str):
    """Pipeline đầy đủ: đọc raw → xử lý → ghi artifacts/<processed_file> (placeholder)."""
    raise NotImplementedError("preprocess.run_preprocess sẽ triển khai ở task sau.")


# ---------------------------------------------------------------------------
# Mock test (chạy được trên máy Mac không có dataset thật — Task 1.5)
# ---------------------------------------------------------------------------

# Định dạng conn.log.labeled mock, dùng để test load_scenario + clean_flows.
# Có cả flow Benign (detailed-label="-") để test bước 1.
_MOCK_CONN_LOG_CLEAN = """\
#separator \\x09
#set_separator	,
#empty_field	(empty)
#unset_field	-
#path	conn.log
#open	2024-01-01-00-00-00
#fields	ts	uid	id.orig_h	id.orig_p	id.resp_h	id.resp_p	proto	service	duration	orig_bytes	resp_bytes	conn_state	local_orig	local_resp	missed_bytes	history	orig_pkts	orig_ip_bytes	resp_pkts	resp_ip_bytes	tunnel_parents   label   detailed-label
#types	time	string	addr	port	addr	port	enum	string	interval	count	count	string	bool	bool	count	string	count	count	count	count	set[string]   string   string
1704067200.123456	C1	192.168.1.10	54321	8.8.8.8	53	udp	dns	0.001	50	80	SF	-	-	0	Dd	1	78	1	78	- Benign -
1704067201.234567	C2	192.168.1.10	54322	8.8.4.4	53	udp	-	0.002	-	120	SF	-	-	0	Dd	1	68	1	68	- Benign -
1704067202.345678	C3	192.168.1.10	4444	45.83.66.1	6667	tcp	-	10.5	200	300	S1	-	-	0	ShADadtaF	5	400	4	500	- Malicious C&C-Mirai
1704067203.456789	C4	192.168.1.10	5555	45.83.66.2	23	tcp	telnet	2.3	100	150	S1	-	-	0	ShADadtaF	3	200	2	250	- Malicious C&C-FileDownload
1704067204.567890	C5	192.168.1.10	33333	198.51.100.5	80	tcp	http	0.5	500	1500	SF	-	-	0	ShADadtaF	10	700	8	900	- Benign -
1704067205.678901	C6	192.168.1.10	33334	198.51.100.6	80	tcp	-	0.01	60	40	S0	-	-	0	Sh	1	88	1	88	- Malicious PartOfAHorizontalPortScan
1704067206.789012	C7	192.168.1.10	33335	198.51.100.7	22	tcp	ssh	1.5	300	800	SF	-	-	0	ShADadtaF	4	500	3	600	- Benign -
#close 2024-01-01-01-00-00
"""


def _create_mock_clean_file(directory: str) -> str:
    """Ghi mock có '-' trong detailed-label ra file tạm, trả về đường dẫn."""
    os.makedirs(directory, exist_ok=True)
    fp = os.path.join(directory, "mock_conn_clean.log.labeled")
    with open(fp, "w", encoding="utf-8") as f:
        f.write(_MOCK_CONN_LOG_CLEAN)
    return fp


def _inspect_clean(df: pd.DataFrame, label: str) -> None:
    """In các thông tin kiểm tra sau clean_flows (dùng cho cả mock & thật)."""
    print(f"\n>>> [{label}] Bước 1: shape SAU clean_flows = {df.shape}")
    print(f">>> [{label}] Bước 2: dtypes SAU khi ép kiểu:")
    print(df.dtypes.to_string())

    print(f"\n>>> [{label}] Bước 3: NaN còn lại mỗi cột (kỳ vọng = 0):")
    nan_counts = df.isna().sum()
    print(nan_counts.to_string())

    print(f"\n>>> [{label}] Bước 4: value_counts của detailed-label:")
    print(df["detailed-label"].value_counts(dropna=False).to_string())

    print(f"\n>>> [{label}] Bước 5: 4 cột đã loại không còn trong df:")
    still_present = [c for c in DROP_COLUMNS if c in df.columns]
    print(f"    Các cột DROP_COLUMNS còn xuất hiện: {still_present} "
          f"(kỳ vọng: [])")


def _assert_clean_invariants(df: pd.DataFrame, label: str) -> None:
    """Assert các bất biến bắt buộc sau clean_flows."""
    # 4 cột đã loại không còn.
    for c in DROP_COLUMNS:
        assert c not in df.columns, (
            f"[{label}] cột '{c}' phải đã bị loại khỏi df."
        )
    # detailed-label phải có "Benign" là 1 lớp thật.
    assert "Benign" in df["detailed-label"].unique(), (
        f"[{label}] 'Benign' phải là 1 lớp của detailed-label sau clean_flows."
    )
    # Không còn NaN ở các cột numeric sau khi fill.
    for c in FLOAT_COLUMNS + INT_COLUMNS:
        assert df[c].isna().sum() == 0, (
            f"[{label}] cột numeric '{c}' còn {df[c].isna().sum()} NaN "
            f"sau clean_flows."
        )
    # dtype đúng.
    for c in FLOAT_COLUMNS:
        if c in df.columns:
            assert str(df[c].dtype) == "float64", (
                f"[{label}] cột '{c}' phải là float64, hiện là {df[c].dtype}."
            )
    for c in INT_COLUMNS:
        if c in df.columns:
            assert str(df[c].dtype) == "int64", (
                f"[{label}] cột '{c}' phải là int64, hiện là {df[c].dtype}."
            )
    # service NaN -> "unknown".
    if "service" in df.columns:
        assert "unknown" in df["service"].astype(str).unique(), (
            f"[{label}] 'unknown' phải xuất hiện trong service sau khi fill."
        )
    # Mỗi cột numeric có NaN ban đầu phải có cờ _missing tương ứng.
    for c in FLOAT_COLUMNS + INT_COLUMNS:
        flag = f"{c}_missing"
        if flag in df.columns:
            assert set(df[flag].unique()).issubset({0, 1}), (
                f"[{label}] cờ '{flag}' chỉ được chứa 0/1, "
                f"có {set(df[flag].unique())}."
            )


def _run_mock_clean_test() -> None:
    """Mock test cho clean_flows — chạy được trên Mac không có file thật."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    tmp_dir = tempfile.mkdtemp(prefix="iot23_clean_mock_")
    mock_path = _create_mock_clean_file(tmp_dir)
    print(f"\n>>> Mock file: {mock_path}")

    # Import trong hàm để tránh vòng tròn & dễ chạy độc lập.
    from src.data_io import load_scenario

    df_raw = load_scenario(mock_path)
    print(f">>> Shape TRƯỚC clean_flows: {df_raw.shape}")

    df_clean = clean_flows(df_raw)
    _inspect_clean(df_clean, "MOCK")

    # Mock cụ thể: 7 dòng, có 4 Benign (label "Benign", detailed-label "-"
    # ban đầu). Sau bước 1, detailed-label phải có 4 dòng "Benign".
    assert df_clean.shape[0] == 7, (
        f"[MOCK] expected 7 dòng, got {df_clean.shape[0]}"
    )
    assert (df_clean["detailed-label"] == "Benign").sum() == 4, (
        "[MOCK] expected 4 dòng 'Benign' ở detailed-label, got "
        f"{(df_clean['detailed-label'] == 'Benign').sum()}"
    )
    # orig_bytes có 1 NaN (dòng C2 "-") -> phải có cờ orig_bytes_missing = 1.
    assert "orig_bytes_missing" in df_clean.columns, (
        "[MOCK] phải có cột 'orig_bytes_missing' do dòng C2 có orig_bytes='-'."
    )
    assert int(df_clean["orig_bytes_missing"].sum()) == 1, (
        "[MOCK] cờ orig_bytes_missing phải có đúng 1 giá trị = 1."
    )
    # service NaN -> "unknown" (3 dòng C2/C3/C6 có service='-').
    assert (df_clean["service"] == "unknown").sum() == 3, (
        "[MOCK] service phải có 3 giá trị 'unknown' sau khi fill."
    )

    _assert_clean_invariants(df_clean, "MOCK")
    print("\n[MOCK TEST clean_flows] Tất cả assertions đều PASS.")


def _run_real_clean_test(path: str) -> None:
    """Test trên file thật — gọi bằng: python -m src.preprocess <path>."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    from src.data_io import load_scenario

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Không tìm thấy file thật: {path}")
    print(f"\n>>> Real file: {path}")

    df_raw = load_scenario(path)
    print(f">>> Shape TRƯỚC clean_flows: {df_raw.shape}")

    df_clean = clean_flows(df_raw)
    _inspect_clean(df_clean, "REAL")

    _assert_clean_invariants(df_clean, "REAL")
    print("\n[REAL TEST clean_flows] Tất cả assertions đều PASS.")


# ---------------------------------------------------------------------------
# Test cho feature engineering (Task 1.6) — dùng chung cho mock & real
# ---------------------------------------------------------------------------

def _assert_fe_invariants(
    df_train_feat: pd.DataFrame,
    df_test_feat: pd.DataFrame,
    preprocessor: Preprocessor,
    label: str,
) -> None:
    """Các bất biến bắt buộc sau fit/transform."""
    # 1. Train và test PHẢI có cùng danh sách cột & cùng thứ tự.
    assert list(df_train_feat.columns) == list(df_test_feat.columns), (
        f"[{label}] train và test có schema khác nhau!"
    )
    assert list(df_train_feat.columns) == (
        ["id.orig_h", "id.resp_h", "ts"]
        + list(preprocessor.feature_columns)
        + ["label", "detailed-label"]
    ), (
        f"[{label}] thứ tự cột không khớp preprocessor.feature_columns."
    )

    # 2. Không sinh cột lạ ở test so với train (đã check ở trên, double-check).
    test_only = set(df_test_feat.columns) - set(df_train_feat.columns)
    train_only = set(df_train_feat.columns) - set(df_test_feat.columns)
    assert not test_only and not train_only, (
        f"[{label}] test_only={test_only}, train_only={train_only}."
    )

    # 3. KHÔNG có NaN ở bất kỳ cột nào (kể cả feature).
    for col in df_train_feat.columns:
        n_nan_tr = int(df_train_feat[col].isna().sum())
        n_nan_te = int(df_test_feat[col].isna().sum())
        assert n_nan_tr == 0, f"[{label}] train còn {n_nan_tr} NaN ở '{col}'."
        assert n_nan_te == 0, f"[{label}] test còn {n_nan_te} NaN ở '{col}'."

    # 4. Cột numeric_scaled trên TRAIN có mean ~ 0, std ~ 1.
    #    Ngoại lệ: cột constant trong train → sklearn StandardScaler set
    #    scale_ = 1.0 (tránh chia 0), transformed = 0 hết, std = 0.
    #    Đây là hành vi đúng, KHÔNG phải bug — bỏ qua check std cho cột đó.
    scaled_cols = [
        c for c in df_train_feat.columns if c.endswith("_scaled")
    ]
    assert len(scaled_cols) == len(preprocessor.numeric_columns), (
        f"[{label}] số cột _scaled ({len(scaled_cols)}) không khớp "
        f"số numeric ({len(preprocessor.numeric_columns)})."
    )
    for col in scaled_cols:
        m = float(df_train_feat[col].mean())
        s = float(df_train_feat[col].std(ddof=0))
        assert abs(m) < 1e-6, (
            f"[{label}] mean cột '{col}' trên train = {m:.6f} (kỳ vọng ~0)."
        )
        if s >= 1e-6:
            assert abs(s - 1.0) < 1e-5, (
                f"[{label}] std cột '{col}' trên train = {s:.6f} (kỳ vọng ~1)."
            )
        # else: cột constant → std = 0 là hợp lệ (xem giải thích trên).

    # 5. Cột one-hot (resp_port_*, proto_*, service_*, conn_state_*) và
    #    *_missing chỉ chứa 0/1.
    binary_prefixes = (
        "resp_port_", "proto_", "service_", "conn_state_",
        "orig_port_is_wellknown",
    )
    for col in df_train_feat.columns:
        is_binary = (
            any(col.startswith(p) for p in binary_prefixes)
            or col.endswith("_missing")
        )
        if is_binary:
            vals = set(df_train_feat[col].unique().tolist())
            vals_te = set(df_test_feat[col].unique().tolist())
            assert vals.issubset({0, 1}), (
                f"[{label}] cột one-hot train '{col}' có giá trị lạ: {vals}."
            )
            assert vals_te.issubset({0, 1}), (
                f"[{label}] cột one-hot test '{col}' có giá trị lạ: {vals_te}."
            )

    # 6. Cột history_n_* phải không âm (đếm ký tự).
    history_cols = [c for c in df_train_feat.columns if c.startswith("history_n_")]
    for col in history_cols:
        assert int(df_train_feat[col].min()) >= 0, (
            f"[{label}] '{col}' có giá trị âm."
        )

    # 7. Trên TEST: mọi giá trị test proto/conn_state KHÔNG có trong train
    #    phải có one-hot cột = 0 (KHÔNG sinh cột mới) — đã đảm bảo bởi
    #    cùng schema; kiểm tra thêm: tổng one-hot theo dòng có thể = 0
    #    với test có giá trị lạ, nhưng KHÔNG được > 1 (vì one-hot exclusive
    #    trong train — giả định test cũng vậy, hoặc nếu không thì tổng = 0).
    for prefix in ("resp_port_", "proto_", "conn_state_"):
        cols = [c for c in df_train_feat.columns if c.startswith(prefix)]
        if not cols:
            continue
        row_sums = df_test_feat[cols].sum(axis=1)
        assert int(row_sums.max()) <= 1, (
            f"[{label}] one-hot '{prefix}*' ở test có dòng tổng > 1."
        )


def _run_fe_mock_test() -> None:
    """Mock test cho fit_preprocessor + transform (Task 1.6)."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    tmp_dir = tempfile.mkdtemp(prefix="iot23_fe_mock_")
    mock_path = _create_mock_clean_file(tmp_dir)
    print(f"\n>>> [FE MOCK] Mock file: {mock_path}")

    from src.data_io import load_scenario

    df_clean = clean_flows(load_scenario(mock_path))

    # Train/test split đơn giản: 5 train / 2 test (mock chỉ 7 dòng).
    df_train = df_clean.iloc[:5].reset_index(drop=True)
    df_test = df_clean.iloc[5:].reset_index(drop=True)
    print(f">>> [FE MOCK] Train: {df_train.shape}, Test: {df_test.shape}")

    pre = fit_preprocessor(df_train)
    print(f">>> [FE MOCK] feature_dim = {pre.feature_dim}")
    print(f">>> [FE MOCK] feature_columns ({len(pre.feature_columns)}):")
    for c in pre.feature_columns:
        print(f"      - {c}")

    df_tr_feat = transform(df_train, pre)
    df_te_feat = transform(df_test, pre)
    _assert_fe_invariants(df_tr_feat, df_te_feat, pre, "FE-MOCK")

    # Sanity cụ thể cho mock.
    # - proto train có 'tcp' + 'udp' → 2 cột proto_*
    assert sorted(pre.proto_categories) == ["tcp", "udp"]
    # - service train: dns, telnet, http, ssh, unknown(=3) → freq ≥ 1/5 = 20%
    #   cho dns/telnet/http/ssh (1/5 = 20%); unknown là 3/5 = 60% nhưng bị
    #   ép vào service_other.
    assert "service_other" in pre.feature_columns
    # - conn_state train (5 dòng đầu): C1=SF, C2=SF, C3=S1, C4=S1, C5=SF.
    #   Chỉ có SF + S1. S0 chỉ xuất hiện ở test (C6).
    assert sorted(pre.conn_state_categories) == ["S1", "SF"], (
        f"[FE MOCK] train phải có conn_state=[S1, SF], got "
        f"{pre.conn_state_categories}."
    )
    # - history chars: 20 cột
    assert len(pre.history_flag_chars) == 20
    assert sum(c.startswith("history_n_") for c in pre.feature_columns) == 20

    # Trên TEST: C6 có conn_state=S0 — KHÔNG có trong train → tất cả cột
    # conn_state_* phải = 0 ở dòng C6.
    c6_idx = df_test.reset_index(drop=True).index[0]  # C6 là dòng test đầu
    cs_cols = [c for c in df_te_feat.columns if c.startswith("conn_state_")]
    assert int(df_te_feat.loc[c6_idx, cs_cols].sum()) == 0, (
        "[FE MOCK] conn_state=S0 ở test phải cho tất cả one-hot = 0."
    )

    # ---- Demo save/load ----
    save_path = os.path.join(tmp_dir, "preprocessor.pkl")
    pre.save(save_path)
    pre_loaded = Preprocessor.load(save_path)
    df_te_feat_loaded = transform(df_test, pre_loaded)
    assert list(df_te_feat_loaded.columns) == list(df_te_feat.columns), (
        "[FE MOCK] schema sau load != schema trước save."
    )
    # So sánh giá trị số (loại trừ cột object vì có thể dtype lệch nhẹ).
    for col in df_te_feat.columns:
        if df_te_feat[col].dtype.kind in "fc":  # float / complex
            assert np.allclose(
                df_te_feat[col].values, df_te_feat_loaded[col].values,
            ), f"[FE MOCK] cột '{col}' khác sau load."
        else:
            assert (df_te_feat[col].values == df_te_feat_loaded[col].values).all(), (
                f"[FE MOCK] cột '{col}' khác sau load."
            )
    print(">>> [FE MOCK] Save/load round-trip OK.")

    print("\n[FE MOCK TEST] Tất cả assertions đều PASS.")


def _run_fe_real_test(path: str) -> None:
    """Test feature engineering trên file thật 34-1 (Task 1.6)."""
    from sklearn.model_selection import train_test_split
    from src.data_io import load_scenario

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Không tìm thấy file thật: {path}")
    print(f"\n>>> [FE REAL] Real file: {path}")

    df_clean = clean_flows(load_scenario(path))
    print(f">>> [FE REAL] Shape sau clean_flows: {df_clean.shape}")

    # 80/20 stratified theo detailed-label (chỉ để test luồng).
    df_train, df_test = train_test_split(
        df_clean,
        test_size=0.2,
        stratify=df_clean["detailed-label"],
        random_state=42,
    )
    print(
        f">>> [FE REAL] Train: {df_train.shape}, Test: {df_test.shape}\n"
        f"    Phân bố detailed-label (train):\n"
        f"{df_train['detailed-label'].value_counts().to_string()}\n"
        f"    Phân bố detailed-label (test):\n"
        f"{df_test['detailed-label'].value_counts().to_string()}"
    )

    pre = fit_preprocessor(df_train)
    print(f"\n>>> [FE REAL] feature_dim = {pre.feature_dim}")
    print(f">>> [FE REAL] proto_categories: {pre.proto_categories}")
    print(f">>> [FE REAL] service_categories (kept, >=1%): "
          f"{pre.service_categories}")
    print(f">>> [FE REAL] conn_state_categories: {pre.conn_state_categories}")
    print(f">>> [FE REAL] numeric_columns: {pre.numeric_columns}")
    print(f">>> [FE REAL] missing_flag_columns: {pre.missing_flag_columns}")
    print(f">>> [FE REAL] scaler.mean_ = "
          f"{np.round(pre.scaler.mean_, 3).tolist()}")
    print(f">>> [FE REAL] scaler.scale_ = "
          f"{np.round(pre.scaler.scale_, 3).tolist()}")

    print(f"\n>>> [FE REAL] Tất cả {len(pre.feature_columns)} cột feature:")
    for i, c in enumerate(pre.feature_columns, 1):
        print(f"      {i:3d}. {c}")

    df_tr_feat = transform(df_train, pre)
    df_te_feat = transform(df_test, pre)
    print(
        f"\n>>> [FE REAL] Shape transform — "
        f"train: {df_tr_feat.shape}, test: {df_te_feat.shape}"
    )

    _assert_fe_invariants(df_tr_feat, df_te_feat, pre, "FE-REAL")

    # In mean/std của cột _scaled trên train để xác nhận bằng mắt.
    scaled_cols = [c for c in df_tr_feat.columns if c.endswith("_scaled")]
    print("\n>>> [FE REAL] Mean/std của cột _scaled trên TRAIN (kỳ vọng ~0 / ~1):")
    for col in scaled_cols:
        m = float(df_tr_feat[col].mean())
        s = float(df_tr_feat[col].std(ddof=0))
        print(f"    {col:30s}  mean={m:+.6f}  std={s:.6f}")

    # Sanity: ở real 34-1, 'irc' (1641/23145=7.1%) là service duy nhất
    # >= 1%; còn lại (unknown=92%, dns=0.8%, http=0.05%, dhcp=0.009%) đều
    # vào service_other.
    assert pre.service_categories == ["irc"], (
        f"[FE REAL] kỳ vọng service_categories=['irc'] ở 34-1, "
        f"got {pre.service_categories}."
    )
    # 4 lớp detailed-label đầy đủ ở cả train & test.
    expected_classes = {"DDoS", "C&C", "Benign", "PartOfAHorizontalPortScan"}
    assert set(df_tr_feat["detailed-label"].unique()) == expected_classes
    assert set(df_te_feat["detailed-label"].unique()) == expected_classes

    # ---- Demo save/load round-trip ----
    tmp_dir = tempfile.mkdtemp(prefix="iot23_fe_real_")
    save_path = os.path.join(tmp_dir, "preprocessor_34-1.pkl")
    pre.save(save_path)
    pre_loaded = Preprocessor.load(save_path)
    df_te_loaded = transform(df_test, pre_loaded)
    assert list(df_te_loaded.columns) == list(df_te_feat.columns), (
        "[FE REAL] schema sau load != schema trước save."
    )
    for col in df_te_feat.columns:
        if df_te_feat[col].dtype.kind in "fc":
            assert np.allclose(
                df_te_feat[col].values, df_te_loaded[col].values,
            ), f"[FE REAL] cột '{col}' khác sau load."
    print("\n>>> [FE REAL] Save/load round-trip OK.")
    print("\n[FE REAL TEST] Tất cả assertions đều PASS.")


if __name__ == "__main__":
    """
    Chạy:
        python -m src.preprocess                       → mock test clean_flows.
        python -m src.preprocess <path>                → real test clean_flows.
        python -m src.preprocess --fe                  → mock test fit/transform.
        python -m src.preprocess --fe <path>           → real test fit/transform.
    """
    args = sys.argv[1:]
    if "--fe" in args:
        args = [a for a in args if a != "--fe"]
        if args:
            logging.basicConfig(
                level=logging.INFO,
                format="[%(levelname)s] %(name)s: %(message)s",
            )
            _run_fe_real_test(args[0])
        else:
            _run_fe_mock_test()
    else:
        if args:
            _run_real_clean_test(args[0])
        else:
            _run_mock_clean_test()
