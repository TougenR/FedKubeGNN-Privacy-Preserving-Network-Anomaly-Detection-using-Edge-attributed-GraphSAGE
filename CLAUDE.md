# CLAUDE.md — IoT-23 FL+GNN · Giai đoạn 1 (Baseline GNN tập trung)

> File này là bộ nhớ dự án. Đọc kỹ trước mọi task. Các quyết định trong mục
> "Quy tắc tiền xử lý" và "Dựng đồ thị" đã được chốt về mặt lý thuyết —
> KHÔNG tự ý thay đổi; nếu thấy mâu thuẫn, hỏi lại người dùng thay vì tự quyết.

## 1. Bối cảnh

Đồ án TTTN nhóm M06 (Nguyễn Khắc Bảo & Nguyễn Chí Hiếu): *Phát hiện hành vi
độc hại trong Kubernetes bằng Federated Learning + GNN trên bộ dữ liệu IoT-23*.
Đồ án có 3 giai đoạn. Repo chứa baseline tập trung Phase 1, pipeline Federated
Learning Phase 2 và Minikube inference PoC Phase 3. Ownership và việc còn lại
được cập nhật tại `docs/HANDOFF_TO_HIEU.md`.

Ràng buộc kế thừa cho GĐ2/GĐ3 (phải tôn trọng ngay từ GĐ1):
- Pipeline tiền xử lý viết dạng **hàm/module tái sử dụng** (không code kiểu
  notebook một lần), vì GĐ3 sẽ dùng lại cho network flow từ Zeek/Cilium Hubble.
  Falco là nguồn syscall/process alert song song, không phải input thay thế.
- Cách xử lý mất cân bằng lớp phải **nhất quán** giữa baseline và các client FL.

## 2. Môi trường & nguyên tắc hạ tầng

- **Phát triển ở local:** MacBook M2 Pro (Apple Silicon, KHÔNG có CUDA). Chỉ để
  viết code và test trên mẫu nhỏ. PyTorch Geometric trên MPS hay lỗi → khi test
  ở local, ép chạy **CPU**.
- **Train thật:** GPU thuê trên vast.ai (Linux + CUDA), lấy code qua `git pull`.
- **Device-agnostic bắt buộc:** không hardcode `.cuda()`. Luôn dùng
  `device = "cuda" if torch.cuda.is_available() else "cpu"` và `.to(device)`.
- **Tách 2 pha rõ ràng:**
  - *Preprocess* (CPU, chạy **một lần**) → lưu ra file trung gian (`.pt`/`.parquet`).
  - *Train* (GPU, chạy **nhiều lần** khi tune) → đọc file trung gian, KHÔNG xử lý
    lại dataset từ đầu mỗi lần.
- Mọi thứ phải **reproducible**: seed numpy + torch + random ngay đầu chương trình.

## 3. Dataset: IoT-23 v1 (định dạng Zeek conn.log.labeled)

- Nguồn chính thức tải file: `https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/<TÊN_SCENARIO>/`
  (bên trong mỗi scenario có thư mục Zeek chứa `conn.log.labeled`).
  **Chỉ tải `conn.log.labeled`, KHÔNG tải file pcap** (pcap nặng GB, không dùng
  ở mức flow). Dùng bản **v1** (khớp với lý thuyết đã chuẩn bị), không dùng v2.
- Mỗi dòng = 1 **flow** (luồng kết nối giữa 2 IP). File có **21 cột**.
- **Cột thứ 21 bị gộp 3 giá trị**: `tunnel_parents label detailed-label`, cách
  nhau bằng khoảng trắng (lỗi định dạng kế thừa từ Zeek). **Phải tách thành 3 cột
  riêng trước khi làm bất cứ gì khác**, nếu không pandas sẽ đọc lệch toàn bộ.
- Header nằm trong các dòng `#fields` / `#types` đầu file Zeek; các dòng bắt đầu
  bằng `#` là metadata, không phải dữ liệu.

**6 scenario đã chọn cho GĐ1** (đa dạng họ malware, tổng dung lượng nhẹ):

| Scenario | Malware |
|---|---|
| CTU-IoT-Malware-Capture-34-1 | Mirai |
| CTU-IoT-Malware-Capture-1-1  | Hide and Seek |
| CTU-IoT-Malware-Capture-3-1  | Muhstik |
| CTU-IoT-Malware-Capture-9-1  | Linux.Hajime |
| CTU-IoT-Malware-Capture-36-1 | Okiru |
| CTU-IoT-Malware-Capture-39-1 | IRCBot |

- **Nhãn để train = `detailed-label`** (bài toán multi-class). Giữ `label`
  (Benign/Malicious) chỉ để đối chiếu nhanh. Bỏ `tunnel_parents`.
## 3b. Dung lượng thật & nơi chạy tiền xử lý (CẬP NHẬT)

- conn.log.labeled KHÔNG nhẹ đồng đều: 34-1 ~2.7MB, 3-1 ~23MB, 1-1 ~141MB,
  nhưng 9-1 / 36-1 / 39-1 mỗi file nhiều GB → tổng 6 file ~13GB.
- BẮT BUỘC đọc theo CHUNK (pandas chunksize / đọc từng dòng), KHÔNG load cả file
  vào RAM. Undersample lớp đa số (PartOfAHorizontalPortScan) NGAY trong lúc đọc
  từng chunk để bóp dung lượng xuống trước khi giữ lại.
- Tiền xử lý FULL (tải 13GB + đọc) chạy trên vast.ai (RAM rộng, tải nhanh), lưu
  ra 1 file .parquet gọn đã undersample → mới đưa file nhỏ này về Mac để dev model.
- Máy Mac M2 chỉ chạy TEST MOCK với dữ liệu giả, không xử lý file thật.
- Mỗi scenario có thể có format #fields khác nhau (tab/space, số cột chênh) →
  đọc tên cột theo dòng #fields của TỪNG file, không hardcode, và không giả định
  6 file giống nhau.

## 4. Quy tắc tiền xử lý (ĐÃ CHỐT — không tự đổi)

- **Xử lý nhãn trước tiên:** đổi `detailed-label` từ `"-"` / `"(empty)"` thành
  `"Benign"` trước khi chuyển token thiếu ở các cột khác thành NaN. Đảo thứ tự
  này sẽ làm mất lớp Benign.
- **Giá trị thiếu:** Zeek dùng ký tự `-` cho "không có giá trị". Chuyển `-` thành
  `NaN` thực sự (không giữ dạng chuỗi) TRƯỚC khi ép kiểu.
  - Cột categorical (`service`): điền `NaN` thành nhãn riêng `"unknown"`
    (bản thân "không nhận diện được service" cũng là tín hiệu).
  - Cột numeric: điền 0, và thêm 1 cột cờ nhị phân đánh dấu "có bị thiếu không".
- **Cột LOẠI KHỎI feature hoàn toàn:** `uid` (định danh, high-cardinality, gây học
  vẹt), `tunnel_parents` (gần như luôn `-`), `local_orig`, `local_resp` (gần như
  không đổi trong IoT-23 → không phân biệt).
- **Địa chỉ IP (`id.orig_h`, `id.resp_h`): TUYỆT ĐỐI KHÔNG đưa vào feature.**
  Chỉ dùng để **định danh node** (xác định ai-nối-với-ai) khi dựng đồ thị. Đây là
  điểm cốt lõi phân biệt GNN với ML truyền thống.
- **Cổng:**
  - `id.orig_p` (cổng nguồn): thường ephemeral, gần như vô nghĩa → bỏ, hoặc chỉ
    giữ 1 cờ nhị phân "well-known port hay không".
  - `id.resp_p` (cổng đích): QUAN TRỌNG (cho biết dịch vụ mục tiêu). Xử lý bằng
    **phân nhóm (bucket)**: well-known (0–1023), registered (1024–49151),
    dynamic (49152+). KHÔNG one-hot toàn bộ 65536 giá trị, KHÔNG coi là số liên tục.
- **Categorical (`proto`, `service`, `conn_state`, `history`):** encode phù hợp.
  `conn_state` và `history` là **2 cột tín hiệu hành vi mạnh nhất** (mô tả hình
  dạng bắt tay TCP → phân biệt scan/DDoS) → ưu tiên làm **đặc trưng cạnh**.
- **Numeric (`duration`, `orig_bytes`, `resp_bytes`, `orig_pkts`, `resp_pkts`,
  `orig_ip_bytes`, `resp_ip_bytes`, `missed_bytes`):** phân phối lệch mạnh
  (heavy-tailed) → **áp dụng `log1p(x)` TRƯỚC, rồi mới standard-scale**. Không
  min-max/z-score trực tiếp (outlier sẽ nén phần còn lại về 0).
- **`ts` (timestamp):** KHÔNG dùng làm feature (không tổng quát hóa được), nhưng
  **GIỮ LẠI** trong dữ liệu đã xử lý — GĐ3 có thể cần để chia đồ thị theo cửa sổ
  thời gian.
- **Fit scaler/encoder trên tập train, chỉ transform trên tập test** (tránh rò rỉ
  dữ liệu). Lưu lại scaler/encoder để tái dùng.

## 5. Mất cân bằng lớp (ĐÃ CHỐT)

IoT-23 lệch lớp tới hàng trăm triệu lần (PartOfAHorizontalPortScan áp đảo,
C&C-Mirai chỉ vài mẫu).
- **KHÔNG dùng SMOTE.** Nội suy giữa 2 flow sẽ tạo "cạnh giả" nối 2 IP chưa từng
  giao tiếp → phá vỡ tính đúng đắn của đồ thị.
- **Ưu tiên:** `class-weighted loss` (gán trọng số cao cho lớp hiếm), và/hoặc
  **undersampling có kiểm soát** lớp đa số. Cả hai không làm hỏng cấu trúc đồ thị.
- Nếu undersample thì làm **trước khi dựng đồ thị** (vì nó bỏ bớt cạnh).

## 6. Dựng đồ thị

- **node = IP**, **cạnh = flow**, đặc trưng hành vi gắn lên **cạnh** (`edge_attr`).
- **Đây là bài toán EDGE classification** (phân loại từng flow = từng cạnh).
  Nhãn `detailed-label` gắn trên **CẠNH**, KHÔNG phải node. Không xây node classification.
- Theo tinh thần **E-GraphSAGE**: đặc trưng node khởi tạo là **vector hằng
  (all-ones)** vì thông tin phân biệt nằm ở cạnh, không ở node.
- Dùng đối tượng `torch_geometric.data.Data`. Cẩn thận bước **map IP → chỉ số
  node (index) liên tục 0..N-1**; `edge_index` phải là tensor `[2, num_edges]`
  kiểu long. Đây là chỗ dễ sai nhất.
- **GĐ1:** gộp toàn bộ flow của một scenario thành **một đồ thị tĩnh duy nhất**
  (cách đơn giản). Không cần chia cửa sổ thời gian ở GĐ1.

## 7. Model

- **E-GraphSAGE (Task 1.9, độ khó ★★★★★):** PyG **không có sẵn**. Phải tự viết
  lớp kế thừa `torch_geometric.nn.MessagePassing`, trong hàm `message()` **ghép
  (concat) đặc trưng node nguồn + đặc trưng cạnh**, rồi mới aggregate. Đây là
  phần tinh vi nhất — làm cẩn thận, test shape từng bước.
- **Baselines đối chứng (Task 1.11):** GCN (`GCNConv`), GraphSAGE gốc
  (`SAGEConv`) — điều chỉnh để phục vụ edge classification (ví dụ: ghép embedding
  2 node đầu-cuối của cạnh + edge_attr rồi đưa qua classifier head).
- **Phương án B nếu E-GraphSAGE quá khó:** dùng `SAGEConv` sẵn có + nối đặc trưng
  cạnh vào đặc trưng node trước khi vào layer; ghi rõ trong báo cáo đây là bản
  "gần đúng" của E-GraphSAGE.

## 8. Đánh giá (Task 1.12)

- Do mất cân bằng cực đoan, **accuracy tổng thể gần như vô nghĩa** (đoán toàn lớp
  đa số vẫn cao). **Chỉ số chính = macro-F1 và F1 theo từng lớp.**
- Bắt buộc có **confusion matrix**, chú ý riêng các lớp hiếm.
- Cũng ghi Precision/Recall theo lớp và AUC-ROC nếu áp dụng được.

## 9. Cấu trúc repo (mục tiêu)

```
.
├── CLAUDE.md               # file này
├── README.md
├── requirements.txt
├── .gitignore              # bỏ qua data/, *.pt, checkpoints/, .venv/
├── config.yaml             # danh sách scenario, đường dẫn, hyperparameter
├── data/                   # conn.log.labeled tải về (gitignore)
├── artifacts/              # file trung gian đã xử lý (.parquet, .pt) (gitignore)
├── checkpoints/            # model đã train (gitignore)
├── scripts/
│   └── download_data.sh    # wget conn.log.labeled 6 scenario
└── src/
    ├── data_io.py          # đọc conn.log.labeled → DataFrame, tách cột 21
    ├── preprocess.py       # làm sạch, encode, scale (hàm tái sử dụng)
    ├── imbalance.py        # tính class weights / undersample
    ├── graph_build.py      # DataFrame → PyG Data, lưu .pt
    ├── model.py            # E-GraphSAGE + baselines GCN/GraphSAGE
    ├── train.py            # vòng train device-agnostic, checkpoint
    └── evaluate.py         # macro-F1, per-class F1, confusion matrix
```

## 10. Quy ước code

- Điều khiển bằng `config.yaml` (scenario nào, hyperparameter nào) — không hardcode.
- Mỗi module là hàm rõ ràng, có docstring, có thể import và test độc lập.
- Ưu tiên `pandas` (hoặc `polars` nếu cần tốc độ) cho xử lý bảng.
- Không dùng notebook cho code chính; notebook (nếu có) chỉ để EDA/khám phá.
- Ghi log tiến trình rõ ràng (dùng `logging` hoặc print có cấu trúc).
