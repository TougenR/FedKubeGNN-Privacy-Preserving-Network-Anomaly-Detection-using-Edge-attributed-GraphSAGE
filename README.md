# IoT-23 FL+GNN — Giai đoạn 1: Baseline GNN tập trung

Đồ án TTTN nhóm M06 (Nguyễn Khắc Bảo & Nguyễn Chí Hiếu):
*Phát hiện hành vi độc hại trong Kubernetes bằng Federated Learning + GNN
trên bộ dữ liệu IoT-23.*

Repo chứa baseline **Giai đoạn 1** do **Nguyễn Khắc Bảo** phụ trách: tiền xử lý
IoT-23 → dựng đồ thị hành vi → huấn luyện **E-GraphSAGE** + 4 baseline
(`gat`, `sage_edge_concat`, `graphsage`, `gcn`) tập trung trên 1 máy.

Pipeline **Giai đoạn 2** nằm trong `src/federated/`: chuẩn bị sáu client IoT-23
với shared train-only preprocessing, chạy FedAvg/FedProx, chọn checkpoint theo
validation và ghi structured JSONL/run artifacts. Core FedAvg/metrics và Flower
runtime không phụ thuộc PyG hay implementation Phase 1. Xem
[kiến trúc Phase 2](docs/PHASE2_ARCHITECTURE.md) trước khi chạy dữ liệu thật.

Preflight và smoke Phase 2:

```bash
python -m src.federated.cli --config configs/phase2/iot23-federated.yaml doctor
python -m unittest discover -s tests/federated -v
python -m src.federated.cli run --task toy --strategy fedavg --rounds 2
```

`doctor` cho biết chính xác raw scenario/dependency/disk nào còn thiếu. Phase 2
cần raw dataset để tạo artifact mới, nhưng không cần tải checkpoint Phase 1;
initial E-GraphSAGE state và centralized reference được tạo lại từ config/seed.

PoC **Giai đoạn 3** nằm trong `phase3_monitoring/`: FastAPI inference, labeled
IoT-23 replay, Docker và Minikube. PoC hiện fail-closed nếu checkpoint và
preprocessor không cùng feature schema; chưa claim zero-day detection hoặc
production cloud readiness. Xem
[README Phase 3](phase3_monitoring/README.md) và
[bàn giao Bảo → Hiếu](docs/HANDOFF_TO_HIEU.md).

## Cấu trúc repo

```
.
├── CLAUDE.md                  # bộ nhớ dự án (đọc kỹ trước khi code)
├── README.md                  # file này
├── requirements.txt
├── .gitignore
├── config.yaml                # cấu hình trung tâm: scenario, đường dẫn, hyperparameter
├── data/                      # conn.log.labeled tải về (gitignore)
├── artifacts/                 # file trung gian đã xử lý (gitignore)
│   ├── phase1_results/        # kết quả GĐ1: CSV + PNG + checkpoint
│   └── eda_summary.csv        # EDA summary (được track)
├── checkpoints/               # model đã train (gitignore)
├── scripts/
│   ├── download_data.sh       # wget 1 file conn.log.labeled (helper của download_all.sh)
│   ├── download_all.sh        # wget TẤT CẢ scenario trong config (khuyến nghị)
│   ├── eda_all_scenarios.py   # EDA TRƯỚC khi train — ma trận hiện diện + gợi ý cap
│   ├── run_full_gpu.sh        # chạy orchestrator trên vast.ai (CUDA check + git backup)
│   ├── test_run_experiments.py    # smoke test orchestrator (2 scenario, 30 epoch)
│   ├── test_multi_scenario.py     # smoke test LOSO harness
│   ├── test_resume_logic.py       # smoke test resume mechanism
│   └── test_*.py              # các smoke test cho từng module src/
└── src/
    ├── data_io.py             # đọc conn.log.labeled → DataFrame, tách cột 21
    ├── preprocess.py          # làm sạch, encode, scale (hàm tái sử dụng)
    ├── imbalance.py           # tính class weights / undersample
    ├── graph_build.py         # DataFrame → PyG Data, lưu .pt
    ├── model.py               # E-GraphSAGE + baselines GCN/GraphSAGE/GAT
    ├── train.py               # vòng train device-agnostic, checkpoint
    ├── evaluate.py            # macro-F1, per-class F1, confusion matrix
    ├── multi_scenario.py      # tầng dữ liệu đa-scenario + LOSO inductive
    └── run_experiments.py     # orchestrator: 2-Phase × 3 protocol × {3 mode, 5 model}
```

## Setup môi trường

### A. Local (MacBook M2 Pro — Apple Silicon, không CUDA)

Mục đích: viết code và test trên mẫu nhỏ. **KHÔNG train thật ở local.**

```bash
# 1. Tạo venv và kích hoạt
python3 -m venv .venv
source .venv/bin/activate

# 2. Cài torch bản CPU-only (PyG trên MPS hay lỗi)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. Cài torch-geometric (bản CPU)
pip install torch-geometric

# 4. Cài các dependency còn lại
pip install -r requirements.txt

# 5. (Tùy chọn) tải dataset bằng script
bash scripts/download_all.sh --apply
```

Trong code luôn dùng `device = "cuda" if torch.cuda.is_available() else "cpu"`
và `.to(device)` — không hardcode `.cuda()`. Trên Mac M2 sẽ tự rơi về CPU.

### B. Trên vast.ai (Linux + CUDA — train thật)

Xem mục **"Chạy full trên vast.ai"** bên dưới.

## Chạy thử nghiệm trên local (CPU)

Smoke test đã có sẵn — chạy được trên Mac, mất khoảng 1–2 phút:

```bash
# EDA trước (không cần GPU):
.venv/bin/python scripts/eda_all_scenarios.py --cap 5000

# Smoke test orchestrator (2 scenario {34-1, 3-1}, 30 epoch):
.venv/bin/python scripts/test_run_experiments.py

# Smoke test LOSO harness:
.venv/bin/python scripts/test_multi_scenario.py

# Smoke test resume logic (xác nhận skip đúng config đã có):
.venv/bin/python scripts/test_resume_logic.py
```

## Chạy full trên vast.ai (Linux + CUDA)

Mục tiêu: train 6 scenario × 3 protocol × (3 mode + 5 model) = **144 training
jobs** với 150 epoch, sinh `results_summary.csv` + confusion matrix PNG đưa vào
báo cáo. **Tổng thời gian ước tính 4–8 giờ trên GPU RTX 3090/4090.**

### Bước 1 — Tạo instance + tmux

Trên giao diện vast.ai, chọn:
- **Image**: `pytorch 2.x + cu121` (hoặc CUDA 12.x mới nhất)
- **GPU**: RTX 3090 / 4090 (24 GB VRAM) hoặc A5000
- **Disk**: ≥ 50 GB (đủ chứa 6 scenario + artifacts)
- **SSH** vào instance, **CHẮC CHẮN chạy trong tmux** (để không bị mất
  session khi mất kết nối mạng):

```bash
tmux new -s fedkube
```

### Bước 2 — Clone + cài dependencies

```bash
git clone <repo-url> fedkube-ids && cd fedkube-ids

# Tạo venv
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# Cài torch ĐÚNG phiên bản CUDA của image.
# Ví dụ image có CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Cài torch-geometric + các optional sparse package
pip install torch-geometric
pip install torch_scatter torch_sparse \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# Phần còn lại
pip install -r requirements.txt
```

> Tra cứu wheel chính xác cho `torch_scatter` / `torch_sparse` tại
> <https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html>.

### Bước 3 — Tải data

```bash
# Idempotent — bỏ qua file đã tải. In kế hoạch trước (dry-run):
bash scripts/download_all.sh
# Khi OK thì tải thật (mất 30-60 phút cho 6 scenario, đặc biệt 39-1 ~10GB):
bash scripts/download_all.sh --apply

# Kiểm tra tổng dung lượng (cảnh báo nếu > 15 GB):
du -sh data/CTU-IoT-Malware-Capture-*/
```

### Bước 4 — EDA (TRƯỚC khi train)

```bash
.venv/bin/python scripts/eda_all_scenarios.py --cap 50000
```

Xem kỹ:
- **Ma trận hiện diện lớp** → biết lớp nào PRIVATE (chỉ xuất hiện ở 1
  scenario, sẽ F1=0 khi LOSO held-out đúng scenario đó).
- **Lớp hiếm toàn cục** → sẽ in warning, cân nhắc `class_weight`.
- **Gợi ý `cap_per_class`** ở mục [4] — chỉnh `config.yaml` nếu cần.

Sau khi EDA ưng ý, đối chiếu với block `experiments.cap_per_class` trong
`config.yaml`.

### Bước 5 — Chạy full experiments

```bash
# Mặc định đọc tất cả từ config.yaml. Có thể override:
bash scripts/run_full_gpu.sh                  # mặc định
bash scripts/run_full_gpu.sh --epochs 100     # chạy 100 epoch thay vì 150
bash scripts/run_full_gpu.sh --cap 20000      # cap_per_class = 20000
bash scripts/run_full_gpu.sh --no-git         # bỏ git backup cuối run
```

Script tự động:
1. **Kiểm tra CUDA** ngay đầu — thoát với lỗi rõ ràng nếu không có GPU
   (tránh vô tình train trên CPU tốn thời gian).
2. **Kiểm tra file data** — thoát nếu scenario nào bị thiếu.
3. **Gọi `python -m src.run_experiments --auto-resume`** — orchestrator
   chạy 24 logical configs × 6 scenarios × 150 epoch.
4. **Save `results_summary.csv` sau MỖI protocol** — nếu instance chết
   giữa chừng, kết quả đến trước đó vẫn còn trên đĩa.
5. **Resume tự động** — chạy lại `bash scripts/run_full_gpu.sh`, script
   sẽ tự BỎ QUA các config đã có trong `results_summary.csv`.
6. **Git backup ở cuối** — `git add artifacts/ && git commit && git push`.
   Nếu push fail (no credential) → in lệnh `rsync/scp` thay thế.

### Bước 6 — Lấy kết quả + dọn instance

```bash
# Trên vast.ai (hoặc local sau khi git pull):
ls -la artifacts/phase1_results/
#   ├── results_summary.csv       ← bảng tổng hợp cuối cùng
#   ├── phase_a_<proto>_egraphsage_3modes.csv      ← Phase A mỗi protocol
#   ├── phase_b_<proto>_mode-<mode>_5models.csv   ← Phase B mỗi protocol
#   └── checkpoints/
#       ├── cm_pooled_*.png                       ← confusion matrix pooled
#       ├── pooled_*_seed42.pt                    ← checkpoint pooled
#       └── confusion_matrix_loso_*_hardest_*.png ← CM LOSO khó nhất

# DESTROY INSTANCE trên vast.ai để KHỎI tốn tiền:
# (giao diện web vast.ai → Destroy)
```

### Nhắc quan trọng

- **LUÔN chạy trong tmux** (Bước 1) — mất kết nối SSH không làm mất job.
- **ĐẶC BIỆT destroy instance SAU KHI push xong** — quên = tiền cứ trừ mỗi giờ.
- **Có thể chạy nhiều lần** `run_full_gpu.sh` an toàn — `--auto-resume`
  đảm bảo không train lại config đã xong.
- **Nếu muốn chạy lại TỪ ĐẦU** (vd sau khi chỉnh hyperparam):
  ```bash
  rm -rf artifacts/phase1_results/
  bash scripts/run_full_gpu.sh
  ```

## Tài liệu tham chiếu

- **[Báo cáo Giai đoạn 1](docs/PHASE1_REPORT.md)** — kết quả thực nghiệm, phân tích, phát hiện chính
- **[Tài liệu bàn giao Giai đoạn 2](docs/HANDOFF_PHASE2.md)** — hướng dẫn cho người làm Federated Learning
- **[Kiến trúc Phase 2](docs/PHASE2_ARCHITECTURE.md)** — contract, trust boundary, cách chạy và validation gates
- **[Phase 3 Minikube PoC](phase3_monitoring/README.md)** — inference contract, replay evaluation và giới hạn
- **[Bàn giao Bảo → Hiếu](docs/HANDOFF_TO_HIEU.md)** — ownership, blocker và thứ tự việc còn lại
- Quyết định thiết kế, quy tắc tiền xử lý, mô hình: xem `CLAUDE.md`.
- Dataset: <https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/>
- PyTorch Geometric: <https://pytorch-geometric.readthedocs.io/>
- vast.ai docs: <https://docs.vast.ai/>
