# Phase 2: IoT-23 Federation

Phase 2 biến đầu ra logic của Phase 1 thành sáu client non-IID theo scenario,
huấn luyện cùng E-GraphSAGE bằng FedAvg/FedProx/FedPer, và lưu đủ bằng chứng để tái
lập hoặc chẩn đoán một run. Historical macro-F1 `0.8773` chỉ là số tham khảo;
benchmark mới fit preprocessing trên train và phải tự tạo centralized result.

## Luồng hệ thống hiện tại

```text
6 x conn.log.labeled
        │ deterministic priority sample, cap/class, SHA-256
        ▼
6 x clean DataFrame ── label-aware 70/10/20 edge masks
        │
        ├── concat TRAIN rows only ──► shared Preprocessor + global labels
        │
        ▼
transform all rows with frozen preprocessor
        │
        ▼
IP nodes + flow edges ──► portable .npy client graphs
        │                  contract JSON/NPZ + checksums + manifest
        ▼
ManifestIoT23Task (server loads contract/state only)
        │
        ├── client selected ──► lazy load exactly one graph
        │
        ▼
30 rounds, 6 clients, full participation, 5 local epochs
        │
        ├── FedAvg: weighted average by train-edge count
        ├── FedProx: FedAvg + local proximal loss, mu=0.01
        └── FedPer: aggregate layers.*, retain head.* per client
        │
        ▼
validation each round ──► best checkpoint ──► test exactly once
        │
        ▼
run.json + JSONL events + rounds.csv/jsonl + summary + .npz checkpoints
```

Đây vẫn là protocol `transductive_edge_mask`: train/validation/test là các
cạnh khác nhau trong cùng graph; message passing được phép nhìn feature và
topology của toàn graph nhưng không nhìn nhãn ngoài mask. Protocol luôn nằm
trong config, manifest, task metadata và event.

## Boundary và extension points

Core chỉ biết `FederatedTask`, named NumPy state, schema và confusion matrix.
PyG/E-GraphSAGE nằm sau adapter. Config nghiêm ngặt chọn rõ:

- `data_source`, `partitioner`, `graph_builder`;
- `model`, `task`, `strategy`, `runtime`, `observer`.

`src/federated/registry.py` từ chối tên lạ và duplicate; không cho config import
Python object tùy ý. Các import cũ trong `src/federated/adapters/` và
`src/federated/flower/` vẫn là compatibility surface.

Các package chính:

- `config/`, `registry.py`: config versioned và extension registry.
- `data/`: sampling, split, preparation, portable graph và manifest validation.
- `tasks/`: manifest-backed lazy IoT-23 task và toy compatibility export.
- `strategies/`, `runtimes/`: FedAvg/FedProx và observed in-process runtime.
- `flower/`: Flower 1.32 Message API, task switching và client/server events.
- `observability/`: backend-neutral observer, JSONL sink, atomic run store.
- `experiments/`: centralized reference, task factory và result comparison.

## Artifact contracts

Prepared dataset:

```text
artifacts/phase2/prepared/<dataset-id>/
├── manifest.json
├── initial_state.npz
├── contract/
│   ├── feature_schema.json, label_schema.json, graph_schema.json
│   ├── model_spec.json, categories.json, learned_arrays.npz
│   └── manifest.json, checksums.json
└── clients/<scenario-id>/
    ├── edge_index.npy, edge_attr.npy, edge_label.npy
    ├── train_mask.npy, val_mask.npy, test_mask.npy
    └── metadata.json, checksums.json
```

Không lưu PyG pickle, raw IP mapping hay tensor dump trong log. Loader dựng lại
`x=ones`, cạnh đảo và message-passing edge attributes. Mọi mask phải boolean,
rời nhau và phủ đúng toàn bộ edge; checksum sai thì fail trước train. Prepared
manifest v2 ràng buộc checksum index của contract và từng client, đồng thời kiểm
tra feature/label schema và số edge dùng chung. Artifact manifest v1 cũ phải được
tạo lại bằng lệnh `prepare` và không được resume như cùng dataset.

Observed run:

```text
artifacts/phase2/runs/<run-id>/
├── run.json, config.snapshot.json, failure.json (nếu lỗi)
├── events/server.jsonl
├── metrics/rounds/round-NNNN.json, rounds.jsonl, rounds.csv, summary.json
└── checkpoints/round-NNNN.npz, best_model.npz
```

Post-training diagnostics reuse those completed artifacts and never evaluate
test data again:

```text
artifacts/phase2/visualizations/<release-id>/
├── round_metrics.csv, final_metrics.csv, visualization_manifest.json
├── federated_learning_curves.{png,pdf}
├── federated_final_metrics.{png,pdf}
├── federated_per_class_f1.{png,pdf}
└── federated_confusion_matrices.{png,pdf}
```

File `metrics/rounds/round-NNNN.json` là commit marker bền vững; JSONL, CSV,
`run.json` và best checkpoint được reconcile từ marker này khi resume sau crash.

Event có UTC timestamp, run/component/strategy/round/client, duration, count,
loss/F1, byte count và digest phù hợp. Field chứa password, token, raw IP,
edge attributes hay tensor bị từ chối. Interface observer không phụ thuộc
backend để Phase 3 có thể thêm OpenTelemetry/Prometheus.

## Chạy

Config chuẩn: `configs/federated/phase2/iot23-federated.yaml`. Global CLI option đứng
trước subcommand:

```bash
python -m src.federated.cli --config configs/federated/phase2/iot23-federated.yaml doctor
python -m src.federated.cli --config configs/federated/phase2/iot23-federated.yaml prepare
python -m src.federated.cli validate --dataset artifacts/phase2/prepared/<dataset-id>

python -m src.federated.cli run --dataset artifacts/phase2/prepared/<dataset-id> --strategy fedavg
python -m src.federated.cli run --dataset artifacts/phase2/prepared/<dataset-id> --strategy fedprox
python -m src.federated.cli centralized --dataset artifacts/phase2/prepared/<dataset-id>
python -m src.federated.cli evaluate --dataset artifacts/phase2/prepared/<dataset-id> --checkpoint <best_model.npz> --split test --output artifacts/phase2/test.json
python -m src.federated.cli compare --runs <central-run> <fedavg-run> <fedprox-run> --output artifacts/phase2/comparison.csv
python -m src.federated.cli visualize --runs <fedavg-run> <fedprox-run> --output artifacts/phase2/visualizations/manual
```

Để smoke không cần PyG/data thật:

```bash
python -m src.federated.cli run --task toy --strategy fedavg --rounds 2
python -m src.federated.cli run --task toy --strategy fedprox --rounds 2
```

Flower dùng unified apps trong `pyproject.toml`; mặc định vẫn là toy để proof:

```bash
flwr run . --stream
```

IoT-23 Flower run cần override `task=iot23_manifest`, `dataset-root` và bật
`save-model`; các hyperparameter benchmark (30 round, 5 local epoch, optimizer,
learning rate và FedProx mu) được lấy trực tiếp từ `phase2-config`, không dùng
toy defaults trong `pyproject.toml`. Mỗi process ghi JSONL riêng dưới
`events-output`. FedAvg, FedProx và FedPer đều dùng full participation, chọn checkpoint
tốt nhất theo validation macro-F1 rồi mới đánh giá test đúng một lần. Global
metric được tính từ tổng fixed-K confusion matrix, không average client macro-F1.
Với FedPer, Flower chỉ vận chuyển `layers.*`; `head.*` được version hóa dưới
`personalized-state-root/<client>/<run>/<model-digest>/` và evaluation fail
closed trước local round đầu tiên. Central best checkpoint vì vậy chỉ là shared
encoder, không phải một global model hoàn chỉnh.
Lệnh `visualize` kiểm tra provenance và trạng thái completed trước khi vẽ. Learning
curve chỉ dùng validation metric theo round; bar chart, per-class F1 và confusion
matrix dùng đúng final-test metric đã lưu, không chạy test lần hai và không tác
động checkpoint selection.

## Validation và giới hạn hiện tại

Local proof chạy bằng:

```bash
python -m unittest discover -s tests/federated -v
python -m compileall -q src/federated
git diff --check
```

Proof bao gồm strict config/registry, integrity/tamper, deterministic sampling
và split, server init không load graph, one-client FedAvg equivalence, sáu-client
FedAvg/FedProx observed runs, FedPer private-state recovery/transport,
communication bytes và best/test artifacts.

Máy hiện tại chưa có raw IoT-23, `torch_geometric` hoặc `flwr`; `doctor` vì vậy
fail rõ ràng trước preparation. Full six-scenario E-GraphSAGE/Flower result và
centralized metric phải chạy trên môi trường có data/dependency; không được xem
synthetic proof hay số Phase 1 cũ là bằng chứng thay thế.
