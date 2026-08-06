# Bàn giao Bảo -> Hiếu

Ngày cập nhật: 2026-07-29

## Mục tiêu chung

Hoàn thiện bằng chứng khoa học cho Federated Learning trước, sau đó nối model FL
đã kiểm chứng vào Minikube inference PoC. Cloud Kubernetes thật là tùy chọn,
không phải điều kiện nghiệm thu PoC hiện tại.

## Bảo đã phụ trách

- Phase 1: pipeline IoT-23, E-GraphSAGE và centralized/LOSO baseline.
- Phase 3 PoC: FastAPI inference, IoT-23 replay, Docker, Minikube Service,
  Deployment inference và Job feeder.
- Phase 3 hardening: artifact contract fail-closed, health probes, response
  provenance, labeled replay evaluator và tài liệu giới hạn.

Kết quả Phase 1 chính:

- Pooled E-GraphSAGE + class weight: macro-F1 `0.8773`.
- LOSO E-GraphSAGE: mean macro-F1 `0.2334`.
- Kết luận đúng: model mạnh trên pooled transductive graph nhưng tổng quát hóa
  sang scenario mới còn yếu.

## Kiến trúc đang có và kiến trúc cần triển khai tiếp

### A. Những gì repository đã có và đã kiểm chứng

```text
IoT-23 conn.log.labeled
        │
        ▼
flow_feeder (replay theo batch)
        │ HTTP /predict
        ▼
FastAPI inference_service
        │ clean + frozen preprocessor
        ▼
batch_local_graph (PyG)
        │
        ▼
E-GraphSAGE checkpoint Phase 1 (.pt)
        │
        ▼
label + confidence + probability map + entropy
```

Luồng trên được đóng gói bằng Docker và chạy bằng Minikube. Manifest hiện tại
chỉ gồm Namespace, inference Deployment, ClusterIP Service và feeder Job.
Feeder vẫn là replay dữ liệu IoT-23; chưa phải network collector thật.

Song song với đó, Phase 2 đã có nền tảng code cho:

```text
6 scenario IoT-23
        ▼
train-only preprocessing + manifest/contract
        ▼
client graph artifacts
        ▼
centralized reference / FedAvg / FedProx
        ▼
best_model.npz + run/metric artifacts
```

Các lệnh và contract đã có, nhưng full run IoT-23 và deployment cloud chưa
được nghiệm thu trên môi trường hiện tại.

### B. Kiến trúc mục tiêu Hiếu cần triển khai

```text
                         CLOUD KUBERNETES - CENTRAL
┌─────────────────────────────────────────────────────────────────────┐
│ Ingress / Load Balancer + TLS hoặc mTLS                             │
│        │                                                            │
│        ▼                                                            │
│ Flower Server (FedAvg/FedProx) ──► Model/Artifact Registry           │
│        │                         (best checkpoint + provenance)      │
│        └──────────────► Prometheus/Grafana hoặc observability backend │
└─────────────────────────────────────────────────────────────────────┘
                         ▲                  │
                         │ secure Flower   │ model update
                         │ communication   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         EDGE KUBERNETES                             │
│                                                                     │
│ Zeek hoặc Cilium Hubble ─► flow converter ─► local graph/inference   │
│                                                                     │
│ Falco DaemonSet ─────────────────────────► runtime syscall alert    │
│                                                                     │
│ Flower Client ───────────── local training + weight synchronization  │
│                                                                     │
│ Decision/Alert router: known-label alert, uncertainty alert và      │
│ runtime alert (ba nguồn được theo dõi riêng)                        │
└─────────────────────────────────────────────────────────────────────┘
```

Phần cần nhớ khi triển khai:

- **Zeek hoặc Cilium Hubble** là nguồn network flow cho E-GraphSAGE.
- **Falco** chỉ cung cấp syscall/process/runtime alert và chạy song song; không
  thay thế flow collector.
- Entropy hiện chỉ là uncertainty score. Chưa được phê duyệt threshold và chưa
  đủ bằng chứng để gọi là zero-day detection.
- Graph lúc inference hiện là `batch_local_graph`, khác với protocol graph khi
  train; rolling-window graph là một hạng mục mở rộng, không được ngầm coi là
  đã hoàn thành.
- Cloud K8s, Ingress/mTLS, model registry, dashboard và autoscaling đều là
  phần triển khai tương lai, chưa có trong PoC hiện tại.

## Blocker cần biết trước khi chạy lại Phase 3

Artifact lịch sử không đủ contract triển khai:

- checkpoint: 55 feature, chưa chứa `feature_columns`;
- preprocessor hiện tại: 53 feature, được tạo lại từ dữ liệu không đầy đủ.

Code mới từ chối cặp artifact này. Không thêm zero-padding. Để mở blocker cần
tạo lại checkpoint và preprocessor từ cùng một run có đủ sáu scenario, hoặc
export một bundle Phase 2 có schema tương đương.

Kết quả replay 392 flow trước hardening chỉ là bằng chứng chẩn đoán:

- 110 Benign, 282 C&C;
- accuracy `0.3903`;
- C&C recall `0.1525`;
- macro-F1 trên hai lớp xuất hiện `0.3720`.

Không dùng số này làm kết quả cuối vì input feature contract khi đó không hợp lệ.
Không claim zero-day detection từ entropy.

## Việc Hiếu làm tiếp, theo thứ tự

### 1. Hoàn thành benchmark Phase 2

1. Bổ sung bốn scenario còn thiếu: `1-1`, `9-1`, `36-1`, `39-1`.
2. Cài Flower đúng phiên bản trong `pyproject.toml`.
3. Chạy `doctor`, `prepare`, `validate`.
4. Chạy centralized reference, FedAvg và FedProx.
5. Chạy test đúng một lần trên validation-best checkpoint.
6. Xuất comparison từ các run tương thích provenance.

Artifact đúng nằm dưới:

```text
artifacts/phase2/prepared/
artifacts/phase2/runs/
```

Không tạo nhánh kết quả song song `artifacts/phase2_results/`.

### 2. Bàn giao model FL cho Phase 3

Phase 2 hiện lưu named NumPy state `.npz`; Phase 3 hiện đọc checkpoint `.pt`.
Chọn một trong hai cách và ghi test:

- thêm Phase 3 loader cho Phase 2 manifest + `best_model.npz`; hoặc
- export validation-best state sang checkpoint `.pt` chứa đầy đủ
  `feature_columns`, `class_to_idx`, model spec và provenance.

Sau đó cấu hình `MODEL_CHECKPOINT_PATH`, `PREPROCESSOR_PATH` và `MODEL_VERSION`,
không sửa hard-code trong app.

### 3. Chạy Phase 4 scientific evaluation

```bash
python -m unittest discover -s tests/application -v
python -m src.application.evaluation.cli --help
python -m src.application.evaluation.report --help
```

Báo cáo tối thiểu: confusion matrix, fixed-K macro-F1, malicious recall, latency
p50/p95 và model/schema digest. Chỉ chọn entropy threshold sau khi có labeled
validation và báo cáo precision/recall hoặc AUROC.

### 4. Mở rộng triển khai nếu còn thời gian

- Rolling-window graph thay cho batch-local graph.
- Zeek hoặc Cilium Hubble làm nguồn network flow.
- Falco chạy song song cho syscall/process alerts.
- Cloud K8s, Ingress/TLS, monitoring backend và autoscaling là phần nâng cao.

## Definition of done chung

- Phase 2 có metric IoT-23 thật cho centralized/FedAvg/FedProx.
- Model Phase 3 truy được provenance về một Phase 1 hoặc Phase 2 run hợp lệ.
- Inference readiness pass, feeder Job `Completed`, không `CrashLoopBackOff`.
- Replay evaluation lưu machine-readable result.
- Báo cáo phân biệt rõ fact, limitation và future work.

Kiến trúc detection hiện tại và các gate local:
[`PHASE4_ARCHITECTURE.md`](PHASE4_ARCHITECTURE.md).
