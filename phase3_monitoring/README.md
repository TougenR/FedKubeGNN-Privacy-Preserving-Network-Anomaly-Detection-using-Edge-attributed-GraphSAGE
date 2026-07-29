# Phase 3: Minikube Inference PoC

Phase 3 chứng minh luồng kỹ thuật:

```text
IoT-23 replay -> FastAPI -> preprocessing -> batch-local PyG graph
              -> E-GraphSAGE -> label + confidence + entropy
```

Đây là PoC triển khai local, không phải hệ thống production và chưa chứng minh
zero-day detection. Nguồn input hiện tại là replay IoT-23; thu thập flow thật
bằng Zeek/Cilium Hubble và Falco chạy song song là công việc sau bàn giao.

## Trạng thái đã kiểm chứng

- FastAPI, Docker, ClusterIP và Minikube đã truyền thông end-to-end.
- Lần chạy trước hardening nhận đủ 392/392 prediction, khoảng 30--73 ms/batch
  50 flow.
- Feeder một lần đã được đổi từ `Deployment` sang Kubernetes `Job`.
- Inference có liveness/readiness và trả model/schema provenance.
- Entropy chỉ là uncertainty score; chưa có alert threshold được phê duyệt.

## Blocker artifact hiện tại

Checkpoint lịch sử cần 55 feature nhưng `preprocessor.pkl` hiện tại có 53
feature; checkpoint cũng chưa lưu `feature_columns`. Image mới vì vậy trả
`/health/ready` HTTP 503 và từ chối prediction. Không được khôi phục workaround
zero-padding.

Checkpoint tạo bởi code hiện tại sẽ lưu `feature_columns`, còn orchestrator
Phase 1 sẽ lưu đúng shared preprocessor vào cùng thư mục checkpoint. Cần đủ sáu
scenario và chạy lại pipeline phù hợp để tạo cặp artifact có cùng provenance.

Kiểm tra artifact trước khi build/deploy:

```bash
python -m phase3_monitoring.inference_service.model_loader
```

Các đường dẫn có thể cấu hình:

```bash
export MODEL_CHECKPOINT_PATH=/absolute/path/model.pt
export PREPROCESSOR_PATH=/absolute/path/preprocessor.pkl
export MODEL_VERSION=phase1-pooled-egraphsage
export INFERENCE_DEVICE=cpu
```

## Chạy local

```bash
python phase3_monitoring/flow_feeder/prepare_sample.py
uvicorn phase3_monitoring.inference_service.app:app --host 0.0.0.0 --port 8000
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready
python phase3_monitoring/flow_feeder/replay_flows.py
```

Đánh giá toàn bộ prediction có nhãn thay vì chỉ xem ba dòng mẫu:

```bash
pip install -r phase3_monitoring/flow_feeder/requirements-eval.txt
python phase3_monitoring/flow_feeder/evaluate_replay.py \
  --url http://localhost:8000/predict \
  --output artifacts/phase3/replay-evaluation.json
```

Report gồm confusion matrix, per-class precision/recall/F1, fixed-K macro-F1,
entropy/confidence summary và latency p50/p95. Report không tự chọn entropy
threshold và không đưa ra claim zero-day.

## Chạy Minikube

```bash
minikube start
eval "$(minikube docker-env)"

# Generated sample_data is ignored by Git; create it before building feeder.
python phase3_monitoring/flow_feeder/prepare_sample.py

docker build -t fedkube-inference:latest \
  -f phase3_monitoring/inference_service/Dockerfile .
docker build -t fedkube-feeder:latest \
  -f phase3_monitoring/flow_feeder/Dockerfile .

# Dọn workload feeder kiểu Deployment cũ, nếu đã từng apply.
kubectl delete deployment fedkube-feeder -n fedkube-ids --ignore-not-found
kubectl apply -f phase3_monitoring/k8s/00-namespace.yaml
kubectl apply -f phase3_monitoring/k8s/inference-service.yaml
kubectl apply -f phase3_monitoring/k8s/inference-deployment.yaml

kubectl wait --for=condition=available deployment/fedkube-inference \
  -n fedkube-ids --timeout=180s

# Job là immutable; xóa job cũ trước mỗi lần replay.
kubectl delete job fedkube-feeder -n fedkube-ids --ignore-not-found
kubectl apply -f phase3_monitoring/k8s/flow-feeder-job.yaml
kubectl logs -f job/fedkube-feeder -n fedkube-ids
```

Nếu inference không `Available`, xem lý do contract:

```bash
kubectl logs deployment/fedkube-inference -n fedkube-ids
kubectl get pods -n fedkube-ids
```

## Contract và giới hạn

- Inference không fit preprocessor khi startup.
- Feature names, order, dimension và label mapping phải khớp checkpoint.
- Graph protocol hiện là `batch_local_graph`, khác protocol graph lớn khi train;
  rolling-window graph và stability test vẫn là follow-up.
- Phase 3 hiện đọc checkpoint `.pt` theo contract Phase 1. Hiếu cần hoàn thành
  adapter/export từ Phase 2 `best_model.npz` trước khi gọi đây là luồng FL
  end-to-end.
- Network flow dùng Zeek/Cilium Hubble. Falco là nguồn syscall/process alert
  song song, không thay trực tiếp flow feeder.

Xem [bàn giao cho Hiếu](../docs/HANDOFF_TO_HIEU.md) để biết ownership và thứ tự
công việc còn lại.
