# Centralized FedPer Detection Application

This package serves one immutable shared E-GraphSAGE encoder with six exact
FedPer heads. It is independent of Flower and the training runtime.

## Boundaries

- `api/`: label-free production HTTP contract and health endpoints.
- `inference/`: bundle validation, trusted sensor routing, and shared-encoder
  prediction.
- `evaluation/`: correctly routed, cross-head, and validation-selected oracle
  scientific protocols. Evaluation labels never enter the production schema.
- `collection/`: Zeek JSON adapters. A Zeek collector observes packets and
  converts connections into `conn` flow records; it does not classify attacks.
- `graph_window/`: sensor-local event-time buffering and label-free graph
  construction.
- `alerting/`: privacy-reduced Elasticsearch documents. The live threshold
  `0.85` was selected from labeled validation and does not trigger blocking.
- `demo_target/`: harmless HTTP target for a user-owned lab.
- `scenario_runner/`: six fixed-target, server-bounded traffic patterns. It
  never accepts an arbitrary host, URL, command, or port.
- `demo_console/`: internal-only responsive UI for scenario control and a
  privacy-reduced live prediction monitor.

The selected lab traffic pattern and the model prediction are intentionally
shown as separate facts. Synthetic flood, beacon, or port-probe patterns are
not ground-truth IoT-23 labels.

The schema-v2 research bundle records the training protocol
`transductive_edge_mask` but has `serving_ready=false` and no serving
`graph_protocol`. The production API requires a serving-ready bundle and
therefore fails readiness until the rolling-window candidate is selected on
validation and promoted into a new immutable bundle.

## Exact local scientific evaluation

```bash
python -m src.federated.exports.fedper_bundle \
  --run-root <fedper-run> \
  --heads-root <exact-heads> \
  --prepared-root <prepared-dataset> \
  --destination artifacts/federated/exports/<bundle-id>

python -m src.application.evaluation.cli \
  --bundle artifacts/federated/exports/<bundle-id> \
  --dataset <prepared-dataset> \
  --output artifacts/application/evaluation/correctly-routed-test.json \
  correctly-routed --split test
```

The exporter never overwrites an existing destination and validates the source
run, model, best round, encoder, six heads, schemas, and every digest before
atomic promotion.
