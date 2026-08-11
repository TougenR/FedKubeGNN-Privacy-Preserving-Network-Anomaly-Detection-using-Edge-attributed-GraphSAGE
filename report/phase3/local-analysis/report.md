# Phase 3A Prepared-Data Findings

- Dataset: `iot23-0d9bbeb9f9ed0a3f`
- Manifest digest: `68fc6fc0cb8974aba1d431113b39dbf82f98457159c04d6a14b22feaa4b0cb89`
- Integrity: contract, initial state, every client checksum, and split-mask coverage verified before analysis.

## Class balance and non-IID structure

| Class | Global support | Clients with class | Severe trigger |
|---|---:|---:|---|
| Benign | 36459 | 6 | no |
| Attack | 6639 | 2 | no |
| C&C | 8252 | 4 | no |
| C&C-HeartBeat | 10000 | 1 | yes |
| DDoS | 10000 | 1 | yes |
| Okiru | 10000 | 1 | yes |
| Okiru-Attack | 3 | 1 | yes |
| PartOfAHorizontalPortScan | 40122 | 5 | no |

The union imbalance ratio is `13374.0:1` because Okiru-Attack has only three observations. C&C-HeartBeat, DDoS, Okiru, and Okiru-Attack are structurally private to one client. This confirms severe imbalance before any model-level treatment.

## Feature and topology checks

- Non-finite transformed feature values: `0`.
- Globally constant/near-constant features: `history_n_s, history_n_g, history_n_G, history_n_W`.
- `duration_missing` positive rate: `0.6358`.
- `orig_bytes_missing` positive rate: `0.6358`.
- `resp_bytes_missing` positive rate: `0.6358`.
- Raw missing values are not retained by the prepared contract; the rates above are measured from explicit transformed missing indicators.

| Client | Nodes | Edges | Unique directed pairs | Parallel edges | Exact duplicate rows |
|---|---:|---:|---:|---:|---:|
| 34-1 | 49 | 18751 | 49 | 18702 | 14005 |
| 1-1 | 19504 | 20008 | 19510 | 498 | 150 |
| 3-1 | 11535 | 20506 | 11562 | 8944 | 1082 |
| 9-1 | 16626 | 20000 | 16647 | 3353 | 657 |
| 36-1 | 10421 | 22666 | 10421 | 12245 | 7615 |
| 39-1 | 14085 | 19544 | 14089 | 5455 | 1988 |

Client 34-1 is an extreme multigraph (49 nodes, 18,751 edges, and only 49 unique directed node pairs). This may be valid scenario structure, but it must be compared with Phase 1 graph construction before treating the federated metric gap as an optimizer-only problem.

## Required next experiments

1. Complete Phase 1/Phase 2 split, learned preprocessor, and graph-membership equivalence checks.
2. Run the exact prepared-data centralized reference to separate data and federation effects.
3. Evaluate global class weights first, then local epoch/learning rate, one alternative loss, and finally class-support-aware aggregation.
4. Keep Okiru-Attack validation metrics marked not estimable; do not duplicate its test example or tune against it.
