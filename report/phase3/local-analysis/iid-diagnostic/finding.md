# Seven-Class IID Versus Natural Non-IID Finding

The controlled experiment isolates natural scenario non-IID as the dominant
cause of the federated metric loss. With the same seven-class model contract,
initial state, optimizer, seed, and `30 rounds × 5 local epochs`, stratified-IID
FedAvg reached test macro-F1 `0.988879`; natural scenario FedAvg reached only
`0.507170`. The centralized upper bound was `0.986555`.

| Benchmark | Validation macro-F1 | Test macro-F1 | Test weighted-F1 | Test accuracy |
|---|---:|---:|---:|---:|
| Centralized-7 | 0.985550 | 0.986555 | 0.985328 | 0.985310 |
| Stratified IID FedAvg-7 | 0.987664 | 0.988879 | 0.986174 | 0.986174 |
| Natural non-IID FedAvg-7 | 0.521828 | 0.507170 | 0.640237 | 0.727471 |

IID is `+0.481709` test macro-F1 above natural non-IID and within `0.002324`
of centralized. In natural non-IID, `C&C-HeartBeat`, `DDoS`, and `Okiru` all
have test F1 `0`; under IID their F1 values are `0.998752`, `0.995480`, and
`0.999750`. This pattern is inconsistent with a fundamental runtime or FedAvg
implementation failure and directly matches the private-class support
structure found in Phase 3A.

The redistribution is diagnostic only. It does not become a production policy
because production Edge data cannot be pooled and reassigned. The next
experiment family should therefore target class-support-aware aggregation or
client update weighting and must be selected on validation across multiple
seeds against the unchanged natural non-IID benchmark.

The immutable provenance, dataset and model digests, per-class metrics, and
evidence hashes are recorded in `summary.json`. Validation selection did not
evaluate test; each fixed round-30 federated state was evaluated on test once
after the comparison was frozen.
