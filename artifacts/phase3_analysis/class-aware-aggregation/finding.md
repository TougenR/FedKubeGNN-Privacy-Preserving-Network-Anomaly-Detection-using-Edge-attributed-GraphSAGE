# Class-Aware Aggregation Finding

Train-support-aware aggregation materially improves the natural seven-class
federated benchmark, but one global model still cannot retain every private
class reliably.

The selected policy gives each globally present class equal total influence on
shared parameters, distributed among clients in proportion to train-only class
support. Final classifier rows exclude clients with zero support for that
class. No validation or test labels determine aggregation weights.

| Seed | FedAvg validation macro-F1 | Selected validation macro-F1 | Delta | Selected test macro-F1 |
|---:|---:|---:|---:|---:|
| 42 | 0.521828 | 0.776278 | +0.254450 | 0.778762 |
| 1337 | 0.521918 | 0.649759 | +0.127841 | 0.652025 |
| 2026 | 0.521611 | 0.782890 | +0.261279 | 0.783504 |

Mean validation macro-F1 increased from `0.521786` to `0.736309`. The frozen
selected checkpoints achieved test macro-F1 `0.738097 ± 0.060893`; each test
checkpoint was evaluated once after the algorithm had passed all three
validation seeds.

Classifier-row filtering alone did not help. Whole-model class-balanced client
weights helped seeds 42 and 2026 but regressed seed 1337. Combining both rules
was necessary to improve every validation seed.

The remaining failure is `C&C-HeartBeat`: global F1 remains zero in every seed.
This is not caused by unusable data or preprocessing. Client `36-1` trained in
isolation reaches validation weighted-F1 `1.0` and F1 `1.0` for both
`C&C-HeartBeat` and `Okiru` by round 8. The global failure therefore supports a
personalized-head/FedPer follow-up instead of more blind global reweighting.

Full selected run artifacts are stored locally under the Git-ignored
`artifacts/phase2/runs/phase3c/`. Digests and compact evidence are recorded in
`summary.json`.
