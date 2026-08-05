# FedPer Personalized-Head Finding

FedPer resolves the dominant natural non-IID failure: the GraphSAGE encoder is shared with sample-weighted FedAvg while every edge retains its complete `head.*` classifier locally.

| Seed | FedAvg val | Class-aware val | FedPer val | FedPer test |
|---:|---:|---:|---:|---:|
| 42 | 0.521828 | 0.776278 | 0.994171 | 0.994073 |
| 1337 | 0.521918 | 0.649759 | 0.994016 | 0.994143 |
| 2026 | 0.521611 | 0.782890 | 0.917721 | 0.923161 |

Mean validation macro-F1 is `0.968636 ± 0.036002`. The three validation-selected bundles achieve test macro-F1 `0.970459 ± 0.033444`; each test split was evaluated once after selection.

`C&C-HeartBeat` test F1 is `1.0`, `1.0`, and `0.685076` for seeds 42, 1337, and 2026. This directly supports the diagnosis that the earlier zero-F1 result came from a globally averaged classifier head, not unusable preprocessing or broken local training.

This is a personalized deployment metric, not a single global-model claim. A known edge must retain its own head; a new edge has no personalized checkpoint and needs calibration or a global fallback. The seed-2026 gap also means initialization sensitivity remains.

Full checkpoints and round logs remain under the Git-ignored `artifacts/phase2/runs/phase3d/`; their hashes are locked in `summary.json`.
