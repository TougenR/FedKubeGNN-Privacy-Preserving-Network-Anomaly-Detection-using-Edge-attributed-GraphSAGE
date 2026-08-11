# FedKubeGNN report and model archive

Snapshot created on 2026-08-06. Start here to browse experiment reports,
figures, metrics, logs, and model checkpoints from Phase 1 through Phase 3.

## Quick reading order

1. [Federated Learning technical report](FEDERATED_LEARNING_REPORT.md)
2. `FEDERATED_LEARNING_REPORT.pdf`
3. [Phase 1 clean benchmark](phase1/clean-benchmark/report.md)
4. [Phase 3 data analysis](phase3/local-analysis/report.md)
5. [IID versus non-IID finding](phase3/local-analysis/iid-diagnostic/finding.md)
6. [Class-aware aggregation finding](phase3/local-analysis/class-aware-aggregation/finding.md)
7. [FedPer finding](phase3/local-analysis/fedper/finding.md)
8. [GKE FedPer final metrics](phase3/experiments/fedper/runs/fedper-20260805T162143974378Z-270b7ffe84/metrics/summary.json)

## Directory layout

```text
report/
├── README.md
├── FEDERATED_LEARNING_REPORT.md
├── FEDERATED_LEARNING_REPORT.pdf
├── appendices/
│   └── ARTIFACT_MANIFEST.md
├── figures/                   # Curated report figures
├── index/
│   ├── SOURCES.md
│   ├── inventory.csv
│   └── SHA256SUMS
├── phase1/
│   ├── README.md
│   ├── historical-benchmark/
│   ├── clean-benchmark/
│   └── clean-reference-model/
├── phase2/
│   ├── README.md
│   └── local-federated-runs/
└── phase3/
    ├── README.md
    ├── local-analysis/
    ├── experiments/
    │   ├── fedavg/
    │   ├── fedprox/
    │   └── fedper/
    └── visualizations/
        ├── f668ed91.../       # FedAvg versus FedProx, eight classes
        └── 83e57c68.../       # FedPer, seven classes
└── tables/                    # Curated and derived report tables
```

## Important model notes

- `best-model.pt` is the exported best model artifact for each GKE strategy.
- `checkpoints/best_model.npz` is the portable NumPy checkpoint selected by
  validation; `round-NNNN.npz` retains every training round.
- The GKE FedPer `best_model.npz` contains the shared GraphSAGE encoder only.
  Its 180 exact private client-head checkpoints and six metadata files are
  archived under
  `phase3/experiments/fedper/private-heads/exact-gke-run-14339380272482304688/`.
- Local FedPer private heads for seeds 42, 1337, and 2026 are available under
  `phase2/local-federated-runs/phase3d/` and are not the same tensor artifacts
  as the completed GKE run.

## Completeness

The GCS Model Artifacts bucket contained 223 objects totaling 14,586,506 bytes;
all 223 are present under `phase3/experiments/` and
`phase3/visualizations/`. The six Edge PVCs contributed another 180 exact
FedPer head checkpoints plus six metadata files. No dataset, credential,
Terraform state, or secret is included.

The finalized archive contains 701 files: the original 674-file evidence
corpus plus 27 report-generated documents, curated figures, and derived tables.

Use `index/SHA256SUMS` to verify local integrity and `index/inventory.csv` to
filter files by phase, category, extension, or size.
