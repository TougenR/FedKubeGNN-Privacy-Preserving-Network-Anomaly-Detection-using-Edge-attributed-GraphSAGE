# Phase 3

## Local analysis

`local-analysis/` contains data-balance and topology audits, centralized and
IID diagnostics, aggregation ablations, multi-seed FedPer reports, figures,
and selected checkpoints.

## GKE experiments

Each strategy under `experiments/` contains:

```text
<strategy>/
├── best-model.pt
├── events/
└── runs/<immutable-run-id>/
    ├── config.snapshot.json
    ├── run.json
    ├── checkpoints/
    │   ├── best_model.npz
    │   └── round-0001.npz ... round-0030.npz
    └── metrics/
        ├── summary.json
        └── validation-round-0001.json ... validation-round-0030.json
```

Final GKE results:

| Strategy | Classes | Best round | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|---:|
| FedAvg | 8 | 22 | 0.738921 | 0.456556 |
| FedProx | 8 | 21 | 0.739045 | 0.456691 |
| FedPer | 7 | 30 | 0.991729 | 0.994073 |

The FedPer score is a known-edge personalized result. Its shared encoder and
all exact GKE private heads are present in this archive. Combine the shared
checkpoint with the matching client's `head-0030.npz` for the final/best model.

## Exact GKE FedPer private heads

`experiments/fedper/private-heads/exact-gke-run-14339380272482304688/`
contains one directory per client. Every directory has `head-0001.npz` through
`head-0030.npz` and `metadata.json`. These files are client-specific model
artifacts and should be handled more carefully than the shared encoder.

## Visualizations

- `visualizations/f668ed91.../`: FedAvg versus FedProx, eight-class run.
- `visualizations/83e57c68.../`: FedPer seven-class run.

Each directory contains PNG and PDF figures, CSV tables, and a provenance
manifest.
