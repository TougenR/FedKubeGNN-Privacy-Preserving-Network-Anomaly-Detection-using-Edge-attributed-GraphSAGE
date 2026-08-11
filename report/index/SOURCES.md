# Source map

| Local path | Source |
|---|---|
| `phase1/historical-benchmark/` | `artifacts/phase1_results/` |
| `phase1/clean-benchmark/` | `artifacts/phase1_clean/report_analysis/` |
| `phase1/clean-reference-model/` | `artifacts/phase1_clean/seed-42-full/` |
| `phase2/local-federated-runs/` | `artifacts/phase2/runs/` |
| `phase3/local-analysis/` | `artifacts/phase3_analysis/` |
| `phase3/experiments/<strategy>/` | `gs://fedlearning-20260729-hn-fedkube-model-artifacts/runs/<strategy>/` |
| `phase3/visualizations/` | `gs://fedlearning-20260729-hn-fedkube-model-artifacts/runs/visualizations/` |
| `phase3/experiments/fedper/private-heads/exact-gke-run-14339380272482304688/` | Six Edge GKE personalized-head PVCs |

Local repository artifacts were copied as hard links to avoid duplicating disk
blocks. GCS objects are independent downloaded files.
