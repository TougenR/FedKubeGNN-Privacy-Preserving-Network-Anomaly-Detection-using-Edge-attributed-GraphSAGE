# Artifact manifest for the Federated Learning report

## Snapshot scope

The source corpus contained 674 files before report-specific figures, tables,
and rendered documents were added. It combines:

- all 223 objects (14,586,506 bytes) from the GCS Model Artifacts bucket;
- Phase 1 clean and historical reports plus the clean reference model;
- Phase 2 local federated runs and checkpoints;
- Phase 3 data, IID/non-IID, aggregation, centralized, and FedPer analyses;
- 180 exact GKE FedPer client-head checkpoints and six Edge metadata files.

No raw dataset, credential, token, Terraform state, or Kubernetes secret is
included.

The finalized archive contains 701 files: the original 674-file corpus plus 27
report-generated documents, curated figures, derived tables, and this manifest.

## Immutable identifiers

| Item | Identifier |
|---|---|
| GCP project | `fedlearning-20260729-hn` |
| GKE FedPer Flower run | `14339380272482304688` |
| Git release commit | `83e57c68fd41996c7c91f9edd5af3a1e1af391c4` |
| Seven-class natural dataset | `iot23-seven-natural-3be7796b1ee27bc3` |
| Dataset digest | `c5ab9c02896c08c9f60e8efb9672a2090cbe595e4c344308f5e4dc2b0e51319a` |
| Model digest | `42642e4cc839c09dfe8519511aa7cf7cdf5ca7350a8dd376e118ee31a6a74bbf` |
| Container image digest | `sha256:4ed1afba8302d595935fd905ed700d6d01040b1fb84e3182e6c47fda86becc7e` |

## Principal evidence

| Claim | Evidence path |
|---|---|
| Phase 1 clean baseline | `phase1/clean-benchmark/report.md` |
| Data balance and topology | `phase3/local-analysis/report.md` |
| Centralized reference | `phase3/local-analysis/centralized-reference/` |
| IID versus natural non-IID | `phase3/local-analysis/iid-diagnostic/` |
| Class-aware aggregation | `phase3/local-analysis/class-aware-aggregation/` |
| FedPer three-seed result | `phase3/local-analysis/fedper/` |
| Exact GKE FedPer summary | `phase3/experiments/fedper/runs/fedper-20260805T162143974378Z-270b7ffe84/metrics/summary.json` |
| Exact GKE shared checkpoints | `phase3/experiments/fedper/runs/fedper-20260805T162143974378Z-270b7ffe84/checkpoints/` |
| Exact GKE private heads | `phase3/experiments/fedper/private-heads/exact-gke-run-14339380272482304688/` |
| GKE figures and tables | `phase3/visualizations/83e57c68fd41996c7c91f9edd5af3a1e1af391c4/` |

The complete per-file inventory is `index/inventory.csv`. SHA-256 values in
`index/SHA256SUMS` were regenerated after the final PDF was rendered.
