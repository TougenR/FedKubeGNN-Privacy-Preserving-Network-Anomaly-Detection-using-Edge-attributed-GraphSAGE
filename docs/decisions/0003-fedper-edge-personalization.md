# 0003 FedPer Edge-Owned Classifier Heads

Date: 2026-08-05

## Status

Accepted

## Context

Natural scenario partitioning is strongly non-IID. Sample FedAvg reached mean
validation macro-F1 `0.521786`; the selected class-aware global model reached
`0.736309` but retained zero F1 for `C&C-HeartBeat` in every seed. A controlled
FedPer experiment sharing only the GraphSAGE encoder reached validation
macro-F1 `0.968636 ± 0.036002` and validation-selected test macro-F1 `0.970459
± 0.033444` over seeds 42, 1337, and 2026.

## Decision

- Use FedPer as the leading natural-non-IID deployment candidate. Share and
  sample-weight aggregate `layers.*`; keep the complete `head.*` classifier at
  its owning Flower client.
- Never include `head.*` in Flower train, validation, or final-test payloads.
- Persist each head on a dedicated Edge PVC under a run, client, and full-model
  digest identity. Use versioned state files and an atomically promoted metadata
  pointer so recovery never selects a partially written checkpoint.
- Cold-start a new Edge from the immutable initial `head.*`. Block evaluation
  and inference until that Edge completes at least one local-training round.
- Keep validation checkpoint selection and one final test evaluation per seed.
  Aggregate client confusion matrices for the deployment-wide metric.
- Store only the shared encoder in the Central model bundle. Private heads stay
  Edge-local and are not copied to Central GCS in the MVP.

## Alternatives Considered

1. Continue tuning one global classifier head, rejected because three-seed
   evidence shows private-class knowledge is lost during global aggregation.
2. Aggregate heads only among clients with class support, retained as the
   global fallback benchmark but rejected as the leading candidate because
   `C&C-HeartBeat` remains at zero F1.
3. Maintain a second globally aggregated bootstrap head, deferred because it
   changes two algorithm families at once and weakens the controlled FedPer
   result.
4. Permit inference immediately from the random initial head, rejected because
   it would report an uncalibrated model as ready.

## Consequences

Positive:

- Private classes remain learnable without moving labels or head parameters
  away from an Edge.
- Flower payloads and Central checkpoints are smaller and no longer expose
  client-specific classifier weights.
- Versioned local persistence supports ClientApp process and pod recovery.

Tradeoffs:

- There is no single portable global model; every known Edge needs its matching
  head.
- A new Edge requires a local calibration round before inference.
- Losing an Edge head PVC requires retraining that head from the initial state.
- Seed 2026 remains weaker than seeds 42 and 1337, so initialization sensitivity
  still requires monitoring.

## Follow-Up

- Run one bounded GKE FedPer release through Argo CD and capture Flower, PVC,
  Kibana, model-artifact, and Argo health evidence.
- Define encrypted Edge-head backup and cross-cluster disaster recovery only
  after the MVP, without changing Central ownership by default.
