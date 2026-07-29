# Execution Plan: Phase 3 PoC And Handoff Hardening

Date: 2026-07-29

## Status

Active — implementation complete; compatible model bundle required for the
end-to-end replay gate.

## Outcome

Make the Minikube inference PoC fail closed on incompatible model artifacts,
run as a healthy one-shot replay workload, expose reproducible validation
evidence, and leave a concise handoff that separates Bảo's completed work from
Hiếu's remaining Phase 2 and deployment work.

## Context

- `docs/PHASE1_REPORT.md`: historical centralized results and limitations.
- `docs/PHASE2_ARCHITECTURE.md`: Phase 2 artifact and evaluation contract.
- `phase3_monitoring/`: current uncommitted inference, feeder, Docker and
  Kubernetes PoC.
- Live Minikube proof on 2026-07-29: inference served 392/392 requests, while
  the feeder Deployment restarted after successful one-shot completion.
- Runtime inspection found a 53-feature preprocessor paired with a 55-feature
  Phase 1 checkpoint. The current zero-padding workaround is not valid model
  provenance.

## Scope

In scope:

- Configurable, fail-closed Phase 3 model/preprocessor loading.
- Health endpoints and Kubernetes probes.
- A Kubernetes Job for one-shot replay.
- Focused contract/API tests and a labeled replay evaluator.
- Concise Phase 1 corrections, Phase 3 documentation and handoff to Hiếu.

Out of scope:

- Downloading the four missing IoT-23 scenarios.
- Fabricating or retraining the missing exact 55-feature preprocessor.
- Running the full Phase 2 IoT-23/Flower benchmark.
- Claiming zero-day detection, deploying a production cloud cluster, or
  integrating Zeek, Cilium Hubble or Falco.

## Approach

1. Replace permissive artifact loading and feature padding with an explicit
   validated runtime bundle assembled from configured paths.
2. Keep uncertainty as a reported score only; do not invent an alert threshold.
3. Make the one-shot feeder a Job and make inference readiness depend on a
   successfully validated model.
4. Add tests around artifact mismatch, API health and prediction shape, plus a
   reusable labeled replay evaluator.
5. Correct overstated documentation and publish one short owner/action handoff.

## Risks And Recovery

- The currently generated preprocessor is incompatible, so hardened startup is
  expected to fail until the exact training preprocessor is supplied. Recovery
  is to set the documented model/preprocessor paths to a compatible bundle, not
  to reintroduce padding.
- Existing Phase 3 files are untracked user work. Changes stay limited to that
  directory and documentation; unrelated dirty files are preserved.
- The old feeder Deployment may remain in an already-applied cluster. The
  README will include the explicit one-time deletion before applying the Job.

## Progress

- [x] Inspect repository authority, live runtime evidence and current artifacts.
- [x] Add fail-closed loading, health endpoints and configurable paths.
- [x] Replace feeder Deployment with a Job.
- [x] Add focused tests and labeled replay evaluation.
- [x] Correct reports and add the handoff document.
- [x] Run focused and repository validation.
- [ ] Supply a compatible full-schema bundle and rerun labeled replay.

## Decisions

- 2026-07-29: Do not choose a default entropy alert threshold without labeled
  validation authority.
- 2026-07-29: Do not repair a 53/55 feature mismatch by position-blind padding.
- 2026-07-29: Configure Phase 3 checkpoint/preprocessor paths rather than
  hard-coding them. A Phase 2 `.npz` loader/export adapter remains an explicit
  handoff item because the current runtime contract is Phase 1 `.pt`.

## Validation

- Focused proof: Phase 3 unit tests for artifact validation and API behavior.
- Integration proof: labeled replay evaluator against a running compatible
  inference service.
- Repository checks: Phase 2 tests, Python compilation and `git diff --check`.

## Result

Implemented fail-closed artifact validation, exact ordered feature schema in
new Phase 1 checkpoints, immutable inference preprocessing, health/provenance
endpoints, one-shot feeder Job, labeled evaluation tooling and focused tests.
Corrected the Phase 1 interpretation and added one concise ownership handoff.

Observed validation:

- Phase 3 unit tests: 8 passed, including compatible-bundle end-to-end loading.
- Phase 2 tests: 46 run, 41 passed and 5 optional Flower tests skipped.
- Phase 1 E-GraphSAGE layer/model and safe-split smoke scripts: passed.
- Kubernetes manifests: client-side dry-run passed for Namespace, Service,
  Deployment and Job.
- Inference and feeder Docker images built successfully under validation-only
  tags without replacing the running Minikube images.
- Uvicorn with current historical artifacts: liveness HTTP 200; readiness HTTP
  503 with the expected missing-`feature_columns` contract error.
- Python compilation and repository diff whitespace checks: passed.

The plan stays active only for the external artifact/evaluation gate. Four raw
scenarios and a compatible 55-feature checkpoint/preprocessor bundle are still
missing, so no post-hardening prediction-quality claim is made.
