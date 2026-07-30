# Execution Plan: Phase 1 Preprocessing Audit

Date: 2026-07-30

## Status

Completed

## Outcome

Produce `docs/PREPROCESSING_AUDIT.md` with evidence-backed classification of
Phase 1 pooled and LOSO preprocessing, historical artifact compatibility, and
the smallest safe fixes required before a three-seed E-GraphSAGE rerun.

## Context

- `docs/PHASE1_REPORT.md`
- `src/preprocess.py`, `src/multi_scenario.py`, `src/run_experiments.py`
- `src/imbalance.py`, `src/train.py`, `src/graph_build.py`
- `phase3_monitoring/inference_service/model_loader.py`

## Scope

In scope:

- Trace real pooled/LOSO data paths and checkpoint contracts.
- Contrast current source with Phase 1 historical artifacts.
- Add only direct regression proof for confirmed preprocessing invariants.

Out of scope:

- Full training, model changes, or broad architecture refactoring.
- Changing a protocol without a separately authorized decision.

## Approach

1. Inspect current code paths, configurations, tests, and historical artifacts.
2. Run minimal targeted probes to verify split-before-fit behavior and contracts.
3. Make only narrowly justified fixes/tests, then write the audit and validation
   result.

## Risks And Recovery

- Historical run source/provenance may be unavailable; classify that evidence as
  UNKNOWN rather than infer it from current code.
- Preserve existing user changes; audit additions are confined to new documents
  and direct regression tests only if a confirmed gap exists.

## Progress

- [x] Establish scope and inspect repository instructions.
- [x] Trace current pooled, LOSO, imbalance, and checkpoint paths.
- [x] Inspect historical artifacts and run targeted probes.
- [x] Add direct helper-level regression proof; do not alter the experiment
  protocol inside a preprocessing audit.
- [x] Write audit and run focused validation.

## Decisions

- 2026-07-30: Treat historical Phase 1 artifacts and current source as separate
  evidence streams because current hardening may postdate the historical run.

## Validation

- Focused proof: relevant existing tests plus any new regression test.
- Integration proof: a toy pooled/LOSO preprocessing probe, without full train.
- Repository checks: `git diff --check`.

## Result

`docs/PREPROCESSING_AUDIT.md` records a FAIL verdict. Pooled/per-scenario
preprocessing and imbalance handling occur before masks; LOSO keeps held-out
flow features out of its preprocessor but leaks non-held-out validation rows
into fit/loss and held-out label vocabulary into the schema. Historical
checkpoint/preprocessor provenance is insufficient and the available 55/53
pair is incompatible. Added helper contract tests pass, but an experiment
protocol correction and runner-level regression tests are required before any
three-seed rerun.
