# Execution Plan: Phase 1 Clean Protocol Remediation

Date: 2026-07-30

## Status

Completed

## Outcome

Provide a separate Phase 1 clean runner for pooled, per-scenario, and LOSO
E-GraphSAGE experiments with split-before-fit preprocessing, fixed taxonomy,
train-only imbalance handling, isolated final test evaluation, and validated
run-scoped bundles under `artifacts/phase1_clean/`.

## Context

- `docs/PREPROCESSING_AUDIT.md`
- `src/run_experiments.py`, `src/multi_scenario.py`
- `src/preprocess.py`, `src/train.py`, `src/imbalance.py`, `src/graph_build.py`
- `phase3_monitoring/inference_service/model_loader.py`

## Scope

In scope:

- Add a clean runner without changing the historical result runner.
- Fixed eight-class taxonomy and preselected `class_weight` path.
- Runner-level leakage and bundle regression tests.
- Toy smoke only; no full IoT-23 training.

Out of scope:

- Model architecture, OOD/open-set behavior, taxonomy changes, full retraining,
  or overwriting `artifacts/phase1_results/`.

## Approach

1. Add deterministic stable row IDs and raw-row split plans.
2. Fit/weight/sample only from training membership and attach preserved masks to
   full transductive graphs.
3. Add pooled/per-scenario and LOSO clean training loops with validation-only
   model selection and one final test/held-out evaluation.
4. Write/validate immutable bundle contracts and add a CLI/smoke command.
5. Add runner-level regression tests and update audit/protocol documentation.

## Risks And Recovery

- The historical runner remains untouched; clean output uses a distinct root.
- Any ambiguous policy is already fixed by the task request. If an unforeseen
  policy choice appears, stop before changing behavior.
- Recovery is removal of the additive clean runner/tests/docs/config section;
  historical artifacts remain unchanged.

## Progress

- [x] Read audit, source paths, current tests, and locked design decisions.
- [x] Implement clean split/preprocessing/imbalance boundaries.
- [x] Implement training, final evaluation, and bundle contract.
- [x] Add runner-level tests and toy smoke.
- [x] Update documentation and complete validation.

## Decisions

- 2026-07-30: Keep `src/run_experiments.py` as the historical runner and expose
  clean behavior through a separate module/CLI so historical reproduction
  cannot be mistaken for the remediated protocol.

## Validation

- Focused proof: `python -m unittest discover -s tests/phase1 -v` — 10
  passed, including all eight runner-level invariants.
- Split proof: `python scripts/test_safe_split.py` — 4 passed.
- Contract proof: `python -m unittest discover -s tests/phase3 -v` — 8
  passed.
- Toy proof: `python -m src.phase1_clean --config config.yaml --toy-smoke` —
  pooled and one-fold LOSO completed one epoch, emitted finite metrics, wrote
  run-scoped temporary bundles, and loaded both through the Phase 3 contract.
- Repository checks: `python -m compileall -q src tests/phase1` and
  `git diff --check` passed.

## Result

The additive `src.phase1_clean` runner now enforces split-before-fit
preprocessing, train-only class weights/optional undersampling, validation-only
model selection, fixed taxonomy, and one final evaluation. It writes validated
run-scoped bundles under a clean output root. Historical Phase 1 source and
artifacts were not changed or overwritten.

The protocol is ready for retraining, but the full run remains intentionally
unattempted: only scenarios `34-1` and `3-1` are available locally.
