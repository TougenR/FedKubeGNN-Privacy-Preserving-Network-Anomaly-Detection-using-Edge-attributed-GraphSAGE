# Execution Plan: Phase 1 Clean Verification and Analysis Tools

Date: 2026-07-30

## Status

Completed

## Outcome

Provide a fail-fast six-scenario dataset verifier and a truthful multi-seed
clean-result analyzer, plus final-split per-sample prediction exports required
for class, binary, confusion-matrix, and entropy analysis.

## Context

- `docs/PHASE1_CLEAN_PROTOCOL.md`
- `src/data_io.py`, `src/preprocess.py`, `src/phase1_clean.py`
- `tests/phase1/`

## Scope

In scope:

- Add `scripts/verify_phase1_dataset.py`.
- Add `scripts/analyze_phase1_clean.py`.
- Add one CSV prediction artifact to new clean bundles without changing
  training, model selection, or evaluation membership.
- Add focused metric, aggregation, and missing-artifact tests.

Out of scope:

- Full IoT-23 training, model/protocol changes, OOD claims, or modification of
  historical artifacts.

## Approach

1. Stream-verify configured Zeek files and emit JSON/CSV manifests.
2. Capture final evaluation logits/probabilities in the existing single
   evaluation pass and write `predictions.csv`.
3. Analyze one or more seed roots with fixed-taxonomy, seen-class, binary, and
   entropy metrics; mark unavailable analyses explicitly.
4. Add toy fixtures and tests, then run script-level smoke checks.

## Risks And Recovery

- Large scenarios are read in chunks; the verifier never loads an entire raw
  file solely for inspection.
- Prediction export is additive and derives from the already-computed final
  forward pass.
- Historical `artifacts/phase1_results/` is outside both tools' write scope.
- Recovery is removal of the two scripts, additive prediction export, tests,
  and this plan.

## Progress

- [x] Inspect current parser, clean bundle, and test conventions.
- [x] Implement dataset verifier.
- [x] Add per-sample prediction export.
- [x] Implement analyzer.
- [x] Add tests, docs, and complete validation.

## Decisions

- 2026-07-30: Use `predictions.csv` with fixed label-specific probability and
  logit columns. CSV avoids a new parquet dependency and remains inspectable.
- 2026-07-30: Prediction rows cover the final evaluation split only; train,
  validation, and test support continue to come from `metadata.json`.
- 2026-07-30: A parser failure is serious when parsing raises, required columns
  are absent, a nonempty file yields no parsed rows, or skipped/bad rows exceed
  the configurable allowance (default zero).

## Validation

- `python -m unittest discover -s tests/phase1 -v` — 18 passed.
- `python -m unittest discover -s tests/phase3 -v` — 8 passed.
- `python scripts/test_safe_split.py` — 4 passed.
- Dataset verifier CLI against current config wrote both manifests and exited
  `2` as required because scenarios `1-1`, `9-1`, `36-1`, and `39-1` are
  missing; the two present files parsed without bad rows.
- Temporary one-epoch pooled + LOSO bundles were analyzed successfully:
  seven requested tables/report outputs plus two confusion figures were
  created, including entropy analysis.
- `python -m compileall -q src scripts tests/phase1`, both CLI `--help`
  commands, and `git diff --check` passed.

## Result

The dataset verifier now streams and fingerprints all required scenarios,
writes JSON/CSV manifests, and fails readiness for missing or seriously
malformed input. The analyzer discovers clean bundles across seed roots,
aggregates fixed/seen/binary metrics and class support, performs entropy
analysis only from available probabilities/logits, and records exact
`NOT_AVAILABLE` requirements otherwise.

New clean bundles include `predictions.csv` captured from the existing single
final evaluation pass. No training behavior or historical artifact was changed,
and no full dataset training was run.
