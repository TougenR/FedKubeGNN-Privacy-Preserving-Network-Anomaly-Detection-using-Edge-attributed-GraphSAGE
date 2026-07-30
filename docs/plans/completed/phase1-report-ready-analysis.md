# Execution Plan: Phase 1 Report-Ready Analysis

Date: 2026-07-30

## Status

Completed

## Outcome

Extend the clean Phase 1 analyzer so one or more existing seed bundle roots
produce report-ready CSV tables, figures, a limitations-aware report, and a
six-figure main-report recommendation without retraining or changing stored
metrics.

## Context

- `scripts/analyze_phase1_clean.py`
- `tests/phase1/test_clean_analysis.py`
- `src/phase1_clean.py`
- `docs/PHASE1_CLEAN_PROTOCOL.md`

The real seed-42 bundles currently exist only on the user's Vast.ai host and
have not yet been copied into this local workspace. Local validation therefore
uses contract-faithful fixtures; final validation against the real three-seed
artifacts is a handoff step after download.

## Scope

In scope:

- Multi-seed inventory and aggregation with explicit single-seed limitations.
- Required pooled, LOSO, support, per-class, entropy, and optional historical
  comparison tables.
- Report-ready Matplotlib figures with fixed taxonomy/order and graceful
  `NOT_AVAILABLE` behavior.
- Prediction export contract audit and future-run-only derivation guidance.
- Focused tests and fixture-based end-to-end validation.

Out of scope:

- Training, experiment protocol, model, stored metrics, or existing bundle
  mutation.
- Fabricating learning curves, probabilities, or historical comparisons.

## Approach

1. Normalize bundle metadata, metrics, history, predictions, support, and
   per-class metrics into reusable frames.
2. Add deterministic multi-seed tables and uncertainty metadata.
3. Add independent report figures and availability notes.
4. Generate `report.md` and `FIGURE_SELECTION.md`.
5. Extend tests and run fixture-based end-to-end analysis.

## Risks And Recovery

- Missing real artifacts: emit exact `NOT_AVAILABLE` reasons; never infer.
- Historical schema drift: compare only clearly identified compatible metrics
  and keep historical rows outside clean aggregates.
- Output collision: analyzer outputs are derived artifacts and may be
  regenerated only in the explicitly supplied analysis directory.
- Recovery: revert the analyzer/tests/docs changes; source bundles remain
  untouched.

## Progress

- [x] Inspect current analyzer and clean bundle contract.
- [x] Implement normalized tables and prediction/history audit.
- [x] Implement report-ready figures and report selection.
- [x] Extend focused tests.
- [x] Validate full Phase 1 suite and document remote three-seed handoff.

## Decisions

- 2026-07-30: Use existing `metrics.json:history` for learning curves; do not
  modify the clean runner because the current bundle already exports it.
- 2026-07-30: Derive class presence from
  `metadata.json:class_support.train`; current predictions already export the
  equivalent `true_class_absent_from_train` and train support.
- 2026-07-30: Run remaining seeds sequentially on one GPU to avoid resource
  contention; aggregation runs only after copying all artifacts locally.

## Validation

- Focused unit tests for metrics, aggregation, entropy, fixed confusion order,
  and missing inputs.
- End-to-end fixture run creating CSV/Markdown/figures.
- Phase 1 test suite, compilation, and `git diff --check`.

## Result

- The analyzer emits the required normalized tables, prediction-export audit,
  entropy detection metrics, report narrative, figure selection, and
  report-ready PNG/PDF figures.
- Validation passed: 26 Phase 1 tests and 8 Phase 3 tests.
- A single-seed contract fixture generated 14 applicable figures; the
  cross-seed stability figure was explicitly marked unavailable.
- Representative pooled comparison, class-availability heatmap, and normalized
  confusion-matrix figures were visually inspected.
- Real seed-42 artifacts remain on Vast.ai, so final real-data validation is a
  handoff step after download; no synthetic values were substituted.
