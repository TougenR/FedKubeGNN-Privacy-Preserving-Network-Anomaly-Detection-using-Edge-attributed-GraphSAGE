# Phase 3A: Data, Preprocessing, and Imbalance Analysis

## Outcome

Produce an evidence-backed explanation of the Phase 1 clean versus federated
metric gap, identify data/preprocessing defects and severe non-IID imbalance,
then validate appropriate treatments before the broader Phase 3 metric-tuning
work in `phase3-metric-improvement.md`.

## Scope and authority

- Analyze the immutable six-scenario prepared dataset currently identified by
  dataset digest
  `68fc6fc0cb8974aba1d431113b39dbf82f98457159c04d6a14b22feaa4b0cb89`.
- Compare against the leakage-remediated Phase 1 implementation and evidence:
  `docs/PHASE1_CLEAN_PROTOCOL.md`, `src/phase1_clean.py`, and
  `artifacts/phase1_clean/report_analysis/`.
- Preserve train-only preprocessing, fixed final-test isolation, eight-class
  reporting, six scenario clients, and the current privacy boundary.
- Do not modify or overwrite historical Phase 1 artifacts.

## Phase 3A.1 — Dataset balance report

Generate machine-readable CSV/JSON and figures for:

- row counts and train/validation/test counts per client and class;
- global and per-client class proportions, imbalance ratios, effective sample
  counts, label entropy, class coverage, and clients-per-class;
- Jensen-Shannon divergence of each client label distribution from the union;
- support-aware warning categories, including structurally private classes,
  zero-support client/class pairs, and ultra-rare classes;
- duplicate stable rows, non-finite values, missingness, and split overlap or
  coverage violations;
- graph nodes, edges, degree/connectivity, isolated nodes, and topology shift by
  scenario.

Observed evidence already requiring treatment:

- DDoS occurs only at client `34-1`.
- C&C-HeartBeat, Okiru, and Okiru-Attack occur only at client `36-1`.
- Okiru-Attack has three total rows; deterministic splitting leaves it with no
  validation/test support on some seeds by construction.

For this subphase, imbalance is severe when any class is confined to one
client, has fewer than 30 global observations, has zero validation support, or
when a client/class pair required by fixed-eight evaluation has zero training
support. The current dataset already meets multiple triggers, so treatment
experiments are required after the baseline and equivalence reports are
generated; no additional decision is needed merely to start those ablations.

## Phase 3A.2 — Phase 1/Phase 2 preprocessing equivalence

Compare, in order:

1. raw source hashes and priority-sampling/cap membership;
2. fixed label taxonomy and index mapping;
3. raw split membership and per-class split support;
4. train-only rows used to fit preprocessing;
5. categorical vocabularies, rare/other/unknown handling, and occurrence rates;
6. scaler means/scales/variances and transformed numeric distributions;
7. exact ordered feature schema, dtype, missing flags, constant/near-constant
   features, and scenario/class distribution shift;
8. graph construction and mask semantics.

The first live check found the same exact ordered 55-feature schema and the same
total per-class population as Phase 1 clean. Phase 1 and Phase 2 currently use
different deterministic split implementations and different label index order;
their effect must be measured rather than assumed harmless.

## Phase 3A.3 — Imbalance treatments

Use the unchanged prepared data and evaluate one treatment family at a time.
Selection uses global validation fixed-eight macro-F1 and per-class recall/F1;
the test split remains untouched until a configuration is selected.

1. Reproduce Phase 1's union-train/global class-weight vector in every client
   and compare it with the current per-client weight vectors.
2. Reduce client drift by comparing one versus five local epochs and tuned
   learning rates at the same total training budget.
3. Test class-balanced/effective-number or focal loss only after the global
   class-weight baseline; do not combine multiple changes in the first ablation.
4. Evaluate aggregation that does not let example-rich clients suppress
   classes present at only one client. Record client update norms and
   class-support-aware contributions before choosing a method.
5. If a class lacks enough validation support (notably Okiru-Attack), report it
   as not estimable and do not optimize against a synthetic or duplicated test
   example. Any augmentation requires a separately recorded data policy and
   provenance.

The execution order is binding for causal attribution: first global class
weights, then local-epoch/learning-rate control, then one alternative loss, and
finally an aggregation treatment informed by recorded update norms and class
support. A treatment is retained only when it improves validation across
multiple seeds without degrading estimable private-class recall or changing
the final-test population.

Undersampling is not the default first treatment because it discards abundant
IoT-23 evidence. Blind row duplication is not considered evidence of improved
generalization. Loss/aggregation corrections are evaluated first.

## Deliverables

- `artifacts/phase3_analysis/data_balance.csv`
- `artifacts/phase3_analysis/class_support_by_client.csv`
- `artifacts/phase3_analysis/preprocessing_comparison.json`
- `artifacts/phase3_analysis/feature_distribution.csv`
- `artifacts/phase3_analysis/graph_topology.csv`
- imbalance and feature-shift figures in PNG/PDF;
- a validation-only ablation table with immutable config/data/model digests;
- a concise finding report mapping each proposed treatment to observed
  evidence.

## Validation and exit criteria

- Analysis can be reproduced from immutable prepared artifacts without raw-data
  mutation.
- Phase 1/Phase 2 differences are explicit and quantified at every preprocessing
  boundary.
- Every severe imbalance claim includes class/client/split support evidence.
- Each treatment has an isolated baseline, validation result, and unchanged
  final-test policy.
- The selected treatment improves validation across multiple seeds or is
  rejected; one-seed gains are not promoted.

## Status

- [x] Phase created from live E2E and Phase 1 clean evidence.
- [x] Exact 55-feature order and total per-class populations checked.
- [x] Structurally private and ultra-rare classes identified.
- [x] Generate the prepared-artifact balance, feature, and topology report.
- [ ] Complete preprocessing learned-state comparison.
- [ ] Run exact-data centralized reference.
- [ ] Implement and validate isolated imbalance treatments.
- [ ] Hand selected treatments to Phase 3 metric-improvement experiments.

## Observed evidence (2026-08-05)

- `analyze-data` verified the immutable manifest, contract, initial state, every
  client checksum, and disjoint/complete split masks before writing artifacts.
  The analysis manifest is bound to dataset digest
  `68fc6fc0cb8974aba1d431113b39dbf82f98457159c04d6a14b22feaa4b0cb89`.
- The union has 121,475 edges and fixed-eight imbalance ratio `13,374:1`:
  Okiru-Attack has only 3 rows. C&C-HeartBeat, DDoS, Okiru, and Okiru-Attack
  each occur at one client only. The severe-treatment trigger is therefore
  confirmed, not hypothetical.
- All 55 transformed features are finite. Four history indicator features are
  globally constant zero. The three explicit numeric missing indicators each
  have global rate `0.6358`, varying from `0.2907` at client `3-1` to `0.8010`
  at client `1-1`; this is a measurable client shift that must be compared with
  Phase 1 learned preprocessing.
- Every client graph is one weak component with zero isolated nodes, but graph
  topology differs sharply. Client `34-1` contains 18,751 edges on 49 nodes
  and only 49 unique directed node pairs; it also contains 18,702 parallel-edge
  instances. Phase 1 graph construction equivalence is required before this is
  classified as valid scenario structure rather than a preprocessing defect.
- The CSV/JSON tables, finding report, three PNG/PDF figures, and SHA-256
  analysis manifest are under `artifacts/phase3_analysis/`. All figures were
  visually inspected. Raw missing values are not retained in the prepared
  contract, so raw missingness remains part of the Phase 1/source comparison.
- The analyzer has two focused overwrite/integrity tests. The complete
  federated suite passes 52 tests with five expected optional-Flower skips;
  Ruff, compilation, CLI discovery, all 12 generated artifact hashes, and
  whitespace validation pass.
- The Phase 1 `preprocessor.pkl` was loaded without compatibility monkeypatches
  in an ephemeral NumPy 2.4.6/scikit-learn 1.9 container. Phase 1 and Phase 2
  have identical ordered features, categorical vocabularies, numeric columns,
  history flags, and missing flags. Their learned scalers are not identical:
  Phase 1 fitted 85,029 train rows while Phase 2 fitted 85,028 due to the split
  algorithm difference. Across the eight numeric columns, maximum absolute
  mean/scale differences are `0.012625`/`0.012627`; the largest relative scale
  difference is `6.27%` (`missed_bytes`). This is measurable preprocessing
  drift, but not yet evidence that it explains the `~0.44` macro-F1 gap.
