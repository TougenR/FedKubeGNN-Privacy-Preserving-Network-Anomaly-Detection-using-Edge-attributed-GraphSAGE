# Phase 3B: Seven-Class IID Versus Natural Non-IID Diagnostic

Date: 2026-08-05

## Status

Completed

## Outcome

Produce three comparable seven-class benchmarks—centralized, stratified IID
FedAvg, and natural scenario non-IID FedAvg—to determine whether the current
federated metric loss is primarily caused by implementation/runtime defects or
by the natural client distribution.

## Context

- The immutable source prepared dataset is `iot23-0d9bbeb9f9ed0a3f`, manifest
  digest `68fc6fc0cb8974aba1d431113b39dbf82f98457159c04d6a14b22feaa4b0cb89`.
- `phase3a-data-analysis-imbalance.md` proves that four classes are private to
  one client and `Okiru-Attack` has only three total examples.
- The exact-data centralized eight-class run reached test macro-F1 `0.869830`,
  while FedAvg reached `0.456556`.
- This is a controlled diagnostic, not a production redistribution policy;
  natural scenario non-IID remains the deployment benchmark.

## Scope

In scope:

- Remove `Okiru-Attack` and remap the original eight-class output to the fixed
  seven-class order.
- Derive an immutable natural seven-class dataset and a deterministic
  stratified-IID train repartition across six clients.
- Keep every retained train flow exactly once without undersampling or
  duplication; preserve validation/test membership and never use them to
  balance clients.
- Namespace nodes by source scenario when scenarios are combined into an IID
  graph.
- Reuse the train-only-fitted preprocessor and derive one deterministic
  seven-class initial state shared by all three benchmarks.
- Run centralized and validation-only FedAvg comparisons with identical model,
  optimizer, seed, and training budget before any final-test evaluation.

Out of scope:

- Moving production data between Edge sites.
- Dirichlet partitions, class-aware aggregation, or new loss functions until
  this diagnostic is complete.
- GCP/GKE changes or billable resource creation.

## Approach

1. Add a transactional derivation tool with integrity manifests and explicit
   flow provenance.
2. Build natural-7 by filtering label 6 and remapping old label 7 to new label
   6 while retaining source client topology and split membership.
3. Build IID-7 by shuffling each union-train class deterministically and
   splitting it into six near-equal parts. Assign each retained validation/test
   flow once to its original scenario owner; those splits are not balanced.
4. When combining source scenarios, remap nodes using
   `(scenario_id, source_node_id)` so equal source node IDs cannot create
   cross-scenario edges.
5. Derive the seven-class model state by deleting output row 6 from
   `head.3.weight` and `head.3.bias`; preserve all other tensors bit-for-bit.
6. Verify class support, split conservation, unique flow assignment, contract
   equality, state equality, determinism, and artifact checksums.
7. Run centralized-7, IID-7 FedAvg, and natural-7 FedAvg under the same budget;
   use validation for diagnosis/selection and touch test only for the frozen
   benchmark comparison.

## Risks And Recovery

- Combining graphs can accidentally collide node IDs. The derivation creates a
  separate node map for every source scenario and records the namespace policy.
- A repartition bug can duplicate or omit flows. Provenance keys
  `(source_client, source_edge_index)` are counted before artifacts are accepted.
- Re-fitting preprocessing would introduce another variable. The immutable
  source learned arrays are reused; the two removed training rows therefore
  influenced the already-fitted scaler, identically in every benchmark.
- Derived datasets are written transactionally to a new destination and never
  overwrite the source. Recovery is deletion of only the incomplete/new derived
  directory; source artifacts remain immutable.

## Progress

- [x] Freeze experimental policy and seven-class label mapping.
- [x] Implement deterministic natural/IID derivation and focused tests.
- [x] Generate and verify separate manifests/digests.
- [x] Run centralized seven-class reference.
- [x] Run IID and natural non-IID FedAvg with equal budget.
- [x] Compare metrics and decide between non-IID mitigation and runtime audit.

## Decisions

- 2026-08-05: The fixed class order is `Benign`, `Attack`, `C&C`,
  `C&C-HeartBeat`, `DDoS`, `Okiru`, `PartOfAHorizontalPortScan`; old class 6
  is removed and old class 7 becomes new class 6.
- 2026-08-05: Reuse the immutable train-only-fitted preprocessor so repartition
  is the only data-variable change. This means the two subsequently removed
  train rows remain part of its learned statistics.
- 2026-08-05: Validation/test split membership is conserved globally and each
  flow stays with its original scenario owner; neither split is balanced.
- 2026-08-05: The seven-class initial state is a deterministic projection of
  the eight-class state, not a new random initialization.

## Validation

- Focused proof: deterministic partition, per-class max/min train support
  difference at most one, no missing/duplicate retained flow, exact split
  conservation, label remapping, node namespace isolation, and state projection.
- Integration proof: both derived manifests load with full checksum/client
  verification and share model-spec and initial-state digests.
- Experiment proof: immutable centralized/FedAvg summaries with dataset,
  contract, model, config, and initial-state digests.
- Repository-required checks: federated tests, Ruff, compile, and whitespace.

## Result

Completed on 2026-08-05.

- Derived and fully verified `iot23-seven-iid-5183e5734796e4cf` with manifest
  digest `7c08f2e53191229b6b2c780c63b36fb440a125bf014b23718d2299d8e6804ecd`
  and `iot23-seven-natural-3be7796b1ee27bc3` with manifest digest
  `c5ab9c02896c08c9f60e8efb9672a2090cbe595e4c344308f5e4dc2b0e51319a`.
- Both datasets share contract digest
  `2b99f6931a0877f34cff7431203d67cbc59d9d6b06e3091850f9c2c560e33de7`,
  model digest
  `42642e4cc839c09dfe8519511aa7cf7cdf5ca7350a8dd376e118ee31a6a74bbf`,
  and initial-state SHA-256
  `090486e2b9a17bf2e74f4f56327c0a8be46a5105e9a0b872d5d72bea0c2a755b`.
- Derivation retained 121,472 flows, removed exactly three `Okiru-Attack`
  flows, preserved global split/class support, produced no duplicate/missing
  retained flow, and limited IID per-class train support variation to one.
- Centralized-7 test macro-F1 was `0.986555`; IID FedAvg-7 was `0.988879`;
  natural non-IID FedAvg-7 was `0.507170`. IID therefore exceeded natural by
  `0.481709` and was within `0.002324` of centralized.
- The diagnosis selects non-IID mitigation, specifically class-support-aware
  aggregation/update weighting, rather than a general runtime audit. Natural
  non-IID remains the production benchmark; IID remains a controlled test.
- Full artifact verification, 59 federated tests (five optional-Flower skips),
  Ruff, compilation, CLI discovery, JSON validation, and whitespace checks
  passed. Evidence is under `artifacts/phase3_analysis/iid-diagnostic/`.
- A repeat of the natural run showed approximately `5e-4` validation macro-F1
  variation at the same nominal round on CPU/PyG, so multi-seed conclusions
  remain required and no rerun state was mislabeled as the original best state.
