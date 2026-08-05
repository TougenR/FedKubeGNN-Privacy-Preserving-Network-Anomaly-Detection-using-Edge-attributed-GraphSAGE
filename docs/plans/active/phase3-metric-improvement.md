# Phase 3 Metric Improvement and Phase 1 Comparison

## Outcome

Explain the gap between the clean Phase 1 pooled E-GraphSAGE result and the
Phase 3 federated result, then improve fixed-eight-class macro-F1 without using
test data for model or hyperparameter selection.

## Repository authority

- `docs/PHASE1_CLEAN_PROTOCOL.md` defines the leakage-remediated Phase 1
  protocol and its accepted transductive limitations.
- `artifacts/phase1_clean/report_analysis/` is the comparison evidence for
  Phase 1 clean runs over seeds 42, 1337, and 2026.
- `configs/phase2/iot23-federated.yaml` defines the current six-scenario FL
  dataset, model, split, optimizer, imbalance, and federation settings.
- `src/federated/data/preparation.py` and
  `src/federated/adapters/phase1_iot23.py` are the current preprocessing and
  local-training implementations.
- Phase 3 GKE/GitOps execution is recorded in
  `docs/plans/completed/phase3-gke-gitops.md`; Argo CD remains the only GKE
  deployer.

## Baseline observations (2026-08-05)

- Phase 1 clean pooled fixed-eight macro-F1 is `0.899981 ± 0.027091` across
  three seeds; seed 42 is `0.928692`.
- The completed Phase 3 FedAvg demonstration has test macro-F1 `0.456556`;
  FedProx has test macro-F1 `0.456691`. Both completed 30 rounds over all six
  clients, and their one-seed difference is too small to support superiority.
- Phase 1 clean and Phase 2 use the same ordered 55-feature schema and the
  same E-GraphSAGE shape (`hidden_dim=64`, two layers, dropout `0.5`).
- The label distribution is strongly non-IID. DDoS exists only at client
  `34-1`; C&C-HeartBeat, Okiru, and Okiru-Attack exist only at client `36-1`;
  Okiru-Attack has only three rows before splitting.
- The current FL adapter computes class weights independently on each client,
  performs five local Adam epochs with a fresh optimizer each round, and then
  Flower aggregates model parameters by local training-example count. Phase 1
  clean instead trains one model with one union-train class-weight vector.
- The current confusion matrices show zero F1 for the four private/ultra-rare
  classes above. This makes aggregation and local objective behavior the first
  hypothesis, not feature-schema drift.

## Ordered work

1. Freeze the final Phase 3 E2E baseline: immutable image/data/model/config
   digests, both run summaries, round CSV, figures, Kibana event counts, and
   final Argo CD health.
2. Complete the prerequisite data, preprocessing, topology, and imbalance work
   in `docs/plans/active/phase3a-data-analysis-imbalance.md`.
3. Run the existing centralized reference on the exact same prepared dataset,
   initial state, split masks, feature contract, and seed. Compare it with both
   Phase 1 clean seed 42 and the federated baseline. This separates data/
   preprocessing effects from federation effects.
4. Audit preprocessing and features without retraining:
   - exact train/validation/test and per-class support by scenario;
   - Phase 1/Phase 2 feature order, categorical vocabularies, scaler arrays,
     missingness, non-finite values, constant/near-constant columns, and
     scenario/class distribution shifts;
   - graph node/edge/degree/connectivity statistics and split-mask integrity;
   - categorical unknown/other rates and whether any useful Phase 1 feature is
     lost or semantically changed.
5. Add experiment observability needed to diagnose client drift: per-client
   train/validation loss and fixed-eight per-class metrics, update norms,
   client-to-global divergence, and class-support metadata per round.
6. Run controlled validation-only ablations, changing one family at a time:
   - union-train/global versus local class weights;
   - one versus five local epochs and learning-rate tuning;
   - aggregation weighting appropriate for private classes;
   - FedProx `mu` tuning and, only if supported by the pinned Flower version,
     server-optimizer strategies such as FedAdam;
   - loss alternatives for extreme rarity only after the weighting baseline.
7. Run the seven-class centralized/IID/natural non-IID diagnostic defined in
   `phase3b-iid-noniid-diagnostic.md` before selecting the next aggregation
   treatment. Treat IID redistribution as causal diagnosis only, never as the
   production data policy.
8. Promote a configuration only from validation metrics. Evaluate the fixed
   final test split exactly once per seed, then compare seeds 42, 1337, and
   2026 against the frozen Phase 1 clean and Phase 3 baselines.

## Cost and safety controls

- Reuse the immutable prepared dataset; do not re-read or recopy raw IoT-23
  unless an audit proves the prepared artifact invalid.
- Use local/in-process or a bounded single-cluster experiment before any
  multi-cluster repetition.
- Every expensive training Job uses an immutable release ID and
  `backoffLimit: 0`; training is disabled in Git immediately after evidence is
  persisted.
- Do not tune on test metrics, overwrite Phase 1 evidence, or claim statistical
  improvement from one seed.

## Acceptance

- A reproducible comparison table covers Phase 1 clean pooled, the exact-data
  centralized reference, FedAvg, and FedProx with fixed-eight macro-F1,
  weighted-F1, accuracy, per-class F1/support, seed, and provenance.
- The metric gap has evidence-backed attribution to preprocessing/features,
  optimization/aggregation, or both.
- At least one validation-selected FL configuration improves the frozen FL
  baseline across three seeds without weakening final-test isolation.
- Figures, CSV/JSON summaries, config snapshots, model checkpoints, and logs
  are stored under immutable run paths.

## Status

- [x] Opened from observed Phase 3 E2E evidence and Phase 1 clean artifacts.
- [x] Freeze the completed Phase 3 baseline.
- [x] Run exact-data centralized reference (`0.869830` test macro-F1).
- [ ] Complete preprocessing/feature/topology audit.
- [x] Add client-drift/update-norm, classifier-row, and local-state
  observability.
- [x] Run controlled validation-only aggregation ablations on natural-7.
- [x] Reject global class weights on validation (`0.380236`, delta `-0.075489`).
- [x] Reject equal-compute one-local-epoch FedAvg (`0.315973`, delta `-0.139753`).
- [x] Complete the seven-class IID versus natural non-IID diagnostic: IID
  `0.988879` versus natural `0.507170` test macro-F1.
- [x] Validate combined class-balanced-client/support-only-head aggregation
  across seeds 42, 1337, and 2026; mean validation macro-F1 `0.736309`, selected
  test macro-F1 `0.738097 ± 0.060893`.
