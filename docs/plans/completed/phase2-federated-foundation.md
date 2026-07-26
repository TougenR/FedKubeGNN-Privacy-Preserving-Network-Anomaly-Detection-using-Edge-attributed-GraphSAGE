# Execution Plan: Phase 2 Federated Foundation

Date: 2026-07-24

## Status

Completed for the federated foundation slice. Full IoT-23/PyG validation remains
an explicit follow-up gate because the required Phase 1 artifacts and dataset
run are not available in this workspace.

## Outcome

Produce a reusable Phase 2 federated-learning implementation whose contracts,
aggregation, metrics, and Flower boundary can be validated without depending on
Phase 1 internals. Connect the current IoT-23/E-GraphSAGE implementation through
an explicit adapter, and reject incompatible data/model artifacts before a
federated round starts.

## Context

- `docs/HANDOFF_PHASE2.md`: intended Phase 2 inputs and experiment questions.
- `docs/PHASE1_REPORT.md`: historical centralized results, not assumed correct
  until independently validated.
- `src/preprocess.py`, `src/graph_build.py`, `src/model.py`: current Phase 1
  implementation behind the adapter boundary.
- `PhanChiaTask_DoAn.md 23-39-29-802.md`: Phase 2 scope (Flower, FedAvg,
  FedProx, communication cost, IID/non-IID analysis).

## Scope

In scope:

- Versioned data, graph, model, parameter, training, and metric contracts.
- Portable JSON/NPZ artifact bundle support with checksums.
- Framework-independent weighted FedAvg and confusion-matrix aggregation.
- A toy task proving the Phase 2 core works without Phase 1 or PyG.
- A Phase 1 IoT-23 adapter with strict compatibility validation.
- Current Flower `ClientApp`/`ServerApp` integration and local simulation
  configuration.
- Focused unit, contract, and smoke tests that can run in the available
  environment; dependency-blocked validation is reported explicitly.

Out of scope for the first verified foundation:

- Refactoring or correcting Phase 1 training behavior without separate proof.
- Full 15-20 GB IoT-23 training runs or reproducing the 0.8773 result locally.
- Cross-client graph partition research, differential privacy, SecAgg+, or
  Kubernetes deployment before the core FedAvg path is verified.

## Approach

1. Add stable contract and artifact types with compatibility validation.
2. Add independent aggregation/evaluation logic and deterministic toy proof.
3. Add the Phase 1 adapter without leaking Phase 1 imports into the core.
4. Add Flower boundary code using the current Message API, isolated from the
   task implementation.
5. Validate progressively: standard-library/unit tests first, optional
   Phase 1/Flower integration when dependencies are available.
6. Document the run contract, known Phase 1 discrepancies, and exact next
   full-data experiment.

## Risks And Recovery

- Phase 1 currently lacks tracked preprocessor/model binary artifacts. The
  adapter will fail closed with actionable errors; it will not fabricate them.
- `torch_geometric` and Flower may be absent locally. Core tests remain runnable
  without them; integration tests will skip with a precise reason until the
  dependencies are installed.
- Pickled Phase 1 classes are refactor-fragile. Portable contract artifacts are
  authoritative at the Phase 2 boundary; legacy pickle loading stays inside the
  Phase 1 adapter only.
- Existing uncommitted Harness files and `.gitignore` edits belong to the user.
  Changes are additive and do not overwrite or revert those files.
- Recovery: remove the additive `src/federated/`, `tests/federated/`, and Phase 2
  config/docs files. No dataset, Phase 1 result, or checkpoint is mutated.

## Progress

- [x] Inspect Phase 1 documents, implementation boundaries, and result artifacts.
- [x] Add versioned contracts and portable artifact bundle support.
- [x] Add independent FedAvg and global metric aggregation.
- [x] Add deterministic toy task and focused proof.
- [x] Add Phase 1 IoT-23 adapter and contract checks.
- [x] Add Flower Message API client/server integration.
- [x] Run available validation and document blocked full-data proof.

## Decisions

- 2026-07-24: Treat Phase 1 as an untrusted upstream provider connected only
  through adapters; Phase 2 core must run against a toy task with Phase 1 absent.
- 2026-07-24: Do not use a pickled `Preprocessor` object as the durable Phase 2
  contract. Store learned categories/scaler arrays in portable versioned files;
  legacy object conversion is adapter-only.
- 2026-07-24: Start with scenario-aligned clients so the first FL baseline does
  not introduce a new cross-client-edge deletion confound.
- 2026-07-24: Compute global macro-F1 from an aggregated confusion matrix rather
  than averaging client macro-F1 scalars.

## Validation

- Focused proof: `python -m unittest discover -s tests/federated -v`
- Integration or end-to-end proof: toy multi-client FedAvg run; optional Flower
  local simulation and Phase 1 adapter checks when dependencies/data exist.
- Repository-required checks: compile new Python modules and preserve a clean
  diff with no unintended Phase 1 or user-owned edits.

## Result

Implemented in `src/federated/` with 20 focused tests, strict contract and
aggregation checks, a deterministic toy FedAvg/FedProx runner, a fail-closed
IoT-23 adapter, and current Flower 1.32.1 ClientApp/ServerApp integration.
The real Flower simulation proof completed with two toy clients for three rounds
and no client failures. See `docs/PHASE2_ARCHITECTURE.md` for the run contract,
trust boundary, evidence, and the next full-data gates.
