# Execution Plan: Phase 2 IoT-23 Federation

Date: 2026-07-26

## Status

Implementation complete with local synthetic proof. Full IoT-23/PyG/Flower
execution remains active because this machine lacks the external inputs.

## Outcome

Provide a modular, observable Phase 2 pipeline that prepares six immutable
scenario-aligned IoT-23 client artifacts, validates their contracts, reruns a
clean centralized reference, and can execute FedAvg followed by FedProx through
Flower without changing the Phase 1 compatibility surface.

## Context

- `docs/PHASE2_ARCHITECTURE.md`: existing trust boundary and verified foundation.
- `docs/decisions/0001-phase2-modularity-observability.md`: accepted extension,
  benchmark, and observability policy.
- `src/federated/`: current contracts, core aggregation, adapters, and Flower apps.
- `config.yaml`: authoritative six scenarios and Phase 1 model/training defaults.

## Scope

In scope:

- Strict registry/config selection for data, partition, model, task, strategy,
  runtime, and observer implementations.
- Structured console/JSONL observability and immutable preparation/run manifests.
- Train-only shared preprocessing and portable six-client graph artifacts.
- Lazy manifest-backed IoT-23 tasks, FedAvg/FedProx Flower execution, per-round
  checkpoints, centralized reference, validation, and comparison commands.
- Synthetic and optional dependency integration proof runnable without raw data.

Out of scope:

- Downloading IoT-23 onto the current disk, Kubernetes deployment, OpenTelemetry
  infrastructure, IID repartition, DP, SecAgg, and solving cross-client edges.

## Approach

1. Install the modular registry/config and observability primitives first.
2. Add deterministic preparation with transactional portable artifacts.
3. Load client graphs lazily behind the existing federated task contract.
4. Add observed/resumable Flower strategies and runtime task switching.
5. Add centralized/equivalence/evaluation commands and focused tests.
6. Run all locally available proof and report external data/dependency gates.

## Risks And Recovery

- Existing Phase 1 imports are compatibility authority; new modules wrap them
  instead of moving or rewriting them.
- The current machine has about 5.7 GB free and lacks PyG/Flower, so doctor must
  fail before full-data work. Synthetic proof remains mandatory.
- Prepared/run outputs live under ignored `artifacts/phase2/`, are written
  transactionally, and never overwrite a non-empty target.
- Recovery removes only the new prepared/run directory by explicit ID; source
  data and Phase 1 results are never deleted.

## Progress

- [x] Add modular config/registry and observability contracts.
- [x] Add train-only preparation and portable artifact validation.
- [x] Add lazy IoT-23 task and runtime switching.
- [x] Add observed/resumable FedAvg/FedProx and experiment outputs.
- [x] Add focused/unit/synthetic smoke proof and documentation.
- [x] Record validation and external full-data limitations.
- [ ] Run the optional Flower tests and full six-scenario benchmark in an
  environment containing IoT-23, PyG, and Flower.
- [x] Remediate review findings: Flower benchmark config/best-test protocol,
  content-root provenance, shared-contract validation, strategy-bound resume,
  crash-consistent rounds, and compatible-run comparison.

## Decisions

- 2026-07-26: Preserve Phase 1 entrypoints and introduce modules incrementally.
- 2026-07-26: Fit one shared preprocessor only on each client's train rows.
- 2026-07-26: Use six scenario-aligned non-IID clients with full participation.
- 2026-07-26: Run FedAvg before FedProx (`mu=0.01`) using 30 rounds x 5 local epochs.
- 2026-07-26: Emit console and JSONL evidence now; reserve telemetry exporters
  for Phase 3 without coupling business logic to an observability backend.
- 2026-07-26: Treat the seven post-implementation review findings as valid.
  Round commit markers, content-root digests, and validation-best Flower state
  become required correctness boundaries rather than optional hardening.

## Validation

- Focused proof: `python -m unittest discover -s tests/federated -v`.
- Integration proof: six-client synthetic prepared artifacts through in-process
  and optional Flower FedAvg/FedProx runs.
- Repository checks: compile touched Python modules and `git diff --check`.

Observed 2026-07-26:

- `python -m unittest discover -s tests/federated -v`: 46 run, 41 passed and 5
  skipped because Flower is not installed in the default project environment.
- With pinned Flower 1.32.1 installed under `/tmp`, all 9 Flower-focused tests
  passed, including a two-round server protocol test proving validation-best
  selection and one final federated test evaluation.
- `python -m compileall -q src/federated tests/federated`: passed.
- Focused `ruff check` on every new/modified Phase 2 Python surface: passed.
- `git diff --check`: passed.
- CLI toy FedAvg completed and produced atomic run status, round JSONL/CSV,
  final summary, per-round checkpoint, best checkpoint, and server events.
- `doctor`: correctly returned not-ready; all six raw scenario files,
  `torch_geometric`, and `flwr` are absent. About 6.05 GB disk was free.

## Result

The planned modular pipeline is implemented. It includes strict config and
registries, unbiased bounded-memory scenario sampling, shared train-only
preprocessing, immutable portable graph artifacts, lazy client loading,
centralized/FedAvg/FedProx commands, validation-selected checkpoints,
failure/resume guards, single final test evaluation, communication accounting,
and console/JSONL observability across preparation, task, runtime, and Flower
boundaries. Post-review remediation additionally binds prepared dataset identity
to client/contract content, validates every graph against the shared schema,
uses Phase 2 settings in Flower, commits resumable rounds crash-consistently,
and rejects comparisons across incompatible provenance.

The plan remains active only for external execution evidence. No real IoT-23
metric or Flower end-to-end claim is made until the missing data/dependencies
are supplied and those commands pass.
