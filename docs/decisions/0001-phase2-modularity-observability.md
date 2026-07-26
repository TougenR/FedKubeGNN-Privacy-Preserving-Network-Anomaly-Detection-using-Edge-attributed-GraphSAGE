# 0001 Phase 2 Modularity, Benchmark, And Observability

Date: 2026-07-26

## Status

Accepted

## Context

Phase 2 must connect the Phase 1 IoT-23/E-GraphSAGE implementation to Flower
without making future data sources, graph builders, models, strategies,
runtimes, or telemetry backends depend on those concrete choices. Historical
Phase 1 preprocessing fits the union of data, while the new benchmark must not
learn feature statistics from validation or test rows. Runs also need enough
observable evidence to diagnose preparation, client, round, aggregation, and
artifact failures.

## Decision

- Add explicit named component registries and strict versioned configuration.
- Preserve existing Phase 1 imports through compatibility adapters.
- Use six scenario-aligned clients, deterministic 70/10/20 edge masks, a single
  preprocessor fitted only on train rows, and a fixed global label vocabulary.
- Keep the current transductive edge-message-passing protocol and identify it
  explicitly in every artifact and result.
- Make structured console events, process-local JSONL, run manifests, digests,
  timing, metrics, checkpoint status, and communication sizes mandatory.
- Do not log raw IPs, flow features, labels per flow, tensors, or credentials.
- Keep observer interfaces backend-neutral; add OpenTelemetry infrastructure in
  Phase 3 rather than coupling Phase 2 to a collector deployment.

## Alternatives Considered

1. Refactor Phase 1 and Phase 2 into a new package in one migration.
2. Reuse the Phase 1 union-fitted preprocessor and historical 0.8773 result.
3. Add only ad-hoc print statements and CSV summaries.

## Consequences

Positive:

- New modules can be selected by config without modifying orchestration code.
- Centralized and federated results share the same leakage-resistant artifacts.
- Failed and resumed runs have machine-readable provenance.

Tradeoffs:

- The new centralized result will not be assumed equal to historical 0.8773.
- Benchmark preparation centrally observes public client train rows once.
- The transductive graph protocol remains a documented limitation.

## Follow-Up

- Add an inductive graph protocol and production OTel observer in later work.
