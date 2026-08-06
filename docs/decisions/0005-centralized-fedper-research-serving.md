# 0005 Centralized FedPer Research Serving

Date: 2026-08-06

## Status

Accepted

## Context

Phase 3 keeps each FedPer classifier head on its owning Edge PVC and stores only
the shared encoder centrally. Phase 4 must evaluate and demonstrate one
centralized detection application using the exact validation-best shared encoder
and all six exact private heads, without depending on Flower or training
infrastructure at runtime.

## Decision

- Permit copies of the six exact private heads in one immutable Phase 4 bundle
  solely for scientific evaluation and a user-owned lab demonstration. This is
  an explicit serving-boundary exception and does not change Phase 3 training
  ownership or its Central GCS contents.
- The bundle contains exactly one validation-best shared encoder and one
  best-round head for each client `1-1`, `3-1`, `9-1`, `34-1`, `36-1`, and
  `39-1`, plus the frozen preprocessor, model/feature/label schemas, and a
  manifest binding every digest and identity.
- Runtime reads only this bundle. It must not access Flower, SuperLink,
  ServerApp, training GCS, training PVCs, or training run-store APIs.
- A trusted server-side mapping selects `sensor_id -> client_id -> head`. Request
  payloads cannot select an arbitrary head and never route based on a predicted
  or ground-truth class.
- Every head retains all seven outputs. Heads are not averaged, concatenated, or
  treated as class-specific experts without cross-head validation evidence.
- Production inference requests contain no ground-truth label. Labeled replay
  uses a separate scientific-evaluation contract.
- Readiness fails closed for missing or mismatched artifacts, invalid routing,
  inconsistent run/model/best-round provenance, schema mismatch, digest failure,
  or a cold-start head.
- Correctly routed evaluation is the deployment protocol. Cross-head evaluation
  measures specialization. An oracle/class-aware upper bound selects its class
  mapping on validation, freezes it, and reads test results once.
- Rolling-window configuration is selected on validation and then recorded as
  the bundle/application graph protocol before final test/live evidence.
- Detection emits privacy-reduced structured events: no raw IP, raw feature,
  ground-truth label, model tensor, credential, or private endpoint enters
  Elasticsearch.
- Entropy is uncertainty only. Phase 4 makes no zero-day or production-readiness
  claim and performs no automatic traffic blocking.

## Consequences

Positive:

- Serving is reproducible without keeping expensive training infrastructure
  alive.
- Exact head routing and provenance are auditable.
- Correctly routed, cross-head, and oracle experiments answer different
  scientific questions without test-set selection leakage.

Tradeoffs:

- Centralized copies weaken the original distributed state boundary and must be
  protected as research artifacts.
- There is still no valid fallback for an unknown sensor or a cold-start client;
  the MVP rejects both.
- Strong known-client results do not imply unseen-client or zero-day
  generalization.

## Security And Operations

- Bundle generation is a read-only export from immutable source artifacts and
  uses temporary output plus atomic promotion.
- Digests cover encoder, every head, schemas, preprocessor, and the manifest's
  client mapping/provenance fields.
- The application deployment receives a configured allow-list mapping; it does
  not trust a caller-provided `client_id` as identity.
- Live attack-pattern tests are limited to infrastructure owned by the user and
  do not trigger automated blocking in the MVP.
