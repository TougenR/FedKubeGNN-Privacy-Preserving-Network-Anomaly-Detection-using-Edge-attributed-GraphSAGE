# 0006 Validation-Calibrated Multi-Head Fusion

Date: 2026-08-09

## Status

Accepted

## Context

The centralized Phase 4 application initially used only the private head mapped
from the trusted sensor identity. GKE live traffic originates from
`sensor-34-1`, so every one of the six lab patterns was decided by head `34-1`
and returned `Benign`. Running the same observed flows through every head proved
the heads are active, but also proved that an `any-head` rule is unsafe: head
`3-1` reports `Attack` above the alert threshold on the known benign baseline.

The user approved using all heads for the centralized detection decision while
retaining trusted routing for provenance and explainability.

## Decision

- Encode each rolling graph once and execute all six exact best-round heads.
- Select class-specific non-negative head weights only from the labeled
  validation replay under the locked rolling-window protocol.
- Select class-specific alert thresholds only from validation while keeping
  the total benign false-alert rate at or below `0.001`, consistent with the
  previously approved false-alert trade-off.
- Freeze the selected policy, its digest, bundle/model/dataset identities,
  graph protocol, head digests, class order, and validation-report digest before
  one locked test evaluation.
- Use the fused probability vector as the primary production prediction. Keep
  the trusted head result and all-head label/confidence summaries as diagnostic
  output; request traffic still cannot select a head.
- Fail readiness if the policy digest, bundle identity, model/dataset digest,
  graph protocol, head set/digests, class order, weights, thresholds, or
  validation provenance does not match.
- Never select a head from the scenario name, predicted class, or ground truth.
  Never implement `any-head` alerting or claim that synthetic HTTP patterns are
  IoT-23 ground truth.
- Elasticsearch may store the fusion policy digest, decision mode, trusted-head
  label, and disagreement count. It still cannot store probability vectors,
  raw features, tensors, labels, or raw network identities.

## Consequences

The centralized demo can use knowledge contained in all six personalized heads
without pretending that a head is an attack-class router. Inference cost grows
only across the small classifier heads because the shared encoder runs once.
The fused decision is a new validation-bound research policy, not a change to
FedPer training and not evidence of zero-day or production readiness.
