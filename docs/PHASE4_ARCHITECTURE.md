# Phase 4 Centralized FedPer Detection

Phase 4 separates scientific model evaluation from a live detection demo. Both
consume the same immutable bundle, but only evaluation accepts ground-truth
labels.

```text
Phase 3 exact artifacts (read-only)
        |
        v
immutable bundle: shared encoder + six heads + frozen preprocessing + digests
        |
        +--> scientific evaluation: correctly routed / cross-head / oracle
        |
        `--> production application
               trusted sensor route
                      |
Zeek conn JSON --> rolling window --> label-free graph --> FedPer prediction
                                                           |
                                                   privacy-reduced event
                                                           |
                                                  Elasticsearch/Kibana
```

## What a Zeek collector is

Zeek is a network-security monitor. It observes packets and produces structured
connection records such as origin/destination, ports, protocol, duration,
packet/byte counts, connection state, and history flags. The collector in this
repository reads Zeek JSON `conn` records and sends them to windowing; Zeek does
not run E-GraphSAGE and does not decide the attack class.

Capturing Kubernetes ingress packets still requires a deployment topology with
appropriate traffic visibility. Host-network/packet-capture privileges are not
enabled by this refactor; that security-sensitive choice is deferred to the
local Minikube acceptance step. An ingress access-log adapter is easier to
deploy but cannot reproduce all Zeek features and therefore cannot silently be
treated as model-equivalent input.

## Current verified state

- The exact schema-v2 GKE research bundle loads without Flower and contains six
  ready round-30 heads plus the validation-best shared encoder. Live API
  readiness intentionally fails until a serving graph protocol is selected.
- Correctly routed evaluation reproduces Phase 3 validation/test metrics:
  macro-F1 `0.9941709665` / `0.9940726453`.
- Cross-head validation/test evidence exists locally under ignored
  `artifacts/application/evaluation/`.
- A validation-selected class/head oracle is kept separate from trusted routing
  and is explicitly non-production.
- Label-bearing production requests return HTTP 422.
- Flow collector/window orchestration and the structured alert router have
  privacy-focused tests; Elasticsearch uses a strict index template.
- The application Docker image builds; application tests and both Helm chart
  lint/template checks pass.

## Open gates

- Select rolling-window stride, lateness, and the 5/15/30/60-second by
  50/100/500/1000-flow candidate using validation evidence.
- Approve a Minikube packet visibility topology before granting Zeek capture
  capabilities.
- Select alert thresholds/severity from labeled validation and false alerts per
  unit time; automatic blocking remains disabled.
- Complete local ingress-to-Kibana acceptance before any GKE sync or cost.
