# Phase 4 IoT-23 Traffic Profile Analysis

## Evidence boundary

The first profile reference was generated from 12,144 validation rows in the
exact Phase 4 replay. All six compressed inputs were verified against their
manifest SHA-256 before analysis. The locked test split was not read.

- Reference digest:
  `1170c604041f572ef17c2cd12b16b5274a086a8d2a90a064c97e2315ff98b170`
- Dataset digest:
  `c5ab9c02896c08c9f60e8efb9672a2090cbe595e4c344308f5e4dc2b0e51319a`
- Graph protocol:
  `rolling-window-v1:duration=60s:max-flows=50:stride=1:lateness=1s`

The reference is authoritative for row-level features in the deterministic
validation sample. The stratified split thins each original source timeline,
so its inter-arrival values are not evidence of natural malware timing. Rolling
density remains useful only as the locked view presented to the model.

## Findings

| Class | Support | Dominant validation fingerprint | Executable-profile consequence |
|---|---:|---|---|
| Benign | 3,644 | UDP 63.7%; unknown service 89.3%; S0 52.9%; heterogeneous ports and six clients | A single HTTP baseline cannot represent the class; retain several controls and measure false alerts. |
| Attack | 663 | TCP/22 100%; SSH 96.1%; SF 100%; median 14 origin and 15 response packets | A bounded completed SSH-session candidate can represent this dataset slice, but `Attack` is still a generic label. |
| C&C | 825 | TCP 99.9%; port 6667 99.8%; S0 68.8%, S3 18.5%, SF 9.3%; response present 29.1% | Requires a fixed IRC-like/no-response mixture; periodic HTTP is not equivalent. |
| C&C-HeartBeat | 1,000 | TCP/57722, unknown service, S0 and SYN-only all 100%; one destination | Use bounded periodic incomplete connections and label timing as model-equivalence only. |
| DDoS | 1,000 | TCP/80 100%; OTH 98.9%; history `C` 98.8%; no response; validation windows saturate at 50 | Completed HTTP flood is structurally wrong. The bounded candidate uses checksum-invalid ACK-only packets on a fixed private target; VM TX/GSO/TSO offload is disabled so checksum-aware Zeek records `OTH`/history `C` without weakening validation for ordinary traffic. |
| Okiru | 1,000 | TCP/37215, unknown service, S0/SYN-only all 100%; one packet, no response, multiple destinations | Requires bounded incomplete connections to multiple fixed lab identities. |
| PartOfAHorizontalPortScan | 4,012 | TCP 100%; S0/SYN-only 99.4%; multiple destinations and usually one port; ports 22/23 dominate | A six-port probe against one host is vertical and does not represent this class. |

Exact categorical distributions, numeric quantiles, missingness, topology, and
rolling evaluator observables are generated locally at
`artifacts/application/traffic-profiles/iot23-validation-v1/reference.json`.

## Scientific acceptance

Profile selection is control-plane metadata and is forbidden from the
production inference request. It cannot select a head, predicted label,
confidence, alert, or graph amplitude.

An executable profile remains a candidate until its Zeek-observed flows pass a
frozen comparison against the validation reference. The comparison will use
the model's exact preprocessed feature order, a deterministic same-class
bootstrap envelope at the candidate sample size, and a nearest-reference check.
Failures remain visible and must not be relabeled for the demo.

The executable catalog exposes a bounded event count and inter-event interval
for every profile. Per-profile limits are server-owned and the complete
scheduled duration cannot exceed two minutes. These controls vary traffic
density only; they cannot modify the fixed target, port, payload, model route,
head, or expected class.

Two claims remain separate:

1. **Model-equivalence:** generated traffic resembles the validation rolling
   view consumed by this exact model.
2. **Natural-timing equivalence:** generated traffic reproduces the original
   malware arrival process.

Phase 4 targets only model-equivalence. Natural-timing equivalence would require
a separate contiguous train-only reconstruction from the official
digest-pinned sources.
