# Exact Prepared-Data Centralized Reference

- Run: `centralized-20260805T132411540234Z-ccb597b7e8`
- Epochs: 150 fixed epochs from the shared immutable initial state
- Dataset digest: `68fc6fc0cb8974aba1d431113b39dbf82f98457159c04d6a14b22feaa4b0cb89`
- Model digest: `0c1faa97cde18b330cc5e1f565a1f80fe3fce4d326525ca1c148712808fa2004`
- Duration: 92.445 seconds on the local CPU container

## Result

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 0.980567 | 0.982101 |
| Fixed-eight macro-F1 | 0.858650 | 0.869830 |
| Weighted-F1 | 0.981330 | 0.982595 |
| Loss | 0.087323 | 0.082986 |

The matching prepared-data centralized result is `0.413274` macro-F1 above
FedAvg and `0.413139` above FedProx. This isolates most of the observed loss to
the federated optimization/aggregation path rather than the prepared feature
representation alone.

Phase 1 clean seed 42 scored `0.928692`. Almost all of the apparent `0.058862`
centralized gap comes from the non-estimable three-row Okiru-Attack class:
Phase 1 happened to score `0.5` on its single test row, while this reference
scores `0.076923`. Excluding only that ultra-rare class for diagnosis (not for
the official fixed-eight metric), Phase 1 and this reference score `0.989933`
and `0.983102`, a difference of only `0.006831`.

This does not authorize dropping Okiru-Attack from official evaluation. It
shows why that one-row result must not be used for model selection and why the
first treatment experiments should target private-class gradient suppression,
local drift, and aggregation.
