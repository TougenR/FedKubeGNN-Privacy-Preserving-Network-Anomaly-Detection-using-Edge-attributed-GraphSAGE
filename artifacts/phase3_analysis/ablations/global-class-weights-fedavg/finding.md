# Global Class-Weight FedAvg Ablation

Decision: **rejected**.

- Dataset digest: `68fc6fc0cb8974aba1d431113b39dbf82f98457159c04d6a14b22feaa4b0cb89`
- Model digest: `0c1faa97cde18b330cc5e1f565a1f80fe3fce4d326525ca1c148712808fa2004`
- Config digest: `daa37bee2a26d4c36ac55a1243d6c9005113d73a921b7fd79c4e9ca2b7f75aa8`
- Protocol: FedAvg, 30 rounds, six clients, five local epochs, Adam 0.001
- Selection data: validation only; `test_evaluations=0`

The best validation fixed-eight macro-F1 is `0.380236` at round 30, versus
the same-runner local-weight FedAvg control `0.455726`. The isolated change
reduces validation macro-F1 by `0.075489`.

The union class weights are
`[0.416477, 2.287667, 1.840433, 1.518357, 1.518357, 1.518357, 5314.25,
0.378440]`. The extreme Okiru-Attack value results from only two training
examples. C&C-HeartBeat, DDoS, and Okiru still have zero validation F1, so
global inverse-frequency weighting alone does not overcome private-class update
suppression during example-weighted aggregation.

This configuration is not promoted and its test split remains unevaluated. The
next isolated experiment is local-epoch/client-drift control using the original
local class-weight baseline.
