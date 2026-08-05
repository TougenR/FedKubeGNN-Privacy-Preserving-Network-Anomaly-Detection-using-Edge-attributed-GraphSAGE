# Validation-Only In-Process FedAvg Control

This control reproduces the frozen `30 rounds × 5 local epochs` local-weight
FedAvg configuration in the same runner used for Phase 3A ablations.

- Best validation macro-F1: `0.455726` at round 25
- Flower baseline validation macro-F1: `0.455775`
- Absolute runner difference: `0.000049`
- Test evaluations: `0`

The close match supports same-runner causal comparisons for the isolated class
weight and local-epoch experiments.
