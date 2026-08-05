# One-Local-Epoch FedAvg Ablation

Decision: **rejected**.

This experiment keeps the same total local epoch budget as the control:
`150 rounds × 1 local epoch` versus `30 rounds × 5 local epochs`. Local class
weights, Adam `0.001`, six-client participation, data, model, seed, and initial
state remain unchanged.

- Best validation macro-F1: `0.315973` at round 84
- Same-runner control: `0.455726` at round 25
- Delta: `-0.139753`
- Test evaluations: `0`

C&C, C&C-HeartBeat, DDoS, Okiru, and Okiru-Attack all have zero validation F1
at the selected round. More frequent aggregation with one local epoch does not
recover private classes and substantially reduces the validation objective.
The configuration is not promoted.
