from dataclasses import dataclass


@dataclass(frozen=True)
class FedAvgPolicy:
    name: str = "fedavg"
    proximal_mu: float = 0.0
