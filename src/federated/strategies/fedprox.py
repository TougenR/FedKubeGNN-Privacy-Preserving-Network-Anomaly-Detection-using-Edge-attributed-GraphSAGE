from dataclasses import dataclass


@dataclass(frozen=True)
class FedProxPolicy:
    proximal_mu: float = 0.01
    name: str = "fedprox"

    def __post_init__(self) -> None:
        if self.proximal_mu <= 0:
            raise ValueError("FedProx proximal_mu must be > 0.")
