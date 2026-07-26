"""Strategy policies independent of a concrete runtime."""

from src.federated.strategies.fedavg import FedAvgPolicy
from src.federated.strategies.fedprox import FedProxPolicy

__all__ = ["FedAvgPolicy", "FedProxPolicy"]
