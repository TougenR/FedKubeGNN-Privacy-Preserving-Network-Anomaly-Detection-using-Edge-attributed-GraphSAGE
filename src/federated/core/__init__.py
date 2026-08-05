"""Framework-independent federated algorithms and evaluation."""

from src.federated.core.aggregation import (
    class_balanced_client_fedavg,
    class_balanced_client_head_fedavg,
    class_balanced_client_weights,
    class_support_head_fedavg,
    weighted_fedavg,
)
from src.federated.core.metrics import (
    aggregate_confusion_matrices,
    classification_metrics,
)
from src.federated.core.simulation import (
    FederatedRunResult,
    FederatedRoundResult,
    run_federated_simulation,
)

__all__ = [
    "FederatedRoundResult",
    "FederatedRunResult",
    "aggregate_confusion_matrices",
    "classification_metrics",
    "run_federated_simulation",
    "class_balanced_client_fedavg",
    "class_balanced_client_head_fedavg",
    "class_balanced_client_weights",
    "class_support_head_fedavg",
    "weighted_fedavg",
]
