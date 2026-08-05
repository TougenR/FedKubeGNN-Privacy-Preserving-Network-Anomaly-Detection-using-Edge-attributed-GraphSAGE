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
    PersonalizedFederatedRunResult,
    merge_personalized_state,
    run_fedper_simulation,
    run_federated_simulation,
    split_personalized_state,
)

__all__ = [
    "FederatedRoundResult",
    "FederatedRunResult",
    "PersonalizedFederatedRunResult",
    "aggregate_confusion_matrices",
    "classification_metrics",
    "run_federated_simulation",
    "run_fedper_simulation",
    "split_personalized_state",
    "merge_personalized_state",
    "class_balanced_client_fedavg",
    "class_balanced_client_head_fedavg",
    "class_balanced_client_weights",
    "class_support_head_fedavg",
    "weighted_fedavg",
]
