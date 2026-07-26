from src.federated.flower.client_app import build_client_app
from src.federated.flower.task_factory import flower_observer, task_factory

app = build_client_app(task_factory, observer_factory=flower_observer)
