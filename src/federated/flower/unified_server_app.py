from src.federated.flower.server_app import build_server_app
from src.federated.flower.task_factory import flower_observer, task_factory

app = build_server_app(task_factory, observer_factory=flower_observer)
