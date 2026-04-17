import logging
from dataclasses import dataclass
from os import getenv
import secrets

from iquana_toolbox.mlflow import MLFlowModelRegistry

from celery_app import app as celery_app
from paths import MLFLOW_URL

logger = logging.getLogger(__name__)

INSTANCE_SEGMENTATION_TASK_TAG = "instance-segmentation"
SERVICE_REGISTRATION_TOKEN = getenv(
    "SERVICE_REGISTRATION_TOKEN",
    getenv("SERVICE_SECRET", "default-secret"),
)


@dataclass
class Backend:
    backend_address: str
    celery_broker_url: str
    mlflow_tracking_uri: str
    service_name: str | None = None


_backend: Backend | None = None
_backend_token: str | None = None

MODEL_REGISTRY = MLFlowModelRegistry(MLFLOW_URL)


def validate_registration_token(registration_token: str) -> bool:
    return secrets.compare_digest(registration_token, SERVICE_REGISTRATION_TOKEN)


def update_celery_config(celery_broker_url: str) -> None:
    celery_app.conf.broker_url = celery_broker_url
    celery_app.conf.result_backend = celery_broker_url


def update_model_registry(tracking_uri: str) -> None:
    global MODEL_REGISTRY
    MODEL_REGISTRY = MLFlowModelRegistry(tracking_uri)


def register_backend(backend: Backend) -> str:
    global _backend, _backend_token
    _backend = backend
    _backend_token = secrets.token_urlsafe(32)
    return _backend_token


def get_backend() -> Backend | None:
    return _backend


def get_backend_token() -> str | None:
    return _backend_token


def get_instance_segmentation_model_names() -> set[str]:
    models = MODEL_REGISTRY.get_models_via_tags({"task": INSTANCE_SEGMENTATION_TASK_TAG})
    return {model["name"] for model in models}


def get_model_registry() -> MLFlowModelRegistry:
    return MODEL_REGISTRY
