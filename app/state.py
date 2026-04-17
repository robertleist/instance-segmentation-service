import logging
from dataclasses import dataclass
from os import getenv
import re
import secrets

from iquana_toolbox.mlflow import MLFlowModelRegistry

from celery_app import app as celery_app
from paths import MLFLOW_URL

logger = logging.getLogger(__name__)

INSTANCE_SEGMENTATION_TASK_TAG = "instance-segmentation"
_raw_registration_token = getenv("SERVICE_REGISTRATION_TOKEN")
SERVICE_REGISTRATION_TOKEN = _raw_registration_token.strip() if _raw_registration_token else None
if _raw_registration_token is not None and not SERVICE_REGISTRATION_TOKEN:
    logger.warning("SERVICE_REGISTRATION_TOKEN is empty; registration will be rejected")
MODEL_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


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
    if not SERVICE_REGISTRATION_TOKEN:
        logger.error("SERVICE_REGISTRATION_TOKEN is not configured")
        return False
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


def get_model_registry() -> MLFlowModelRegistry:
    return MODEL_REGISTRY


def is_valid_model_key(model_key: str) -> bool:
    return bool(MODEL_KEY_PATTERN.fullmatch(model_key))
