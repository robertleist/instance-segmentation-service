import logging

from iquana_service_core import create_service_app

from paths import SERVICE_NAME, SERVICE_DESCRIPTION, ALLOWED_ORIGINS
from app.state import MODEL_REGISTRY
from app.routes.inference import router as inference_router
from app.routes.training import router as training_router

logger = logging.getLogger(__name__)


def create_app():
    return create_service_app(
        title=SERVICE_NAME,
        description=SERVICE_DESCRIPTION,
        task="instance-segmentation",
        registry=MODEL_REGISTRY,
        models_package="models",
        inference_routers=[inference_router, training_router],
        hf_login=False,
        allowed_origins=ALLOWED_ORIGINS,
    )
