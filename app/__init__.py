import logging

from iquana_service_core import create_service_app

from paths import SERVICE_NAME, SERVICE_DESCRIPTION, ALLOWED_ORIGINS
from app.state import MODEL_REGISTRY
from app.routes.inference import router as inference_router
from app.routes.training import router as training_router
from util.registry_util import _PENDING_REGISTRATIONS

logger = logging.getLogger(__name__)


def _register_models(registry) -> None:
    """Register every class collected via the @register_base_model decorator.

    The model modules are imported as a side effect of importing the service
    routers above, which populates ``_PENDING_REGISTRATIONS``.
    """
    registry.register_models(_PENDING_REGISTRATIONS)


def create_app():
    return create_service_app(
        title=SERVICE_NAME,
        description=SERVICE_DESCRIPTION,
        task="instance-segmentation",
        registry=MODEL_REGISTRY,
        register_models=_register_models,
        inference_routers=[inference_router, training_router],
        hf_login=False,
        allowed_origins=ALLOWED_ORIGINS,
    )
