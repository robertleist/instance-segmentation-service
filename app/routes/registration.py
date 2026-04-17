from fastapi import APIRouter, HTTPException
from iquana_toolbox.schemas.networking.http.services import ServiceRegistrationRequest

from app.state import (
    Backend,
    register_backend,
    update_celery_config,
    update_model_registry,
    validate_registration_token,
)

router = APIRouter()


@router.post("/register")
async def register_service(request: ServiceRegistrationRequest):
    if not validate_registration_token(request.registration_token):
        raise HTTPException(status_code=401, detail="Invalid registration token")

    update_celery_config(request.celery_broker_url)
    update_model_registry(request.mlflow_tracking_uri)

    backend = Backend(
        backend_address=request.backend_url,
        celery_broker_url=request.celery_broker_url,
        mlflow_tracking_uri=request.mlflow_tracking_uri,
        service_name=request.service_name,
    )
    backend_token = register_backend(backend)

    return {
        "success": True,
        "message": "Service registered successfully",
        "service_id": request.service_name,
        "backend_token": backend_token,
    }
