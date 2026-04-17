import logging

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, HTTPException
from mlflow.exceptions import RestException
from pydantic import BaseModel, Field

from app.dependencies import get_current_backend
from app.state import Backend, INSTANCE_SEGMENTATION_TASK_TAG, get_model_registry, is_valid_model_key
from app.tasks import train_model

logger = logging.getLogger(__name__)

router = APIRouter()


class TrainingRequest(BaseModel):
    model_id: str
    dataset_path: str
    params: dict = Field(default_factory=dict)


@router.post("/train")
async def start_training(
    request: TrainingRequest,
    backend: Backend = Depends(get_current_backend),
):
    """Start a training job asynchronously via Celery."""
    try:
        if not is_valid_model_key(request.model_id):
            raise HTTPException(status_code=400, detail="Invalid model registry key format")
        model_registry = get_model_registry()
        try:
            registered_model = model_registry.client.get_registered_model(request.model_id)
        except RestException as exc:
            raise HTTPException(status_code=404, detail="Model not found in registry") from exc

        tags = getattr(registered_model, "tags", {})
        if tags.get("task") != INSTANCE_SEGMENTATION_TASK_TAG:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{request.model_id}' is not tagged with task:instance-segmentation",
            )
        task = train_model.delay(
            request.model_id,
            request.dataset_path,
            request.params,
            backend.mlflow_tracking_uri,
        )
        return {"task_id": task.id}
    except Exception as exc:
        logger.error("Failed to start training for backend '%s': %s", backend.backend_address, exc)
        raise HTTPException(status_code=500, detail="Failed to start training") from exc


@router.delete("/train/{task_id}")
async def cancel_training(task_id: str):
    """Cancel a training job. This requires that the Celery worker supports task revocation and that the training task checks for revocation status."""
    task = AsyncResult(task_id)
    task.revoke(terminate=True)
    return {"message": "Training cancelled"}
