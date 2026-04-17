import logging

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.dependencies import get_current_backend
from app.state import Backend, get_instance_segmentation_model_names
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
        if request.model_id not in get_instance_segmentation_model_names():
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
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to start training for backend '%s': %s", backend.backend_address, exc)
        raise HTTPException(status_code=500, detail="Failed to start training") from exc


@router.delete("/train/{task_id}")
async def cancel_training(task_id: str):
    """Cancel a training job. This requires that the Celery worker supports task revocation and that the training task checks for revocation status."""
    task = AsyncResult(task_id)
    task.revoke(terminate=True)
    return {"message": "Training cancelled"}
