import logging

from celery.result import AsyncResult
from fastapi import APIRouter, Body, HTTPException

from app.state import MODEL_REGISTRY
from app.tasks import train_model
from iquana_toolbox.schemas.training import InstanceSegmentationTrainingRequest

from util.validate_model import validate_model

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/train")
async def start_training(
    request: InstanceSegmentationTrainingRequest
):
    """Start a training job asynchronously. Delegates the training tasks to Celery workers."""
    validate_model(request)
    task = train_model.delay(
        model_registry_key=request.model_registry_key,
        request_dict=request.model_dump(),  # serialize to dict for Celery/Redis
    )
    return {"task_id": task.id}


@router.delete("/train/{task_id}")
async def cancel_training(task_id: str):
    """Cancel a training job. This requires that the Celery worker supports task revocation and that the training task checks for revocation status."""
    task = AsyncResult(task_id)
    task.revoke(terminate=True)
    return {"message": "Training cancelled"}