import logging

from celery.result import AsyncResult
from fastapi import APIRouter, Body, HTTPException

from app.state import MODEL_REGISTRY
from app.tasks import train_model

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/train")
async def start_training(
    # TODO: Replace with your service-specific training request schema.
    request: dict = Body(...)
):
    """Start a training job asynchronously. Delegates the training tasks to Celery workers."""
    try:
        task = train_model.delay(
            request.get("model_id"),
            request.get("dataset_path"),
            request.get("params", {}),
            MODEL_REGISTRY.tracking_uri,
        )
        return {"task_id": task.id}
    except AttributeError as exc:
        logger.error("Training request schema mismatch: %s", exc)
        raise HTTPException(status_code=400, detail="Invalid training request payload") from exc
    except Exception as exc:
        logger.error("Failed to start training for backend '%s': %s", backend.backend_address, exc)
        raise HTTPException(status_code=500, detail="Failed to start training") from exc


@router.delete("/train/{task_id}")
async def cancel_training(task_id: str):
    """Cancel a training job. This requires that the Celery worker supports task revocation and that the training task checks for revocation status."""
    task = AsyncResult(task_id)
    task.revoke(terminate=True)
    return {"message": "Training cancelled"}