import json

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from iquana_toolbox.schemas.networking.http.services import InstanceSegmentationRequest

from app.dependencies import get_current_backend
from app.state import (
    Backend,
    INSTANCE_SEGMENTATION_TASK_TAG,
    get_instance_segmentation_model_names,
    get_model_registry,
)


router = APIRouter()


def _serialize_result(result):
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if isinstance(result, np.ndarray):
        return result.tolist()
    if isinstance(result, (dict, list, str, int, float, bool)) or result is None:
        return result
    try:
        json.dumps(result)
        return result
    except TypeError:
        return str(result)


def _run_inference(model, request: InstanceSegmentationRequest):
    if hasattr(model, "predict"):
        try:
            return model.predict(request)
        except TypeError:
            try:
                return model.predict(request.image, request.label)
            except TypeError:
                return model.predict(request.image)
    if callable(model):
        try:
            return model(request)
        except TypeError:
            return model(request.image)
    raise HTTPException(status_code=500, detail="Loaded model is not callable")


@router.post("/inference")
async def inference(
    request: InstanceSegmentationRequest,
    backend: Backend = Depends(get_current_backend),
):
    model_names = get_instance_segmentation_model_names()
    if request.model_registry_key not in model_names:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model_registry_key}' is not tagged with task:instance-segmentation",
        )

    model_registry = get_model_registry()
    versions = list(model_registry.client.search_model_versions(f"name='{request.model_registry_key}'"))
    if not versions:
        raise HTTPException(status_code=404, detail="No registered versions found for model")
    latest_version = max(versions, key=lambda version: int(version.version))
    model = model_registry.get_model_by_version(request.model_registry_key, latest_version.version)

    result = _run_inference(model, request)
    return {
        "model_registry_key": request.model_registry_key,
        "model_version": latest_version.version,
        "mlflow_tracking_uri": backend.mlflow_tracking_uri,
        "prediction": _serialize_result(result),
    }


@router.get("/models")
async def get_available_models(backend: Backend = Depends(get_current_backend)):
    return {
        "task_tag": INSTANCE_SEGMENTATION_TASK_TAG,
        "mlflow_tracking_uri": backend.mlflow_tracking_uri,
        "models": get_model_registry().get_models_via_tags({"task": INSTANCE_SEGMENTATION_TASK_TAG}),
    }
