import inspect
import json
import logging

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from iquana_toolbox.schemas.networking.http.services import InstanceSegmentationRequest
from mlflow.exceptions import RestException

from app.dependencies import get_current_backend
from app.state import (
    Backend,
    INSTANCE_SEGMENTATION_TASK_TAG,
    get_model_registry,
    is_valid_model_key,
)

logger = logging.getLogger(__name__)

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
    predictor = getattr(model, "predict", None)
    if predictor is None and callable(model):
        predictor = model
    if predictor is None:
        raise HTTPException(status_code=500, detail="Loaded model is not callable")

    signature = inspect.signature(predictor)
    accepts_varargs = any(
        parameter.kind == inspect.Parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    )
    predictor_parameter_count = len(signature.parameters)
    candidate_calls = [
        (2, lambda: predictor(request.image, request.label)),
        (1, lambda: predictor(request)),
        (1, lambda: predictor(request.image)),
    ]

    for required_parameters, execute in candidate_calls:
        if not accepts_varargs and required_parameters > predictor_parameter_count:
            continue
        try:
            result = execute()
            logger.debug("Inference executed with %d-argument predictor signature", required_parameters)
            return result
        except TypeError:
            continue

    raise HTTPException(status_code=500, detail="Model inference signature is unsupported")


def _get_latest_version(versions):
    if not versions:
        raise HTTPException(status_code=404, detail="No registered versions found for model")

    def sort_key(version):
        creation_timestamp = getattr(version, "creation_timestamp", 0) or 0
        try:
            version_number = int(str(version.version))
        except (TypeError, ValueError):
            version_number = 0
        return creation_timestamp, version_number

    return max(versions, key=sort_key)


@router.post("/inference")
async def inference(
    request: InstanceSegmentationRequest,
    backend: Backend = Depends(get_current_backend),
):
    if not is_valid_model_key(request.model_registry_key):
        raise HTTPException(status_code=400, detail="Invalid model registry key format")

    model_registry = get_model_registry()
    try:
        registered_model = model_registry.client.get_registered_model(request.model_registry_key)
    except RestException as exc:
        raise HTTPException(status_code=404, detail="Model not found in registry") from exc

    tags = getattr(registered_model, "tags", {})
    if tags.get("task") != INSTANCE_SEGMENTATION_TASK_TAG:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model_registry_key}' is not tagged with task:instance-segmentation",
        )

    versions = list(getattr(registered_model, "latest_versions", []))
    latest_version = _get_latest_version(versions)
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
