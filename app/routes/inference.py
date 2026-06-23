from fastapi import APIRouter, HTTPException
from iquana_toolbox.schemas.database.contours import Contour

from app.state import MODEL_REGISTRY
from iquana_toolbox.schemas.networking.http.services import InstanceSegmentationRequest

from iquana_toolbox.ai.base_classes import InstanceSegmentationModel
from util.validate_model import validate_model

router = APIRouter()
session_router = APIRouter(prefix="/annotation_session", tags=["annotation_session"])


@router.post("/inference")
async def inference(request: InstanceSegmentationRequest) -> list[Contour]:
    """
        Load a model from mlflow registry and perform inference on the provided image.
        This is a placeholder implementation. The actual logic will depend on the model and inference requirements.
    """
    # Validates the model selection
    validate_model(request)
    model = MODEL_REGISTRY.get_model_by_alias(request.model_registry_key, "latest")
    return model.predict(request)


@session_router.post("/run")
async def run_inference(request: InstanceSegmentationRequest):
    """Run instance segmentation for an annotation session.

    Returns the detected instances wrapped in the ``{success, message, result}``
    envelope the annotation gateway expects from every session backend.
    """
    validate_model(request)
    model = MODEL_REGISTRY.get_model_by_alias(request.model_registry_key, "latest")
    contours = model.predict(request)
    return {
        "success": True,
        "message": f"Detected {len(contours)} instances for user {request.user_id}",
        "result": contours,
    }
