from fastapi import APIRouter, HTTPException
from iquana_toolbox.schemas.database.contours import Contour

from app.state import MODEL_REGISTRY
from iquana_toolbox.schemas.networking.http.services import InstanceSegmentationRequest

from models.base_model import BaseInstanceSegmentationModel
from util.validate_model import validate_model

router = APIRouter()

@router.post("/inference")
async def inference(request: InstanceSegmentationRequest) -> list[Contour]:
    """
        Load a model from mlflow registry and perform inference on the provided image.
        This is a placeholder implementation. The actual logic will depend on the model and inference requirements.
    """
    # Validates the model selection
    validate_model(request)
    model: BaseInstanceSegmentationModel = MODEL_REGISTRY.get_model_by_alias(request.model_registry_key, "latest")
    return model.inference(request)
