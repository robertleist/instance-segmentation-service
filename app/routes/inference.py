from fastapi import APIRouter
from app.state import MODEL_REGISTRY
from iquana_toolbox.schemas.networking.http.services import BaseServiceRequest


router = APIRouter()

@router.post("/inference")
async def inference(request: BaseServiceRequest):
    """ Load a model from mlflow registry and perform inference on the provided image. This is a placeholder implementation. The actual logic will depend on the model and inference requirements."""
    # Load a model like this. Note that this caches the loading, too!
    model = MODEL_REGISTRY.get_model(request.model_registry_key)
    # TODO: Implement your inference!
    raise NotImplementedError("Inference endpoint not implemented yet")

