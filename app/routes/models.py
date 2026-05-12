from logging import getLogger

from fastapi import APIRouter

from app.state import MODEL_REGISTRY

logger = getLogger(__name__)
session_router = APIRouter(prefix="/annotation_session", tags=["annotation_session"])
router = APIRouter()


@router.get("/models/all")
async def list_models():
    """ Lists all available models in the registry. """
    available_models = MODEL_REGISTRY.get_models_via_tags(tags={
        "task": "instance-segmentation",
    })
    return {
        "success": True,
        "message": f"Retrieved {len(available_models)} available models.",
        "result": available_models}


@router.get("/models/all/available")
async def list_models():
    """ Lists all available models in the registry. """
    available_models = MODEL_REGISTRY.get_models_via_tags(tags={
        "task": "instance-segmentation",
        "status": "ready"
    })
    return {
        "success": True,
        "message": f"Retrieved {len(available_models)} available models.",
        "result": available_models}


@router.get("/models/{model_registry_key}")
async def get_model(model_registry_key: str):
    model_info = MODEL_REGISTRY.get_model_info(model_registry_key)
    return {
        "success": True,
        "message": "Retrieved model information.",
        "result": model_info
    }


@session_router.get("/models/{model_registry_key}/preload")
async def load_model(model_registry_key: str, user_id: str):
    """ Loads a model into the cache if not already loaded. This is a convenience endpoint; models are loaded
        automatically when needed, but this can be called at the start
        of an annotation session to preload the model."""
    MODEL_REGISTRY.get_model_by_alias(model_registry_key, "latest")
    return {
        "success": True,
        "message": f"Loaded {model_registry_key} model information.",
    }
