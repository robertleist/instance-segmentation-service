from typing import Union

from fastapi import HTTPException
from iquana_toolbox.schemas.networking.http.services import InstanceSegmentationRequest
from iquana_toolbox.schemas.training import InstanceSegmentationTrainingRequest
from app.state import MODEL_REGISTRY


def validate_model(request: Union[InstanceSegmentationRequest, InstanceSegmentationTrainingRequest]):
    # Load a model like this. Note that this caches the loading, too!
    model_info = MODEL_REGISTRY.get_model_info(request.model_registry_key)
    if model_info["task"] != "instance-segmentation":
        raise HTTPException(status_code=400,
                            detail=f"Model {model_info["name"]} is not an instance segmentation model.")
    # Check whether the model can predict the class
    if not request.label.name in model_info["label"]:
        raise HTTPException(status_code=400,
                            detail=f"Model {model_info["name"]} predicts {model_info['label']} not requested label {request.label.name}.")
    if type(request) == InstanceSegmentationTrainingRequest and not model_info["trainable"]:
        raise HTTPException(status_code=400,
                            detail=f"Model {model_info["name"]} is not trainable!")
