import mlflow
from iquana_toolbox.ai.base_classes import InstanceSegmentationModel

from celery_app import app
import logging
from iquana_toolbox.mlflow import MLFlowModelRegistry
from paths import MLFLOW_URL
from iquana_toolbox.schemas.training import InstanceSegmentationTrainingRequest

logger = logging.getLogger(__name__)

@app.task(bind=True)
def train_and_register_model(self, request_dict: dict):
    """
    Generic training dispatcher. Loads the model from the registry and
    delegates all training logic to the model's own train() method.
    """
    try:
        registry: MLFlowModelRegistry = MLFlowModelRegistry(MLFLOW_URL)

        # Reconstruct the typed request inside the worker
        request = InstanceSegmentationTrainingRequest.model_validate(request_dict)
        model: InstanceSegmentationModel = registry.get_model_by_alias(request.model_registry_key, "latest")._model_impl

        self.update_state(state='PROGRESS', meta={'status': 'training started'})
        with mlflow.start_run(run_id=self.id):
            model.train(request)
            # You need to specify the dataset_id and user_id or else the model does not get logged.
            model.model_info.tags["dataset_id"] = request.dataset_id
            model.model_info.tags["user_id"] = request.user_id
            registry.register_model(model)

        return {"status": "completed"}
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise self.retry(countdown=60, max_retries=3)