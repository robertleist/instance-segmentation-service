import mlflow

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
        model = registry.get_model_by_alias(request.model_registry_key, "latest")

        # Copy tags from the existing model and add new ones for this training run
        # Note: Tags only get added when the training finishes.
        old_tags = model.tags
        new_tags = old_tags.copy()
        new_tags["dataset_id"] = request.dataset_id
        new_tags["created_by"] = request.user_id
        new_tags["label"] = request.label

        self.update_state(state='PROGRESS', meta={'status': 'training started'})
        with mlflow.start_run(run_id=self.id):
            model.train(request)
            new_model = mlflow.pyfunc.log_model(
                python_model=self,
                artifact_path="model",
                registered_model_name=request.model_registry_key,
                tags=new_tags,
            )

        return {"status": "completed", "model": new_model.model_id}
    except Exception as e:
        logger.error(f"Training failed for {model_registry_key}: {e}")
        raise self.retry(countdown=60, max_retries=3)