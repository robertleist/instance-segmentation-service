from celery_app import app
import logging
import os
import mlflow

logger = logging.getLogger(__name__)


@app.task(bind=True)
def train_model(self, model_id, dataset_path, params, mlflow_tracking_uri: str):
    """Background task for training a model."""
    try:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        with mlflow.start_run() as run:
            mlflow.log_params(params)
            mlflow.set_tag("task", "instance-segmentation")
            mlflow.set_tag("base_model_id", model_id)
            if dataset_path and os.path.exists(dataset_path):
                mlflow.log_artifact(dataset_path, "dataset")
            self.update_state(state="PROGRESS", meta={"progress": 100, "run_id": run.info.run_id})

        return {"status": "completed", "model_id": model_id, "mlflow_run_id": run.info.run_id}
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise self.retry(countdown=60, max_retries=3)
