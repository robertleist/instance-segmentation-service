import logging

from iquana_toolbox.mlflow import MLFlowModelRegistry

from paths import MLFLOW_URL

logger = logging.getLogger(__name__)

MODEL_REGISTRY = MLFlowModelRegistry(MLFLOW_URL)
