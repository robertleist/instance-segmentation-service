import logging
from typing import Optional

import mlflow
import numpy as np
import torch
from iquana_toolbox.ai.dataloaders import get_coco_instance_segmentation_dataset
from iquana_toolbox.mlflow import MLFlowModelRegistry
from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.networking.http.services import InstanceSegmentationRequest
from iquana_toolbox.schemas.training import InstanceSegmentationTrainingRequest
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    Mask2FormerConfig,
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
)

from models.base_model import BaseInstanceSegmentationModel

logger = logging.getLogger(__name__)

# Lightest HuggingFace Mask2Former variant for instance segmentation.
# Swap for e.g. "facebook/mask2former-swin-base-coco-instance" for higher accuracy.
DEFAULT_HF_MODEL = "facebook/mask2former-swin-tiny-coco-instance"


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class Mask2FormerSegmentationModel(BaseInstanceSegmentationModel):
    """
    Instance segmentation wrapper backed by Mask2Former (HuggingFace Transformers).

    This is an integration wrapper — NOT a bare nn.Module.  It owns the full
    inference and training pipelines, including data fetching, fine-tuning, and
    MLflow logging.

    Each instance is scoped to a **single label class** (one model per label):
    - Inference: all detected masks are mapped to `request.label`.
    - Training: ground-truth instances are always stored under class index 0.

    Loading
    -------
    ``model_name_or_path`` accepts anything ``from_pretrained`` understands:
    - A HuggingFace Hub id  →  auto-downloads weights (requires internet the first time).
    - A local directory  →  where a previous ``save_pretrained`` was written, e.g. an
      MLflow artifact path fetched by the model registry.

    Training input contract
    -----------------------
    ``request.gt_instances[i]`` must be a URL that returns a **JSON list of
    Contour dicts** for the image at ``request.image_urls[i]``:

        [{"x": [0.1, ...], "y": [0.3, ...], "label_id": 5, ...}, ...]

    Coordinates in Contour are already normalised to [0, 1].
    """

    DEFAULT_HYPERPARAMETERS: dict = {
        "epochs": 10,
        "batch_size": 2,
        "lr": 1e-5,
        "weight_decay": 1e-4,
    }

    def __init__(
            self,
            model_name_or_path: str = DEFAULT_HF_MODEL,
            mlflow_tracking_uri: str = "http://localhost:5000",
            model_name: str = "mask2former-seg",
            device: Optional[str] = None,
    ):
        """
        Args:
            model_name_or_path:    HuggingFace Hub id or local directory with
                                   saved model + processor weights.
            mlflow_tracking_uri:   MLflow server for logging trained runs.
            model_name:            Registered model name in MLflow.
            device:                "cuda", "cpu", or None (auto-detect).
        """
        self.model_name_or_path = model_name_or_path
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = Mask2FormerImageProcessor.from_pretrained(
            model_name_or_path,
            ignore_mismatched_sizes=True,
        )

        # Reinitialise the classification head for a single class so pretrained
        # COCO weights are used for the backbone + pixel decoder, but the
        # classifier is fresh — correct for domain-specific fine-tuning.
        config = Mask2FormerConfig.from_pretrained(model_name_or_path)
        config.num_labels = 1
        self.hf_model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_name_or_path,
            config=config,
            ignore_mismatched_sizes=True,
        ).to(self.device)

        logger.info("Loaded Mask2Former from '%s' on %s", model_name_or_path, self.device)

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------

    def inference(self, request: InstanceSegmentationRequest) -> list[Contour]:
        """
        Run instance segmentation on a single image.

        ``request.image`` is a cached np.ndarray fetched from ``request.image_url``.
        Each detected mask is converted to a normalised Contour (x/y ∈ [0, 1]).
        """
        image: np.ndarray = request.image  # (H, W, 3) uint8 RGB
        h, w = image.shape[:2]

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        self.hf_model.eval()
        with torch.no_grad():
            outputs = self.hf_model(**inputs)

        # post_process_instance_segmentation returns a list, one entry per image
        result = self.processor.post_process_instance_segmentation(
            outputs,
            target_sizes=[(h, w)],
            threshold=0.5,
        )[0]

        # segmentation: (H, W) int tensor — each unique value is one instance id
        # segments_info: [{"id": int, "label_id": int, "score": float}, ...]
        segmentation_map = result["segmentation"].cpu().numpy()

        contours: list[Contour] = []
        for seg in result["segments_info"]:
            binary_mask = (segmentation_map == seg["id"]).astype(np.uint8)
            contour = Contour.from_binary_mask(
                binary_mask=binary_mask.astype(bool),
                only_return_biggest_contour=True,
                label_id=request.label.id,
                confidence=float(seg.get("score", 1.0)),
                added_by=str(request.user_id),
            )
            contours.append(contour)

        return contours

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def train(self, request: InstanceSegmentationTrainingRequest) -> None:
        """
        Full training pipeline — runs synchronously inside a Celery worker.

        Steps:
            1. Load COCO dataset from image folder and annotations.
            2. Split into train/val.
            3. Fine-tune Mask2Former with AdamW optimizer.
            4. Log trained weights and metrics to MLflow.
            5. Register model version in MLflow model registry.
        """
        if not request.image_folder_path:
            raise ValueError("image_folder_path is required for training.")
        if not request.annotation_file_url:
            raise ValueError("annotation_file_url is required for training.")

        params = {**self.DEFAULT_HYPERPARAMETERS, **request.hyper_parameter}

        logger.info(
            "Loading COCO dataset from '%s' with annotations '%s'…",
            request.image_folder_path,
            request.annotation_file_url,
        )

        # Load dataset using iquana-toolbox helper
        dataset = get_coco_instance_segmentation_dataset(
            image_folder=request.image_folder_path,
            annotation_file=request.annotation_file_url,
        )

        # Train model
        final_loss = self._train(dataset, params)

        logger.info("Fine-tuning complete (final loss: %.4f). Logging to MLflow…", final_loss)


    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------

    def _train(
            self,
            train_dataset: torch.utils.data.Dataset,
            params: dict,
    ) -> float:
        """
        Fine-tune Mask2Former on the training dataset.
        Returns the average loss of the last epoch.
        """
        mlflow.log_params(params)
        mlflow.log_param("num_samples", len(train_dataset))
        loader = DataLoader(
            train_dataset,
            batch_size=params["batch_size"],
            shuffle=True,
            collate_fn=self._collate_batch,
        )

        optimizer = AdamW(
            self.hf_model.parameters(),
            lr=params["lr"],
            weight_decay=params["weight_decay"],
        )

        last_loss = 0.0
        self.hf_model.train()

        for epoch in range(params["epochs"]):
            epoch_loss = 0.0
            for batch_idx, batch in enumerate(loader):
                # Extract images and annotations from batch
                images = batch["images"]  # list of np.ndarray (H, W, 3)
                annotations = batch["annotations"]  # list of dicts with "masks" and "labels"

                # Process inputs and annotations
                inputs = self.processor(
                    images=images,
                    annotations=annotations,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                optimizer.zero_grad()
                outputs = self.hf_model(**inputs)
                outputs.loss.backward()
                optimizer.step()

                epoch_loss += outputs.loss.item()

                if (batch_idx + 1) % max(1, len(loader) // 5) == 0:
                    logger.info(
                        "Epoch %d/%d, batch %d/%d — loss: %.4f",
                        epoch + 1,
                        params["epochs"],
                        batch_idx + 1,
                        len(loader),
                        outputs.loss.item(),
                    )

            last_loss = epoch_loss / max(len(loader), 1)
            mlflow.log_metric("loss", last_loss)
            logger.info("Epoch %d/%d — avg loss: %.4f", epoch + 1, params["epochs"], last_loss)

        self.hf_model.eval()
        return last_loss


    # -----------------------------------------------------------------------
    # Batch processing helper
    # -----------------------------------------------------------------------

    @staticmethod
    def _collate_batch(batch: list[dict]) -> dict:
        """
        Collate batch of COCO dataset items into images and annotations.
        Expected batch items: {"image": np.ndarray, "annotations": list[dict]}
        """
        images = [item["image"] for item in batch]
        annotations = [item["annotations"] for item in batch]
        return {"images": images, "annotations": annotations}
