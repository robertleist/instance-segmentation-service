import json
import logging
import os
from typing import Optional, Any

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from iquana_toolbox.ai.dataloaders import get_coco_instance_segmentation_dataset
from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.database.labels import Label
from iquana_toolbox.schemas.networking.http.services import InstanceSegmentationRequest
from iquana_toolbox.schemas.training import InstanceSegmentationTrainingRequest, HyperParameter
from iquana_toolbox.ai.base_classes import InstanceSegmentationModel, InstanceSegmentationModelInfo
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    Mask2FormerConfig,
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
)

from iquana_service_core import register_model

logger = logging.getLogger(__name__)

# Lightest HuggingFace Mask2Former variant for instance segmentation.
# Swap for e.g. "facebook/mask2former-swin-base-coco-instance" for higher accuracy.
DEFAULT_HF_MODEL = "facebook/mask2former-swin-tiny-coco-instance"


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

@register_model
class Mask2Former(InstanceSegmentationModel):
    """
    Multiclass instance segmentation wrapper backed by Mask2Former (HuggingFace).

    This is an integration wrapper — NOT a bare nn.Module. It owns the full
    inference and training pipelines, including data loading, fine-tuning, and
    MLflow logging.

    Multiclass
    ----------
    A trained model predicts every label it was trained on (``request.labels``).
    The class-index <-> label mapping is persisted alongside the weights (a
    ``label_mapping.json`` artifact) so inference can resolve a predicted class
    index back to the real dataset :class:`Label` (id, name, ...). Single-class
    training is just the edge case of a one-element ``labels`` list.

    Nested instances
    ----------------
    Mask2Former predicts a flat set of (mask, class); it has no hierarchy concept.
    Training masks are built per-instance and may overlap, so nested/contained
    instances are kept. At inference, per-query binary masks are decoded
    (``return_binary_maps=True``) so overlapping/nested predictions survive; the
    parent/child structure is reconstructed downstream by geometric containment.
    """
    model_info = InstanceSegmentationModelInfo(
        registry_key="mask2former",
        name="Mask2Former",
        description=(
            "The Mask2Former model was proposed in Masked-attention Mask Transformer for Universal Image Segmentation by"
            " Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, Rohit Girdhar. Mask2Former is a "
            "unified framework for panoptic, instance and semantic segmentation and features significant performance and "
            "efficiency improvements over MaskFormer."
        ),
        usage_tip="Works well with general domain images. Trains one multiclass model over the dataset's labels.",
        info_url=r"https://huggingface.co/docs/transformers/model_doc/mask2former",
        tags={
            "task": "instance-segmentation",
            "status": "ready",
            "trainable": "true",
            "domain": "general",
            "publisher": "meta",
        },
        badges=["fast", "pretrained"],
        trainable=True,
        label_ids=[],
        training_parameters=[
            HyperParameter(key="epochs", label="Epochs", default_value=10, type="int",
                           description="Number of passes over the dataset.",
                           min_value=1, max_value=100, step=1),
            HyperParameter(key="batch_size", label="Batch size", default_value=2, type="int",
                           description="Images per training step.",
                           min_value=1, max_value=16, step=1),
            HyperParameter(key="lr", label="Learning rate", default_value=1e-5, type="float",
                           description="AdamW learning rate."),
            HyperParameter(key="weight_decay", label="Weight decay", default_value=1e-4, type="float",
                           description="AdamW weight decay."),
            HyperParameter(key="image_size", label="Image size", default_value=512, type="int",
                           description="Square size images are resized to for training.",
                           options=[256, 512, 768, 1024]),
        ],
    )
    default_hyperparameters: dict = {
        "epochs": 10,
        "batch_size": 2,
        "lr": 1e-5,
        "weight_decay": 1e-4,
        "image_size": 512,
    }

    # Live HF objects can't be cloudpickled (transformers attaches ContextVar-backed
    # forward hooks). They are stripped from the pickle and rebuilt in ``load_context``
    # from the ``hf_weights`` artifact written by ``get_artifacts``.
    _unpicklable_attrs = ("hf_model", "processor")

    def __init__(
            self,
            model_name_or_path: str = DEFAULT_HF_MODEL,
            device: Optional[str] = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # class index -> Label, populated on train() / restored in load_context().
        self.idx_to_label: dict[int, Label] = {}

        # We don't initialize hf_model or processor here if we are loading via MLflow
        # because load_context will handle it.
        if model_name_or_path:
            self._setup_model(model_name_or_path)

    def _setup_model(
            self,
            path: str,
            num_labels: Optional[int] = None,
            id2label: Optional[dict] = None,
            label2id: Optional[dict] = None,
    ):
        """Initialize HF components from a path.

        ``num_labels`` (with optional ``id2label``/``label2id``) rebuilds the
        classification head for a fresh number of classes — used when fine-tuning a
        base checkpoint for the dataset's labels. When loading a previously
        fine-tuned model, pass nothing so the saved config (and its class count) is
        kept.
        """
        self.processor = Mask2FormerImageProcessor.from_pretrained(path)
        config = Mask2FormerConfig.from_pretrained(path)
        if num_labels is not None:
            config.num_labels = num_labels
            if id2label is not None:
                config.id2label = id2label
            if label2id is not None:
                config.label2id = label2id
        self.hf_model = Mask2FormerForUniversalSegmentation.from_pretrained(
            path, config=config, ignore_mismatched_sizes=True
        ).to(self.device)

    def get_artifacts(self, tmp_dir: str) -> dict[str, str]:
        """Persist the fine-tuned HF weights + processor and the class->label mapping.

        Called by the registry at log time; the returned paths are exposed back as
        ``context.artifacts[...]`` in ``load_context``.
        """
        weights_path = os.path.join(tmp_dir, "hf_weights")
        self.hf_model.save_pretrained(weights_path)
        self.processor.save_pretrained(weights_path)

        mapping_path = os.path.join(tmp_dir, "label_mapping.json")
        with open(mapping_path, "w", encoding="utf-8") as fp:
            json.dump({str(idx): label.model_dump() for idx, label in self.idx_to_label.items()}, fp)

        return {"hf_weights": weights_path, "label_mapping": mapping_path}

    def load_context(self, context):
        """Restore weights + class->label mapping. Called by MLflow at load time."""
        weights_path = context.artifacts["hf_weights"]
        logger.info(f"Loading weights from MLflow context: {weights_path}")
        self._setup_model(weights_path)

        mapping_path = context.artifacts.get("label_mapping")
        if mapping_path and os.path.exists(mapping_path):
            with open(mapping_path, encoding="utf-8") as fp:
                raw = json.load(fp)
            self.idx_to_label = {int(idx): Label(**data) for idx, data in raw.items()}

    # -- Inference ----------------------------------------------------------

    def predict(self,
                context: Any,
                model_input,
                params=None) -> list[Contour]:
        """
        Run multiclass instance segmentation on a single image.

        ``model_input`` is a single :class:`InstanceSegmentationRequest` (left
        untyped because MLflow's signature inference recurses infinitely on the
        recursive ``Label.children`` field at log time).

        Each detected instance is mapped, via the persisted class->label mapping,
        to the dataset :class:`Label` it represents. If ``request.label`` is set, only
        instances of that label are returned. Per-query masks are decoded so
        overlapping/nested instances are preserved.
        """
        request: InstanceSegmentationRequest = model_input
        image: np.ndarray = request.image  # (H, W, 3) uint8 RGB
        h, w = image.shape[:2]
        threshold = (params or {}).get("threshold", 0.5)

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        self.hf_model.eval()
        with torch.no_grad():
            outputs = self.hf_model(**inputs)

        result = self.processor.post_process_instance_segmentation(
            outputs,
            target_sizes=[(h, w)],
            threshold=threshold,
            return_binary_maps=True,
        )[0]

        segmentation = result["segmentation"]  # (num_instances, H, W) binary, or (H, W) fallback
        seg_np = segmentation.cpu().numpy() if hasattr(segmentation, "cpu") else np.asarray(segmentation)

        filter_label_id = request.label.id if request.label is not None else None

        contours: list[Contour] = []
        for i, seg in enumerate(result["segments_info"]):
            label = self.idx_to_label.get(int(seg["label_id"]))
            if label is None:
                continue
            if filter_label_id is not None and label.id != filter_label_id:
                continue

            if seg_np.ndim == 3:
                binary_mask = seg_np[i].astype(bool)
            else:
                binary_mask = (seg_np == seg["id"])

            contour = Contour.from_binary_mask(
                binary_mask=binary_mask,
                only_return_biggest_contour=True,
                label_id=label.id,
                confidence=float(seg.get("score", 1.0)),
                added_by=str(request.user_id),
            )
            contours.append(contour)

        return contours

    # -- Training -----------------------------------------------------------

    def train(self, request: InstanceSegmentationTrainingRequest, **kwargs) -> None:
        """
        Full training pipeline — runs synchronously inside a Celery worker.

        Steps:
            1. Build the class-index <-> label mapping from ``request.labels``.
            2. Rebuild the Mask2Former head for that number of classes.
            3. Load the COCO dataset and fine-tune with AdamW.
            4. Record the trained label set on ``model_info`` for registration.
        """
        if not request.image_folder_path:
            raise ValueError("image_folder_path is required for training.")
        if not request.annotation_file_url:
            raise ValueError("annotation_file_url is required for training.")
        if not request.labels:
            raise ValueError("At least one label is required for training.")

        params = {**self.default_hyperparameters, **request.hyper_parameter}

        # Contiguous class indices over the requested labels.
        label2idx = {label.id: idx for idx, label in enumerate(request.labels)}
        self.idx_to_label = {idx: label for idx, label in enumerate(request.labels)}
        id2label = {idx: label.name for idx, label in enumerate(request.labels)}
        label2id = {label.name: idx for idx, label in enumerate(request.labels)}

        logger.info(
            "Fine-tuning Mask2Former on %d label(s) from '%s' (annotations '%s')…",
            len(request.labels), request.image_folder_path, request.annotation_file_url,
        )

        # Rebuild the classification head for this dataset's labels.
        self._setup_model(self.model_name_or_path, num_labels=len(request.labels),
                          id2label=id2label, label2id=label2id)

        # Train at a fixed square size so per-instance masks align with pixel_values.
        size = int(params["image_size"])
        self.processor.do_resize = True
        self.processor.size = {"height": size, "width": size}

        dataset = get_coco_instance_segmentation_dataset(
            image_folder=request.image_folder_path,
            annotation_file=request.annotation_file_url,
        )

        final_loss = self._train(dataset, params, label2idx)

        # Record the trained classes so the registered model carries them.
        self.model_info.label_ids = [label.id for label in request.labels]
        self.model_info.tags["label_ids"] = str([label.id for label in request.labels])

        logger.info("Fine-tuning complete (final loss: %.4f).", final_loss)

    def _train(
            self,
            train_dataset: torch.utils.data.Dataset,
            params: dict,
            label2idx: dict[int, int],
    ) -> float:
        """Fine-tune Mask2Former. Returns the average loss of the last epoch."""
        mlflow.log_params(params)
        mlflow.log_param("num_samples", len(train_dataset))
        mlflow.log_param("num_classes", len(label2idx))

        loader = DataLoader(
            train_dataset,
            batch_size=int(params["batch_size"]),
            shuffle=True,
            collate_fn=self._collate_batch,
        )
        optimizer = AdamW(
            self.hf_model.parameters(),
            lr=float(params["lr"]),
            weight_decay=float(params["weight_decay"]),
        )

        last_loss = 0.0
        self.hf_model.train()

        for epoch in range(int(params["epochs"])):
            epoch_loss = 0.0
            for batch_idx, batch in enumerate(loader):
                pixel_values, mask_labels, class_labels = self._build_inputs(
                    batch["images"], batch["targets"], label2idx
                )

                optimizer.zero_grad()
                outputs = self.hf_model(
                    pixel_values=pixel_values,
                    mask_labels=mask_labels,
                    class_labels=class_labels,
                )
                outputs.loss.backward()
                optimizer.step()

                epoch_loss += outputs.loss.item()

                if (batch_idx + 1) % max(1, len(loader) // 5) == 0:
                    logger.info(
                        "Epoch %d/%d, batch %d/%d — loss: %.4f",
                        epoch + 1, int(params["epochs"]), batch_idx + 1, len(loader),
                        outputs.loss.item(),
                    )

            last_loss = epoch_loss / max(len(loader), 1)
            # Log loss + completed-epoch counter so progress consumers (the gateway's
            # MLflow-backed SSE stream) can render a progress bar.
            mlflow.log_metric("loss", last_loss, step=epoch + 1)
            mlflow.log_metric("epoch", epoch + 1, step=epoch + 1)
            logger.info("Epoch %d/%d — avg loss: %.4f", epoch + 1, int(params["epochs"]), last_loss)

        self.hf_model.eval()
        return last_loss

    def _build_inputs(self, images: list, targets: list, label2idx: dict[int, int]):
        """Build ``(pixel_values, mask_labels, class_labels)`` for one batch.

        Images are preprocessed by the (fixed-square) processor; each instance's
        binary mask is resized to the resulting spatial size so overlapping masks
        are preserved (we build ``mask_labels`` directly instead of letting the
        processor collapse instances into one segmentation map).
        """
        proc = self.processor(images=images, return_tensors="pt")
        pixel_values = proc["pixel_values"].to(self.device)
        out_h, out_w = pixel_values.shape[-2:]

        mask_labels: list[torch.Tensor] = []
        class_labels: list[torch.Tensor] = []

        for target in targets:
            inst_masks: list[torch.Tensor] = []
            inst_classes: list[int] = []
            for mask_np, category_id in zip(target["masks"], target["labels"]):
                if category_id not in label2idx:
                    continue  # label not part of this model's class set
                m = torch.from_numpy(mask_np.astype(np.float32))[None, None]
                m = F.interpolate(m, size=(out_h, out_w), mode="nearest")[0, 0]
                inst_masks.append(m)
                inst_classes.append(label2idx[category_id])

            if inst_masks:
                mask_labels.append(torch.stack(inst_masks).to(self.device))
                class_labels.append(torch.tensor(inst_classes, dtype=torch.long, device=self.device))
            else:
                mask_labels.append(torch.zeros((0, out_h, out_w), device=self.device))
                class_labels.append(torch.zeros((0,), dtype=torch.long, device=self.device))

        return pixel_values, mask_labels, class_labels

    @staticmethod
    def _collate_batch(batch: list[tuple]) -> dict:
        """Collate ``(image_np, target)`` items into parallel lists."""
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        return {"images": images, "targets": targets}
