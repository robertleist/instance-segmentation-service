"""DINOv3-backed instance segmentation that learns fast.

Unlike ``Mask2Former`` (which fine-tunes its whole Swin backbone end-to-end), this
model keeps a **frozen** DINOv3 backbone and trains only a small query-based mask
decoder on top of its dense patch features. Far fewer trainable parameters means it
converges in a handful of epochs and on much less labelled data — that is the point.

Architecture
------------
``frozen DINOv3 -> patch feature grid (B, C, Hp, Wp) -> lightweight decoder``

The decoder is a miniature Mask2Former head:
  * ``num_queries`` learnable object queries attend to the patch tokens through a
    small Transformer decoder;
  * a class head maps each query to ``num_classes + 1`` logits (+1 = "no object");
  * a mask-embedding head dots each query against per-patch pixel embeddings to
    produce a low-resolution mask, upsampled to image size at inference.

Training uses Hungarian matching between queries and ground-truth instances, with a
classification + mask-BCE + dice set loss (the Mask2Former recipe, simplified).

Integration notes (shared with ``Mask2Former``)
------------------------------------------------
* This is an MLflow ``pyfunc`` model. The live torch modules can't be cloudpickled,
  so ``backbone``/``head`` are stripped in ``__getstate__`` (via ``_unpicklable_attrs``)
  and rebuilt in ``load_context``. Only the *head* weights are saved as artifacts —
  the frozen DINOv3 reloads from its HuggingFace id, keeping artifacts tiny.
* The class-index <-> :class:`Label` mapping is persisted as ``label_mapping.json`` so
  predictions resolve back to real dataset labels, exactly like ``Mask2Former``.
"""

import json
import logging
import os
from typing import Any, Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from iquana_toolbox.ai.backbones.dinov3 import DEFAULT_DINOV3_MODEL, DINOv3Backbone
from iquana_toolbox.ai.base_classes import InstanceSegmentationModel, InstanceSegmentationModelInfo
from iquana_toolbox.ai.dataloaders import get_coco_instance_segmentation_dataset
from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.database.labels import Label
from iquana_toolbox.schemas.networking.http.services import InstanceSegmentationRequest
from iquana_toolbox.schemas.training import HyperParameter, InstanceSegmentationTrainingRequest
from scipy.optimize import linear_sum_assignment
from torch.optim import AdamW
from torch.utils.data import DataLoader

from iquana_service_core import register_model

import paths

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight query-based mask decoder
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    """A small multi-layer perceptron (used for the per-query mask embedding)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU(inplace=True)]
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _MaskDecoderHead(nn.Module):
    """Maps a patch-feature grid to per-query class logits and mask logits.

    Output shapes for an input grid ``(B, C, Hp, Wp)``:
      * ``class_logits``: ``(B, num_queries, num_classes + 1)``
      * ``mask_logits``:  ``(B, num_queries, Hp, Wp)``  (sigmoid -> instance masks)
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        d_model: int = 256,
        mask_dim: int = 256,
        num_queries: int = 100,
        num_decoder_layers: int = 3,
        nhead: int = 8,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes

        # Project patch tokens into the decoder's working dimension (the "memory").
        self.input_proj = nn.Linear(in_dim, d_model)
        self.query_embed = nn.Embedding(num_queries, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.class_head = nn.Linear(d_model, num_classes + 1)  # +1 = "no object"
        self.mask_embed = _MLP(d_model, d_model, mask_dim, num_layers=3)
        # Per-patch pixel embeddings; dotted with each query's mask embedding.
        self.pixel_embed = nn.Linear(in_dim, mask_dim)

    def forward(self, feature_grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, hp, wp = feature_grid.shape
        tokens = feature_grid.flatten(2).transpose(1, 2)  # (B, Hp*Wp, C)

        memory = self.input_proj(tokens)                          # (B, N, d_model)
        queries = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)  # (B, Q, d_model)
        hs = self.decoder(queries, memory)                        # (B, Q, d_model)

        class_logits = self.class_head(hs)                        # (B, Q, K+1)
        mask_embed = self.mask_embed(hs)                          # (B, Q, mask_dim)

        pixel = self.pixel_embed(tokens)                          # (B, N, mask_dim)
        pixel = pixel.transpose(1, 2).reshape(b, -1, hp, wp)      # (B, mask_dim, Hp, Wp)
        mask_logits = torch.einsum("bqd,bdhw->bqhw", mask_embed, pixel)  # (B, Q, Hp, Wp)
        return class_logits, mask_logits


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

@register_model
class DinoV3InstanceSegmenter(InstanceSegmentationModel):
    """Instance segmentation on frozen DINOv3 features with a trainable query head.

    A base instance has a randomly initialised head and must be trained on a dataset's
    labels before it can predict (``predict`` returns ``[]`` until then). It is still
    registered as ``status="ready"`` so it shows up in the model selection: the gateway
    only ever lists ``status=="ready"`` models (``GET /models/all/available``), so a
    ``not_ready`` model would be invisible — including for *picking it to train*.
    Training rebuilds the head for the requested number of classes and only optimises
    the head's parameters — the DINOv3 backbone stays frozen throughout.
    """

    model_info = InstanceSegmentationModelInfo(
        registry_key="dinov3_instance",
        name="DINOv3 Instance Segmenter",
        description=(
            "Instance segmentation built on a **frozen** DINOv3 self-supervised vision "
            "backbone (Meta, 2025) with a lightweight query-based mask decoder. Because "
            "only the small decoder is trained, it converges much faster and on less data "
            "than fine-tuning a full backbone. Note: DINOv3 weights are gated and ship "
            "under Meta's DINOv3 License (commercial-use restrictions apply)."
        ),
        usage_tip=(
            "Train on your dataset's labels before use. Fast to fine-tune; great when "
            "labelled data is scarce. Masks are coarser than Mask2Former at low image_size — "
            "raise image_size if edge precision matters."
        ),
        info_url="https://huggingface.co/docs/transformers/model_doc/dinov3",
        tags={
            "task": "instance-segmentation",
            # "ready" so the model is selectable — the gateway only lists ready models.
            # The untrained base predicts nothing until trained (predict returns []).
            "status": "ready",
            "trainable": "true",
            "domain": "general",
            "publisher": "meta",
        },
        badges=["fast-training", "foundation-model"],
        trainable=True,
        status="ready",
        label_ids=[],
        training_parameters=[
            HyperParameter(key="epochs", label="Epochs", default_value=20, type="int",
                           description="Number of passes over the dataset.",
                           min_value=1, max_value=200, step=1),
            HyperParameter(key="batch_size", label="Batch size", default_value=4, type="int",
                           description="Images per training step.",
                           min_value=1, max_value=16, step=1),
            HyperParameter(key="lr", label="Learning rate", default_value=1e-4, type="float",
                           description="AdamW learning rate for the decoder head."),
            HyperParameter(key="weight_decay", label="Weight decay", default_value=1e-4, type="float",
                           description="AdamW weight decay."),
            HyperParameter(key="image_size", label="Image size", default_value=768, type="int",
                           description="Square size images are resized to (must be a multiple of 16).",
                           options=[256, 512, 768, 1024]),
        ],
    )
    default_hyperparameters: dict = {
        "epochs": 20,
        "batch_size": 4,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "image_size": 768,
    }

    # Live torch/HF modules can't be cloudpickled; stripped here and rebuilt in
    # load_context (backbone from its HF id, head from saved weights).
    _unpicklable_attrs = ("backbone", "head")

    # Set-loss / matcher weights (Mask2Former-style).
    _cost_class, _cost_mask, _cost_dice = 2.0, 5.0, 5.0
    _w_mask, _w_dice, _no_object_weight = 5.0, 5.0, 0.1

    def __init__(
        self,
        model_id: str = DEFAULT_DINOV3_MODEL,
        image_size: int = 768,
        num_queries: int = 100,
        device: Optional[str] = None,
    ):
        self.model_id = model_id
        self.image_size = image_size
        self.num_queries = num_queries
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # class index -> Label, populated on train() / restored in load_context().
        self.idx_to_label: dict[int, Label] = {}
        self.num_classes = 0

        # During MLflow load the instance is unpickled (no __init__) and these are
        # rebuilt in load_context instead; guard so that path doesn't double-build.
        if model_id:
            self._build_backbone()
            self._build_head(self.num_classes)

    # -- construction helpers ----------------------------------------------

    def _build_backbone(self) -> None:
        self.backbone = DINOv3Backbone(
            model_id=self.model_id,
            image_size=self.image_size,
            token=paths.HF_ACCESS_TOKEN,
            device=self.device
        )

    def _build_head(self, num_classes: int) -> None:
        """(Re)create the decoder head for ``num_classes`` and move it to the device."""
        self.num_classes = num_classes
        self.head = _MaskDecoderHead(
            in_dim=self.backbone.hidden_size,
            num_classes=num_classes,
            num_queries=self.num_queries,
        ).to(self.device)

    # -- MLflow persistence -------------------------------------------------

    def get_artifacts(self, tmp_dir: str) -> dict[str, str]:
        """Persist only the trained head weights + config + class->label mapping.

        The frozen DINOv3 backbone is *not* saved; it is reloaded from ``model_id``
        in ``load_context``, so artifacts stay small.
        """
        head_path = os.path.join(tmp_dir, "head_weights.pt")
        torch.save(self.head.state_dict(), head_path)

        config_path = os.path.join(tmp_dir, "head_config.json")
        with open(config_path, "w", encoding="utf-8") as fp:
            json.dump(
                {
                    "model_id": self.model_id,
                    "image_size": self.image_size,
                    "num_queries": self.num_queries,
                    "num_classes": self.num_classes,
                },
                fp,
            )

        mapping_path = os.path.join(tmp_dir, "label_mapping.json")
        with open(mapping_path, "w", encoding="utf-8") as fp:
            json.dump({str(i): lab.model_dump() for i, lab in self.idx_to_label.items()}, fp)

        return {"head_weights": head_path, "head_config": config_path, "label_mapping": mapping_path}

    def load_context(self, context):
        """Rebuild the frozen backbone + trained head from artifacts (MLflow load time)."""
        with open(context.artifacts["head_config"], encoding="utf-8") as fp:
            cfg = json.load(fp)
        self.model_id = cfg["model_id"]
        self.image_size = cfg["image_size"]
        self.num_queries = cfg["num_queries"]

        self._build_backbone()
        self._build_head(cfg["num_classes"])
        self.head.load_state_dict(torch.load(context.artifacts["head_weights"], map_location=self.device))
        self.head.eval()

        mapping_path = context.artifacts.get("label_mapping")
        if mapping_path and os.path.exists(mapping_path):
            with open(mapping_path, encoding="utf-8") as fp:
                raw = json.load(fp)
            self.idx_to_label = {int(i): Label(**data) for i, data in raw.items()}

    # -- inference ----------------------------------------------------------

    def predict(self, context: Any, model_input, params=None) -> list[Contour]:
        """Run instance segmentation on a single :class:`InstanceSegmentationRequest`.

        (``model_input`` is left untyped because MLflow's signature inference recurses
        infinitely on the recursive ``Label.children`` field — same reason as Mask2Former.)
        """
        request: InstanceSegmentationRequest = model_input
        if self.num_classes == 0:
            # Untrained base model: nothing meaningful to predict.
            return []

        image: np.ndarray = request.image  # (H, W, 3) uint8 RGB
        h, w = image.shape[:2]
        threshold = (params or {}).get("threshold", 0.5)

        self.head.eval()
        with torch.no_grad():
            features = self.backbone(self.backbone.preprocess([image]))
            class_logits, mask_logits = self.head(features)

        class_logits = class_logits[0]          # (Q, K+1)
        mask_logits = mask_logits[0]            # (Q, Hp, Wp)
        # Probability over real classes only (drop the no-object column).
        probs = class_logits.softmax(-1)[:, :-1]
        scores, label_idx = probs.max(-1)       # (Q,), (Q,)
        keep = scores > threshold
        if keep.sum() == 0:
            return []

        # Upsample only the kept queries' masks to the original image size.
        masks = F.interpolate(
            mask_logits[keep].unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False
        ).squeeze(1)
        masks = masks.sigmoid() > 0.5

        filter_label_id = request.label.id if request.label is not None else None
        contours: list[Contour] = []
        for mask, idx, score in zip(masks, label_idx[keep], scores[keep]):
            label = self.idx_to_label.get(int(idx))
            if label is None:
                continue
            if filter_label_id is not None and label.id != filter_label_id:
                continue
            binary_mask = mask.cpu().numpy().astype(bool)
            if not binary_mask.any():
                continue
            contours.append(
                Contour.from_binary_mask(
                    binary_mask=binary_mask,
                    only_return_biggest_contour=True,
                    label_id=label.id,
                    confidence=float(score),
                    added_by=str(request.user_id),
                )
            )
        return contours

    # -- training -----------------------------------------------------------

    def train(self, request: InstanceSegmentationTrainingRequest, **kwargs) -> None:
        """Train only the decoder head on the dataset's labels; backbone stays frozen."""
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

        # Resize the backbone's input grid, then build a fresh head for this class count.
        self.image_size = int(params["image_size"])
        self.backbone.set_image_size(self.image_size)
        self._build_head(len(request.labels))

        logger.info(
            "Training DINOv3 head on %d label(s) from '%s' (annotations '%s')…",
            len(request.labels), request.image_folder_path, request.annotation_file_url,
        )

        dataset = get_coco_instance_segmentation_dataset(
            image_folder=request.image_folder_path,
            annotation_file=request.annotation_file_url,
        )
        final_loss = self._train(dataset, params, label2idx)

        # Record the trained classes + flip status to ready for registration/inference.
        self.model_info.label_ids = [label.id for label in request.labels]
        self.model_info.tags["label_ids"] = str([label.id for label in request.labels])
        self.model_info.status = "ready"
        self.model_info.tags["status"] = "ready"

        logger.info("Training complete (final loss: %.4f).", final_loss)

    def _train(self, dataset, params: dict, label2idx: dict[int, int]) -> float:
        mlflow.log_params(params)
        mlflow.log_param("num_samples", len(dataset))
        mlflow.log_param("num_classes", len(label2idx))
        mlflow.log_param("backbone", self.model_id)
        mlflow.log_param("frozen_backbone", True)

        loader = DataLoader(
            dataset,
            batch_size=int(params["batch_size"]),
            shuffle=True,
            collate_fn=self._collate_batch,
        )
        # Only the head is optimised — the frozen backbone has no trainable params.
        optimizer = AdamW(
            self.head.parameters(),
            lr=float(params["lr"]),
            weight_decay=float(params["weight_decay"]),
        )

        grid = self.image_size // self.backbone.patch_size
        last_loss = 0.0
        self.backbone.model.eval()
        self.head.train()

        for epoch in range(int(params["epochs"])):
            epoch_loss = 0.0
            for batch_idx, batch in enumerate(loader):
                # Frozen backbone forward (no grad); head forward (with grad).
                with torch.no_grad():
                    features = self.backbone(self.backbone.preprocess(batch["images"]))
                class_logits, mask_logits = self.head(features)

                targets = self._build_targets(batch["targets"], label2idx, grid, grid)
                loss = self._set_loss(class_logits, mask_logits, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                if (batch_idx + 1) % max(1, len(loader) // 5) == 0:
                    logger.info(
                        "Epoch %d/%d, batch %d/%d — loss: %.4f",
                        epoch + 1, int(params["epochs"]), batch_idx + 1, len(loader), loss.item(),
                    )

            last_loss = epoch_loss / max(len(loader), 1)
            # Per-epoch loss + counter so the gateway's MLflow-backed SSE progress bar works.
            mlflow.log_metric("loss", last_loss, step=epoch + 1)
            mlflow.log_metric("epoch", epoch + 1, step=epoch + 1)
            logger.info("Epoch %d/%d — avg loss: %.4f", epoch + 1, int(params["epochs"]), last_loss)

        self.head.eval()
        return last_loss

    def _build_targets(self, raw_targets: list, label2idx: dict[int, int], hp: int, wp: int) -> list[dict]:
        """Build per-image ``{"labels": (n,), "masks": (n, Hp, Wp)}`` at grid resolution.

        GT masks are downsampled to the patch grid so matching and the mask loss are
        computed cheaply at low resolution (predictions are upsampled only at inference).
        """
        targets: list[dict] = []
        for target in raw_targets:
            masks: list[torch.Tensor] = []
            classes: list[int] = []
            for mask_np, category_id in zip(target["masks"], target["labels"]):
                if category_id not in label2idx:
                    continue  # label not part of this model's class set
                m = torch.from_numpy(mask_np.astype(np.float32))[None, None]
                m = F.interpolate(m, size=(hp, wp), mode="nearest")[0, 0]
                masks.append(m)
                classes.append(label2idx[category_id])

            if masks:
                targets.append({
                    "labels": torch.tensor(classes, dtype=torch.long, device=self.device),
                    "masks": torch.stack(masks).to(self.device),
                })
            else:
                targets.append({
                    "labels": torch.zeros(0, dtype=torch.long, device=self.device),
                    "masks": torch.zeros(0, hp, wp, device=self.device),
                })
        return targets

    # -- set loss (Hungarian matching + classification/mask/dice) -----------

    def _set_loss(self, class_logits: torch.Tensor, mask_logits: torch.Tensor, targets: list[dict]) -> torch.Tensor:
        b, q, _ = class_logits.shape
        k = self.num_classes
        cls_loss = class_logits.new_zeros(())
        mask_loss = class_logits.new_zeros(())
        dice_loss = class_logits.new_zeros(())
        num_with_masks = 0

        for i in range(b):
            tgt_labels = targets[i]["labels"]
            tgt_masks = targets[i]["masks"]
            q_idx, t_idx = self._match(class_logits[i], mask_logits[i], tgt_labels, tgt_masks)

            # Classification: every query defaults to "no object" (class index = k);
            # matched queries take their target class.
            tgt_classes = torch.full((q,), k, dtype=torch.long, device=self.device)
            if q_idx.numel() > 0:
                tgt_classes[q_idx] = tgt_labels[t_idx]
            weight = torch.ones(k + 1, device=self.device)
            weight[k] = self._no_object_weight
            cls_loss = cls_loss + F.cross_entropy(class_logits[i], tgt_classes, weight=weight)

            # Mask losses only on matched (query, gt) pairs.
            if t_idx.numel() > 0:
                pred = mask_logits[i][q_idx]   # (m, Hp, Wp)
                gt = tgt_masks[t_idx]          # (m, Hp, Wp)
                mask_loss = mask_loss + F.binary_cross_entropy_with_logits(pred, gt)
                dice_loss = dice_loss + self._dice_loss(pred, gt)
                num_with_masks += 1

        cls_loss = cls_loss / b
        if num_with_masks > 0:
            mask_loss = mask_loss / num_with_masks
            dice_loss = dice_loss / num_with_masks
        return cls_loss + self._w_mask * mask_loss + self._w_dice * dice_loss

    @torch.no_grad()
    def _match(self, class_logits_i, mask_logits_i, tgt_labels, tgt_masks):
        """Hungarian matching for one image. Returns (query_indices, target_indices)."""
        empty = torch.empty(0, dtype=torch.long, device=self.device)
        if tgt_labels.numel() == 0:
            return empty, empty

        probs = class_logits_i.softmax(-1)               # (Q, K+1)
        cost_class = -probs[:, tgt_labels]               # (Q, n)
        pred_flat = mask_logits_i.flatten(1)             # (Q, P)
        tgt_flat = tgt_masks.flatten(1)                  # (n, P)
        cost_mask = self._pairwise_bce(pred_flat, tgt_flat)
        cost_dice = self._pairwise_dice(pred_flat, tgt_flat)

        cost = (self._cost_class * cost_class
                + self._cost_mask * cost_mask
                + self._cost_dice * cost_dice).cpu().numpy()
        q_idx, t_idx = linear_sum_assignment(cost)
        return (torch.as_tensor(q_idx, dtype=torch.long, device=self.device),
                torch.as_tensor(t_idx, dtype=torch.long, device=self.device))

    @staticmethod
    def _pairwise_bce(pred_logits: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Mean per-pixel BCE between every (query, gt) pair. Shapes (Q,P),(n,P)->(Q,n)."""
        p = pred_logits.shape[1]
        pos = F.binary_cross_entropy_with_logits(pred_logits, torch.ones_like(pred_logits), reduction="none")
        neg = F.binary_cross_entropy_with_logits(pred_logits, torch.zeros_like(pred_logits), reduction="none")
        return (pos @ tgt.T + neg @ (1.0 - tgt).T) / p

    @staticmethod
    def _pairwise_dice(pred_logits: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """1 - dice between every (query, gt) pair. Shapes (Q,P),(n,P)->(Q,n)."""
        pred = pred_logits.sigmoid()
        numerator = 2.0 * (pred @ tgt.T)
        denom = pred.sum(-1)[:, None] + tgt.sum(-1)[None, :]
        return 1.0 - (numerator + 1.0) / (denom + 1.0)

    @staticmethod
    def _dice_loss(pred_logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Mean dice loss over matched pairs. Shapes (m,Hp,Wp)."""
        pred = pred_logits.sigmoid().flatten(1)
        gt = gt.flatten(1)
        numerator = 2.0 * (pred * gt).sum(-1)
        denom = pred.sum(-1) + gt.sum(-1)
        return (1.0 - (numerator + 1.0) / (denom + 1.0)).mean()

    @staticmethod
    def _collate_batch(batch: list[tuple]) -> dict:
        """Collate ``(image_np, target)`` items into parallel lists (same as Mask2Former)."""
        return {"images": [item[0] for item in batch], "targets": [item[1] for item in batch]}
