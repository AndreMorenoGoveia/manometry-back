"""The trained study-level network, rebuilt for inference.

The architecture and the preprocessing here are a transcription of the winning
training run (``run_fusion_heavy.py`` / ``common.py`` in
``manometry-classification``): a ConvNeXt-Base ImageNet-22k encoder over the ten
swallow panels, a per-swallow instance head, and the ``chicago_set`` fusion —
a Chicago v4.0 aggregator over the per-swallow posteriors and image-calibrated
metrics, concatenated with a Set Transformer over the swallow embeddings.

Two deliberate differences from the training file, neither of which changes a
single weight or activation:

* the encoder is built with ``pretrained=False``.  Every parameter is overwritten
  by the checkpoint a moment later, so downloading the ImageNet weights would
  only add a network dependency to start-up.
* gradient checkpointing is off.  It is a training-memory device with no effect
  on outputs.

``tests/test_inference_parity.py`` re-scores the held-out examinations of the
run and asserts the probabilities match the ones stored with the checkpoint,
which is what keeps this transcription honest.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch import nn

from .vendor.fusion_models import FusionModel

# Cache/crop geometry of the training loader: every panel is decoded once into a
# 256 square and the encoder reads a 224 window out of it.  Evaluation uses the
# centre window; the training run selected centre over multi-crop TTA on the
# validation examinations, so the centre window is what the reported score is.
CACHE_SIZE = 256
IMAGE_SIZE = 224
EVAL_OFFSET = (CACHE_SIZE - IMAGE_SIZE) // 2
MAX_SWALLOWS = 10

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

TIMM_ENCODERS = {
    "convnext_small_in22k": "convnext_small.fb_in22k_ft_in1k",
    "convnext_base_in22k": "convnext_base.fb_in22k_ft_in1k",
    "convnextv2_base_in22k": "convnextv2_base.fcmae_ft_in22k_in1k",
    "swin_small_in22k": "swin_small_patch4_window7_224.ms_in22k_ft_in1k",
    "swin_base_in22k": "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
    "vit_base_in21k": "vit_base_patch16_224.augreg_in21k_ft_in1k",
    "densenet201": "densenet201.tv_in1k",
    "resnet101": "resnet101.tv2_in1k",
}


def build_encoder(name: str, *, drop_path: float = 0.0, pretrained: bool = False) -> tuple[nn.Module, int]:
    """A headless encoder and its embedding width."""
    if name in TIMM_ENCODERS:
        import timm

        model = timm.create_model(
            TIMM_ENCODERS[name], pretrained=pretrained, num_classes=0, drop_path_rate=drop_path
        )
        return model, int(model.num_features)
    if name == "resnet18":
        from torchvision import models

        model = models.resnet18(weights=None)
        dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, dim
    if name == "convnext_tiny":
        from torchvision import models

        model = models.convnext_tiny(weights=None)
        dim = model.classifier[2].in_features
        model.classifier = nn.Sequential(model.classifier[0], model.classifier[1])
        return model, dim
    raise ValueError(f"Encoder nao suportado: {name}")


class HeavyExamModel(nn.Module):
    """Heavy encoder + the study-level fusion head."""

    def __init__(
        self,
        encoder_name: str,
        fusion: str,
        num_classes: int,
        metric_dim: int,
        *,
        drop_path: float = 0.0,
        hidden: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder, dim = build_encoder(encoder_name, drop_path=drop_path)
        self.embedding_dim = dim
        self.instance_head = nn.Sequential(
            nn.LayerNorm(dim), nn.Dropout(dropout), nn.Linear(dim, num_classes)
        )
        self.fusion = FusionModel(
            fusion, dim, num_classes, metric_dim, hidden=hidden, dropout=dropout
        )

    def forward(
        self, images: torch.Tensor, mask: torch.Tensor, study: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, swallows = images.shape[:2]
        flat = images.reshape(batch * swallows, *images.shape[2:])
        embeddings = self.encoder(flat).reshape(batch, swallows, -1)
        instance_logits = self.instance_head(embeddings)
        return self.fusion(embeddings, instance_logits, mask, study), instance_logits


# ------------------------------------------------------------- preprocessing


def to_cache_window(rgb: np.ndarray) -> np.ndarray:
    """One cropped topography -> the uint8 224 window the encoder was fed.

    Decode to a 256 square with bilinear resampling, then read the centre 224
    window, exactly as ``common.cache_images`` and ``ExamImageDataset`` do.
    """
    square = Image.fromarray(np.ascontiguousarray(rgb.astype(np.uint8))).resize(
        (CACHE_SIZE, CACHE_SIZE), Image.BILINEAR
    )
    cached = np.asarray(square, dtype=np.uint8)
    top = left = EVAL_OFFSET
    return np.ascontiguousarray(cached[top : top + IMAGE_SIZE, left : left + IMAGE_SIZE])


def stack_exam(windows: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad the swallows of one examination into the model's fixed ten slots."""
    images = torch.zeros(1, MAX_SWALLOWS, IMAGE_SIZE, IMAGE_SIZE, 3, dtype=torch.uint8)
    mask = torch.zeros(1, MAX_SWALLOWS, dtype=torch.bool)
    for slot, window in enumerate(windows[:MAX_SWALLOWS]):
        images[0, slot] = torch.from_numpy(window)
        mask[0, slot] = True
    return images, mask


def to_model_input(batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    """uint8 NHWC -> normalised NCHW float on ``device`` (evaluation path)."""
    images = batch.to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0)
    images = (images - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
    return images.contiguous(memory_format=torch.channels_last)
