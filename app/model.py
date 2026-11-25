"""The JBHI proposed model, rebuilt for inference.

Transcribed from ``studies/cbm_2026`` in ``manometry-classification``: a ConvNeXt-Tiny swallow encoder trained
multitask (``common.MultitaskSwallowClassifier``: a diagnosis head plus per-swallow measurement heads supervised by
the software's own DCI, IRP, DL, break and pan-pressurization values), read out at the examination level by a
multinomial logistic over exact Chicago count events of the predicted states (``jbhi_fusion.readout_features``).
The encoder is built without the ImageNet weights, which the checkpoint overwrites anyway.

``tests/test_inference_parity.py`` re-scores the fold's held-out examinations and asserts the probabilities match
the ones stored with the bundle, which is what keeps this transcription honest.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models

from .vendor.jbhi_readout import AUX_SIZES

# Cache/crop geometry of the training loader: every panel is decoded once into a 256 square and the encoder reads
# the centre 224 window out of it.
CACHE_SIZE = 256
IMAGE_SIZE = 224
EVAL_OFFSET = (CACHE_SIZE - IMAGE_SIZE) // 2
MAX_SWALLOWS = 10

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class MultitaskSwallowModel(nn.Module):
    """ConvNeXt-Tiny + diagnosis head + measurement heads, with the parameter names of the training class."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.encoder = models.convnext_tiny(weights=None)
        dim = self.encoder.classifier[2].in_features
        self.encoder.classifier = nn.Sequential(self.encoder.classifier[0], self.encoder.classifier[1])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(0.2), nn.Linear(dim, num_classes))
        self.aux = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, sum(AUX_SIZES)))

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(images)
        return self.head(features), self.aux(features)


# ------------------------------------------------------------- preprocessing


def to_cache_square(rgb: np.ndarray) -> np.ndarray:
    """One cropped topography decoded into the uint8 256 square of the cache.

    Bilinear resampling to 256, exactly as ``common.cache_images`` does; the
    encoder then reads a 224 window out of it.
    """
    square = Image.fromarray(np.ascontiguousarray(rgb.astype(np.uint8))).resize(
        (CACHE_SIZE, CACHE_SIZE), Image.BILINEAR
    )
    return np.asarray(square, dtype=np.uint8)


def window(square: np.ndarray, offset: tuple[int, int] = (EVAL_OFFSET, EVAL_OFFSET)) -> np.ndarray:
    top, left = offset
    return np.ascontiguousarray(square[top : top + IMAGE_SIZE, left : left + IMAGE_SIZE])


def to_cache_window(rgb: np.ndarray) -> np.ndarray:
    """The centre 224 window of one cropped topography."""
    return window(to_cache_square(rgb))


def to_model_input(batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    """uint8 NHWC -> normalised NCHW float on ``device`` (evaluation path)."""
    images = batch.to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0)
    images = (images - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
    return images.contiguous(memory_format=torch.channels_last)
