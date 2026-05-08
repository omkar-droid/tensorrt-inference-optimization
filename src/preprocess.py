"""Shared preprocessing for ResNet18/34 on ImageNet.

Calibration, validation, and inference must all use identical preprocessing —
otherwise INT8 calibration will produce a poorly-calibrated engine.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# ImageNet normalization constants (the same torchvision uses)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_SIZE = 224


def preprocess_image(path: Path | str) -> np.ndarray:
    """Load an image and convert to NCHW float32 tensor (1, 3, 224, 224).

    Steps: resize short side to 256, center-crop 224, scale to [0, 1],
    normalize with ImageNet mean/std, transpose to CHW.
    """
    img = Image.open(path).convert("RGB")

    # Resize so the shorter side is 256, preserving aspect ratio.
    w, h = img.size
    if w < h:
        new_w, new_h = 256, int(256 * h / w)
    else:
        new_w, new_h = int(256 * w / h), 256
    img = img.resize((new_w, new_h), Image.BILINEAR)

    # Center crop 224x224.
    left = (new_w - INPUT_SIZE) // 2
    top = (new_h - INPUT_SIZE) // 2
    img = img.crop((left, top, left + INPUT_SIZE, top + INPUT_SIZE))

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    return arr[np.newaxis, ...]    # add batch dim
