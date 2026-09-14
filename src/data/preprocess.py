"""Frame preprocessing for the World Models (2018) CarRacing reproduction.

The paper does not fully specify this pipeline, so every choice here is
documented and must be verified visually (see visualize_frames).

Defaults:
    crop_bottom = 12   # 96 -> 84 rows; drops the on-screen HUD indicator bars
    size        = 64   # paper: VAE operates on 64x64 frames
    resize      = PIL BILINEAR  (NOT nearest / NOT cv2 INTER_NEAREST)
    dtype       = uint8 in [0, 255]; float conversion happens in the dataloader
"""

from __future__ import annotations

import numpy as np
from PIL import Image

try:                                    # Pillow >= 9.1
    _RESAMPLE = Image.Resampling.BILINEAR
except AttributeError:                  # older Pillow
    _RESAMPLE = Image.BILINEAR

FRAME_SIZE = 64
CROP_BOTTOM = 12


def preprocess_frame(
    frame: np.ndarray,
    crop_bottom: int = CROP_BOTTOM,
    size: int = FRAME_SIZE,
    grayscale: bool = False,
) -> np.ndarray:
    """96x96x3 uint8 -> 64x64x3 (or 64x64) uint8."""
    if not isinstance(frame, np.ndarray):
        frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        raise ValueError(f"expected uint8 frame, got {frame.dtype}")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3), got {frame.shape}")

    h = frame.shape[0]
    if crop_bottom:
        if not 0 <= crop_bottom < h:
            raise ValueError(f"crop_bottom={crop_bottom} invalid for height {h}")
        frame = frame[: h - crop_bottom, :, :]

    img = Image.fromarray(frame)
    img = img.resize((size, size), resample=_RESAMPLE)

    if grayscale:
        out = np.asarray(img.convert("L"), dtype=np.uint8)
    else:
        out = np.asarray(img, dtype=np.uint8)

    if out.shape[0] != size or out.shape[1] != size:
        raise RuntimeError(f"resize produced {out.shape}, expected {size}x{size}")
    return out


def preprocess_episode(frames: np.ndarray, **kw) -> np.ndarray:
    """(T, 96, 96, 3) uint8 -> (T, 64, 64, 3) uint8."""
    if frames.shape[0] == 0:
        h, w = frames.shape[1:3]
        c = 1 if kw.get("grayscale") else 3
        return np.empty((0, kw.get("size", FRAME_SIZE),
                         kw.get("size", FRAME_SIZE), c), np.uint8)
    return np.stack([preprocess_frame(f, **kw) for f in frames])


def to_float(batch: np.ndarray) -> np.ndarray:
    """uint8 [0,255] -> float32 [0,1]. Call this in the dataloader, never on disk."""
    return batch.astype(np.float32) / 255.0