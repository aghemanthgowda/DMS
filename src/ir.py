"""Infrared (IR/NIR) camera frame preprocessing.

Near-infrared cameras are the standard for in-vehicle DMS because they work in
darkness and see through many sunglasses. IR frames are typically single-channel;
this module normalises them and expands them to the 3-channel RGB form the
MediaPipe ``FaceLandmarker`` expects, with optional CLAHE contrast enhancement
for low-light robustness.
"""

from __future__ import annotations

import cv2
import numpy as np


def to_three_channel(frame: np.ndarray) -> np.ndarray:
    """Expand a single-channel IR frame to three identical channels.

    Args:
        frame: A ``(H, W)``, ``(H, W, 1)`` or already ``(H, W, 3)`` image.

    Returns:
        An ``(H, W, 3)`` image.

    Raises:
        ValueError: If the frame shape is unsupported.
    """
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    if frame.ndim == 3 and frame.shape[2] == 1:
        return cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2RGB)
    if frame.ndim == 3 and frame.shape[2] == 3:
        return frame
    raise ValueError(f"unsupported IR frame shape {frame.shape}")


def enhance_contrast(
    gray: np.ndarray, clip_limit: float = 2.0, tile: int = 8
) -> np.ndarray:
    """Apply CLAHE to a single-channel image to boost low-light contrast.

    Args:
        gray: A ``(H, W)`` ``uint8`` image.
        clip_limit: CLAHE contrast-clipping limit.
        tile: Side length of the CLAHE tile grid.

    Returns:
        The contrast-enhanced ``(H, W)`` image.

    Raises:
        ValueError: If ``gray`` is not single-channel.
    """
    if gray.ndim != 2:
        raise ValueError("enhance_contrast expects a single-channel image")
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return clahe.apply(gray)


def preprocess_ir(frame: np.ndarray, enhance: bool = True) -> np.ndarray:
    """Preprocess an IR frame into a MediaPipe-ready 3-channel RGB image.

    Args:
        frame: The raw IR frame (grayscale or 3-channel).
        enhance: Whether to apply CLAHE contrast enhancement first.

    Returns:
        An ``(H, W, 3)`` RGB image.
    """
    if frame.ndim == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    elif frame.ndim == 3 and frame.shape[2] == 1:
        gray = frame[:, :, 0]
    else:
        gray = frame
    if enhance:
        gray = enhance_contrast(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
