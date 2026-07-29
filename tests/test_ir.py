"""Tests for IR frame preprocessing."""

from __future__ import annotations

import numpy as np
import pytest

from src.ir import enhance_contrast, preprocess_ir, to_three_channel


def test_to_three_channel_from_grayscale() -> None:
    out = to_three_channel(np.zeros((8, 8), dtype=np.uint8))
    assert out.shape == (8, 8, 3)


def test_to_three_channel_passthrough_rgb() -> None:
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    assert to_three_channel(rgb).shape == (8, 8, 3)


def test_to_three_channel_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        to_three_channel(np.zeros((8, 8, 4), dtype=np.uint8))


def test_enhance_contrast_keeps_shape_and_dtype() -> None:
    gray = np.arange(64, dtype=np.uint8).reshape(8, 8)
    out = enhance_contrast(gray)
    assert out.shape == (8, 8)
    assert out.dtype == np.uint8


def test_enhance_contrast_rejects_multichannel() -> None:
    with pytest.raises(ValueError):
        enhance_contrast(np.zeros((8, 8, 3), dtype=np.uint8))


def test_preprocess_ir_returns_rgb() -> None:
    out = preprocess_ir(np.full((16, 16), 50, dtype=np.uint8))
    assert out.shape == (16, 16, 3)
