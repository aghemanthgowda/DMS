"""Unit tests for the geometric metrics using synthetic landmarks.

The tests build small landmark arrays with known geometry so EAR/MAR can be
checked against hand-computed values without a camera or MediaPipe.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import LandmarkIndices
from src.metrics import (
    average_ear,
    eye_aspect_ratio,
    landmarks_to_array,
    mouth_aspect_ratio,
)

INDICES = LandmarkIndices()


def _blank(n: int = 478) -> np.ndarray:
    """Return an ``(n, 2)`` array of zeros to populate at specific indices."""
    return np.zeros((n, 2), dtype=np.float64)


def _set_eye(landmarks: np.ndarray, indices: tuple[int, ...], opening: float) -> None:
    """Place a rectangular eye of width 4 and vertical ``opening`` at ``indices``.

    Points are laid out so that ``|p1 - p4| = 4`` and each vertical pair spans
    ``opening``, giving ``EAR = opening / 4``.
    """
    p1, p2, p3, p4, p5, p6 = indices
    half = opening / 2.0
    landmarks[p1] = (0.0, 0.0)        # left corner
    landmarks[p4] = (4.0, 0.0)        # right corner
    landmarks[p2] = (1.0, half)       # upper-left lid
    landmarks[p3] = (3.0, half)       # upper-right lid
    landmarks[p5] = (3.0, -half)      # lower-right lid
    landmarks[p6] = (1.0, -half)      # lower-left lid


def test_eye_aspect_ratio_matches_manual_value() -> None:
    landmarks = _blank()
    _set_eye(landmarks, INDICES.left_eye, opening=2.0)
    # vertical = 2 + 2 = 4, horizontal = 4 -> EAR = 4 / (2 * 4) = 0.5
    assert eye_aspect_ratio(landmarks, INDICES.left_eye) == pytest.approx(0.5)


def test_open_eye_has_larger_ear_than_closed_eye() -> None:
    open_eye = _blank()
    closed_eye = _blank()
    _set_eye(open_eye, INDICES.left_eye, opening=2.4)
    _set_eye(closed_eye, INDICES.left_eye, opening=0.4)
    ear_open = eye_aspect_ratio(open_eye, INDICES.left_eye)
    ear_closed = eye_aspect_ratio(closed_eye, INDICES.left_eye)
    assert ear_open > ear_closed
    assert ear_closed == pytest.approx(0.1)


def test_average_ear_is_mean_of_both_eyes() -> None:
    landmarks = _blank()
    _set_eye(landmarks, INDICES.left_eye, opening=2.0)   # EAR 0.5
    _set_eye(landmarks, INDICES.right_eye, opening=1.0)  # EAR 0.25
    assert average_ear(
        landmarks, INDICES.left_eye, INDICES.right_eye
    ) == pytest.approx(0.375)


def test_mouth_aspect_ratio_matches_manual_value() -> None:
    landmarks = _blank()
    _set_eye(landmarks, INDICES.mouth, opening=3.0)
    # vertical = 3 + 3 = 6, horizontal = 4 -> MAR = 6 / (2 * 4) = 0.75
    assert mouth_aspect_ratio(landmarks, INDICES.mouth) == pytest.approx(0.75)


def test_degenerate_horizontal_span_returns_zero() -> None:
    landmarks = _blank()
    # All eye points coincide -> zero horizontal span -> guarded to 0.0.
    assert eye_aspect_ratio(landmarks, INDICES.left_eye) == 0.0


def test_aspect_ratio_rejects_wrong_shape() -> None:
    landmarks = _blank()
    with pytest.raises(ValueError):
        # Only five indices -> not a (6, 2) slice.
        eye_aspect_ratio(landmarks, INDICES.left_eye[:5])


def test_landmarks_to_array_scales_to_pixels() -> None:
    class _LM:
        def __init__(self, x: float, y: float) -> None:
            self.x = x
            self.y = y

    normalised = [_LM(0.5, 0.25), _LM(1.0, 1.0)]
    pixels = landmarks_to_array(normalised, image_width=640, image_height=480)
    assert pixels.shape == (2, 2)
    assert pixels[0] == pytest.approx([320.0, 120.0])
    assert pixels[1] == pytest.approx([640.0, 480.0])
