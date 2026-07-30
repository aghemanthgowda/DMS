"""Tests for the hand-to-face gesture heuristics."""

from __future__ import annotations

import numpy as np

from src.gesture import detect_hand_gesture, hand_near_point, min_distance

# A synthetic face ~100px wide: mouth centre and two ears.
MOUTH = (100.0, 120.0)
LEFT_EAR = (150.0, 100.0)
RIGHT_EAR = (50.0, 100.0)
FACE_WIDTH = 100.0


def _hand_at(x: float, y: float) -> np.ndarray:
    """A small cluster of 5 hand points around (x, y)."""
    return np.array([[x, y], [x + 2, y], [x, y + 2], [x - 2, y], [x, y - 2]], dtype=np.float64)


def test_min_distance_empty_is_inf() -> None:
    assert min_distance(np.empty((0, 2)), MOUTH) == float("inf")


def test_hand_near_point_true_and_false() -> None:
    # Nearest hand point sits 6px from the mouth.
    hand = _hand_at(100.0, 128.0)
    assert hand_near_point(hand, MOUTH, max_distance=10.0) is True
    assert hand_near_point(hand, MOUTH, max_distance=1.0) is False


def test_hand_at_mouth_detected() -> None:
    gesture = detect_hand_gesture(_hand_at(100.0, 125.0), MOUTH, LEFT_EAR, RIGHT_EAR, FACE_WIDTH)
    assert gesture.hand_at_mouth is True
    assert gesture.hand_at_ear is False


def test_hand_at_ear_detected() -> None:
    gesture = detect_hand_gesture(_hand_at(150.0, 100.0), MOUTH, LEFT_EAR, RIGHT_EAR, FACE_WIDTH)
    assert gesture.hand_at_ear is True


def test_hand_resting_low_triggers_nothing() -> None:
    # Hand far below the face (on the wheel) -> no gesture.
    gesture = detect_hand_gesture(_hand_at(100.0, 400.0), MOUTH, LEFT_EAR, RIGHT_EAR, FACE_WIDTH)
    assert gesture.hand_at_mouth is False
    assert gesture.hand_at_ear is False
