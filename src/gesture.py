"""Hand-to-face gesture heuristics (phone use / smoking proxies).

These are geometric heuristics, **not** trained object detectors: a hand held at
the mouth is a proxy for smoking or a phone/food-to-mouth gesture, and a hand at
the ear is a proxy for a phone call. Distances are scaled by the detected face
width so the thresholds are roughly distance-invariant.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np


class HandGesture(NamedTuple):
    """Flags for hand-to-face gestures detected on a single frame."""

    hand_at_mouth: bool  # possible smoking / phone-to-mouth / eating
    hand_at_ear: bool    # possible phone call


def min_distance(points_px: np.ndarray, target_px: Sequence[float]) -> float:
    """Return the smallest distance from any point to ``target_px``.

    Args:
        points_px: ``(N, 2)`` array of pixel coordinates.
        target_px: The ``(x, y)`` point to measure to.

    Returns:
        The minimum Euclidean distance, or ``inf`` if there are no points.
    """
    if len(points_px) == 0:
        return float("inf")
    deltas = np.asarray(points_px, dtype=np.float64) - np.asarray(target_px, dtype=np.float64)
    return float(np.min(np.linalg.norm(deltas, axis=1)))


def hand_near_point(
    hand_points_px: np.ndarray, target_px: Sequence[float], max_distance: float
) -> bool:
    """Return ``True`` if any hand point is within ``max_distance`` of the target."""
    return min_distance(hand_points_px, target_px) <= max_distance


def detect_hand_gesture(
    hand_points_px: np.ndarray,
    mouth_px: Sequence[float],
    left_ear_px: Sequence[float],
    right_ear_px: Sequence[float],
    face_width_px: float,
    mouth_factor: float = 0.55,
    ear_factor: float = 0.50,
) -> HandGesture:
    """Classify a hand's proximity to the mouth and ears.

    Args:
        hand_points_px: ``(21, 2)`` hand landmark pixels.
        mouth_px: Mouth-centre pixel coordinate.
        left_ear_px: Left-ear/side-of-face pixel coordinate.
        right_ear_px: Right-ear/side-of-face pixel coordinate.
        face_width_px: Face width in pixels (used to scale thresholds).
        mouth_factor: Mouth threshold as a fraction of face width.
        ear_factor: Ear threshold as a fraction of face width.

    Returns:
        A :class:`HandGesture` with the two proximity flags.
    """
    max_mouth = mouth_factor * face_width_px
    max_ear = ear_factor * face_width_px
    at_mouth = hand_near_point(hand_points_px, mouth_px, max_mouth)
    at_ear = hand_near_point(hand_points_px, left_ear_px, max_ear) or hand_near_point(
        hand_points_px, right_ear_px, max_ear
    )
    return HandGesture(hand_at_mouth=at_mouth, hand_at_ear=at_ear)
