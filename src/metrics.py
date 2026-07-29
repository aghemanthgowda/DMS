"""Geometric driver-state metrics computed from facial landmarks.

This module is deliberately dependency-light (NumPy only) so the core maths is
unit-testable without a camera, MediaPipe, or a display. Landmarks are supplied
as an ``(N, 2)`` array of pixel coordinates; helpers convert MediaPipe's
normalised landmark lists into that form.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

_EPSILON = 1e-6


def landmarks_to_array(
    face_landmarks: Sequence[object],
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Convert a MediaPipe normalised-landmark list to a pixel ``(N, 2)`` array.

    Args:
        face_landmarks: Sequence of objects exposing ``.x`` and ``.y`` in the
            normalised ``[0, 1]`` range, as returned by ``FaceLandmarker``.
        image_width: Frame width in pixels.
        image_height: Frame height in pixels.

    Returns:
        An ``(N, 2)`` ``float64`` array of pixel coordinates.
    """
    return np.array(
        [(lm.x * image_width, lm.y * image_height) for lm in face_landmarks],
        dtype=np.float64,
    )


def _aspect_ratio(points: np.ndarray) -> float:
    """Return the 6-point aspect ratio for an eye- or mouth-shaped polygon.

    The points must be ordered ``(p1, p2, p3, p4, p5, p6)`` where ``p1`` and
    ``p4`` span the horizontal axis and ``(p2, p6)`` / ``(p3, p5)`` are the two
    vertical pairs::

        ratio = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)

    Args:
        points: A ``(6, 2)`` array of coordinates.

    Returns:
        The aspect ratio, or ``0.0`` when the horizontal span is degenerate.

    Raises:
        ValueError: If ``points`` is not shaped ``(6, 2)``.
    """
    if points.shape != (6, 2):
        raise ValueError(f"expected a (6, 2) array, got {points.shape}")
    p1, p2, p3, p4, p5, p6 = points
    horizontal = float(np.linalg.norm(p1 - p4))
    if horizontal < _EPSILON:
        return 0.0
    vertical = float(np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5))
    return vertical / (2.0 * horizontal)


def eye_aspect_ratio(landmarks: np.ndarray, eye_indices: Sequence[int]) -> float:
    """Compute the Eye Aspect Ratio (EAR) for one eye.

    Args:
        landmarks: ``(N, 2)`` array of pixel coordinates.
        eye_indices: The six landmark indices in ``(p1..p6)`` order.

    Returns:
        The EAR; smaller values indicate a more-closed eye.
    """
    return _aspect_ratio(landmarks[list(eye_indices)])


def mouth_aspect_ratio(landmarks: np.ndarray, mouth_indices: Sequence[int]) -> float:
    """Compute the Mouth Aspect Ratio (MAR).

    Args:
        landmarks: ``(N, 2)`` array of pixel coordinates.
        mouth_indices: The six landmark indices in ``(p1..p6)`` order.

    Returns:
        The MAR; larger values indicate a more-open mouth (e.g. a yawn).
    """
    return _aspect_ratio(landmarks[list(mouth_indices)])


def average_ear(
    landmarks: np.ndarray,
    left_eye_indices: Sequence[int],
    right_eye_indices: Sequence[int],
) -> float:
    """Return the mean EAR across both eyes."""
    left = eye_aspect_ratio(landmarks, left_eye_indices)
    right = eye_aspect_ratio(landmarks, right_eye_indices)
    return 0.5 * (left + right)


# ---------------------------------------------------------------------------
# Blendshape cross-checks
#
# MediaPipe's FaceLandmarker can emit 52 ARKit-style blendshape scores in
# ``[0, 1]``. These learned scores give an independent second opinion on the
# purely geometric EAR/MAR, which helps in poses where the 2D landmark geometry
# alone is ambiguous.
# ---------------------------------------------------------------------------

BLINK_BLENDSHAPES: tuple[str, str] = ("eyeBlinkLeft", "eyeBlinkRight")
JAW_OPEN_BLENDSHAPE: str = "jawOpen"


def blendshape_scores(face_blendshapes: Sequence[object]) -> dict[str, float]:
    """Convert a MediaPipe blendshape category list to a ``{name: score}`` map.

    Args:
        face_blendshapes: Sequence of objects exposing ``.category_name`` and
            ``.score``, as found in ``FaceLandmarkerResult.face_blendshapes[i]``.

    Returns:
        A mapping from blendshape name to its score in ``[0, 1]``.
    """
    return {shape.category_name: float(shape.score) for shape in face_blendshapes}


def blink_score(scores: dict[str, float]) -> float:
    """Return the mean of the left/right eye-blink blendshape scores."""
    left = scores.get(BLINK_BLENDSHAPES[0], 0.0)
    right = scores.get(BLINK_BLENDSHAPES[1], 0.0)
    return 0.5 * (left + right)


def eyes_closed_from_blendshapes(
    scores: dict[str, float], threshold: float = 0.5
) -> bool:
    """Return ``True`` when the eye-blink blendshapes indicate closed eyes.

    Args:
        scores: Blendshape map from :func:`blendshape_scores`.
        threshold: Mean blink score at or above which eyes are deemed closed.
    """
    return blink_score(scores) >= threshold


def yawn_from_blendshapes(
    scores: dict[str, float], threshold: float = 0.5
) -> bool:
    """Return ``True`` when the ``jawOpen`` blendshape indicates a yawn.

    Args:
        scores: Blendshape map from :func:`blendshape_scores`.
        threshold: ``jawOpen`` score at or above which a yawn is deemed present.
    """
    return scores.get(JAW_OPEN_BLENDSHAPE, 0.0) >= threshold
