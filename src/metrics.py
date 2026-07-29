"""Geometric driver-state metrics computed from facial landmarks.

This module is deliberately dependency-light (NumPy only) so the core maths is
unit-testable without a camera, MediaPipe, or a display. Landmarks are supplied
as an ``(N, 2)`` array of pixel coordinates; helpers convert MediaPipe's
normalised landmark lists into that form.
"""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
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


# ---------------------------------------------------------------------------
# Head pose estimation (cv2.solvePnP against a generic 3D face model)
# ---------------------------------------------------------------------------

# A canonical, roughly life-sized 3D face model in millimetres. The origin sits
# near the nose tip; +x is to the subject's left, +y is up, +z is toward the
# camera. These six points pair with the MediaPipe indices in
# :data:`HEAD_POSE_LANDMARKS`.
MODEL_POINTS_3D: np.ndarray = np.array(
    [
        (0.0, 0.0, 0.0),        # nose tip
        (0.0, -63.6, -12.5),    # chin
        (-43.3, 32.7, -26.0),   # left eye outer corner
        (43.3, 32.7, -26.0),    # right eye outer corner
        (-28.9, -28.9, -24.1),  # left mouth corner
        (28.9, -28.9, -24.1),   # right mouth corner
    ],
    dtype=np.float64,
)

# MediaPipe FaceLandmarker indices matching MODEL_POINTS_3D, in the same order.
HEAD_POSE_LANDMARKS: tuple[int, int, int, int, int, int] = (1, 152, 33, 263, 61, 291)


def default_camera_matrix(image_width: int, image_height: int) -> np.ndarray:
    """Return an approximate pinhole camera matrix for a webcam.

    Uses the image width as the focal length and the image centre as the
    principal point, which is a standard assumption when the camera is
    uncalibrated.

    Args:
        image_width: Frame width in pixels.
        image_height: Frame height in pixels.

    Returns:
        A ``(3, 3)`` intrinsic camera matrix.
    """
    focal_length = float(image_width)
    return np.array(
        [
            [focal_length, 0.0, image_width / 2.0],
            [0.0, focal_length, image_height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _rotation_to_euler(rotation_matrix: np.ndarray) -> tuple[float, float, float]:
    """Decompose a rotation matrix into ``(yaw, pitch, roll)`` degrees.

    Args:
        rotation_matrix: A ``(3, 3)`` rotation matrix.

    Returns:
        ``(yaw, pitch, roll)`` in degrees, rotations about the y, x and z axes.
    """
    angles = cv2.RQDecomp3x3(rotation_matrix)[0]
    pitch, yaw, roll = float(angles[0]), float(angles[1]), float(angles[2])
    return yaw, pitch, roll


def estimate_head_pose(
    landmarks: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: Optional[np.ndarray] = None,
) -> Optional[tuple[float, float, float]]:
    """Estimate head orientation with ``cv2.solvePnP``.

    Args:
        landmarks: ``(N, 2)`` array of pixel coordinates.
        camera_matrix: A ``(3, 3)`` intrinsic camera matrix (see
            :func:`default_camera_matrix`).
        dist_coeffs: Optional lens distortion coefficients; zeros if ``None``.

    Returns:
        ``(yaw, pitch, roll)`` in degrees, or ``None`` if the solver fails.
    """
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    image_points = np.ascontiguousarray(
        landmarks[list(HEAD_POSE_LANDMARKS)], dtype=np.float64
    )
    success, rotation_vector, _ = cv2.solvePnP(
        MODEL_POINTS_3D,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    return _rotation_to_euler(rotation_matrix)
