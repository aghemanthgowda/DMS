"""Drawing helpers for landmark overlays (face mesh, hand skeletons).

Kept dependency-light (OpenCV + NumPy only) and connection-agnostic: the caller
passes the landmark points and the connection index pairs, so this module never
imports MediaPipe and stays unit-testable without a display.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import cv2
import numpy as np

Color = tuple[int, int, int]


def draw_connections(
    frame: np.ndarray,
    points_px: np.ndarray,
    connections: Iterable[Sequence[int]],
    color: Color = (0, 255, 0),
    thickness: int = 1,
) -> None:
    """Draw line segments between connected landmark points, in place.

    Args:
        frame: BGR image to draw on.
        points_px: ``(N, 2)`` array of pixel coordinates.
        connections: Iterable of ``(start_index, end_index)`` pairs.
        color: BGR line colour.
        thickness: Line thickness in pixels.
    """
    count = len(points_px)
    for start, end in connections:
        if 0 <= start < count and 0 <= end < count:
            p1 = (int(points_px[start][0]), int(points_px[start][1]))
            p2 = (int(points_px[end][0]), int(points_px[end][1]))
            cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)


def draw_points(
    frame: np.ndarray,
    points_px: np.ndarray,
    color: Color = (0, 255, 0),
    radius: int = 1,
) -> None:
    """Draw a filled dot at each landmark point, in place.

    Args:
        frame: BGR image to draw on.
        points_px: ``(N, 2)`` array of pixel coordinates.
        color: BGR dot colour.
        radius: Dot radius in pixels.
    """
    for point in points_px:
        cv2.circle(frame, (int(point[0]), int(point[1])), radius, color, -1, cv2.LINE_AA)


def draw_box(
    frame: np.ndarray,
    box: Sequence[int],
    label: str,
    color: Color = (0, 165, 255),
    thickness: int = 2,
) -> None:
    """Draw a labelled bounding box, in place.

    Args:
        frame: BGR image to draw on.
        box: ``(x1, y1, x2, y2)`` pixel coordinates.
        label: Text label to render above the box.
        color: BGR box/label colour.
        thickness: Box line thickness in pixels.
    """
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(
        frame,
        label,
        (x1, max(0, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )
