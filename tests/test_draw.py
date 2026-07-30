"""Tests for the landmark drawing helpers."""

from __future__ import annotations

import numpy as np

from src.draw import draw_box, draw_connections, draw_points


def _blank_image() -> np.ndarray:
    return np.zeros((50, 50, 3), dtype=np.uint8)


def test_draw_connections_marks_pixels() -> None:
    frame = _blank_image()
    points = np.array([[5, 5], [45, 45]], dtype=np.float64)
    draw_connections(frame, points, [(0, 1)], color=(0, 255, 0), thickness=1)
    assert frame.any()  # some pixels were drawn


def test_draw_connections_ignores_out_of_range_indices() -> None:
    frame = _blank_image()
    points = np.array([[5, 5]], dtype=np.float64)
    # Connection references index 1 which doesn't exist -> no crash, no draw.
    draw_connections(frame, points, [(0, 1)])
    assert not frame.any()


def test_draw_points_marks_pixels() -> None:
    frame = _blank_image()
    points = np.array([[25, 25]], dtype=np.float64)
    draw_points(frame, points, color=(0, 0, 255), radius=3)
    assert frame.any()


def test_draw_box_marks_pixels() -> None:
    frame = _blank_image()
    draw_box(frame, (5, 5, 40, 40), "phone 0.90", color=(0, 165, 255))
    assert frame.any()
