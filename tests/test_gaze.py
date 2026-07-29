"""Tests for gaze-region classification."""

from __future__ import annotations

from src.gaze import GazeRegion, classify_gaze


def test_forward_when_within_thresholds() -> None:
    assert classify_gaze(yaw=5.0, pitch=5.0) is GazeRegion.FORWARD


def test_right_and_left() -> None:
    assert classify_gaze(yaw=30.0, pitch=0.0) is GazeRegion.RIGHT
    assert classify_gaze(yaw=-30.0, pitch=0.0) is GazeRegion.LEFT


def test_up_and_down() -> None:
    assert classify_gaze(yaw=0.0, pitch=25.0) is GazeRegion.UP
    assert classify_gaze(yaw=0.0, pitch=-25.0) is GazeRegion.DOWN


def test_yaw_takes_precedence_over_pitch() -> None:
    assert classify_gaze(yaw=40.0, pitch=40.0) is GazeRegion.RIGHT
