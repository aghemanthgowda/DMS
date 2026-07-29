"""Gaze-region classification from head-pose angles.

Turns continuous yaw/pitch head-pose angles into a coarse gaze region, useful for
distinguishing "eyes on road" (FORWARD) from mirror checks, phone glances (DOWN),
etc. Yaw dominates pitch, since side-to-side is the primary road-attention axis.
"""

from __future__ import annotations

from enum import Enum


class GazeRegion(Enum):
    """Coarse direction the driver's head is pointing."""

    FORWARD = "FORWARD"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UP = "UP"
    DOWN = "DOWN"


def classify_gaze(
    yaw: float,
    pitch: float,
    yaw_threshold: float = 20.0,
    pitch_threshold: float = 15.0,
) -> GazeRegion:
    """Classify a gaze region from head yaw and pitch (degrees).

    Args:
        yaw: Head yaw in degrees (negative left, positive right).
        pitch: Head pitch in degrees (negative down, positive up).
        yaw_threshold: Absolute yaw beyond which the driver is looking aside.
        pitch_threshold: Absolute pitch beyond which the driver is looking up/down.

    Returns:
        The classified :class:`GazeRegion`. Yaw takes precedence over pitch.
    """
    if yaw > yaw_threshold:
        return GazeRegion.RIGHT
    if yaw < -yaw_threshold:
        return GazeRegion.LEFT
    if pitch > pitch_threshold:
        return GazeRegion.UP
    if pitch < -pitch_threshold:
        return GazeRegion.DOWN
    return GazeRegion.FORWARD
