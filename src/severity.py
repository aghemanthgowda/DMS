"""Fuse drowsiness and distraction signals into a single alert severity."""

from __future__ import annotations

from enum import IntEnum

from .state import DrowsinessState


class Severity(IntEnum):
    """Overall alert severity, ordered from least to most urgent."""

    NONE = 0
    LOW = 1
    HIGH = 2
    CRITICAL = 3


def compute_severity(
    state: DrowsinessState,
    distracted: bool,
    perclos: float,
    warn_perclos: float = 0.25,
    critical_perclos: float = 0.70,
) -> Severity:
    """Combine the drowsiness state, distraction flag and PERCLOS into a level.

    Args:
        state: Current drowsiness state.
        distracted: Whether sustained distraction is flagged.
        perclos: Current PERCLOS fraction in ``[0, 1]``.
        warn_perclos: PERCLOS above which an otherwise-alert driver is LOW.
        critical_perclos: PERCLOS at which a drowsy driver escalates to CRITICAL.

    Returns:
        The computed :class:`Severity`.
    """
    drowsy = state is DrowsinessState.DROWSY
    if drowsy and (distracted or perclos >= critical_perclos):
        return Severity.CRITICAL
    if drowsy or distracted:
        return Severity.HIGH
    if perclos >= warn_perclos:
        return Severity.LOW
    return Severity.NONE
