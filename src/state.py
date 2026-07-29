"""Drowsiness state tracking.

The scaffold implementation estimates PERCLOS (the proportion of time the eyes
are closed) over a fixed-length *frame* window and drives a small state machine.
A later change replaces the frame window with a true time-based rolling window.
"""

from __future__ import annotations

from collections import deque
from enum import Enum

from .config import Thresholds


class DrowsinessState(Enum):
    """Coarse driver-alertness classification."""

    AWAKE = "AWAKE"
    DROWSY = "DROWSY"


class PerclosTracker:
    """Rolling PERCLOS estimate over the most recent ``window_frames`` frames."""

    def __init__(self, window_frames: int) -> None:
        """Create a tracker.

        Args:
            window_frames: Number of recent frames to average over. Must be > 0.

        Raises:
            ValueError: If ``window_frames`` is not positive.
        """
        if window_frames <= 0:
            raise ValueError("window_frames must be positive")
        self._window: deque[int] = deque(maxlen=window_frames)

    def update(self, eyes_closed: bool) -> float:
        """Record a frame and return the current PERCLOS in ``[0, 1]``."""
        self._window.append(1 if eyes_closed else 0)
        return self.value

    @property
    def value(self) -> float:
        """Current PERCLOS fraction; ``0.0`` before any samples are recorded."""
        if not self._window:
            return 0.0
        return sum(self._window) / len(self._window)

    def reset(self) -> None:
        """Discard all accumulated samples."""
        self._window.clear()


class DrowsinessMonitor:
    """Combine per-frame EAR/MAR into a debounced drowsiness state."""

    def __init__(self, thresholds: Thresholds) -> None:
        self._thresholds = thresholds
        self._perclos = PerclosTracker(thresholds.perclos_window_frames)
        self._state = DrowsinessState.AWAKE

    @property
    def state(self) -> DrowsinessState:
        """The most recently computed state."""
        return self._state

    @property
    def perclos(self) -> float:
        """The current PERCLOS fraction."""
        return self._perclos.value

    def update(self, ear: float, mar: float) -> DrowsinessState:
        """Advance the state machine by one frame.

        Args:
            ear: Current (averaged) eye aspect ratio.
            mar: Current mouth aspect ratio.

        Returns:
            The updated :class:`DrowsinessState`.
        """
        eyes_closed = ear < self._thresholds.ear_closed
        perclos = self._perclos.update(eyes_closed)
        if perclos >= self._thresholds.perclos_drowsy:
            self._state = DrowsinessState.DROWSY
        else:
            self._state = DrowsinessState.AWAKE
        return self._state

    def is_yawning(self, mar: float) -> bool:
        """Return ``True`` when the mouth aspect ratio indicates a yawn."""
        return mar > self._thresholds.mar_yawn
