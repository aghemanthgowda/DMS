"""Drowsiness state tracking.

PERCLOS (the proportion of time the eyes are closed) is estimated over a true
*time-based* rolling window rather than a fixed frame count, so the measure is
robust to a varying frame rate. A small hysteresis state machine converts the
continuous PERCLOS signal into a debounced AWAKE/DROWSY classification.
"""

from __future__ import annotations

from collections import deque
from enum import Enum
from typing import Optional

from .config import Thresholds


class DrowsinessState(Enum):
    """Coarse driver-alertness classification."""

    AWAKE = "AWAKE"
    DROWSY = "DROWSY"


class PerclosTracker:
    """Rolling PERCLOS estimate over the most recent ``window_seconds`` seconds.

    Samples are stored as ``(timestamp, eyes_closed)`` pairs in a deque; on each
    update, samples older than the window are evicted. PERCLOS is the fraction of
    retained samples in which the eyes were closed.
    """

    def __init__(self, window_seconds: float = 60.0) -> None:
        """Create a tracker.

        Args:
            window_seconds: Length of the rolling window in seconds. Must be > 0.

        Raises:
            ValueError: If ``window_seconds`` is not positive.
        """
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self._window_seconds = window_seconds
        self._samples: deque[tuple[float, bool]] = deque()

    def update(self, eyes_closed: bool, timestamp: float) -> float:
        """Record a sample and return the current PERCLOS in ``[0, 1]``.

        Args:
            eyes_closed: Whether the eyes are closed on this frame.
            timestamp: Monotonic time of the frame, in seconds.

        Returns:
            The updated PERCLOS fraction.
        """
        self._samples.append((timestamp, bool(eyes_closed)))
        cutoff = timestamp - self._window_seconds
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()
        return self.value

    @property
    def value(self) -> float:
        """Current PERCLOS fraction; ``0.0`` before any samples are recorded."""
        if not self._samples:
            return 0.0
        closed = sum(1 for _, eyes_closed in self._samples if eyes_closed)
        return closed / len(self._samples)

    @property
    def sample_count(self) -> int:
        """Number of samples currently inside the window."""
        return len(self._samples)

    def reset(self) -> None:
        """Discard all accumulated samples."""
        self._samples.clear()


class DrowsinessMonitor:
    """Combine per-frame EAR/MAR into a debounced drowsiness state.

    The state machine uses hysteresis: it enters :attr:`DrowsinessState.DROWSY`
    once PERCLOS reaches ``perclos_drowsy`` and only returns to
    :attr:`DrowsinessState.AWAKE` once it falls back to ``perclos_recover``,
    preventing rapid flip-flopping near a single threshold.
    """

    def __init__(self, thresholds: Thresholds) -> None:
        self._thresholds = thresholds
        self._perclos = PerclosTracker(thresholds.perclos_window_seconds)
        self._state = DrowsinessState.AWAKE

    @property
    def state(self) -> DrowsinessState:
        """The most recently computed state."""
        return self._state

    @property
    def perclos(self) -> float:
        """The current PERCLOS fraction."""
        return self._perclos.value

    def update(self, ear: float, mar: float, timestamp: float) -> DrowsinessState:
        """Advance the state machine by one frame.

        Args:
            ear: Current (averaged) eye aspect ratio.
            mar: Current mouth aspect ratio.
            timestamp: Monotonic time of the frame, in seconds.

        Returns:
            The updated :class:`DrowsinessState`.
        """
        eyes_closed = ear < self._thresholds.ear_closed
        perclos = self._perclos.update(eyes_closed, timestamp)

        if self._state is DrowsinessState.AWAKE:
            if perclos >= self._thresholds.perclos_drowsy:
                self._state = DrowsinessState.DROWSY
        else:  # currently DROWSY
            if perclos <= self._thresholds.perclos_recover:
                self._state = DrowsinessState.AWAKE
        return self._state

    def is_yawning(self, mar: float) -> bool:
        """Return ``True`` when the mouth aspect ratio indicates a yawn."""
        return mar > self._thresholds.mar_yawn


class EyeClosureTracker:
    """Fire an immediate alarm when the eyes stay closed past a short threshold.

    This is the low-latency counterpart to PERCLOS: PERCLOS integrates eye
    closure over ~60s to judge fatigue, whereas this tracker reacts within about
    a second to a sustained closure (a microsleep), so a driver who shuts their
    eyes is flagged almost immediately.
    """

    def __init__(self, alarm_seconds: float) -> None:
        """Create the tracker.

        Args:
            alarm_seconds: Continuous closed duration that raises the alarm.

        Raises:
            ValueError: If ``alarm_seconds`` is not positive.
        """
        if alarm_seconds <= 0:
            raise ValueError("alarm_seconds must be positive")
        self._alarm_seconds = alarm_seconds
        self._closed_since: Optional[float] = None
        self._duration = 0.0
        self._alarm = False

    @property
    def alarm(self) -> bool:
        """Whether the microsleep alarm is currently active."""
        return self._alarm

    @property
    def closed_duration(self) -> float:
        """How long the eyes have been continuously closed, in seconds."""
        return self._duration

    def update(self, eyes_closed: bool, timestamp: float) -> bool:
        """Update with the latest closure state and return the alarm flag.

        Args:
            eyes_closed: Whether the eyes are closed on this frame.
            timestamp: Monotonic time of the frame, in seconds.

        Returns:
            ``True`` once the eyes have been closed for ``alarm_seconds``.
        """
        if eyes_closed:
            if self._closed_since is None:
                self._closed_since = timestamp
            self._duration = timestamp - self._closed_since
            self._alarm = self._duration >= self._alarm_seconds
        else:
            self._closed_since = None
            self._duration = 0.0
            self._alarm = False
        return self._alarm


class DistractionTracker:
    """Flag distraction when the head yaw stays off-axis for a sustained time.

    A single glance away from the road should not raise an alert; only a yaw
    that exceeds ``yaw_threshold_deg`` continuously for ``sustained_seconds``
    counts as distraction.
    """

    def __init__(self, yaw_threshold_deg: float, sustained_seconds: float) -> None:
        self._yaw_threshold_deg = yaw_threshold_deg
        self._sustained_seconds = sustained_seconds
        self._off_axis_since: Optional[float] = None
        self._distracted = False

    @property
    def distracted(self) -> bool:
        """Whether distraction is currently flagged."""
        return self._distracted

    def update(self, yaw_deg: float, timestamp: float) -> bool:
        """Update with the latest head yaw and return the distraction flag.

        Args:
            yaw_deg: Current head yaw in degrees.
            timestamp: Monotonic time of the frame, in seconds.

        Returns:
            ``True`` once the yaw has been off-axis for the sustained duration.
        """
        if abs(yaw_deg) > self._yaw_threshold_deg:
            if self._off_axis_since is None:
                self._off_axis_since = timestamp
            elapsed = timestamp - self._off_axis_since
            self._distracted = elapsed >= self._sustained_seconds
        else:
            self._off_axis_since = None
            self._distracted = False
        return self._distracted
