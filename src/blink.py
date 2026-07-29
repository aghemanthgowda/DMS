"""Blink detection and blink-rate estimation from an EAR stream.

Blink *rate* is itself a fatigue signal: rates drift as drivers tire. A blink is
counted on a hysteresis cycle (EAR dips below a closed threshold, then recovers
above an open threshold) to avoid double-counting a single blink.
"""

from __future__ import annotations

from collections import deque


class BlinkCounter:
    """Count blinks from an EAR stream using open/closed hysteresis."""

    def __init__(self, closed_threshold: float, open_threshold: float) -> None:
        """Create a counter.

        Args:
            closed_threshold: EAR below this starts a blink.
            open_threshold: EAR above this (after a dip) completes a blink.

        Raises:
            ValueError: If ``closed_threshold >= open_threshold``.
        """
        if closed_threshold >= open_threshold:
            raise ValueError("closed_threshold must be < open_threshold")
        self._closed = closed_threshold
        self._open = open_threshold
        self._eyes_closed = False
        self.count = 0

    def update(self, ear: float) -> int:
        """Update with the latest EAR and return the running blink count."""
        if not self._eyes_closed and ear < self._closed:
            self._eyes_closed = True
        elif self._eyes_closed and ear > self._open:
            self._eyes_closed = False
            self.count += 1
        return self.count


class BlinkRateTracker:
    """Estimate blinks-per-minute over a rolling time window."""

    def __init__(self, window_seconds: float = 60.0) -> None:
        """Create a tracker.

        Args:
            window_seconds: Rolling window length in seconds. Must be > 0.

        Raises:
            ValueError: If ``window_seconds`` is not positive.
        """
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self._window_seconds = window_seconds
        self._blink_times: deque[float] = deque()

    def record_blink(self, timestamp: float) -> None:
        """Record a blink occurring at ``timestamp`` (seconds)."""
        self._blink_times.append(timestamp)
        self._evict(timestamp)

    def rate_per_minute(self, timestamp: float) -> float:
        """Return the current blink rate per minute at ``timestamp``."""
        self._evict(timestamp)
        return len(self._blink_times) * 60.0 / self._window_seconds

    def _evict(self, now: float) -> None:
        cutoff = now - self._window_seconds
        while self._blink_times and self._blink_times[0] < cutoff:
            self._blink_times.popleft()
