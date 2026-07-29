"""Lightweight online smoothers for noisy per-frame metrics.

EAR/MAR jitter frame-to-frame; smoothing them reduces spurious state flips.
"""

from __future__ import annotations

import statistics
from collections import deque


class ExponentialSmoother:
    """Exponential moving average of a scalar stream."""

    def __init__(self, alpha: float) -> None:
        """Create a smoother.

        Args:
            alpha: Smoothing factor in ``(0, 1]``; higher tracks faster.

        Raises:
            ValueError: If ``alpha`` is outside ``(0, 1]``.
        """
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self._alpha = alpha
        self._value: float | None = None

    def update(self, sample: float) -> float:
        """Incorporate ``sample`` and return the smoothed value."""
        if self._value is None:
            self._value = float(sample)
        else:
            self._value = self._alpha * sample + (1.0 - self._alpha) * self._value
        return self._value

    @property
    def value(self) -> float:
        """Current smoothed value; ``0.0`` before any samples."""
        return 0.0 if self._value is None else self._value


class RollingMedian:
    """Rolling median over the most recent ``window`` samples."""

    def __init__(self, window: int) -> None:
        """Create a rolling-median filter.

        Args:
            window: Number of recent samples to consider. Must be > 0.

        Raises:
            ValueError: If ``window`` is not positive.
        """
        if window <= 0:
            raise ValueError("window must be positive")
        self._buffer: deque[float] = deque(maxlen=window)

    def update(self, sample: float) -> float:
        """Add ``sample`` and return the median of the current window."""
        self._buffer.append(float(sample))
        return statistics.median(self._buffer)
