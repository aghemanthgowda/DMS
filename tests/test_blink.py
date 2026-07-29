"""Tests for blink counting and blink-rate estimation."""

from __future__ import annotations

import pytest

from src.blink import BlinkCounter, BlinkRateTracker


def test_blink_counter_counts_one_cycle() -> None:
    counter = BlinkCounter(closed_threshold=0.2, open_threshold=0.3)
    assert counter.update(0.35) == 0  # open
    assert counter.update(0.10) == 0  # closing (dip)
    assert counter.update(0.35) == 1  # recovered -> one blink


def test_blink_counter_ignores_flutter_without_full_recovery() -> None:
    counter = BlinkCounter(closed_threshold=0.2, open_threshold=0.3)
    counter.update(0.10)  # closed
    counter.update(0.25)  # between thresholds, not a full recovery
    assert counter.count == 0


def test_blink_counter_rejects_inverted_thresholds() -> None:
    with pytest.raises(ValueError):
        BlinkCounter(closed_threshold=0.3, open_threshold=0.2)


def test_blink_rate_per_minute() -> None:
    tracker = BlinkRateTracker(window_seconds=60.0)
    for second in range(10):
        tracker.record_blink(float(second))
    # 10 blinks in a 60s window -> 10 per minute.
    assert tracker.rate_per_minute(10.0) == pytest.approx(10.0)


def test_blink_rate_evicts_old_blinks() -> None:
    tracker = BlinkRateTracker(window_seconds=10.0)
    tracker.record_blink(0.0)
    tracker.record_blink(1.0)
    # Far in the future: both old blinks evicted.
    assert tracker.rate_per_minute(100.0) == pytest.approx(0.0)


def test_blink_rate_rejects_bad_window() -> None:
    with pytest.raises(ValueError):
        BlinkRateTracker(window_seconds=-1.0)
