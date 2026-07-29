"""Unit tests for the time-based PERCLOS tracker and drowsiness state machine."""

from __future__ import annotations

import pytest

from src.config import Thresholds
from src.state import (
    DistractionTracker,
    DrowsinessMonitor,
    DrowsinessState,
    PerclosTracker,
)


def test_perclos_is_fraction_of_closed_samples() -> None:
    tracker = PerclosTracker(window_seconds=60.0)
    for t in range(10):
        # 4 of 10 samples closed -> PERCLOS 0.4.
        tracker.update(eyes_closed=(t < 4), timestamp=float(t))
    assert tracker.value == pytest.approx(0.4)


def test_perclos_evicts_samples_outside_window() -> None:
    tracker = PerclosTracker(window_seconds=10.0)
    # Old closed samples that should fall out of the window.
    for t in range(5):
        tracker.update(eyes_closed=True, timestamp=float(t))
    # Newer open samples well past the 10s window.
    for t in range(100, 110):
        tracker.update(eyes_closed=False, timestamp=float(t))
    # Only the recent (open) samples remain -> PERCLOS 0.0.
    assert tracker.value == pytest.approx(0.0)
    assert tracker.sample_count == 10


def test_perclos_rejects_non_positive_window() -> None:
    with pytest.raises(ValueError):
        PerclosTracker(window_seconds=0.0)


def test_empty_tracker_reports_zero() -> None:
    assert PerclosTracker(window_seconds=60.0).value == 0.0


def _thresholds() -> Thresholds:
    return Thresholds(
        ear_closed=0.21,
        perclos_drowsy=0.40,
        perclos_recover=0.30,
        perclos_window_seconds=60.0,
    )


def test_monitor_enters_drowsy_when_perclos_exceeds_threshold() -> None:
    monitor = DrowsinessMonitor(_thresholds())
    closed_ear = 0.10  # below ear_closed
    state = DrowsinessState.AWAKE
    for t in range(10):
        state = monitor.update(ear=closed_ear, mar=0.0, timestamp=float(t))
    assert state is DrowsinessState.DROWSY
    assert monitor.perclos == pytest.approx(1.0)


def test_monitor_hysteresis_does_not_flip_between_thresholds() -> None:
    monitor = DrowsinessMonitor(_thresholds())
    # Drive firmly into DROWSY.
    for t in range(10):
        monitor.update(ear=0.10, mar=0.0, timestamp=float(t))
    assert monitor.state is DrowsinessState.DROWSY

    # Add open-eye samples until PERCLOS sits between recover (0.30) and
    # drowsy (0.40): 6 closed of 17 samples ~= 0.353. Still DROWSY.
    t = 10
    while monitor.perclos > 0.40:
        monitor.update(ear=0.30, mar=0.0, timestamp=float(t))
        t += 1
    assert 0.30 < monitor.perclos <= 0.40
    assert monitor.state is DrowsinessState.DROWSY


def test_monitor_recovers_when_perclos_drops_below_recover() -> None:
    monitor = DrowsinessMonitor(_thresholds())
    for t in range(10):
        monitor.update(ear=0.10, mar=0.0, timestamp=float(t))
    assert monitor.state is DrowsinessState.DROWSY

    t = 10
    for _ in range(40):
        monitor.update(ear=0.30, mar=0.0, timestamp=float(t))
        t += 1
    assert monitor.perclos <= 0.30
    assert monitor.state is DrowsinessState.AWAKE


def test_is_yawning_uses_mar_threshold() -> None:
    monitor = DrowsinessMonitor(Thresholds(mar_yawn=0.6))
    assert monitor.is_yawning(0.8) is True
    assert monitor.is_yawning(0.4) is False


def test_distraction_requires_sustained_off_axis_yaw() -> None:
    tracker = DistractionTracker(yaw_threshold_deg=30.0, sustained_seconds=2.0)
    # Off-axis but not yet sustained long enough.
    assert tracker.update(yaw_deg=40.0, timestamp=0.0) is False
    assert tracker.update(yaw_deg=40.0, timestamp=1.5) is False
    # Sustained beyond 2 seconds -> distracted.
    assert tracker.update(yaw_deg=40.0, timestamp=2.1) is True


def test_distraction_resets_when_gaze_returns() -> None:
    tracker = DistractionTracker(yaw_threshold_deg=30.0, sustained_seconds=2.0)
    tracker.update(yaw_deg=40.0, timestamp=0.0)
    assert tracker.update(yaw_deg=40.0, timestamp=3.0) is True
    # Gaze returns to centre -> flag clears and timer resets.
    assert tracker.update(yaw_deg=5.0, timestamp=3.5) is False
    assert tracker.distracted is False
    # A brief glance is not enough to re-trigger immediately.
    assert tracker.update(yaw_deg=40.0, timestamp=3.6) is False


def test_distraction_ignores_within_threshold_yaw() -> None:
    tracker = DistractionTracker(yaw_threshold_deg=30.0, sustained_seconds=1.0)
    for t in range(10):
        assert tracker.update(yaw_deg=10.0, timestamp=float(t)) is False
