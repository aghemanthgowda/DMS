"""Unit tests for the time-based PERCLOS tracker and drowsiness state machine."""

from __future__ import annotations

import pytest

from src.config import Thresholds
from src.state import (
    DistractionTracker,
    DrowsinessMonitor,
    DrowsinessState,
    EyeClosureTracker,
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
    for t in range(5):
        tracker.update(eyes_closed=True, timestamp=float(t))
    for t in range(100, 110):
        tracker.update(eyes_closed=False, timestamp=float(t))
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
    for t in range(10):
        monitor.update(ear=0.10, mar=0.0, timestamp=float(t))
    assert monitor.state is DrowsinessState.DROWSY

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
    assert tracker.update(yaw_deg=40.0, timestamp=0.0) is False
    assert tracker.update(yaw_deg=40.0, timestamp=1.5) is False
    assert tracker.update(yaw_deg=40.0, timestamp=2.1) is True


def test_distraction_resets_when_gaze_returns() -> None:
    tracker = DistractionTracker(yaw_threshold_deg=30.0, sustained_seconds=2.0)
    tracker.update(yaw_deg=40.0, timestamp=0.0)
    assert tracker.update(yaw_deg=40.0, timestamp=3.0) is True
    assert tracker.update(yaw_deg=5.0, timestamp=3.5) is False
    assert tracker.distracted is False
    assert tracker.update(yaw_deg=40.0, timestamp=3.6) is False


def test_distraction_ignores_within_threshold_yaw() -> None:
    tracker = DistractionTracker(yaw_threshold_deg=30.0, sustained_seconds=1.0)
    for t in range(10):
        assert tracker.update(yaw_deg=10.0, timestamp=float(t)) is False


def test_eye_closure_alarm_fires_quickly() -> None:
    tracker = EyeClosureTracker(alarm_seconds=1.0)
    assert tracker.update(eyes_closed=True, timestamp=0.0) is False
    assert tracker.update(eyes_closed=True, timestamp=0.5) is False
    assert tracker.update(eyes_closed=True, timestamp=1.1) is True
    assert tracker.closed_duration == pytest.approx(1.1)


def test_eye_closure_resets_when_eyes_open() -> None:
    tracker = EyeClosureTracker(alarm_seconds=1.0)
    tracker.update(eyes_closed=True, timestamp=0.0)
    assert tracker.update(eyes_closed=True, timestamp=2.0) is True
    assert tracker.update(eyes_closed=False, timestamp=2.1) is False
    assert tracker.closed_duration == 0.0


def test_eye_closure_does_not_fire_for_a_blink() -> None:
    tracker = EyeClosureTracker(alarm_seconds=1.0)
    assert tracker.update(eyes_closed=True, timestamp=0.0) is False
    assert tracker.update(eyes_closed=True, timestamp=0.3) is False
    assert tracker.update(eyes_closed=False, timestamp=0.4) is False


def test_eye_closure_rejects_bad_threshold() -> None:
    with pytest.raises(ValueError):
        EyeClosureTracker(alarm_seconds=0.0)
