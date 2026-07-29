"""Tests for the severity fusion logic."""

from __future__ import annotations

from src.severity import Severity, compute_severity
from src.state import DrowsinessState

AWAKE = DrowsinessState.AWAKE
DROWSY = DrowsinessState.DROWSY


def test_none_when_alert_and_low_perclos() -> None:
    assert compute_severity(AWAKE, distracted=False, perclos=0.05) is Severity.NONE


def test_low_when_perclos_creeps_up() -> None:
    assert compute_severity(AWAKE, distracted=False, perclos=0.30) is Severity.LOW


def test_high_when_drowsy_or_distracted() -> None:
    assert compute_severity(DROWSY, distracted=False, perclos=0.45) is Severity.HIGH
    assert compute_severity(AWAKE, distracted=True, perclos=0.05) is Severity.HIGH


def test_critical_when_drowsy_and_distracted() -> None:
    assert compute_severity(DROWSY, distracted=True, perclos=0.45) is Severity.CRITICAL


def test_critical_when_drowsy_and_perclos_extreme() -> None:
    assert compute_severity(DROWSY, distracted=False, perclos=0.8) is Severity.CRITICAL


def test_severity_is_ordered() -> None:
    assert Severity.NONE < Severity.LOW < Severity.HIGH < Severity.CRITICAL
