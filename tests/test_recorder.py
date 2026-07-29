"""Tests for the JSONL event recorder."""

from __future__ import annotations

from pathlib import Path

from src.recorder import EventRecorder


def test_record_then_read_round_trip(tmp_path: Path) -> None:
    recorder = EventRecorder(tmp_path / "events.jsonl")
    recorder.record({"state": "DROWSY", "perclos": 0.5})
    recorder.record({"state": "AWAKE", "perclos": 0.1})
    events = recorder.read_all()
    assert events == [
        {"perclos": 0.5, "state": "DROWSY"},
        {"perclos": 0.1, "state": "AWAKE"},
    ]


def test_read_all_missing_file_is_empty(tmp_path: Path) -> None:
    recorder = EventRecorder(tmp_path / "nested" / "events.jsonl")
    assert recorder.read_all() == []


def test_parent_directory_is_created(tmp_path: Path) -> None:
    recorder = EventRecorder(tmp_path / "deep" / "dir" / "events.jsonl")
    recorder.record({"ok": True})
    assert recorder.path.exists()
