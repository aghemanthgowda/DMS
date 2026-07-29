"""Tests for the CLI argument parser."""

from __future__ import annotations

from src.cli import build_parser


def test_defaults() -> None:
    args = build_parser().parse_args([])
    assert args.camera == 0
    assert args.calibrate is False
    assert args.no_audio is False
    assert args.ir is False
    assert args.log_level == "INFO"
    assert args.record is None


def test_flags_and_values() -> None:
    args = build_parser().parse_args(
        ["--camera", "2", "--calibrate", "--no-audio", "--ir",
         "--log-level", "DEBUG", "--record", "events.jsonl"]
    )
    assert args.camera == 2
    assert args.calibrate is True
    assert args.no_audio is True
    assert args.ir is True
    assert args.log_level == "DEBUG"
    assert args.record == "events.jsonl"
