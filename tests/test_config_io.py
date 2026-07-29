"""Tests for Config JSON serialisation."""

from __future__ import annotations

from pathlib import Path

from src.config import Config, Thresholds
from src.config_io import config_to_dict, load_config, save_config


def test_round_trip_preserves_config(tmp_path: Path) -> None:
    original = Config(camera_index=2, thresholds=Thresholds(ear_closed=0.18))
    path = tmp_path / "config.json"
    save_config(original, path)
    restored = load_config(path)
    assert restored == original


def test_config_to_dict_is_json_friendly() -> None:
    data = config_to_dict(Config())
    assert data["camera_index"] == 0
    assert data["thresholds"]["ear_closed"] == Config().thresholds.ear_closed


def test_load_with_partial_data_uses_defaults(tmp_path: Path) -> None:
    path = tmp_path / "partial.json"
    path.write_text('{"camera_index": 5}', encoding="utf-8")
    config = load_config(path)
    assert config.camera_index == 5
    assert config.thresholds == Thresholds()  # defaults filled in
