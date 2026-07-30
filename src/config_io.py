"""Serialise the DMS :class:`~src.config.Config` to and from JSON.

Lets a driver's calibrated thresholds and camera settings be saved to disk and
reloaded on the next run instead of being re-entered each time.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import Config, LandmarkIndices, Thresholds


def config_to_dict(config: Config) -> dict[str, Any]:
    """Return a plain-``dict`` representation of ``config``."""
    return asdict(config)


def save_config(config: Config, path: str | Path) -> None:
    """Write ``config`` to ``path`` as indented JSON."""
    Path(path).write_text(
        json.dumps(config_to_dict(config), indent=2), encoding="utf-8"
    )


def load_config(path: str | Path) -> Config:
    """Load a :class:`Config` from a JSON file, filling gaps with defaults.

    Args:
        path: Path to a JSON file produced by :func:`save_config` (or a partial
            subset of the same shape).

    Returns:
        The reconstructed :class:`Config`.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    thresholds = Thresholds(**data.get("thresholds", {}))
    indices_data = data.get("indices", {})
    indices = LandmarkIndices(
        **{
            key: (tuple(value) if isinstance(value, list) else value)
            for key, value in indices_data.items()
        }
    )
    scalars = {
        key: value
        for key, value in data.items()
        if key not in ("thresholds", "indices")
    }
    return Config(thresholds=thresholds, indices=indices, **scalars)
