"""Structured logging helpers for the DMS."""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def get_logger(name: str = "dms", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger, attaching a stream handler only once.

    Args:
        name: Logger name.
        level: Logging level (e.g. ``logging.INFO``).

    Returns:
        A :class:`logging.Logger` with a single stream handler.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def level_from_name(name: str) -> int:
    """Translate a level name (e.g. ``"INFO"``) to its numeric value.

    Args:
        name: Case-insensitive level name.

    Returns:
        The numeric logging level.

    Raises:
        ValueError: If ``name`` is not a recognised level.
    """
    value = logging.getLevelName(name.upper())
    if not isinstance(value, int):
        raise ValueError(f"unknown log level {name!r}")
    return value
