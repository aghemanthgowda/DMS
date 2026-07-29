"""Tests for logging helpers."""

from __future__ import annotations

import logging

import pytest

from src.logging_utils import get_logger, level_from_name


def test_get_logger_returns_logger() -> None:
    logger = get_logger("dms-test")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "dms-test"


def test_get_logger_does_not_duplicate_handlers() -> None:
    first = get_logger("dms-test-nodup")
    handler_count = len(first.handlers)
    second = get_logger("dms-test-nodup")
    assert second is first
    assert len(second.handlers) == handler_count


def test_level_from_name_valid() -> None:
    assert level_from_name("info") == logging.INFO
    assert level_from_name("DEBUG") == logging.DEBUG


def test_level_from_name_invalid() -> None:
    with pytest.raises(ValueError):
        level_from_name("not-a-level")
