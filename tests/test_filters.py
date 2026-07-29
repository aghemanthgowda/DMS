"""Tests for the online signal smoothers."""

from __future__ import annotations

import pytest

from src.filters import ExponentialSmoother, RollingMedian


def test_ema_first_sample_is_passthrough() -> None:
    smoother = ExponentialSmoother(alpha=0.5)
    assert smoother.update(1.0) == pytest.approx(1.0)


def test_ema_blends_toward_new_samples() -> None:
    smoother = ExponentialSmoother(alpha=0.5)
    smoother.update(0.0)
    assert smoother.update(1.0) == pytest.approx(0.5)
    assert smoother.update(1.0) == pytest.approx(0.75)


def test_ema_rejects_bad_alpha() -> None:
    with pytest.raises(ValueError):
        ExponentialSmoother(alpha=0.0)
    with pytest.raises(ValueError):
        ExponentialSmoother(alpha=1.5)


def test_ema_default_value_is_zero() -> None:
    assert ExponentialSmoother(alpha=0.3).value == 0.0


def test_rolling_median() -> None:
    median = RollingMedian(window=3)
    assert median.update(1.0) == pytest.approx(1.0)
    assert median.update(3.0) == pytest.approx(2.0)
    assert median.update(2.0) == pytest.approx(2.0)
    # Window slides: {3, 2, 10} -> median 3.
    assert median.update(10.0) == pytest.approx(3.0)


def test_rolling_median_rejects_bad_window() -> None:
    with pytest.raises(ValueError):
        RollingMedian(window=0)
