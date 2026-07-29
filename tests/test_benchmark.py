"""Tests for the metrics micro-benchmark."""

from __future__ import annotations

import pytest

from src.benchmark import benchmark_average_ear


def test_benchmark_returns_positive_throughput() -> None:
    ips = benchmark_average_ear(100)
    assert ips > 0.0


def test_benchmark_rejects_non_positive_iterations() -> None:
    with pytest.raises(ValueError):
        benchmark_average_ear(0)
