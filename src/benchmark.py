"""Micro-benchmark for the geometric metrics.

Useful for sanity-checking that the per-frame maths is far cheaper than the
camera/inference budget on a target machine.
"""

from __future__ import annotations

import time

import numpy as np

from src.config import LandmarkIndices
from src.metrics import average_ear


def benchmark_average_ear(
    iterations: int, landmarks: np.ndarray | None = None
) -> float:
    """Return the throughput of :func:`average_ear` in iterations per second.

    Args:
        iterations: Number of calls to time. Must be positive.
        landmarks: Optional ``(478, 2)`` landmark array; a fixed random array is
            generated when omitted.

    Returns:
        Iterations per second (``inf`` if timing resolution rounds to zero).

    Raises:
        ValueError: If ``iterations`` is not positive.
    """
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    indices = LandmarkIndices()
    if landmarks is None:
        landmarks = np.random.default_rng(0).random((478, 2))
    start = time.perf_counter()
    for _ in range(iterations):
        average_ear(landmarks, indices.left_eye, indices.right_eye)
    elapsed = time.perf_counter() - start
    return iterations / elapsed if elapsed > 0 else float("inf")


def main() -> None:
    """Print the average-EAR throughput."""
    ips = benchmark_average_ear(10_000)
    print(f"average_ear: {ips:,.0f} iterations/sec")


if __name__ == "__main__":
    main()
