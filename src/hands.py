"""MediaPipe Tasks ``HandLandmarker`` wrapper (LIVE_STREAM mode).

Mirrors :class:`~src.landmarks.FaceLandmarkerStream`: frames are submitted
asynchronously and the newest 21-point-per-hand result is stored behind a lock
for the main loop to read.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import TracebackType
from typing import Optional

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


class HandLandmarkerStream:
    """Async multi-hand landmarker."""

    def __init__(self, model_path: str, num_hands: int = 2) -> None:
        """Create the landmarker from a downloaded ``.task`` bundle.

        Args:
            model_path: Path to the ``hand_landmarker`` ``.task`` bundle.
            num_hands: Maximum number of hands to track.

        Raises:
            FileNotFoundError: If the model bundle does not exist.
        """
        if not Path(model_path).is_file():
            raise FileNotFoundError(
                f"hand model bundle not found at {model_path!r}; run "
                "scripts/download_model.py first"
            )
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=num_hands,
            result_callback=self._on_result,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._lock = threading.Lock()
        self._latest: Optional[vision.HandLandmarkerResult] = None

    def _on_result(
        self,
        result: vision.HandLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ) -> None:
        """LIVE_STREAM result callback; stores the newest result."""
        del output_image, timestamp_ms  # unused, required by the callback signature
        with self._lock:
            self._latest = result

    def detect_async(self, frame_rgb: np.ndarray, timestamp_ms: int) -> None:
        """Submit an RGB frame for asynchronous hand detection."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._landmarker.detect_async(mp_image, timestamp_ms)

    def latest_result(self) -> Optional[vision.HandLandmarkerResult]:
        """Return the most recent result, or ``None`` if none has arrived."""
        with self._lock:
            return self._latest

    def close(self) -> None:
        """Release the underlying landmarker."""
        self._landmarker.close()

    def __enter__(self) -> "HandLandmarkerStream":
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.close()
