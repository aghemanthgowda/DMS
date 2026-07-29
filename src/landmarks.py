"""MediaPipe Tasks ``FaceLandmarker`` wrapper (LIVE_STREAM mode).

This uses the *current* MediaPipe Tasks vision API
(``mediapipe.tasks.python.vision.FaceLandmarker``) rather than the deprecated
``mp.solutions.face_mesh`` solution. The landmarker runs in
``RunningMode.LIVE_STREAM``: frames are submitted asynchronously and results are
delivered to a callback, which stores the latest result behind a lock for the
main loop to consume.
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


class FaceLandmarkerStream:
    """Async, single-face landmarker returning landmarks and blendshapes."""

    def __init__(self, model_path: str, num_faces: int = 1) -> None:
        """Create the landmarker from a downloaded ``.task`` bundle.

        Args:
            model_path: Path to the ``face_landmarker_v2`` ``.task`` bundle.
            num_faces: Maximum number of faces to track.

        Raises:
            FileNotFoundError: If the model bundle does not exist.
        """
        if not Path(model_path).is_file():
            raise FileNotFoundError(
                f"model bundle not found at {model_path!r}; run "
                "scripts/download_model.py first"
            )

        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_faces=num_faces,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            result_callback=self._on_result,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._lock = threading.Lock()
        self._latest: Optional[vision.FaceLandmarkerResult] = None

    def _on_result(
        self,
        result: vision.FaceLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ) -> None:
        """LIVE_STREAM result callback; stores the newest result."""
        del output_image, timestamp_ms  # unused, required by the callback signature
        with self._lock:
            self._latest = result

    def detect_async(self, frame_rgb: np.ndarray, timestamp_ms: int) -> None:
        """Submit an RGB frame for asynchronous detection.

        Args:
            frame_rgb: Contiguous ``uint8`` RGB image.
            timestamp_ms: Monotonically increasing frame timestamp in ms.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._landmarker.detect_async(mp_image, timestamp_ms)

    def latest_result(self) -> Optional[vision.FaceLandmarkerResult]:
        """Return the most recent result, or ``None`` if none has arrived."""
        with self._lock:
            return self._latest

    def close(self) -> None:
        """Release the underlying landmarker."""
        self._landmarker.close()

    def __enter__(self) -> "FaceLandmarkerStream":
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.close()
