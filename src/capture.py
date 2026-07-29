"""Threaded OpenCV camera capture.

Reading frames on a dedicated thread decouples camera I/O from the inference
loop, so a slow ``VideoCapture.read`` never stalls landmark processing. The
newest frame is always available via :meth:`VideoStream.read`.
"""

from __future__ import annotations

import threading
from types import TracebackType
from typing import Optional

import cv2
import numpy as np


class VideoStream:
    """A background-threaded wrapper around :class:`cv2.VideoCapture`."""

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
    ) -> None:
        """Open the camera and configure the requested resolution.

        Args:
            camera_index: Index passed to :class:`cv2.VideoCapture`.
            width: Requested capture width in pixels.
            height: Requested capture height in pixels.

        Raises:
            RuntimeError: If the camera cannot be opened.
        """
        self._capture = cv2.VideoCapture(camera_index)
        if not self._capture.isOpened():
            raise RuntimeError(f"could not open camera index {camera_index}")
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "VideoStream":
        """Start the background grab thread and return ``self``."""
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()
        return self

    def _update(self) -> None:
        """Continuously grab frames until :meth:`stop` is called."""
        while self._running:
            ok, frame = self._capture.read()
            if not ok:
                continue
            with self._lock:
                self._frame = frame

    def read(self) -> Optional[np.ndarray]:
        """Return a copy of the most recent frame, or ``None`` if not ready."""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def stop(self) -> None:
        """Stop the grab thread and release the camera."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._capture.release()

    def __enter__(self) -> "VideoStream":
        return self.start()

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.stop()
