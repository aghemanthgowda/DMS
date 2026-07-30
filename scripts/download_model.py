"""Download the MediaPipe ``.task`` bundles used by the DMS into ``models/``.

Downloads both the face-landmarker and hand-landmarker bundles. They are
intentionally not committed (gitignored). Run once before starting the monitor::

    python scripts/download_model.py
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

MODELS: dict[str, str] = {
    "face_landmarker_v2.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    ),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    ),
}
MODELS_DIR = Path("models")


def download(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest``, creating parent directories as needed.

    Args:
        url: Source URL of the ``.task`` bundle.
        dest: Destination path on disk.

    Raises:
        urllib.error.URLError: If the download fails.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}\n  -> {dest}")
    with urllib.request.urlopen(url) as response, dest.open("wb") as out_file:
        out_file.write(response.read())
    print(f"Saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def main() -> int:
    """Download every model bundle that is not already present."""
    for filename, url in MODELS.items():
        dest = MODELS_DIR / filename
        if dest.exists():
            print(f"{dest} already exists; skipping.")
            continue
        try:
            download(url, dest)
        except urllib.error.URLError as error:
            print(f"Download failed for {filename}: {error}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
