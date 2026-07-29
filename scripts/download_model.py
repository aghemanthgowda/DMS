"""Download the MediaPipe ``face_landmarker_v2`` task bundle into ``models/``.

The ``.task`` bundle is intentionally not committed to the repository (it is
gitignored). Run this once before starting the monitor::

    python scripts/download_model.py
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
DEFAULT_DEST = Path("models") / "face_landmarker_v2.task"


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
    """Parse arguments and download the model bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", default=MODEL_URL, help="model bundle URL to download"
    )
    parser.add_argument(
        "--dest", type=Path, default=DEFAULT_DEST, help="destination path"
    )
    args = parser.parse_args()

    if args.dest.exists():
        print(f"{args.dest} already exists; nothing to do.")
        return 0

    try:
        download(args.url, args.dest)
    except urllib.error.URLError as error:
        print(f"Download failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
