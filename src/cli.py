"""Reusable command-line argument parser for the DMS entry points."""

from __future__ import annotations

import argparse

from .config import Config


def build_parser() -> argparse.ArgumentParser:
    """Build the DMS argument parser.

    Returns:
        An :class:`argparse.ArgumentParser` with all DMS options registered.
    """
    parser = argparse.ArgumentParser(description="Driver Monitoring System")
    parser.add_argument(
        "--camera", type=int, default=0, help="camera index (default: 0)"
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="sample the driver's baseline EAR for 5 seconds at startup",
    )
    parser.add_argument(
        "--no-audio", action="store_true", help="disable the audio alarm"
    )
    parser.add_argument(
        "--ir", action="store_true", help="treat the camera as an IR/NIR source"
    )
    parser.add_argument(
        "--model",
        default=Config.model_path,
        help="path to the face_landmarker_v2 .task bundle",
    )
    parser.add_argument(
        "--log-level", default="INFO", help="logging level (e.g. INFO, DEBUG)"
    )
    parser.add_argument(
        "--record",
        default=None,
        help="optional path to a JSONL file for recording driver-state events",
    )
    return parser
