"""Append-only JSONL recorder for driver-state events.

Persisting events (state transitions, alarms, distraction flags) to a JSON Lines
file gives a lightweight audit trail that can be replayed or analysed offline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class EventRecorder:
    """Append driver-state events to a JSON Lines file."""

    def __init__(self, path: str | Path) -> None:
        """Create a recorder, ensuring the parent directory exists.

        Args:
            path: Destination ``.jsonl`` file path.
        """
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        """The file events are written to."""
        return self._path

    def record(self, event: dict[str, Any]) -> None:
        """Append a single event as one JSON line.

        Args:
            event: A JSON-serialisable mapping.
        """
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        """Return every recorded event, or an empty list if none exist yet."""
        if not self._path.exists():
            return []
        with self._path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
