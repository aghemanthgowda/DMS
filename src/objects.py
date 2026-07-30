"""YOLO-based in-cabin object detection for safety-relevant items.

Detects objects such as a phone, food/drink, cigarette, seatbelt, and sunglasses
and maps raw detector labels onto DMS *safety categories*. A pretrained COCO
model (e.g. ``yolov8n.pt``) already covers phone/food/drink out of the box;
cigarette, seatbelt and sunglasses require a custom-trained model (see
``training/``) whose class names are also mapped here.

The heavy ``ultralytics`` import is done lazily inside :class:`ObjectDetector`
so the pure mapping helpers stay importable (and unit-testable) without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

# Raw detector class name (lower-case) -> DMS safety category.
SAFETY_CLASS_MAP: dict[str, str] = {
    # COCO classes available from a stock model:
    "cell phone": "phone",
    "cellphone": "phone",
    "phone": "phone",
    "mobile phone": "phone",
    "bottle": "drink",
    "cup": "drink",
    "wine glass": "drink",
    "sandwich": "food",
    "banana": "food",
    "apple": "food",
    "pizza": "food",
    "donut": "food",
    "hot dog": "food",
    "cake": "food",
    # Custom-trained classes (see training/):
    "cigarette": "cigarette",
    "smoking": "cigarette",
    "seatbelt": "seatbelt",
    "seat belt": "seatbelt",
    "no_seatbelt": "no_seatbelt",
    "sunglasses": "sunglasses",
    "goggles": "sunglasses",
}

# Categories whose presence is a distraction / safety violation.
VIOLATION_CATEGORIES: frozenset[str] = frozenset(
    {"phone", "cigarette", "food", "drink", "no_seatbelt"}
)


@dataclass(frozen=True)
class Detection:
    """A single detected object mapped to a DMS safety category."""

    label: str  # safety category (see SAFETY_CLASS_MAP values)
    confidence: float
    box: tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixels


def to_safety_category(raw_label: str) -> Optional[str]:
    """Map a raw detector label to a safety category, or ``None`` if irrelevant."""
    return SAFETY_CLASS_MAP.get(raw_label.strip().lower())


def safety_violations(detections: Sequence[Detection]) -> set[str]:
    """Return the set of violation categories present in ``detections``."""
    return {d.label for d in detections if d.label in VIOLATION_CATEGORIES}


class ObjectDetector:
    """Thin wrapper around an Ultralytics YOLO model returning :class:`Detection`."""

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.35) -> None:
        """Load a YOLO model.

        Args:
            model_path: Path to a ``.pt`` weights file (stock COCO or custom).
            confidence: Minimum confidence for a detection to be kept.

        Raises:
            ImportError: If ``ultralytics`` is not installed.
        """
        from ultralytics import YOLO  # lazy import — heavy optional dependency

        self._model = YOLO(model_path)
        self._confidence = confidence
        self._names = self._model.names

    def _class_name(self, class_id: int) -> str:
        names = self._names
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        return str(names[class_id])

    def detect(self, frame_bgr) -> list[Detection]:
        """Run detection on a BGR frame and return mapped safety detections."""
        results = self._model.predict(
            frame_bgr, conf=self._confidence, verbose=False
        )
        detections: list[Detection] = []
        for result in results:
            for box in result.boxes:
                category = to_safety_category(self._class_name(int(box.cls[0])))
                if category is None:
                    continue
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                detections.append(
                    Detection(category, float(box.conf[0]), (x1, y1, x2, y2))
                )
        return detections
