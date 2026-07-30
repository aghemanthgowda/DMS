"""Tests for the object-detection mapping/decision logic.

These cover the pure helpers only; the YOLO model itself (ultralytics) is a heavy
optional dependency exercised at runtime, not in unit tests.
"""

from __future__ import annotations

from src.objects import Detection, safety_violations, to_safety_category


def test_maps_known_coco_labels() -> None:
    assert to_safety_category("cell phone") == "phone"
    assert to_safety_category("Bottle") == "drink"
    assert to_safety_category("banana") == "food"


def test_maps_custom_trained_labels() -> None:
    assert to_safety_category("cigarette") == "cigarette"
    assert to_safety_category("seat belt") == "seatbelt"
    assert to_safety_category("sunglasses") == "sunglasses"


def test_unknown_label_is_none() -> None:
    assert to_safety_category("chair") is None
    assert to_safety_category("person") is None


def test_safety_violations_selects_only_violations() -> None:
    detections = [
        Detection("phone", 0.9, (0, 0, 10, 10)),
        Detection("seatbelt", 0.8, (0, 0, 10, 10)),  # present belt: not a violation
        Detection("cigarette", 0.7, (0, 0, 10, 10)),
        Detection("sunglasses", 0.6, (0, 0, 10, 10)),  # tracked but not a violation
    ]
    assert safety_violations(detections) == {"phone", "cigarette"}


def test_no_seatbelt_is_a_violation() -> None:
    assert safety_violations([Detection("no_seatbelt", 0.9, (0, 0, 1, 1))]) == {
        "no_seatbelt"
    }
