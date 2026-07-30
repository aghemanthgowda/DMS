"""Configuration objects and tunable thresholds for the DMS pipeline.

All tunable values live here as immutable dataclasses so they can be passed
around explicitly, serialised, or overridden from the command line without
reaching for module-level globals.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LandmarkIndices:
    """MediaPipe FaceLandmarker (478-point) indices used by the metrics.

    Each eye tuple is ordered ``(p1, p2, p3, p4, p5, p6)`` where ``p1`` and
    ``p4`` are the horizontal eye corners and the remaining points are the
    upper/lower lid pairs, matching the classic 6-point EAR formulation.
    The mouth tuple follows the same ``(p1..p6)`` convention.
    """

    left_eye: tuple[int, int, int, int, int, int] = (362, 385, 387, 263, 373, 380)
    right_eye: tuple[int, int, int, int, int, int] = (33, 160, 158, 133, 153, 144)
    mouth: tuple[int, int, int, int, int, int] = (61, 0, 13, 291, 14, 17)

    mouth_center: int = 13
    """Upper-lip-centre landmark, used as the mouth reference for gestures."""

    left_ear: int = 454
    """Left side-of-face landmark (~ear), used for phone/head-width geometry."""

    right_ear: int = 234
    """Right side-of-face landmark (~ear), used for phone/head-width geometry."""


@dataclass(frozen=True)
class Thresholds:
    """Decision thresholds for the drowsiness/distraction logic."""

    ear_closed: float = 0.21
    """EAR below this value marks the eye as closed."""

    mar_yawn: float = 0.60
    """MAR above this value marks the mouth as yawning."""

    perclos_drowsy: float = 0.40
    """PERCLOS at/above which the driver enters the DROWSY state."""

    perclos_recover: float = 0.30
    """PERCLOS at/below which the driver returns to AWAKE (hysteresis)."""

    perclos_window_seconds: float = 60.0
    """Length of the rolling time window used to estimate PERCLOS."""

    eye_closed_alarm_seconds: float = 1.0
    """Continuous eye-closure duration that triggers an immediate microsleep alarm."""

    calibration_seconds: float = 5.0
    """Seconds of baseline sampling when ``--calibrate`` is used."""

    calibration_ear_factor: float = 0.75
    """Closed-eye threshold as a fraction of the calibrated open-eye baseline."""

    blink_blendshape: float = 0.50
    """Mean eye-blink blendshape score at/above which eyes are deemed closed."""

    jaw_open_blendshape: float = 0.50
    """``jawOpen`` blendshape score at/above which a yawn is deemed present."""

    yaw_distraction_deg: float = 30.0
    """Absolute head yaw (degrees) beyond which the driver is looking away."""

    distraction_seconds: float = 2.0
    """Seconds of sustained off-axis yaw before distraction is flagged."""

    hand_mouth_factor: float = 0.55
    """Hand-to-mouth distance threshold as a fraction of face width."""

    hand_ear_factor: float = 0.50
    """Hand-to-ear distance threshold as a fraction of face width."""

    hand_eye_factor: float = 0.45
    """Hand-to-eye distance threshold as a fraction of face width (eye rubbing)."""


@dataclass(frozen=True)
class Config:
    """Top-level runtime configuration."""

    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    model_path: str = "models/face_landmarker_v2.task"
    hand_model_path: str = "models/hand_landmarker.task"
    max_hands: int = 2
    object_model_path: str = "yolov8n.pt"
    object_confidence: float = 0.30
    detect_every_n_frames: int = 5
    hand_tracking_enabled: bool = True
    object_detection_enabled: bool = True
    debug: bool = False
    alarm_sound: str = "assets/alarm.wav"
    audio_enabled: bool = True
    calibrate: bool = False
    thresholds: Thresholds = field(default_factory=Thresholds)
    indices: LandmarkIndices = field(default_factory=LandmarkIndices)
