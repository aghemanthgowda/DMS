"""Driver Monitoring System entry point.

Wires the threaded camera, the MediaPipe FaceLandmarker stream, the geometric
metrics, the drowsiness state machine, and the audio alarm into a single live
loop with an on-frame overlay.

Usage::

    python main.py --camera 0
    python main.py --calibrate        # sample a baseline EAR for 5 seconds
    python main.py --no-audio
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import cv2
import numpy as np

from src.alert import AudioAlarm
from src.capture import VideoStream
from src.config import Config, Thresholds
from src.draw import draw_box, draw_connections, draw_points
from src.gesture import detect_hand_gesture
from src.hands import HandLandmarkerStream
from src.objects import ObjectDetector, safety_violations
from src.landmarks import FaceLandmarkerStream
from src.metrics import (
    average_ear,
    blendshape_scores,
    default_camera_matrix,
    estimate_head_pose,
    landmarks_to_array,
    mouth_aspect_ratio,
    yawn_from_blendshapes,
)
from src.state import (
    DistractionTracker,
    DrowsinessMonitor,
    DrowsinessState,
    EyeClosureTracker,
)

# Import the landmark connection constants directly from their submodules.
# ``import mediapipe as mp; mp.solutions`` is not always populated (e.g. on some
# Windows/newer builds it raises AttributeError), so we avoid that access path.
try:
    from mediapipe.python.solutions.face_mesh_connections import (
        FACEMESH_TESSELATION as FACE_CONNECTIONS,
    )
    from mediapipe.python.solutions.hands_connections import HAND_CONNECTIONS
except ImportError:  # pragma: no cover - overlay lines are optional
    FACE_CONNECTIONS = frozenset()
    HAND_CONNECTIONS = frozenset()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
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
        "--model",
        default=Config.model_path,
        help="path to the face_landmarker_v2 .task bundle",
    )
    return parser.parse_args()


def calibrate_baseline(
    stream: VideoStream,
    landmarker: FaceLandmarkerStream,
    config: Config,
) -> float:
    """Sample the driver's open-eye EAR baseline for ``calibration_seconds``.

    Args:
        stream: A started :class:`VideoStream`.
        landmarker: A live :class:`FaceLandmarkerStream`.
        config: Active configuration (for indices and duration).

    Returns:
        The median EAR observed during calibration, or ``0.0`` if no face was
        seen.
    """
    print(
        f"Calibrating: keep your eyes open for "
        f"{config.thresholds.calibration_seconds:.0f} seconds..."
    )
    samples: list[float] = []
    start = time.monotonic()
    while time.monotonic() - start < config.thresholds.calibration_seconds:
        frame = stream.read()
        if frame is None:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarker.detect_async(rgb, int(time.monotonic() * 1000))
        result = landmarker.latest_result()
        if result is None or not result.face_landmarks:
            continue
        points = landmarks_to_array(
            result.face_landmarks[0], frame.shape[1], frame.shape[0]
        )
        samples.append(
            average_ear(points, config.indices.left_eye, config.indices.right_eye)
        )
    baseline = float(np.median(samples)) if samples else 0.0
    print(f"Calibration complete: baseline EAR = {baseline:.3f}")
    return baseline


def draw_overlay(
    frame: np.ndarray,
    fps: float,
    ear: float,
    mar: float,
    perclos: float,
    state: DrowsinessState,
    yawn: bool = False,
    yaw: float = 0.0,
    distracted: bool = False,
    microsleep: bool = False,
    eyes_closed_seconds: float = 0.0,
    hand_at_mouth: bool = False,
    hand_at_ear: bool = False,
    detected_objects: tuple[str, ...] = (),
    object_alert: bool = False,
) -> None:
    """Draw the metrics HUD onto ``frame`` in place."""
    alert = (
        state is DrowsinessState.DROWSY
        or distracted
        or microsleep
        or hand_at_mouth
        or hand_at_ear
        or object_alert
    )
    colour = (0, 0, 255) if alert else (0, 255, 0)
    status = state.value
    if microsleep:
        status += " | MICROSLEEP"
    if distracted:
        status += " | DISTRACTED"
    if hand_at_mouth:
        status += " | PHONE/SMOKING"
    if hand_at_ear:
        status += " | PHONE-CALL"
    lines = [
        f"FPS:     {fps:5.1f}",
        f"EAR:     {ear:5.3f}",
        f"MAR:     {mar:5.3f}",
        f"PERCLOS: {perclos * 100:5.1f}%",
        f"EYES-SHUT: {eyes_closed_seconds:4.1f}s",
        f"YAW:     {yaw:5.1f} deg",
        f"HAND@MOUTH: {'yes' if hand_at_mouth else 'no'}",
        f"HAND@EAR: {'yes' if hand_at_ear else 'no'}",
        f"OBJECTS: {', '.join(detected_objects) if detected_objects else 'none'}",
        f"STATE:   {status}",
        f"YAWN:    {'yes' if yawn else 'no'}",
    ]
    for i, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (10, 30 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            colour,
            2,
            cv2.LINE_AA,
        )


def run(config: Config) -> None:
    """Run the live monitoring loop until the user presses ``q``."""
    with VideoStream(
        config.camera_index, config.frame_width, config.frame_height
    ).start() as stream, FaceLandmarkerStream(config.model_path) as landmarker:
        thresholds = config.thresholds
        if config.calibrate:
            baseline = calibrate_baseline(stream, landmarker, config)
            if baseline > 0.0:
                thresholds = replace(
                    thresholds,
                    ear_closed=baseline * thresholds.calibration_ear_factor,
                )
                config = replace(config, thresholds=thresholds)

        monitor = DrowsinessMonitor(thresholds)
        closure = EyeClosureTracker(thresholds.eye_closed_alarm_seconds)
        distraction = DistractionTracker(
            thresholds.yaw_distraction_deg, thresholds.distraction_seconds
        )
        alarm = AudioAlarm(config.alarm_sound, enabled=config.audio_enabled)
        camera_matrix = default_camera_matrix(
            config.frame_width, config.frame_height
        )

        # Hand tracking is optional: run without it if the model is missing.
        hand_stream: HandLandmarkerStream | None = None
        try:
            hand_stream = HandLandmarkerStream(
                config.hand_model_path, num_hands=config.max_hands
            )
        except FileNotFoundError:
            print(
                "Hand model not found; running without hand tracking. "
                "Run scripts/download_model.py to enable it."
            )

        # Object detection is optional: skip if ultralytics/model is unavailable.
        detector: ObjectDetector | None = None
        try:
            detector = ObjectDetector(
                config.object_model_path, config.object_confidence
            )
        except (ImportError, FileNotFoundError, OSError) as error:
            print(f"Object detection disabled ({error}); pip install ultralytics.")

        detections: list = []
        frame_index = 0
        last_time = time.monotonic()
        fps = 0.0

        try:
            while True:
                frame = stream.read()
                if frame is None:
                    continue

                now = time.monotonic()
                dt = now - last_time
                last_time = now
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

                width, height = frame.shape[1], frame.shape[0]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp_ms = int(now * 1000)
                landmarker.detect_async(rgb, timestamp_ms)
                if hand_stream is not None:
                    hand_stream.detect_async(rgb, timestamp_ms)
                result = landmarker.latest_result()

                ear = mar = 0.0
                yawn = False
                yaw = 0.0
                distracted = False
                eyes_closed = False
                points = None
                face_present = result is not None and bool(result.face_landmarks)
                if face_present:
                    points = landmarks_to_array(
                        result.face_landmarks[0], width, height
                    )
                    ear = average_ear(
                        points, config.indices.left_eye, config.indices.right_eye
                    )
                    eyes_closed = ear < config.thresholds.ear_closed
                    mar = mouth_aspect_ratio(points, config.indices.mouth)
                    # Cross-check the geometric MAR against the learned jawOpen
                    # blendshape: a yawn is flagged only when both agree.
                    yawn_geom = monitor.is_yawning(mar)
                    yawn_bs = False
                    if result.face_blendshapes:
                        scores = blendshape_scores(result.face_blendshapes[0])
                        yawn_bs = yawn_from_blendshapes(
                            scores, config.thresholds.jaw_open_blendshape
                        )
                    yawn = yawn_geom and yawn_bs

                    pose = estimate_head_pose(points, camera_matrix)
                    if pose is not None:
                        yaw, _pitch, _roll = pose
                        distracted = distraction.update(yaw, now)

                # Collect hand landmarks (pixel space) for drawing + gestures.
                hands_px: list[np.ndarray] = []
                if hand_stream is not None:
                    hand_result = hand_stream.latest_result()
                    if hand_result is not None and hand_result.hand_landmarks:
                        hands_px = [
                            landmarks_to_array(hand, width, height)
                            for hand in hand_result.hand_landmarks
                        ]

                # Phone / smoking heuristic from hand-to-face proximity.
                hand_at_mouth = False
                hand_at_ear = False
                if face_present and points is not None and hands_px:
                    idx = config.indices
                    face_width = float(
                        np.linalg.norm(points[idx.left_ear] - points[idx.right_ear])
                    )
                    for hand_pts in hands_px:
                        gesture = detect_hand_gesture(
                            hand_pts,
                            points[idx.mouth_center],
                            points[idx.left_ear],
                            points[idx.right_ear],
                            face_width,
                            config.thresholds.hand_mouth_factor,
                            config.thresholds.hand_ear_factor,
                        )
                        hand_at_mouth = hand_at_mouth or gesture.hand_at_mouth
                        hand_at_ear = hand_at_ear or gesture.hand_at_ear

                # YOLO object detection, run every N frames (heavier than the
                # landmark passes). Detections persist between runs for display.
                frame_index += 1
                if (
                    detector is not None
                    and frame_index % config.detect_every_n_frames == 0
                ):
                    detections = detector.detect(frame)
                violations = safety_violations(detections)

                # Low-latency microsleep alarm: reacts within ~1s of the eyes
                # closing, independent of the slower PERCLOS fatigue measure.
                microsleep = closure.update(eyes_closed, now)
                state = monitor.update(ear, mar, now)
                if state is DrowsinessState.DROWSY or distracted or microsleep:
                    alarm.start()
                else:
                    alarm.stop()

                # Overlay the landmark maps for the demo.
                if face_present and points is not None:
                    draw_connections(
                        frame, points, FACE_CONNECTIONS, color=(0, 255, 0), thickness=1
                    )
                for hand_pts in hands_px:
                    draw_connections(
                        frame, hand_pts, HAND_CONNECTIONS, color=(255, 0, 0), thickness=2
                    )
                    draw_points(frame, hand_pts, color=(0, 0, 255), radius=3)
                for det in detections:
                    box_colour = (0, 0, 255) if det.label in violations else (0, 165, 255)
                    draw_box(frame, det.box, f"{det.label} {det.confidence:.2f}", box_colour)

                draw_overlay(
                    frame,
                    fps,
                    ear,
                    mar,
                    monitor.perclos,
                    state,
                    yawn,
                    yaw,
                    distracted,
                    microsleep,
                    closure.closed_duration,
                    hand_at_mouth,
                    hand_at_ear,
                    tuple(sorted({det.label for det in detections})),
                    bool(violations),
                )
                cv2.imshow("Driver Monitoring System", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            alarm.close()
            if hand_stream is not None:
                hand_stream.close()
            cv2.destroyAllWindows()


def main() -> None:
    """Parse arguments and start monitoring."""
    args = parse_args()
    config = Config(
        camera_index=args.camera,
        model_path=args.model,
        audio_enabled=not args.no_audio,
        calibrate=args.calibrate,
    )
    run(config)


if __name__ == "__main__":
    main()
