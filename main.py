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
from src.state import DistractionTracker, DrowsinessMonitor, DrowsinessState


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
) -> None:
    """Draw the metrics HUD onto ``frame`` in place."""
    alert = state is DrowsinessState.DROWSY or distracted
    colour = (0, 0, 255) if alert else (0, 255, 0)
    status = state.value + (" | DISTRACTED" if distracted else "")
    lines = [
        f"FPS:     {fps:5.1f}",
        f"EAR:     {ear:5.3f}",
        f"MAR:     {mar:5.3f}",
        f"PERCLOS: {perclos * 100:5.1f}%",
        f"YAW:     {yaw:5.1f} deg",
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
        distraction = DistractionTracker(
            thresholds.yaw_distraction_deg, thresholds.distraction_seconds
        )
        alarm = AudioAlarm(config.alarm_sound, enabled=config.audio_enabled)
        camera_matrix = default_camera_matrix(
            config.frame_width, config.frame_height
        )
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

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                landmarker.detect_async(rgb, int(now * 1000))
                result = landmarker.latest_result()

                ear = mar = 0.0
                yawn = False
                yaw = 0.0
                distracted = False
                if result is not None and result.face_landmarks:
                    points = landmarks_to_array(
                        result.face_landmarks[0], frame.shape[1], frame.shape[0]
                    )
                    ear = average_ear(
                        points, config.indices.left_eye, config.indices.right_eye
                    )
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

                state = monitor.update(ear, mar, now)
                if state is DrowsinessState.DROWSY or distracted:
                    alarm.start()
                else:
                    alarm.stop()

                draw_overlay(
                    frame, fps, ear, mar, monitor.perclos, state, yawn, yaw, distracted
                )
                cv2.imshow("Driver Monitoring System", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            alarm.close()
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
