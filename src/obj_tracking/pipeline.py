"""YOLO detection and DeepSORT tracking for video streams."""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def parse_source(value: str) -> str | int:
    """Treat a non-negative integer source as a camera index."""
    return int(value) if value.isdecimal() else value


def _scalar(value: Any) -> float:
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _coordinates(value: Any) -> list[float]:
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], Sequence):
        value = value[0]
    if len(value) != 4:
        raise ValueError(f"Expected four box coordinates, received {len(value)}")
    return [float(coordinate) for coordinate in value]


def detections_from_result(result: Any, names: Mapping[int, str] | Sequence[str]) -> list[tuple[list[float], float, str]]:
    """Convert one Ultralytics result into DeepSORT's xywh detections."""
    detections: list[tuple[list[float], float, str]] = []
    for box in result.boxes:
        x1, y1, x2, y2 = _coordinates(box.xyxy[0])
        confidence = _scalar(box.conf[0])
        class_id = int(_scalar(box.cls[0]))
        label = names[class_id]
        detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, str(label)))
    return detections


def _valid_confidence(value: str) -> float:
    try:
        confidence = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("confidence must be a number between 0 and 1") from exc
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise argparse.ArgumentTypeError("confidence must be a number between 0 and 1")
    return confidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Track objects in a video or webcam stream with YOLO and DeepSORT."
    )
    parser.add_argument("--source", required=True, help="Video path or camera index (for example, 0)")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model or weights path")
    parser.add_argument("--output", help="Optional path for the annotated output video")
    parser.add_argument("--confidence", type=_valid_confidence, default=0.3, help="YOLO confidence threshold")
    parser.add_argument("--no-display", action="store_true", help="Do not open a preview window")
    return parser


def run_pipeline(
    source: str | int,
    model_path: str,
    output_path: str | None,
    confidence: float,
    display: bool,
) -> int:
    try:
        import cv2
        from deep_sort_realtime.deepsort_tracker import DeepSort
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Pipeline dependencies are missing. Install them with `python -m pip install -e .`."
        ) from exc

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open video source: {source}")

    output = None
    window_name = "YOLO + DeepSORT"
    try:
        model = YOLO(model_path)
        tracker = DeepSort()
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0

        if output_path:
            destination = Path(output_path)
            if isinstance(source, str) and Path(source).exists():
                if destination.resolve() == Path(source).resolve():
                    raise ValueError("Output path must be different from the input video path")
            destination.parent.mkdir(parents=True, exist_ok=True)
            output = cv2.VideoWriter(
                str(destination),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            if not output.isOpened():
                raise RuntimeError(f"Could not create output video: {destination}")

        frame_count = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            result = model.predict(frame, conf=confidence, verbose=False)[0]
            detections = detections_from_result(result, model.names)
            tracks = tracker.update_tracks(detections, frame=frame)
            for track in tracks:
                if not track.is_confirmed():
                    continue
                left, top, right, bottom = (int(value) for value in track.to_ltrb())
                cv2.rectangle(frame, (left, top), (right, bottom), (40, 220, 40), 2)
                label = f"ID {track.track_id}"
                class_name = track.get_det_class()
                if class_name:
                    label = f"{class_name} {label}"
                cv2.putText(
                    frame,
                    label,
                    (left, max(20, top - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (40, 220, 40),
                    2,
                )

            if output is not None:
                output.write(frame)
            if display:
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_count += 1
    finally:
        capture.release()
        if output is not None:
            output.release()
        if display:
            cv2.destroyAllWindows()

    return frame_count


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.no_display and not args.output:
        parser.error("--no-display requires --output so processed frames are saved")

    try:
        count = run_pipeline(
            source=parse_source(args.source),
            model_path=args.model,
            output_path=args.output,
            confidence=args.confidence,
            display=not args.no_display,
        )
    except (RuntimeError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(f"Processed {count} frame(s)")


if __name__ == "__main__":
    main()
