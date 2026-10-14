"""HTTP API for uploaded-video and webcam object tracking."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from obj_tracking.pipeline import process_frame, run_pipeline

SUPPORTED_VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
MAX_VIDEO_SIZE = 500 * 1024 * 1024


def create_app(model_path: str = "yolo11n.pt") -> FastAPI:
    app = FastAPI(
        title="Real-Time Object Tracking API",
        description="Track uploaded videos or stream annotated webcam frames with YOLO and DeepSORT.",
        version="0.1.0",
    )
    app.state.model_path = model_path
    app.state.model = None
    app.state.model_lock = Lock()
    app.state.inference_lock = Lock()
    frontend_dir = Path(__file__).parent / "static"

    def get_runtime() -> tuple[Any, Any, Any]:
        try:
            import cv2
            from deep_sort_realtime.deepsort_tracker import DeepSort
            from ultralytics import YOLO
        except ImportError as exc:
            raise HTTPException(
                status_code=503,
                detail="Tracking dependencies are missing. Install the project with `python -m pip install -e .`.",
            ) from exc

        if app.state.model is None:
            with app.state.model_lock:
                if app.state.model is None:
                    app.state.model = YOLO(app.state.model_path)
        return cv2, app.state.model, DeepSort

    @app.get("/health")
    def health() -> dict[str, str | bool]:
        return {"status": "ok", "model_loaded": app.state.model is not None}

    @app.post("/track", response_class=FileResponse)
    def track_upload(
        request: Request,
        video: UploadFile = File(..., description="Video file to process"),
        confidence: float = Form(0.3, ge=0, le=1),
    ) -> FileResponse:
        suffix = Path(video.filename or "").suffix.lower()
        if suffix not in SUPPORTED_VIDEO_SUFFIXES:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported video extension. Use one of: {', '.join(sorted(SUPPORTED_VIDEO_SUFFIXES))}",
            )

        temporary_directory = tempfile.mkdtemp(prefix="obj-tracking-")
        input_path = Path(temporary_directory) / f"input{suffix}"
        output_path = Path(temporary_directory) / "tracked.mp4"
        try:
            with input_path.open("wb") as destination:
                size = 0
                while chunk := video.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_VIDEO_SIZE:
                        raise HTTPException(status_code=413, detail="Video uploads must be 500 MB or smaller")
                    destination.write(chunk)

            _, model, tracker_type = get_runtime()
            frame_count = run_pipeline(
                source=str(input_path),
                model_path=request.app.state.model_path,
                output_path=str(output_path),
                confidence=confidence,
                display=False,
                model=model,
                tracker=tracker_type(),
                inference_lock=request.app.state.inference_lock,
            )
            if frame_count == 0:
                raise HTTPException(status_code=422, detail="No video frames could be read from the upload")
        except HTTPException:
            shutil.rmtree(temporary_directory, ignore_errors=True)
            raise
        except (RuntimeError, ValueError) as exc:
            shutil.rmtree(temporary_directory, ignore_errors=True)
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename="tracked.mp4",
            background=BackgroundTask(shutil.rmtree, temporary_directory, ignore_errors=True),
        )

    @app.get("/stream")
    def stream_webcam(
        source: int = Query(0, ge=0, description="Local webcam device index"),
        confidence: float = Query(0.3, ge=0, le=1),
    ) -> StreamingResponse:
        cv2, model, tracker_type = get_runtime()
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            capture.release()
            raise HTTPException(status_code=503, detail=f"Could not open webcam source {source}")
        tracker = tracker_type()

        def frames():
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    frame = process_frame(
                        frame,
                        model,
                        tracker,
                        confidence,
                        app.state.inference_lock,
                    )
                    encoded, buffer = cv2.imencode(".jpg", frame)
                    if not encoded:
                        raise RuntimeError("Could not encode webcam frame as JPEG")
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            finally:
                capture.release()

        return StreamingResponse(
            frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/", include_in_schema=False)
    def frontend() -> FileResponse:
        return FileResponse(frontend_dir / "index.html")

    app.mount("/assets", StaticFiles(directory=frontend_dir), name="assets")
    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the object tracking HTTP API.")
    parser.add_argument("--host", default="127.0.0.1", help="Interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model or weights path")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    try:
        import uvicorn
    except ImportError as exc:
        parser.exit(1, f"error: Install the project dependencies with `python -m pip install -e .`: {exc}\n")

    uvicorn.run(create_app(args.model), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
