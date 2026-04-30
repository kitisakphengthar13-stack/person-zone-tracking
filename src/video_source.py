from __future__ import annotations

import time
from dataclasses import dataclass

import cv2

from config import AppConfig


@dataclass(frozen=True)
class FramePacket:
    frame: object
    frame_index: int
    timestamp: float


class VideoSource:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.capture: cv2.VideoCapture | None = None
        self.frame_index = 0
        self.fps = 0.0
        self.width = 0
        self.height = 0

    def open(self) -> None:
        if self.config.source_type == "video":
            if self.config.source_path is None:
                raise ValueError("source_path is required for video input.")
            if not self.config.source_path.exists():
                raise FileNotFoundError(
                    f"Video file not found: {self.config.source_path}"
                )
            self.capture = cv2.VideoCapture(str(self.config.source_path))
            source_label = str(self.config.source_path)
        elif self.config.source_type == "webcam":
            self.capture = cv2.VideoCapture(int(self.config.camera_id))
            source_label = f"camera {self.config.camera_id}"
        else:
            raise ValueError(f"Unsupported source_type: {self.config.source_type}")

        if self.capture is None or not self.capture.isOpened():
            raise RuntimeError(f"Unable to open {self.config.source_type}: {source_label}")

        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if self.fps <= 0:
            self.fps = 30.0

        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.frame_index = 0

    def read(self) -> FramePacket | None:
        if self.capture is None:
            raise RuntimeError("Video source is not open.")

        ok, frame = self.capture.read()
        if not ok or frame is None:
            return None

        current_index = self.frame_index
        timestamp = (
            time.time()
            if self.config.source_type == "webcam"
            else current_index / max(self.fps, 1e-6)
        )
        self.frame_index += 1
        return FramePacket(frame=frame, frame_index=current_index, timestamp=timestamp)

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def is_open(self) -> bool:
        return self.capture is not None and self.capture.isOpened()
