from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from ultralytics import YOLO

from config import AppConfig
from tracker import YOLOBuiltInTracker
from utils import log_info, log_warn


@dataclass(frozen=True)
class Detection:
    bbox: tuple[float, float, float, float]
    class_id: int
    class_name: str
    confidence: float
    track_id: int | None


class Detector:
    def __init__(self, config: AppConfig) -> None:
        if not config.model_path.exists():
            raise FileNotFoundError(
                "YOLO model file not found: "
                f"{config.model_path}. Place your weights there or pass --model-path. "
                "No model weights are downloaded automatically."
            )

        self.config = config
        self.model = YOLO(str(config.model_path))
        self.class_names = self._extract_class_names(self.model.names)
        self.target_classes = {name.lower() for name in config.target_classes}
        self.warned_missing_track_ids = False
        self.tracker = YOLOBuiltInTracker(
            model=self.model,
            conf=config.conf,
            iou=config.iou,
            device=config.device,
            imgsz=config.imgsz,
            tracker_config=config.tracker_config,
        )
        log_info(f"Loaded YOLO model: {config.model_path}")

    def detect(self, frame: object) -> list[Detection]:
        results = self.tracker.track(frame)
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = _to_numpy(boxes.xyxy)
        class_ids = _to_numpy(boxes.cls)
        confidences = _to_numpy(boxes.conf)
        track_ids = _to_numpy(getattr(boxes, "id", None))

        if track_ids is None and not self.warned_missing_track_ids:
            log_warn(
                "YOLO tracking did not return track IDs for this frame. "
                "Dwell time will be skipped for detections without IDs."
            )
            self.warned_missing_track_ids = True

        detections: list[Detection] = []
        for index, bbox in enumerate(xyxy):
            class_id = int(class_ids[index])
            class_name = self.class_names.get(class_id, str(class_id))
            confidence = float(confidences[index])

            if confidence < self.config.conf:
                continue
            if class_name.lower() not in self.target_classes:
                continue

            detections.append(
                Detection(
                    bbox=tuple(float(v) for v in bbox),
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    track_id=_safe_track_id(track_ids, index),
                )
            )

        return detections

    @staticmethod
    def _extract_class_names(names: Any) -> dict[int, str]:
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}
        if isinstance(names, (list, tuple)):
            return {index: str(name) for index, name in enumerate(names)}
        return {}


def _to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _safe_track_id(track_ids: np.ndarray | None, index: int) -> int | None:
    if track_ids is None:
        return None
    try:
        value = float(track_ids[index])
    except (TypeError, ValueError, IndexError):
        return None
    if np.isnan(value):
        return None
    return int(value)
