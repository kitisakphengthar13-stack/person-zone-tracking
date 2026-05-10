from __future__ import annotations

from typing import Any

import numpy as np
from ultralytics import YOLO

from config import AppConfig
from posture_classifier import COCO_KEYPOINT_NAMES
from pose_types import Keypoint, PosePerson
from tracker import YOLOBuiltInTracker
from utils import log_info


class YOLOPoseDetector:
    def __init__(self, config: AppConfig) -> None:
        if not config.posture.model_path.exists():
            raise FileNotFoundError(
                "YOLO pose model file not found: "
                f"{config.posture.model_path}. Place your weights there or configure "
                "posture.model_path. No model weights are downloaded automatically."
            )

        self.config = config
        self.model = YOLO(str(config.posture.model_path))
        self.tracker = YOLOBuiltInTracker(
            model=self.model,
            conf=config.conf,
            iou=config.iou,
            device=config.device,
            imgsz=config.imgsz,
            tracker_config=config.tracker_config,
        )
        log_info(f"Loaded YOLO pose model: {config.posture.model_path}")

    def detect(self, frame: object) -> list[PosePerson]:
        results = self.tracker.track(frame)
        return extract_pose_people(
            results,
            person_class_name=self.config.posture.person_class_name,
        )


def extract_pose_people(
    results: list[Any] | None,
    person_class_name: str = "person",
) -> list[PosePerson]:
    if not results:
        return []
    people: list[PosePerson] = []
    for result in results:
        people.extend(pose_result_to_people(result, person_class_name=person_class_name))
    return people


def pose_result_to_people(
    result: Any,
    person_class_name: str = "person",
) -> list[PosePerson]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or _safe_len(boxes) == 0:
        return []

    xyxy = _to_numpy(getattr(boxes, "xyxy", None))
    confidences = _to_numpy(getattr(boxes, "conf", None))
    track_ids = _to_numpy(getattr(boxes, "id", None))
    keypoint_xy = _to_numpy(getattr(getattr(result, "keypoints", None), "xy", None))
    keypoint_conf = _to_numpy(getattr(getattr(result, "keypoints", None), "conf", None))

    if xyxy is None:
        return []

    people: list[PosePerson] = []
    for index, bbox in enumerate(xyxy):
        people.append(
            PosePerson(
                bbox=tuple(float(value) for value in bbox),
                confidence=_safe_float_at(confidences, index, default=0.0),
                track_id=_safe_track_id(track_ids, index),
                keypoints=_extract_keypoints(keypoint_xy, keypoint_conf, index),
                class_name=person_class_name,
                class_id=0,
            )
        )
    return people


def _extract_keypoints(
    keypoint_xy: np.ndarray | None,
    keypoint_conf: np.ndarray | None,
    person_index: int,
) -> dict[str, Keypoint]:
    if keypoint_xy is None:
        return {}
    try:
        person_keypoints = keypoint_xy[person_index]
    except (IndexError, TypeError):
        return {}

    keypoints: dict[str, Keypoint] = {}
    for keypoint_index, point in enumerate(person_keypoints):
        keypoint_name = COCO_KEYPOINT_NAMES.get(keypoint_index)
        if keypoint_name is None:
            continue
        if len(point) < 2:
            continue
        confidence = _safe_keypoint_confidence(keypoint_conf, person_index, keypoint_index)
        keypoints[keypoint_name] = Keypoint(
            x=float(point[0]),
            y=float(point[1]),
            confidence=confidence,
            visible=confidence > 0.0,
        )
    return keypoints


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


def _safe_len(value: Any) -> int:
    try:
        return len(value)
    except TypeError:
        return 0


def _safe_float_at(
    values: np.ndarray | None,
    index: int,
    default: float,
) -> float:
    if values is None:
        return default
    try:
        return float(values[index])
    except (IndexError, TypeError, ValueError):
        return default


def _safe_keypoint_confidence(
    keypoint_conf: np.ndarray | None,
    person_index: int,
    keypoint_index: int,
) -> float:
    if keypoint_conf is None:
        return 0.0
    try:
        return float(keypoint_conf[person_index][keypoint_index])
    except (IndexError, TypeError, ValueError):
        return 0.0


def _safe_track_id(track_ids: np.ndarray | None, index: int) -> int | None:
    if track_ids is None:
        return None
    try:
        value = float(track_ids[index])
    except (IndexError, TypeError, ValueError):
        return None
    if np.isnan(value):
        return None
    return int(value)
