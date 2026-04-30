from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from utils import ensure_parent_dir, log_info


@dataclass(frozen=True)
class Zone:
    id: str
    name: str
    points: list[tuple[int, int]]
    target_classes: list[str] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], index: int) -> "Zone":
        zone_id = str(data.get("id") or f"zone_{index + 1}")
        name = str(data.get("name") or zone_id)
        points = _validate_points(data.get("points"), zone_id)
        target_classes = data.get("target_classes")
        if target_classes:
            target_classes = [str(item).strip() for item in target_classes if str(item).strip()]
        else:
            target_classes = None
        return cls(id=zone_id, name=name, points=points, target_classes=target_classes)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "id": self.id,
            "name": self.name,
            "points": [[int(x), int(y)] for x, y in self.points],
        }
        if self.target_classes:
            data["target_classes"] = list(self.target_classes)
        return data


@dataclass(frozen=True)
class ZoneMatch:
    zone_id: str
    zone_name: str
    detection: Any
    anchor_point: tuple[float, float]
    overlap_ratio: float = 0.0
    match_method: str = "center"


class ZoneManager:
    def __init__(self, zones: list[Zone], global_target_classes: list[str]) -> None:
        self.zones = zones
        self.global_target_classes = list(global_target_classes)
        self._validate_unique_zone_ids()

    @classmethod
    def from_file(cls, path: Path, global_target_classes: list[str]) -> "ZoneManager":
        if not path.exists():
            raise FileNotFoundError(
                f"Zones file not found: {path}. Run with --draw-zones true to create it."
            )

        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        raw_zones = data.get("zones") if isinstance(data, dict) else data
        if not isinstance(raw_zones, list):
            raise ValueError("Zones JSON must contain a 'zones' list.")

        zones = [Zone.from_dict(zone_data, index) for index, zone_data in enumerate(raw_zones)]
        manager = cls(zones=zones, global_target_classes=global_target_classes)
        log_info(f"Loaded {len(manager.zones)} zone(s) from {path}")
        return manager

    @staticmethod
    def save_zones(zones: list[Zone], path: Path) -> None:
        for zone in zones:
            _validate_points(zone.points, zone.id)
        ensure_parent_dir(path)
        payload = {"version": 1, "zones": [zone.to_dict() for zone in zones]}
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
        log_info(f"Saved {len(zones)} zone(s) to {path}")

    def match_detections(self, detections: list[Any]) -> list[ZoneMatch]:
        matches: list[ZoneMatch] = []

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            center_point = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

            for zone in self.zones:
                allowed_classes = self._allowed_classes(zone)
                if detection.class_name.lower() not in allowed_classes:
                    continue

                if _point_in_polygon(center_point, zone.points):
                    matches.append(
                        ZoneMatch(
                            zone_id=zone.id,
                            zone_name=zone.name,
                            detection=detection,
                            anchor_point=center_point,
                            overlap_ratio=0.0,
                            match_method="center",
                        )
                    )
                    continue

                overlap_ratio = _bbox_polygon_overlap_ratio(detection.bbox, zone.points)
                if overlap_ratio >= _adaptive_overlap_threshold(detection.bbox):
                    matches.append(
                        ZoneMatch(
                            zone_id=zone.id,
                            zone_name=zone.name,
                            detection=detection,
                            anchor_point=center_point,
                            overlap_ratio=overlap_ratio,
                            match_method="overlap",
                        )
                    )

        return matches

    def _allowed_classes(self, zone: Zone) -> set[str]:
        classes = zone.target_classes or self.global_target_classes
        return {class_name.lower() for class_name in classes}

    def _validate_unique_zone_ids(self) -> None:
        seen: set[str] = set()
        for zone in self.zones:
            if zone.id in seen:
                raise ValueError(f"Duplicate zone id found: {zone.id}")
            seen.add(zone.id)


def _validate_points(points: Any, zone_id: str) -> list[tuple[int, int]]:
    if not isinstance(points, list):
        raise ValueError(f"Zone {zone_id} must contain a list of points.")
    if len(points) < 3:
        raise ValueError(f"Zone {zone_id} must contain at least 3 points.")

    parsed_points: list[tuple[int, int]] = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"Zone {zone_id} has an invalid point: {point!r}")
        x, y = point
        parsed_points.append((int(x), int(y)))
    return parsed_points


def _point_in_polygon(point: tuple[float, float], polygon: list[tuple[int, int]]) -> bool:
    contour = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(contour, point, measureDist=False) >= 0


def _bbox_area(bbox: Any) -> float:
    x1, y1, x2, y2 = bbox
    width = float(x2) - float(x1)
    height = float(y2) - float(y1)
    if width <= 0.0 or height <= 0.0:
        return 0.0
    return width * height


def _adaptive_overlap_threshold(bbox: Any) -> float:
    area = _bbox_area(bbox)
    if area < 5_000.0:
        return 0.05
    if area < 25_000.0:
        return 0.10
    return 0.20


def _bbox_polygon_overlap_ratio(bbox: Any, polygon: list[tuple[int, int]]) -> float:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    bbox_area = _bbox_area((x1, y1, x2, y2))
    if bbox_area <= 0.0:
        return 0.0

    roi_width = max(1, int(np.ceil(x2 - x1)))
    roi_height = max(1, int(np.ceil(y2 - y1)))
    shifted_polygon = np.array(
        [[(float(px) - x1, float(py) - y1) for px, py in polygon]],
        dtype=np.int32,
    )

    mask = np.zeros((roi_height, roi_width), dtype=np.uint8)
    cv2.fillPoly(mask, shifted_polygon, 255)

    overlap_area = float(cv2.countNonZero(mask))
    return min(1.0, overlap_area / bbox_area)
