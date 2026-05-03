from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from typing import Any, Iterable


BBox = tuple[float, float, float, float]

DEFAULT_PPE_REGIONS = {
    "helmet": "head",
    "hardhat": "head",
    "mask": "head",
    "vest": "torso",
    "safety_vest": "torso",
    "boots": "lower_body",
    "gloves": "full_body",
}

VALID_PERSON_REGIONS = {"head", "torso", "lower_body", "full_body"}


@dataclass(frozen=True)
class PPEMatchConfig:
    center_inside_person: bool = True
    min_person_overlap_ratio: float = 0.02
    min_region_overlap_ratio: float = 0.05
    max_center_distance_ratio: float = 0.60
    ppe_regions: dict[str, str] = field(default_factory=dict)


def split_person_and_ppe_detections(
    detections: Iterable[Any],
    person_classes: Iterable[str],
    ppe_classes: Iterable[str],
) -> tuple[list[Any], list[Any]]:
    person_class_set = _normalize_class_set(person_classes)
    ppe_class_set = _normalize_class_set(ppe_classes)

    persons: list[Any] = []
    ppe_items: list[Any] = []

    for detection in detections:
        class_name = _class_name(detection)
        if class_name in person_class_set:
            persons.append(detection)
        elif class_name in ppe_class_set:
            ppe_items.append(detection)

    return persons, ppe_items


def bbox_area(bbox: Any) -> float:
    x1, y1, x2, y2 = _bbox(bbox)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_intersection(bbox_a: Any, bbox_b: Any) -> BBox | None:
    ax1, ay1, ax2, ay2 = _bbox(bbox_a)
    bx1, by1, bx2, by2 = _bbox(bbox_b)

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def bbox_intersection_area(bbox_a: Any, bbox_b: Any) -> float:
    intersection = bbox_intersection(bbox_a, bbox_b)
    if intersection is None:
        return 0.0
    return bbox_area(intersection)


def bbox_overlap_ratio(bbox_a: Any, bbox_b: Any, denominator: str = "a") -> float:
    intersection_area = bbox_intersection_area(bbox_a, bbox_b)
    if denominator == "b":
        base_area = bbox_area(bbox_b)
    elif denominator == "min":
        base_area = min(bbox_area(bbox_a), bbox_area(bbox_b))
    else:
        base_area = bbox_area(bbox_a)

    if base_area <= 0.0:
        return 0.0
    return min(1.0, intersection_area / base_area)


def bbox_center(bbox: Any) -> tuple[float, float]:
    x1, y1, x2, y2 = _bbox(bbox)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def point_inside_bbox(point: tuple[float, float], bbox: Any) -> bool:
    x, y = point
    x1, y1, x2, y2 = _bbox(bbox)
    return x1 <= x <= x2 and y1 <= y <= y2


def person_body_region(person_bbox: Any, region: str) -> BBox:
    normalized_region = region.strip().lower()
    if normalized_region not in VALID_PERSON_REGIONS:
        raise ValueError(f"Unsupported person body region: {region}")

    x1, y1, x2, y2 = _bbox(person_bbox)
    height = max(0.0, y2 - y1)

    if normalized_region == "head":
        return (x1, y1, x2, y1 + height * 0.30)
    if normalized_region == "torso":
        return (x1, y1 + height * 0.25, x2, y1 + height * 0.75)
    if normalized_region == "lower_body":
        return (x1, y1 + height * 0.50, x2, y2)
    return (x1, y1, x2, y2)


def match_ppe_to_persons(
    persons: Iterable[Any],
    ppe_items: Iterable[Any],
    config: PPEMatchConfig | None = None,
) -> dict[int, dict[str, list[Any]]]:
    match_config = config or PPEMatchConfig()
    tracked_persons = [person for person in persons if _track_id(person) is not None]
    matches: dict[int, dict[str, list[Any]]] = {
        int(_track_id(person)): {} for person in tracked_persons
    }

    for ppe_item in ppe_items:
        ppe_type = _class_name(ppe_item)
        best_person = _best_person_for_ppe(ppe_item, tracked_persons, match_config)
        if best_person is None:
            continue

        track_id = int(_track_id(best_person))
        matches.setdefault(track_id, {}).setdefault(ppe_type, []).append(ppe_item)

    return matches


def _best_person_for_ppe(
    ppe_item: Any,
    persons: list[Any],
    config: PPEMatchConfig,
) -> Any | None:
    best_person = None
    best_score = 0.0

    for person in persons:
        score = _match_score(person, ppe_item, config)
        if score > best_score:
            best_person = person
            best_score = score

    return best_person


def _match_score(person: Any, ppe_item: Any, config: PPEMatchConfig) -> float:
    person_bbox = _bbox(person.bbox)
    ppe_bbox = _bbox(ppe_item.bbox)
    ppe_area = bbox_area(ppe_bbox)
    if bbox_area(person_bbox) <= 0.0 or ppe_area <= 0.0:
        return 0.0

    ppe_center = bbox_center(ppe_bbox)
    center_inside = point_inside_bbox(ppe_center, person_bbox)
    person_overlap = bbox_overlap_ratio(ppe_bbox, person_bbox, denominator="a")
    if config.center_inside_person and not center_inside:
        if person_overlap < config.min_person_overlap_ratio:
            return 0.0
    elif person_overlap < config.min_person_overlap_ratio:
        return 0.0

    region = _region_for_ppe(_class_name(ppe_item), config)
    region_bbox = person_body_region(person_bbox, region)
    region_overlap = bbox_overlap_ratio(ppe_bbox, region_bbox, denominator="a")
    if region_overlap < config.min_region_overlap_ratio:
        return 0.0

    distance_score = _center_distance_score(ppe_center, person_bbox, config)
    if distance_score <= 0.0:
        return 0.0

    return region_overlap * 0.70 + distance_score * 0.30


def _center_distance_score(
    ppe_center: tuple[float, float],
    person_bbox: BBox,
    config: PPEMatchConfig,
) -> float:
    person_center = bbox_center(person_bbox)
    x1, y1, x2, y2 = person_bbox
    max_distance = max(x2 - x1, y2 - y1) * config.max_center_distance_ratio
    if max_distance <= 0.0:
        return 0.0

    distance = hypot(ppe_center[0] - person_center[0], ppe_center[1] - person_center[1])
    if distance > max_distance:
        return 0.0
    return 1.0 - (distance / max_distance)


def _region_for_ppe(class_name: str, config: PPEMatchConfig) -> str:
    region = config.ppe_regions.get(class_name) or DEFAULT_PPE_REGIONS.get(class_name)
    return region or "full_body"


def _normalize_class_set(class_names: Iterable[str]) -> set[str]:
    return {str(class_name).strip().lower() for class_name in class_names if str(class_name).strip()}


def _class_name(detection: Any) -> str:
    return str(detection.class_name).strip().lower()


def _track_id(detection: Any) -> int | None:
    value = getattr(detection, "track_id", None)
    if value is None:
        return None
    return int(value)


def _bbox(value: Any) -> BBox:
    if hasattr(value, "bbox"):
        value = value.bbox
    x1, y1, x2, y2 = value
    return (float(x1), float(y1), float(x2), float(y2))
