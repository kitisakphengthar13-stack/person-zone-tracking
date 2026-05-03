from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal


ComplianceState = Literal["compliant", "non_compliant", "unknown"]

COMPLIANT: ComplianceState = "compliant"
NON_COMPLIANT: ComplianceState = "non_compliant"
UNKNOWN: ComplianceState = "unknown"

RAW_PRESENT_PPE_CLASSES = {
    "Hardhat": "hardhat",
    "Safety Vest": "safety_vest",
    "Mask": "mask",
}

RAW_MISSING_PPE_CLASSES = {
    "NO-Hardhat": "hardhat",
    "NO-Safety Vest": "safety_vest",
    "NO-Mask": "mask",
}


@dataclass(frozen=True)
class PersonPPEStatus:
    track_id: int | None
    person_detection: Any
    matched_items: dict[str, list[Any]]
    required_items: set[str]
    missing_items: set[str]
    compliance_state: ComplianceState


def evaluate_person_compliance(
    person_detection: Any,
    matched_items: dict[str, list[Any]] | None,
    required_items: Iterable[str],
) -> PersonPPEStatus:
    track_id = _safe_track_id(getattr(person_detection, "track_id", None))
    normalized_required = _normalize_required_items(required_items)
    normalized_matched = normalize_matched_items(matched_items or {})

    if track_id is None:
        return PersonPPEStatus(
            track_id=None,
            person_detection=person_detection,
            matched_items=normalized_matched,
            required_items=normalized_required,
            missing_items=set(),
            compliance_state=UNKNOWN,
        )

    present_items = _present_items(normalized_matched)
    explicit_missing_items = _explicit_missing_items(normalized_matched)
    missing_items = (normalized_required - present_items) | (
        explicit_missing_items & normalized_required
    )
    compliance_state = NON_COMPLIANT if missing_items else COMPLIANT

    return PersonPPEStatus(
        track_id=track_id,
        person_detection=person_detection,
        matched_items=normalized_matched,
        required_items=normalized_required,
        missing_items=missing_items,
        compliance_state=compliance_state,
    )


def evaluate_all_compliance(
    person_detections: Iterable[Any],
    matched_items_by_track: dict[int, dict[str, list[Any]]] | None,
    required_items: Iterable[str],
) -> list[PersonPPEStatus]:
    matches_by_track = matched_items_by_track or {}
    statuses: list[PersonPPEStatus] = []

    for person_detection in person_detections:
        track_id = _safe_track_id(getattr(person_detection, "track_id", None))
        matched_items = matches_by_track.get(track_id, {}) if track_id is not None else {}
        statuses.append(
            evaluate_person_compliance(
                person_detection=person_detection,
                matched_items=matched_items,
                required_items=required_items,
            )
        )

    return statuses


def normalize_matched_items(matched_items: dict[str, list[Any]]) -> dict[str, list[Any]]:
    normalized: dict[str, list[Any]] = {}

    for raw_key, detections in matched_items.items():
        normalized_key = normalize_ppe_item_name(raw_key)
        if normalized_key is None:
            continue

        normalized.setdefault(normalized_key, []).extend(list(detections))
        for detection in detections:
            detection_key = normalize_ppe_item_name(getattr(detection, "class_name", ""))
            if detection_key is None or detection_key == normalized_key:
                continue
            normalized.setdefault(detection_key, []).append(detection)

    return normalized


def normalize_ppe_item_name(value: str) -> str | None:
    raw_value = str(value).strip()
    if not raw_value:
        return None
    if raw_value in RAW_PRESENT_PPE_CLASSES:
        return RAW_PRESENT_PPE_CLASSES[raw_value]
    if raw_value in RAW_MISSING_PPE_CLASSES:
        return f"missing:{RAW_MISSING_PPE_CLASSES[raw_value]}"

    normalized = raw_value.lower().replace("-", "_").replace(" ", "_")
    if normalized.startswith("no_"):
        return f"missing:{normalized.removeprefix('no_')}"
    return normalized


def _normalize_required_items(required_items: Iterable[str]) -> set[str]:
    normalized_items: set[str] = set()
    for item in required_items:
        normalized = normalize_ppe_item_name(item)
        if normalized is None:
            continue
        normalized_items.add(normalized.removeprefix("missing:"))
    return normalized_items


def _present_items(matched_items: dict[str, list[Any]]) -> set[str]:
    return {
        item_name
        for item_name, detections in matched_items.items()
        if detections and not item_name.startswith("missing:")
    }


def _explicit_missing_items(matched_items: dict[str, list[Any]]) -> set[str]:
    return {
        item_name.removeprefix("missing:")
        for item_name, detections in matched_items.items()
        if detections and item_name.startswith("missing:")
    }


def _safe_track_id(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
