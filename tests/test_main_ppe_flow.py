from __future__ import annotations

from types import SimpleNamespace

from compliance import NON_COMPLIANT
from config import PPEMatchingConfig
from detector import Detection
from events import PPE_VIOLATION
from main import _create_violation_tracker, _process_frame_analytics
from violation_engine import ViolationStateTracker
from zone_manager import ZoneMatch


class FakeZoneManager:
    def __init__(self) -> None:
        self.received_detections: list[object] | None = None

    def match_detections(self, detections: list[object]) -> list[ZoneMatch]:
        self.received_detections = detections
        return [
            ZoneMatch(
                zone_id="zone_1",
                zone_name="Zone 1",
                detection=detection,
                anchor_point=(50.0, 100.0),
            )
            for detection in detections
        ]


class FakeDwellTracker:
    def __init__(self) -> None:
        self.updated_matches: list[ZoneMatch] | None = None
        self.updated_timestamp: float | None = None

    def update(self, zone_matches: list[ZoneMatch], timestamp: float) -> None:
        self.updated_matches = zone_matches
        self.updated_timestamp = timestamp


def _detection(
    class_name: str,
    bbox: tuple[float, float, float, float],
    track_id: int | None = None,
) -> Detection:
    return Detection(
        bbox=bbox,
        class_id=0,
        class_name=class_name,
        confidence=0.90,
        track_id=track_id,
    )


def _ppe_config(enabled: bool = True, required_items: list[str] | None = None) -> object:
    return SimpleNamespace(
        ppe=SimpleNamespace(
            enabled=enabled,
            person_classes=["Person"],
            ppe_classes=[
                "Hardhat",
                "Safety Vest",
                "Mask",
                "NO-Hardhat",
                "NO-Safety Vest",
                "NO-Mask",
            ],
            required_items=required_items or ["hardhat", "safety_vest"],
            matching=PPEMatchingConfig(
                center_inside_person=True,
                min_person_overlap_ratio=0.02,
                min_region_overlap_ratio=0.05,
                max_center_distance_ratio=0.60,
                ppe_regions={},
            ),
        )
    )


def test_create_violation_tracker_uses_configured_thresholds() -> None:
    config = SimpleNamespace(
        ppe=SimpleNamespace(enabled=True),
        violations=SimpleNamespace(
            enabled=True,
            min_violation_seconds=4.5,
            clear_after_seconds=2.5,
            max_missing_track_seconds=3.5,
            emit_unknown_ppe=False,
        ),
    )

    tracker = _create_violation_tracker(config)

    assert tracker is not None
    assert tracker.min_violation_seconds == 4.5
    assert tracker.clear_after_seconds == 2.5
    assert tracker.max_missing_track_seconds == 3.5
    assert tracker.emit_unknown_ppe is False


def test_create_violation_tracker_disabled_when_ppe_disabled() -> None:
    config = SimpleNamespace(
        ppe=SimpleNamespace(enabled=False),
        violations=SimpleNamespace(
            enabled=True,
            min_violation_seconds=4.5,
            clear_after_seconds=2.5,
            max_missing_track_seconds=3.5,
            emit_unknown_ppe=True,
        ),
    )

    assert _create_violation_tracker(config) is None


def test_ppe_disabled_uses_original_detections_for_zone_matching() -> None:
    person = _detection("Person", (0, 0, 100, 200), track_id=7)
    vest = _detection("Safety Vest", (25, 70, 75, 135))
    detections = [person, vest]
    zone_manager = FakeZoneManager()
    dwell_tracker = FakeDwellTracker()

    analytics = _process_frame_analytics(
        config=_ppe_config(enabled=False),
        detections=detections,
        zone_manager=zone_manager,
        dwell_tracker=dwell_tracker,
        violation_tracker=None,
        timestamp=10.0,
    )

    assert zone_manager.received_detections == detections
    assert analytics.detections_for_display == detections
    assert len(analytics.zone_matches) == 2
    assert analytics.ppe_statuses == []
    assert analytics.violation_events == []
    assert dwell_tracker.updated_matches == analytics.zone_matches
    assert dwell_tracker.updated_timestamp == 10.0


def test_ppe_enabled_filters_zone_matching_to_person_detections() -> None:
    person = _detection("Person", (0, 0, 100, 200), track_id=7)
    hardhat = _detection("Hardhat", (35, 5, 65, 35))
    vest = _detection("Safety Vest", (25, 70, 75, 135))
    zone_manager = FakeZoneManager()

    analytics = _process_frame_analytics(
        config=_ppe_config(),
        detections=[person, hardhat, vest],
        zone_manager=zone_manager,
        dwell_tracker=FakeDwellTracker(),
        violation_tracker=ViolationStateTracker(),
        timestamp=10.0,
    )

    assert zone_manager.received_detections == [person]
    assert analytics.detections_for_display == [person, hardhat, vest]
    assert len(analytics.zone_matches) == 1
    assert analytics.ppe_statuses[0].compliance_state == "compliant"
    assert analytics.violation_events == []


def test_ppe_enabled_produces_statuses_and_violation_events_without_model() -> None:
    person = _detection("Person", (0, 0, 100, 200), track_id=7)
    vest = _detection("Safety Vest", (25, 70, 75, 135))
    config = _ppe_config(required_items=["hardhat", "safety_vest"])
    zone_manager = FakeZoneManager()
    dwell_tracker = FakeDwellTracker()
    violation_tracker = ViolationStateTracker(min_violation_seconds=0.5)

    first = _process_frame_analytics(
        config=config,
        detections=[person, vest],
        zone_manager=zone_manager,
        dwell_tracker=dwell_tracker,
        violation_tracker=violation_tracker,
        timestamp=0.0,
    )
    second = _process_frame_analytics(
        config=config,
        detections=[person, vest],
        zone_manager=zone_manager,
        dwell_tracker=dwell_tracker,
        violation_tracker=violation_tracker,
        timestamp=0.5,
    )

    assert first.violation_events == []
    assert second.ppe_statuses[0].compliance_state == NON_COMPLIANT
    assert second.ppe_statuses[0].missing_items == {"hardhat"}
    assert len(second.violation_events) == 1
    assert second.violation_events[0].violation_type == PPE_VIOLATION
