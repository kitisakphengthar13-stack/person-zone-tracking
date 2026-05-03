from __future__ import annotations

from detector import Detection
from compliance import COMPLIANT, NON_COMPLIANT, UNKNOWN, PersonPPEStatus
from events import ACTIVE, CLEARED, PENDING, PPE_VIOLATION, UNKNOWN_PPE
from violation_engine import ViolationStateTracker
from zone_manager import ZoneMatch


def _detection(track_id: int | None = 7) -> Detection:
    return Detection(
        bbox=(0, 0, 100, 200),
        class_id=5,
        class_name="Person",
        confidence=0.90,
        track_id=track_id,
    )


def _zone_match(track_id: int | None = 7, zone_id: str = "zone_1") -> ZoneMatch:
    detection = _detection(track_id)
    return ZoneMatch(
        zone_id=zone_id,
        zone_name="Zone 1",
        detection=detection,
        anchor_point=(50.0, 100.0),
    )


def _status(
    track_id: int | None = 7,
    state: str = NON_COMPLIANT,
    missing_items: set[str] | None = None,
) -> PersonPPEStatus:
    return PersonPPEStatus(
        track_id=track_id,
        person_detection=_detection(track_id),
        matched_items={},
        required_items={"hardhat", "safety_vest"},
        missing_items=missing_items if missing_items is not None else {"hardhat"},
        compliance_state=state,
    )


def test_compliant_person_inside_zone_emits_no_event() -> None:
    tracker = ViolationStateTracker(min_violation_seconds=1.0)

    events = tracker.update(
        zone_matches=[_zone_match()],
        ppe_statuses=[_status(state=COMPLIANT, missing_items=set())],
        timestamp=0.0,
    )

    assert events == []
    assert tracker.records == {}


def test_non_compliant_person_inside_zone_starts_pending_without_event() -> None:
    tracker = ViolationStateTracker(min_violation_seconds=2.0)

    events = tracker.update(
        zone_matches=[_zone_match()],
        ppe_statuses=[_status()],
        timestamp=0.0,
    )

    assert events == []
    record = next(iter(tracker.records.values()))
    assert record.state == PENDING
    assert record.missing_items == {"hardhat"}


def test_persistent_non_compliance_emits_one_active_event() -> None:
    tracker = ViolationStateTracker(min_violation_seconds=2.0)
    tracker.update([_zone_match()], [_status()], 0.0)

    events = tracker.update([_zone_match()], [_status()], 2.0)

    assert len(events) == 1
    event = events[0]
    assert event.event_type == "active"
    assert event.state == ACTIVE
    assert event.violation_type == PPE_VIOLATION
    assert event.zone_id == "zone_1"
    assert event.track_id == 7
    assert event.missing_items == {"hardhat"}
    assert event.started_at == 0.0
    assert event.duration_seconds == 2.0


def test_active_violation_does_not_emit_duplicate_active_events() -> None:
    tracker = ViolationStateTracker(min_violation_seconds=1.0)
    tracker.update([_zone_match()], [_status()], 0.0)
    first_events = tracker.update([_zone_match()], [_status()], 1.0)
    repeated_events = tracker.update([_zone_match()], [_status()], 2.0)

    assert len(first_events) == 1
    assert repeated_events == []


def test_active_violation_clears_only_after_clear_threshold() -> None:
    tracker = ViolationStateTracker(min_violation_seconds=1.0, clear_after_seconds=1.0)
    tracker.update([_zone_match()], [_status()], 0.0)
    tracker.update([_zone_match()], [_status()], 1.0)

    early_events = tracker.update([], [_status()], 1.5)
    cleared_events = tracker.update([], [_status()], 2.0)

    assert early_events == []
    assert len(cleared_events) == 1
    event = cleared_events[0]
    assert event.event_type == "cleared"
    assert event.state == CLEARED
    assert event.ended_at == 2.0


def test_one_frame_miss_does_not_emit_active_event() -> None:
    tracker = ViolationStateTracker(min_violation_seconds=2.0, clear_after_seconds=1.0)

    assert tracker.update([_zone_match()], [_status()], 0.0) == []
    assert tracker.update([_zone_match()], [_status(state=COMPLIANT, missing_items=set())], 0.5) == []
    assert tracker.update([_zone_match()], [_status()], 2.1) == []

    record = next(iter(tracker.records.values()))
    assert record.state == PENDING
    assert record.first_seen == 2.1


def test_person_outside_zone_emits_no_ppe_zone_violation() -> None:
    tracker = ViolationStateTracker(min_violation_seconds=0.0)

    events = tracker.update(
        zone_matches=[],
        ppe_statuses=[_status()],
        timestamp=0.0,
    )

    assert events == []
    assert tracker.records == {}


def test_unknown_compliance_state_is_handled_safely() -> None:
    tracker = ViolationStateTracker(min_violation_seconds=2.0)

    events = tracker.update(
        zone_matches=[_zone_match()],
        ppe_statuses=[_status(state=UNKNOWN, missing_items=set())],
        timestamp=0.0,
    )

    assert events == []
    key = next(iter(tracker.records))
    record = tracker.records[key]
    assert key.violation_type == UNKNOWN_PPE
    assert record.state == PENDING


def test_unknown_compliance_tracking_can_be_disabled() -> None:
    tracker = ViolationStateTracker(emit_unknown_ppe=False)

    events = tracker.update(
        zone_matches=[_zone_match()],
        ppe_statuses=[_status(state=UNKNOWN, missing_items=set())],
        timestamp=0.0,
    )

    assert events == []
    assert tracker.records == {}
