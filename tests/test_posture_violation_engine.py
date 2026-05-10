from __future__ import annotations

from types import SimpleNamespace

from pose_types import PosePerson, PostureState, PostureStatus
from posture_events import (
    ACTIVE,
    CLEARED,
    PENDING,
    SITTING_VIOLATION_ACTIVE,
    SITTING_VIOLATION_CLEARED,
)
from posture_violation_engine import PostureViolationTracker


def _pose_person(track_id: int | None = 7) -> PosePerson:
    return PosePerson(
        bbox=(0.0, 0.0, 100.0, 200.0),
        confidence=0.90,
        track_id=track_id,
    )


def _status(
    state: PostureState,
    track_id: int | None = 7,
) -> PostureStatus:
    person = _pose_person(track_id)
    return PostureStatus(
        track_id=track_id,
        pose_person=person,
        state=state,
        confidence=0.80,
    )


def _zone_match(track_id: int | None = 7, zone_id: str = "zone_1") -> object:
    return SimpleNamespace(
        zone_id=zone_id,
        zone_name="Work Zone",
        detection=SimpleNamespace(track_id=track_id),
    )


def _first_record(tracker: PostureViolationTracker):
    return next(iter(tracker.records.values()))


def test_standing_inside_zone_emits_no_event() -> None:
    tracker = PostureViolationTracker()

    events = tracker.update([_zone_match()], [_status(PostureState.STANDING)], 0.0)

    assert events == []
    assert tracker.records == {}


def test_sitting_under_threshold_is_pending_without_active_event() -> None:
    tracker = PostureViolationTracker(min_violation_seconds=30.0)

    assert tracker.update([_zone_match()], [_status(PostureState.SITTING)], 0.0) == []
    assert tracker.update([_zone_match()], [_status(PostureState.SITTING)], 29.9) == []

    record = _first_record(tracker)
    assert record.state == PENDING
    assert record.first_sitting_seen == 0.0


def test_sitting_over_threshold_emits_one_active_event() -> None:
    tracker = PostureViolationTracker(min_violation_seconds=30.0)
    tracker.update([_zone_match()], [_status(PostureState.SITTING)], 0.0)

    events = tracker.update([_zone_match()], [_status(PostureState.SITTING)], 30.1)

    assert len(events) == 1
    event = events[0]
    assert event.event_type == SITTING_VIOLATION_ACTIVE
    assert event.state == ACTIVE
    assert event.zone_id == "zone_1"
    assert event.zone_name == "Work Zone"
    assert event.track_id == 7
    assert event.started_at == 0.0
    assert event.duration_seconds == 30.1


def test_active_violation_does_not_emit_duplicate_active_events() -> None:
    tracker = PostureViolationTracker(min_violation_seconds=30.0)
    tracker.update([_zone_match()], [_status(PostureState.SITTING)], 0.0)
    first_events = tracker.update([_zone_match()], [_status(PostureState.SITTING)], 30.1)
    repeated_events = tracker.update([_zone_match()], [_status(PostureState.SITTING)], 35.0)

    assert len(first_events) == 1
    assert repeated_events == []


def test_standing_less_than_clear_threshold_does_not_clear() -> None:
    tracker = PostureViolationTracker(min_violation_seconds=1.0, clear_after_seconds=2.0)
    tracker.update([_zone_match()], [_status(PostureState.SITTING)], 0.0)
    tracker.update([_zone_match()], [_status(PostureState.SITTING)], 1.1)

    events = tracker.update([_zone_match()], [_status(PostureState.STANDING)], 2.0)

    assert events == []
    assert _first_record(tracker).state == ACTIVE


def test_standing_for_clear_threshold_emits_cleared_event() -> None:
    tracker = PostureViolationTracker(min_violation_seconds=1.0, clear_after_seconds=2.0)
    tracker.update([_zone_match()], [_status(PostureState.SITTING)], 0.0)
    tracker.update([_zone_match()], [_status(PostureState.SITTING)], 1.1)
    tracker.update([_zone_match()], [_status(PostureState.STANDING)], 2.0)

    events = tracker.update([_zone_match()], [_status(PostureState.STANDING)], 4.0)

    assert len(events) == 1
    assert events[0].event_type == SITTING_VIOLATION_CLEARED
    assert events[0].state == CLEARED
    assert events[0].ended_at == 4.0


def test_unknown_within_grace_preserves_pending_state() -> None:
    tracker = PostureViolationTracker(min_violation_seconds=30.0, unknown_grace_seconds=3.0)
    tracker.update([_zone_match()], [_status(PostureState.SITTING)], 0.0)
    tracker.update([_zone_match()], [_status(PostureState.UNKNOWN)], 1.0)

    events = tracker.update([_zone_match()], [_status(PostureState.SITTING)], 3.0)

    assert events == []
    assert _first_record(tracker).first_sitting_seen == 0.0


def test_unknown_beyond_grace_does_not_confirm_new_sitting_violation() -> None:
    tracker = PostureViolationTracker(min_violation_seconds=30.0, unknown_grace_seconds=3.0)
    tracker.update([_zone_match()], [_status(PostureState.SITTING)], 0.0)
    tracker.update([_zone_match()], [_status(PostureState.UNKNOWN)], 1.0)
    tracker.update([_zone_match()], [_status(PostureState.UNKNOWN)], 5.0)

    events = tracker.update([_zone_match()], [_status(PostureState.SITTING)], 30.5)

    assert events == []
    assert _first_record(tracker).state == PENDING


def test_missing_track_shorter_than_grace_preserves_active_state() -> None:
    tracker = PostureViolationTracker(min_violation_seconds=1.0, max_missing_track_seconds=2.0)
    tracker.update([_zone_match()], [_status(PostureState.SITTING)], 0.0)
    tracker.update([_zone_match()], [_status(PostureState.SITTING)], 1.1)

    events = tracker.update([], [_status(PostureState.SITTING)], 2.0)

    assert events == []
    assert _first_record(tracker).state == ACTIVE


def test_sitting_outside_zone_is_ignored_by_default() -> None:
    tracker = PostureViolationTracker(ignore_outside_zones=True, min_violation_seconds=0.0)

    events = tracker.update([], [_status(PostureState.SITTING)], 10.0)

    assert events == []
    assert tracker.records == {}


def test_track_id_none_is_ignored_safely() -> None:
    tracker = PostureViolationTracker(min_violation_seconds=0.0)

    events = tracker.update([_zone_match(track_id=None)], [_status(PostureState.SITTING, None)], 10.0)

    assert events == []
    assert tracker.records == {}
