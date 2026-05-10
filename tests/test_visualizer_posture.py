from __future__ import annotations

import numpy as np

from detector import Detection
from dwell_time import DwellTimeTracker
from pose_types import PosePerson, PostureState, PostureStatus
from posture_events import ACTIVE, SITTING_VIOLATION_ACTIVE, PostureViolationEvent
from visualizer import (
    Visualizer,
    build_active_posture_event_by_track,
    build_posture_status_by_track,
    format_posture_status_label,
    format_sitting_duration,
)


def _pose_person(track_id: int | None = 7) -> PosePerson:
    return PosePerson(
        bbox=(10.0, 20.0, 100.0, 220.0),
        confidence=0.90,
        track_id=track_id,
    )


def _status(
    state: PostureState,
    track_id: int | None = 7,
    sitting_seconds: float = 0.0,
) -> PostureStatus:
    person = _pose_person(track_id)
    return PostureStatus(
        track_id=track_id,
        pose_person=person,
        state=state,
        confidence=0.80,
        sitting_seconds=sitting_seconds,
    )


def _active_event(track_id: int = 7) -> PostureViolationEvent:
    return PostureViolationEvent(
        event_type=SITTING_VIOLATION_ACTIVE,
        zone_id="zone_1",
        zone_name="Zone 1",
        track_id=track_id,
        state=ACTIVE,
        posture_state=PostureState.SITTING,
        started_at=0.0,
        ended_at=None,
        duration_seconds=31.2,
    )


def test_format_posture_label_for_standing_status() -> None:
    assert format_posture_status_label(_status(PostureState.STANDING)) == "standing"


def test_format_posture_label_for_sitting_status_without_event() -> None:
    assert format_posture_status_label(_status(PostureState.SITTING)) == "sitting"
    assert format_posture_status_label(
        _status(PostureState.SITTING, sitting_seconds=12.4)
    ) == "sitting 12.4s"


def test_format_posture_label_for_unknown_status() -> None:
    assert format_posture_status_label(_status(PostureState.UNKNOWN)) == "unknown posture"


def test_format_posture_label_for_active_sitting_violation_event() -> None:
    assert format_posture_status_label(
        _status(PostureState.SITTING, sitting_seconds=31.2),
        _active_event(),
    ) == "sitting violation 31.2s"


def test_helper_functions_handle_none_and_empty_inputs() -> None:
    assert build_posture_status_by_track(None) == {}
    assert build_posture_status_by_track([]) == {}
    assert build_active_posture_event_by_track(None) == {}
    assert build_active_posture_event_by_track([]) == {}
    assert format_posture_status_label(None) == ""
    assert format_sitting_duration(None) == "0.0s"


def test_status_and_event_builders_index_by_track_id() -> None:
    status = _status(PostureState.SITTING, track_id=7)
    unknown_track_status = _status(PostureState.UNKNOWN, track_id=None)
    event = _active_event(track_id=7)

    assert build_posture_status_by_track([status, unknown_track_status]) == {7: status}
    assert build_active_posture_event_by_track([event]) == {7: event}


def test_visualizer_draw_remains_backward_compatible_without_posture_data() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    detection = Detection(
        bbox=(10.0, 20.0, 100.0, 220.0),
        class_id=0,
        class_name="person",
        confidence=0.90,
        track_id=7,
    )

    output = Visualizer().draw(
        frame=frame,
        zones=[],
        detections=[detection],
        zone_matches=[],
        dwell_tracker=DwellTimeTracker(),
    )

    assert output.shape == frame.shape
    assert np.count_nonzero(output) > 0
