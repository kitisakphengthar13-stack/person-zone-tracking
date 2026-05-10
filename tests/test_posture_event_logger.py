from __future__ import annotations

import json
from pathlib import Path

from pose_types import PostureState
from posture_event_logger import append_posture_events_jsonl, posture_event_to_dict
from posture_events import (
    ACTIVE,
    CLEARED,
    SITTING_VIOLATION_ACTIVE,
    SITTING_VIOLATION_CLEARED,
    PostureViolationEvent,
)


def _event(
    event_type: str = SITTING_VIOLATION_ACTIVE,
    state: str = ACTIVE,
    ended_at: float | None = None,
) -> PostureViolationEvent:
    return PostureViolationEvent(
        event_type=event_type,
        zone_id="zone_1",
        zone_name="Work Zone",
        track_id=7,
        state=state,
        posture_state=PostureState.SITTING,
        started_at=10.0,
        ended_at=ended_at,
        duration_seconds=31.2,
    )


def test_posture_event_serializes_to_json_safe_dict() -> None:
    data = posture_event_to_dict(_event(), emitted_at=41.2)

    assert data == {
        "emitted_at": 41.2,
        "event_type": SITTING_VIOLATION_ACTIVE,
        "zone_id": "zone_1",
        "zone_name": "Work Zone",
        "track_id": 7,
        "state": ACTIVE,
        "posture_state": "sitting",
        "started_at": 10.0,
        "ended_at": None,
        "duration_seconds": 31.2,
    }
    json.dumps(data)


def test_append_one_event_creates_jsonl_file(tmp_path: Path) -> None:
    path = tmp_path / "events" / "posture_events.jsonl"

    append_posture_events_jsonl([_event()], path, emitted_at=41.2)

    rows = path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["emitted_at"] == 41.2


def test_append_multiple_events_writes_one_json_object_per_line(tmp_path: Path) -> None:
    path = tmp_path / "posture_events.jsonl"

    append_posture_events_jsonl(
        [
            _event(),
            _event(
                event_type=SITTING_VIOLATION_CLEARED,
                state=CLEARED,
                ended_at=45.0,
            ),
        ],
        path,
    )

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert [row["event_type"] for row in rows] == [
        SITTING_VIOLATION_ACTIVE,
        SITTING_VIOLATION_CLEARED,
    ]
    assert rows[1]["ended_at"] == 45.0


def test_empty_event_list_does_not_create_file(tmp_path: Path) -> None:
    path = tmp_path / "posture_events.jsonl"

    append_posture_events_jsonl([], path, emitted_at=10.0)

    assert not path.exists()


def test_parent_directory_is_created_automatically(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "events" / "posture_events.jsonl"

    append_posture_events_jsonl([_event()], path)

    assert path.exists()
