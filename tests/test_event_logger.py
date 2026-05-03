from __future__ import annotations

import json
from pathlib import Path

from event_logger import append_violation_events_jsonl, violation_event_to_dict
from events import ACTIVE, CLEARED, PPE_VIOLATION, ViolationEvent


def _event(
    event_type: str = "active",
    state: str = ACTIVE,
    missing_items: set[str] | None = None,
) -> ViolationEvent:
    return ViolationEvent(
        event_type=event_type,
        zone_id="zone_1",
        zone_name="Zone 1",
        track_id=7,
        violation_type=PPE_VIOLATION,
        state=state,
        missing_items=missing_items or {"safety_vest", "hardhat"},
        started_at=1.0,
        ended_at=None if event_type == "active" else 4.0,
        duration_seconds=3.0,
    )


def test_violation_event_serializes_to_json_safe_dict() -> None:
    data = violation_event_to_dict(_event(), emitted_at=3.0)

    assert data == {
        "emitted_at": 3.0,
        "event_type": "active",
        "violation_type": PPE_VIOLATION,
        "state": ACTIVE,
        "zone_id": "zone_1",
        "zone_name": "Zone 1",
        "track_id": 7,
        "missing_items": ["hardhat", "safety_vest"],
        "started_at": 1.0,
        "ended_at": None,
        "duration_seconds": 3.0,
    }
    json.dumps(data)


def test_missing_items_are_sorted_lists() -> None:
    data = violation_event_to_dict(_event(missing_items={"mask", "hardhat"}))

    assert data["missing_items"] == ["hardhat", "mask"]


def test_append_one_event_creates_jsonl_file(tmp_path: Path) -> None:
    path = tmp_path / "events" / "violations.jsonl"

    append_violation_events_jsonl([_event()], path, emitted_at=3.0)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["emitted_at"] == 3.0


def test_append_multiple_events_writes_one_json_object_per_line(tmp_path: Path) -> None:
    path = tmp_path / "violations.jsonl"

    append_violation_events_jsonl(
        [
            _event(),
            _event(event_type="cleared", state=CLEARED, missing_items={"hardhat"}),
        ],
        path,
        emitted_at=5.0,
    )

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert [row["event_type"] for row in rows] == ["active", "cleared"]
    assert rows[1]["missing_items"] == ["hardhat"]


def test_empty_event_list_does_not_create_file(tmp_path: Path) -> None:
    path = tmp_path / "violations.jsonl"

    append_violation_events_jsonl([], path, emitted_at=5.0)

    assert not path.exists()
