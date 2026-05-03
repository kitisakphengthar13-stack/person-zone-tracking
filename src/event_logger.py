from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from events import ViolationEvent
from utils import ensure_parent_dir


def violation_event_to_dict(
    event: ViolationEvent,
    emitted_at: float | None = None,
) -> dict[str, object]:
    data: dict[str, object] = {
        "event_type": event.event_type,
        "violation_type": event.violation_type,
        "state": event.state,
        "zone_id": event.zone_id,
        "zone_name": event.zone_name,
        "track_id": event.track_id,
        "missing_items": sorted(event.missing_items),
        "started_at": event.started_at,
        "ended_at": event.ended_at,
        "duration_seconds": event.duration_seconds,
    }
    if emitted_at is not None:
        data["emitted_at"] = emitted_at
    return data


def append_violation_events_jsonl(
    events: Iterable[ViolationEvent],
    path: Path,
    emitted_at: float | None = None,
) -> None:
    event_list = list(events)
    if not event_list:
        return

    ensure_parent_dir(path)
    with path.open("a", encoding="utf-8") as file:
        for event in event_list:
            data = violation_event_to_dict(event, emitted_at=emitted_at)
            file.write(json.dumps(data, sort_keys=True) + "\n")
