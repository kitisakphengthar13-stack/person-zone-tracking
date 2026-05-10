from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from posture_events import PostureViolationEvent
from utils import ensure_parent_dir


def posture_event_to_dict(
    event: PostureViolationEvent,
    emitted_at: float | None = None,
) -> dict[str, object]:
    data: dict[str, object] = {
        "event_type": event.event_type,
        "zone_id": event.zone_id,
        "zone_name": event.zone_name,
        "track_id": event.track_id,
        "state": event.state,
        "posture_state": event.posture_state.value,
        "started_at": event.started_at,
        "ended_at": event.ended_at,
        "duration_seconds": event.duration_seconds,
    }
    if emitted_at is not None:
        data["emitted_at"] = emitted_at
    return data


def append_posture_events_jsonl(
    events: Iterable[PostureViolationEvent],
    path: Path,
    emitted_at: float | None = None,
) -> None:
    event_list = list(events)
    if not event_list:
        return

    ensure_parent_dir(path)
    with path.open("a", encoding="utf-8") as file:
        for event in event_list:
            file.write(json.dumps(posture_event_to_dict(event, emitted_at), sort_keys=True))
            file.write("\n")
