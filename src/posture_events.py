from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pose_types import PostureState


PostureViolationState = Literal["pending", "active", "cleared"]
PostureViolationEventType = Literal[
    "sitting_violation_active",
    "sitting_violation_cleared",
]

PENDING: PostureViolationState = "pending"
ACTIVE: PostureViolationState = "active"
CLEARED: PostureViolationState = "cleared"

SITTING_VIOLATION_ACTIVE: PostureViolationEventType = "sitting_violation_active"
SITTING_VIOLATION_CLEARED: PostureViolationEventType = "sitting_violation_cleared"


@dataclass(frozen=True)
class PostureViolationKey:
    zone_id: str
    track_id: int


@dataclass
class PostureViolationRecord:
    state: PostureViolationState
    first_sitting_seen: float
    last_seen: float
    zone_name: str
    active_since: float | None = None
    clear_started_at: float | None = None
    unknown_started_at: float | None = None
    missing_started_at: float | None = None
    emitted_active: bool = False


@dataclass(frozen=True)
class PostureViolationEvent:
    event_type: PostureViolationEventType
    zone_id: str
    zone_name: str
    track_id: int
    state: PostureViolationState
    posture_state: PostureState
    started_at: float
    ended_at: float | None
    duration_seconds: float
