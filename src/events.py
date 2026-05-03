from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


ViolationState = Literal["no_violation", "pending", "active", "cleared"]
ViolationType = Literal[
    "ppe_violation",
    "zone_violation",
    "zone_and_ppe_violation",
    "unknown_ppe",
]
ViolationEventType = Literal["active", "cleared"]

NO_VIOLATION: ViolationState = "no_violation"
PENDING: ViolationState = "pending"
ACTIVE: ViolationState = "active"
CLEARED: ViolationState = "cleared"

PPE_VIOLATION: ViolationType = "ppe_violation"
ZONE_VIOLATION: ViolationType = "zone_violation"
ZONE_AND_PPE_VIOLATION: ViolationType = "zone_and_ppe_violation"
UNKNOWN_PPE: ViolationType = "unknown_ppe"


@dataclass(frozen=True)
class ViolationKey:
    zone_id: str
    track_id: int
    violation_type: ViolationType


@dataclass
class ViolationRecord:
    state: ViolationState
    first_seen: float
    last_seen: float
    active_since: float | None = None
    cleared_since: float | None = None
    emitted: bool = False
    missing_items: set[str] = field(default_factory=set)
    zone_name: str = ""


@dataclass(frozen=True)
class ViolationEvent:
    event_type: ViolationEventType
    zone_id: str
    zone_name: str
    track_id: int
    violation_type: ViolationType
    state: ViolationState
    missing_items: set[str]
    started_at: float
    ended_at: float | None
    duration_seconds: float
