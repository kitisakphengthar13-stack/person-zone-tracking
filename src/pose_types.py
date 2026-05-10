from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PostureState(str, Enum):
    STANDING = "standing"
    SITTING = "sitting"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Keypoint:
    x: float
    y: float
    confidence: float
    visible: bool = True

    def is_usable(self, min_confidence: float) -> bool:
        return self.visible and self.confidence >= min_confidence


@dataclass(frozen=True)
class PosePerson:
    bbox: tuple[float, float, float, float]
    confidence: float
    track_id: int | None
    keypoints: dict[str, Keypoint] = field(default_factory=dict)
    class_name: str = "person"
    class_id: int = 0


@dataclass(frozen=True)
class PostureStatus:
    track_id: int | None
    pose_person: PosePerson
    state: PostureState
    confidence: float
    reason: str = ""
    zone_id: str | None = None
    sitting_seconds: float = 0.0
    violation_active: bool = False
