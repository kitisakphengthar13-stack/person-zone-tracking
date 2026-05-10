from __future__ import annotations

from dataclasses import dataclass
from math import acos, degrees, hypot

from pose_types import Keypoint, PosePerson, PostureState, PostureStatus


COCO_KEYPOINT_NAMES = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}

CORE_REQUIRED_KEYPOINTS = (
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
)

ANKLE_KEYPOINTS = ("left_ankle", "right_ankle")


@dataclass(frozen=True)
class PostureClassificationRules:
    min_keypoint_confidence: float = 0.35
    standing_knee_angle_min: float = 150.0
    sitting_knee_angle_max: float = 145.0
    sitting_hip_to_knee_y_ratio_max: float = 0.35
    squat_hip_to_knee_y_ratio_min: float = 0.55
    standing_lower_leg_dx_ratio_max: float = 0.15
    max_torso_angle_degrees: float = 45.0


def classify_posture(
    person: PosePerson,
    rules: PostureClassificationRules | None = None,
) -> PostureStatus:
    active_rules = rules or PostureClassificationRules()
    if not validate_keypoint_quality(
        person,
        CORE_REQUIRED_KEYPOINTS,
        active_rules.min_keypoint_confidence,
    ):
        return _status(person, PostureState.UNKNOWN, 0.0, "missing required upper/lower body keypoints")

    shoulder_mid = midpoint(
        get_keypoint(person, "left_shoulder"),
        get_keypoint(person, "right_shoulder"),
    )
    hip_mid = midpoint(get_keypoint(person, "left_hip"), get_keypoint(person, "right_hip"))
    knee_mid = midpoint(get_keypoint(person, "left_knee"), get_keypoint(person, "right_knee"))

    torso_angle = compute_torso_angle(shoulder_mid, hip_mid)
    if torso_angle > active_rules.max_torso_angle_degrees:
        return _status(person, PostureState.UNKNOWN, 0.25, "torso angle is too ambiguous")

    left_knee_angle = _safe_joint_angle(person, "left_hip", "left_knee", "left_ankle")
    right_knee_angle = _safe_joint_angle(person, "right_hip", "right_knee", "right_ankle")
    knee_angles = [angle for angle in (left_knee_angle, right_knee_angle) if angle is not None]

    hip_to_knee_y = abs(knee_mid.y - hip_mid.y)
    torso_height = max(1.0, abs(hip_mid.y - shoulder_mid.y))
    hip_to_knee_y_ratio = hip_to_knee_y / torso_height

    if _has_usable_ankle(person, active_rules.min_keypoint_confidence):
        average_knee_angle = sum(knee_angles) / len(knee_angles) if knee_angles else 0.0
        if (
            average_knee_angle >= active_rules.standing_knee_angle_min
            and hip_to_knee_y_ratio >= active_rules.squat_hip_to_knee_y_ratio_min
            and _lower_leg_dx_ratio(person, torso_height) <= active_rules.standing_lower_leg_dx_ratio_max
        ):
            return _status(person, PostureState.STANDING, 0.85, "upright body with mostly straight knees")

        if (
            average_knee_angle <= active_rules.sitting_knee_angle_max
            and hip_to_knee_y_ratio <= active_rules.sitting_hip_to_knee_y_ratio_max
        ):
            return _status(person, PostureState.SITTING, 0.80, "bent knees with seated hip-knee geometry")

        return _status(person, PostureState.UNKNOWN, 0.35, "ambiguous lower-body geometry")

    if hip_to_knee_y_ratio <= active_rules.sitting_hip_to_knee_y_ratio_max:
        return _status(person, PostureState.SITTING, 0.60, "seated hip-knee geometry without ankle confirmation")

    return _status(person, PostureState.UNKNOWN, 0.20, "ankles missing and posture is ambiguous")


def validate_keypoint_quality(
    person: PosePerson,
    required_names: tuple[str, ...] | list[str],
    min_confidence: float,
) -> bool:
    return all(
        (keypoint := get_keypoint(person, name)) is not None
        and keypoint.is_usable(min_confidence)
        for name in required_names
    )


def get_keypoint(person: PosePerson, name: str) -> Keypoint | None:
    return person.keypoints.get(name)


def midpoint(first: Keypoint, second: Keypoint) -> Keypoint:
    return Keypoint(
        x=(first.x + second.x) / 2.0,
        y=(first.y + second.y) / 2.0,
        confidence=min(first.confidence, second.confidence),
        visible=first.visible and second.visible,
    )


def distance(first: Keypoint, second: Keypoint) -> float:
    return hypot(second.x - first.x, second.y - first.y)


def compute_joint_angle(first: Keypoint, joint: Keypoint, third: Keypoint) -> float:
    first_vector = (first.x - joint.x, first.y - joint.y)
    third_vector = (third.x - joint.x, third.y - joint.y)
    first_length = hypot(*first_vector)
    third_length = hypot(*third_vector)
    if first_length <= 0.0 or third_length <= 0.0:
        return 0.0

    dot = first_vector[0] * third_vector[0] + first_vector[1] * third_vector[1]
    cosine = max(-1.0, min(1.0, dot / (first_length * third_length)))
    return degrees(acos(cosine))


def compute_torso_angle(shoulder_midpoint: Keypoint, hip_midpoint: Keypoint) -> float:
    dx = abs(shoulder_midpoint.x - hip_midpoint.x)
    dy = abs(hip_midpoint.y - shoulder_midpoint.y)
    if dy <= 0.0:
        return 90.0
    return degrees(acos(dy / max(distance(shoulder_midpoint, hip_midpoint), 1e-6)))


def _safe_joint_angle(
    person: PosePerson,
    first_name: str,
    joint_name: str,
    third_name: str,
) -> float | None:
    first = get_keypoint(person, first_name)
    joint = get_keypoint(person, joint_name)
    third = get_keypoint(person, third_name)
    if first is None or joint is None or third is None:
        return None
    return compute_joint_angle(first, joint, third)


def _has_usable_ankle(person: PosePerson, min_confidence: float) -> bool:
    return any(
        (keypoint := get_keypoint(person, name)) is not None
        and keypoint.is_usable(min_confidence)
        for name in ANKLE_KEYPOINTS
    )


def _lower_leg_dx_ratio(person: PosePerson, torso_height: float) -> float:
    ratios: list[float] = []
    for knee_name, ankle_name in (
        ("left_knee", "left_ankle"),
        ("right_knee", "right_ankle"),
    ):
        knee = get_keypoint(person, knee_name)
        ankle = get_keypoint(person, ankle_name)
        if knee is None or ankle is None:
            continue
        ratios.append(abs(ankle.x - knee.x) / max(1.0, torso_height))
    if not ratios:
        return 1.0
    return sum(ratios) / len(ratios)


def _status(
    person: PosePerson,
    state: PostureState,
    confidence: float,
    reason: str,
) -> PostureStatus:
    return PostureStatus(
        track_id=person.track_id,
        pose_person=person,
        state=state,
        confidence=confidence,
        reason=reason,
    )
