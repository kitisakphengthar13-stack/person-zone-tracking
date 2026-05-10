from __future__ import annotations

from pose_types import Keypoint, PosePerson, PostureState
from posture_classifier import (
    classify_posture,
    compute_joint_angle,
    distance,
    midpoint,
    validate_keypoint_quality,
)


def _kp(x: float, y: float, confidence: float = 0.90, visible: bool = True) -> Keypoint:
    return Keypoint(x=x, y=y, confidence=confidence, visible=visible)


def _person(keypoints: dict[str, Keypoint], track_id: int | None = 7) -> PosePerson:
    return PosePerson(
        bbox=(0.0, 0.0, 100.0, 220.0),
        confidence=0.95,
        track_id=track_id,
        keypoints=keypoints,
    )


def _standing_keypoints() -> dict[str, Keypoint]:
    return {
        "left_shoulder": _kp(40, 40),
        "right_shoulder": _kp(60, 40),
        "left_hip": _kp(42, 105),
        "right_hip": _kp(58, 105),
        "left_knee": _kp(43, 165),
        "right_knee": _kp(57, 165),
        "left_ankle": _kp(44, 215),
        "right_ankle": _kp(56, 215),
    }


def _sitting_keypoints() -> dict[str, Keypoint]:
    return {
        "left_shoulder": _kp(40, 40),
        "right_shoulder": _kp(60, 40),
        "left_hip": _kp(42, 110),
        "right_hip": _kp(58, 110),
        "left_knee": _kp(28, 126),
        "right_knee": _kp(72, 126),
        "left_ankle": _kp(28, 176),
        "right_ankle": _kp(72, 176),
    }


def _crouching_keypoints() -> dict[str, Keypoint]:
    return {
        "left_shoulder": _kp(40, 40),
        "right_shoulder": _kp(60, 40),
        "left_hip": _kp(42, 110),
        "right_hip": _kp(58, 110),
        "left_knee": _kp(42, 155),
        "right_knee": _kp(58, 155),
        "left_ankle": _kp(25, 188),
        "right_ankle": _kp(75, 188),
    }


def test_midpoint_distance_and_joint_angle_helpers() -> None:
    first = _kp(0, 0)
    joint = _kp(0, 1)
    third = _kp(1, 1)

    middle = midpoint(first, third)

    assert middle.x == 0.5
    assert middle.y == 0.5
    assert distance(first, third) > 1.4
    assert compute_joint_angle(first, joint, third) == 90.0


def test_validate_keypoint_quality_requires_presence_and_confidence() -> None:
    person = _person(_standing_keypoints())

    assert validate_keypoint_quality(person, ["left_hip", "right_hip"], 0.35) is True
    assert validate_keypoint_quality(person, ["left_hip", "missing"], 0.35) is False
    assert validate_keypoint_quality(person, ["left_hip"], 0.99) is False


def test_synthetic_standing_skeleton_classifies_as_standing() -> None:
    status = classify_posture(_person(_standing_keypoints()))

    assert status.state == PostureState.STANDING
    assert status.track_id == 7
    assert "standing" not in status.reason.lower() or status.confidence > 0.0


def test_synthetic_sitting_skeleton_classifies_as_sitting() -> None:
    status = classify_posture(_person(_sitting_keypoints()))

    assert status.state == PostureState.SITTING
    assert status.confidence > 0.0
    assert "seated" in status.reason


def test_missing_hips_returns_unknown() -> None:
    keypoints = _standing_keypoints()
    del keypoints["left_hip"]
    del keypoints["right_hip"]

    status = classify_posture(_person(keypoints))

    assert status.state == PostureState.UNKNOWN


def test_missing_knees_returns_unknown() -> None:
    keypoints = _standing_keypoints()
    del keypoints["left_knee"]
    del keypoints["right_knee"]

    status = classify_posture(_person(keypoints))

    assert status.state == PostureState.UNKNOWN


def test_low_confidence_required_keypoint_returns_unknown() -> None:
    keypoints = _standing_keypoints()
    keypoints["left_hip"] = _kp(42, 105, confidence=0.10)

    status = classify_posture(_person(keypoints))

    assert status.state == PostureState.UNKNOWN


def test_only_upper_body_visible_returns_unknown() -> None:
    keypoints = {
        "left_shoulder": _kp(40, 40),
        "right_shoulder": _kp(60, 40),
    }

    status = classify_posture(_person(keypoints))

    assert status.state == PostureState.UNKNOWN


def test_crouching_or_squatting_like_pose_is_not_confidently_sitting() -> None:
    status = classify_posture(_person(_crouching_keypoints()))

    assert status.state == PostureState.UNKNOWN


def test_person_without_track_id_does_not_crash() -> None:
    status = classify_posture(_person(_sitting_keypoints(), track_id=None))

    assert status.track_id is None
    assert status.state == PostureState.SITTING
