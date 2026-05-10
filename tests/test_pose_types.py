from __future__ import annotations

from pose_types import Keypoint, PosePerson, PostureState, PostureStatus


def test_keypoint_stores_coordinates_confidence_and_visibility() -> None:
    keypoint = Keypoint(x=12.5, y=30.0, confidence=0.82, visible=True)

    assert keypoint.x == 12.5
    assert keypoint.y == 30.0
    assert keypoint.confidence == 0.82
    assert keypoint.visible is True
    assert keypoint.is_usable(0.50) is True
    assert keypoint.is_usable(0.90) is False


def test_pose_person_exposes_zone_compatible_detection_fields() -> None:
    nose = Keypoint(x=20.0, y=10.0, confidence=0.75)
    pose_person = PosePerson(
        bbox=(10.0, 5.0, 100.0, 200.0),
        confidence=0.91,
        track_id=7,
        keypoints={"nose": nose},
    )

    assert pose_person.bbox == (10.0, 5.0, 100.0, 200.0)
    assert pose_person.class_name == "person"
    assert pose_person.confidence == 0.91
    assert pose_person.track_id == 7
    assert pose_person.keypoints["nose"] == nose


def test_posture_status_links_state_to_pose_person() -> None:
    pose_person = PosePerson(
        bbox=(10.0, 5.0, 100.0, 200.0),
        confidence=0.91,
        track_id=7,
    )

    status = PostureStatus(
        track_id=7,
        pose_person=pose_person,
        state=PostureState.SITTING,
        confidence=0.80,
        reason="synthetic test posture",
        zone_id="zone_1",
        sitting_seconds=12.5,
    )

    assert status.track_id == 7
    assert status.pose_person == pose_person
    assert status.state == PostureState.SITTING
    assert status.confidence == 0.80
    assert status.reason == "synthetic test posture"
    assert status.zone_id == "zone_1"
    assert status.sitting_seconds == 12.5
    assert status.violation_active is False
