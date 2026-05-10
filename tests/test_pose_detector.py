from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pose_detector import extract_pose_people, pose_result_to_people


class FakeBoxes:
    def __init__(
        self,
        xyxy: list[list[float]],
        conf: list[float] | None = None,
        ids: list[float] | None = None,
    ) -> None:
        self.xyxy = np.asarray(xyxy, dtype=float)
        self.conf = None if conf is None else np.asarray(conf, dtype=float)
        self.id = None if ids is None else np.asarray(ids, dtype=float)

    def __len__(self) -> int:
        return len(self.xyxy)


def _keypoints(
    count: int,
    confidence: float = 0.80,
) -> tuple[np.ndarray, np.ndarray]:
    xy = np.zeros((1, 17, 2), dtype=float)
    conf = np.full((1, 17), confidence, dtype=float)
    for index in range(count):
        xy[0, index] = [float(index * 10), float(index * 10 + 5)]
    return xy, conf


def _result(
    boxes: FakeBoxes | None,
    keypoint_xy: np.ndarray | None = None,
    keypoint_conf: np.ndarray | None = None,
) -> object:
    keypoints = None
    if keypoint_xy is not None:
        keypoints = SimpleNamespace(xy=keypoint_xy, conf=keypoint_conf)
    return SimpleNamespace(boxes=boxes, keypoints=keypoints)


def test_one_pose_detection_converts_to_pose_person() -> None:
    xy, conf = _keypoints(17, confidence=0.85)
    result = _result(
        boxes=FakeBoxes([[10, 20, 110, 220]], conf=[0.91], ids=[7]),
        keypoint_xy=xy,
        keypoint_conf=conf,
    )

    people = pose_result_to_people(result)

    assert len(people) == 1
    person = people[0]
    assert person.bbox == (10.0, 20.0, 110.0, 220.0)
    assert person.confidence == 0.91
    assert person.track_id == 7
    assert person.class_name == "person"


def test_multiple_pose_detections_convert_to_multiple_pose_people() -> None:
    xy = np.zeros((2, 17, 2), dtype=float)
    conf = np.full((2, 17), 0.75, dtype=float)
    result = _result(
        boxes=FakeBoxes(
            [[10, 20, 110, 220], [200, 30, 300, 230]],
            conf=[0.91, 0.88],
            ids=[7, 8],
        ),
        keypoint_xy=xy,
        keypoint_conf=conf,
    )

    people = pose_result_to_people(result, person_class_name="worker")

    assert [person.track_id for person in people] == [7, 8]
    assert [person.confidence for person in people] == [0.91, 0.88]
    assert [person.class_name for person in people] == ["worker", "worker"]


def test_missing_track_id_is_handled_safely() -> None:
    xy, conf = _keypoints(17)
    result = _result(
        boxes=FakeBoxes([[10, 20, 110, 220]], conf=[0.91], ids=None),
        keypoint_xy=xy,
        keypoint_conf=conf,
    )

    people = pose_result_to_people(result)

    assert len(people) == 1
    assert people[0].track_id is None


def test_keypoint_names_coordinates_and_confidence_are_preserved() -> None:
    xy, conf = _keypoints(17, confidence=0.67)
    result = _result(
        boxes=FakeBoxes([[10, 20, 110, 220]], conf=[0.91], ids=[7]),
        keypoint_xy=xy,
        keypoint_conf=conf,
    )

    person = pose_result_to_people(result)[0]

    assert person.keypoints["left_shoulder"].x == 50.0
    assert person.keypoints["left_shoulder"].y == 55.0
    assert person.keypoints["left_shoulder"].confidence == 0.67
    assert person.keypoints["right_hip"].x == 120.0
    assert person.keypoints["left_knee"].x == 130.0


def test_missing_keypoints_does_not_crash() -> None:
    result = _result(
        boxes=FakeBoxes([[10, 20, 110, 220]], conf=[0.91], ids=[7]),
        keypoint_xy=None,
        keypoint_conf=None,
    )

    people = pose_result_to_people(result)

    assert len(people) == 1
    assert people[0].keypoints == {}


def test_partial_keypoints_are_preserved_without_crashing() -> None:
    xy, conf = _keypoints(3, confidence=0.50)
    result = _result(
        boxes=FakeBoxes([[10, 20, 110, 220]], conf=[0.91], ids=[7]),
        keypoint_xy=xy[:, :3, :],
        keypoint_conf=conf[:, :3],
    )

    person = pose_result_to_people(result)[0]

    assert sorted(person.keypoints) == ["left_eye", "nose", "right_eye"]


def test_missing_boxes_returns_empty_list() -> None:
    assert pose_result_to_people(_result(boxes=None)) == []


def test_extract_pose_people_handles_empty_results() -> None:
    assert extract_pose_people(None) == []
    assert extract_pose_people([]) == []
