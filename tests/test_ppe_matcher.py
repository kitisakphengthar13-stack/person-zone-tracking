from __future__ import annotations

from detector import Detection
from ppe_matcher import (
    PPEMatchConfig,
    bbox_area,
    bbox_intersection,
    bbox_overlap_ratio,
    match_ppe_to_persons,
    person_body_region,
    split_person_and_ppe_detections,
)


def _detection(
    class_name: str,
    bbox: tuple[float, float, float, float],
    track_id: int | None = None,
) -> Detection:
    return Detection(
        bbox=bbox,
        class_id=0,
        class_name=class_name,
        confidence=0.90,
        track_id=track_id,
    )


def test_split_person_and_ppe_detections() -> None:
    person = _detection("person", (0, 0, 100, 200), track_id=1)
    helmet = _detection("helmet", (30, 0, 70, 25))
    car = _detection("car", (200, 200, 300, 300))

    persons, ppe_items = split_person_and_ppe_detections(
        [person, helmet, car],
        person_classes={"person"},
        ppe_classes={"helmet", "vest"},
    )

    assert persons == [person]
    assert ppe_items == [helmet]


def test_bbox_utilities_handle_intersection_and_overlap() -> None:
    bbox_a = (0, 0, 100, 100)
    bbox_b = (50, 50, 150, 150)

    assert bbox_area(bbox_a) == 10_000.0
    assert bbox_intersection(bbox_a, bbox_b) == (50.0, 50.0, 100.0, 100.0)
    assert bbox_overlap_ratio(bbox_a, bbox_b) == 0.25


def test_person_body_regions_use_expected_vertical_slices() -> None:
    person_bbox = (0, 0, 100, 200)

    assert person_body_region(person_bbox, "head") == (0.0, 0.0, 100.0, 60.0)
    assert person_body_region(person_bbox, "torso") == (0.0, 50.0, 100.0, 150.0)
    assert person_body_region(person_bbox, "lower_body") == (0.0, 100.0, 100.0, 200.0)
    assert person_body_region(person_bbox, "full_body") == (0.0, 0.0, 100.0, 200.0)


def test_match_ppe_to_tracked_person_by_body_region() -> None:
    person = _detection("person", (0, 0, 100, 200), track_id=7)
    helmet = _detection("helmet", (35, 5, 65, 35))
    vest = _detection("vest", (25, 70, 75, 135))

    matches = match_ppe_to_persons([person], [helmet, vest])

    assert matches == {7: {"helmet": [helmet], "vest": [vest]}}


def test_match_ppe_ignores_items_outside_person() -> None:
    person = _detection("person", (0, 0, 100, 200), track_id=7)
    helmet = _detection("helmet", (200, 5, 230, 35))

    matches = match_ppe_to_persons([person], [helmet])

    assert matches == {7: {}}


def test_match_ppe_skips_untracked_persons() -> None:
    person = _detection("person", (0, 0, 100, 200), track_id=None)
    helmet = _detection("helmet", (35, 5, 65, 35))

    assert match_ppe_to_persons([person], [helmet]) == {}


def test_match_ppe_assigns_item_to_best_person() -> None:
    first_person = _detection("person", (0, 0, 100, 200), track_id=1)
    second_person = _detection("person", (90, 0, 190, 200), track_id=2)
    vest = _detection("vest", (115, 70, 155, 135))

    matches = match_ppe_to_persons([first_person, second_person], [vest])

    assert matches == {1: {}, 2: {"vest": [vest]}}


def test_match_ppe_respects_custom_region_mapping() -> None:
    person = _detection("person", (0, 0, 100, 200), track_id=7)
    badge = _detection("badge", (40, 80, 60, 100))
    config = PPEMatchConfig(ppe_regions={"badge": "torso"})

    matches = match_ppe_to_persons([person], [badge], config)

    assert matches == {7: {"badge": [badge]}}


def test_match_ppe_uses_raw_ppe_model_class_regions() -> None:
    person = _detection("Person", (0, 0, 100, 200), track_id=7)
    hardhat = _detection("Hardhat", (35, 5, 65, 35))
    no_hardhat = _detection("NO-Hardhat", (36, 10, 66, 40))
    vest = _detection("Safety Vest", (25, 70, 75, 135))
    no_vest = _detection("NO-Safety Vest", (26, 75, 76, 140))

    matches = match_ppe_to_persons(
        [person],
        [hardhat, no_hardhat, vest, no_vest],
    )

    assert matches == {
        7: {
            "hardhat": [hardhat],
            "no-hardhat": [no_hardhat],
            "safety vest": [vest],
            "no-safety vest": [no_vest],
        }
    }


def test_raw_safety_vest_outside_torso_region_is_not_matched() -> None:
    person = _detection("Person", (0, 0, 100, 200), track_id=7)
    vest_near_head = _detection("Safety Vest", (25, 5, 75, 35))

    matches = match_ppe_to_persons([person], [vest_near_head])

    assert matches == {7: {}}
