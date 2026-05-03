from __future__ import annotations

from detector import Detection
from compliance import (
    COMPLIANT,
    NON_COMPLIANT,
    UNKNOWN,
    evaluate_all_compliance,
    evaluate_person_compliance,
    normalize_ppe_item_name,
)


def _detection(
    class_name: str,
    bbox: tuple[float, float, float, float] = (0, 0, 100, 200),
    track_id: int | None = None,
) -> Detection:
    return Detection(
        bbox=bbox,
        class_id=0,
        class_name=class_name,
        confidence=0.90,
        track_id=track_id,
    )


def test_required_ppe_present_is_compliant_with_actual_model_class_names() -> None:
    person = _detection("Person", track_id=7)
    hardhat = _detection("Hardhat")
    vest = _detection("Safety Vest")
    mask = _detection("Mask")

    status = evaluate_person_compliance(
        person_detection=person,
        matched_items={
            "Hardhat": [hardhat],
            "Safety Vest": [vest],
            "Mask": [mask],
        },
        required_items={"hardhat", "safety_vest", "mask"},
    )

    assert status.track_id == 7
    assert status.compliance_state == COMPLIANT
    assert status.missing_items == set()
    assert status.matched_items == {
        "hardhat": [hardhat],
        "safety_vest": [vest],
        "mask": [mask],
    }


def test_missing_hardhat_is_non_compliant() -> None:
    person = _detection("Person", track_id=7)
    vest = _detection("Safety Vest")

    status = evaluate_person_compliance(
        person_detection=person,
        matched_items={"Safety Vest": [vest]},
        required_items={"hardhat", "safety_vest"},
    )

    assert status.compliance_state == NON_COMPLIANT
    assert status.missing_items == {"hardhat"}


def test_missing_multiple_required_items_reports_all_missing_items() -> None:
    person = _detection("Person", track_id=7)

    status = evaluate_person_compliance(
        person_detection=person,
        matched_items={},
        required_items={"hardhat", "safety_vest", "mask"},
    )

    assert status.compliance_state == NON_COMPLIANT
    assert status.missing_items == {"hardhat", "safety_vest", "mask"}


def test_no_required_ppe_is_compliant() -> None:
    person = _detection("Person", track_id=7)

    status = evaluate_person_compliance(
        person_detection=person,
        matched_items={},
        required_items=set(),
    )

    assert status.compliance_state == COMPLIANT
    assert status.required_items == set()
    assert status.missing_items == set()


def test_person_without_track_id_is_unknown() -> None:
    person = _detection("Person", track_id=None)

    status = evaluate_person_compliance(
        person_detection=person,
        matched_items={},
        required_items={"hardhat"},
    )

    assert status.track_id is None
    assert status.compliance_state == UNKNOWN
    assert status.missing_items == set()


def test_empty_matched_items_does_not_crash() -> None:
    person = _detection("Person", track_id=7)

    status = evaluate_person_compliance(
        person_detection=person,
        matched_items=None,
        required_items={"hardhat"},
    )

    assert status.compliance_state == NON_COMPLIANT
    assert status.matched_items == {}
    assert status.missing_items == {"hardhat"}


def test_multiple_persons_are_evaluated_independently() -> None:
    first_person = _detection("Person", track_id=1)
    second_person = _detection("Person", track_id=2)
    hardhat = _detection("Hardhat")
    vest = _detection("Safety Vest")

    statuses = evaluate_all_compliance(
        person_detections=[first_person, second_person],
        matched_items_by_track={
            1: {"Hardhat": [hardhat], "Safety Vest": [vest]},
            2: {"Hardhat": [hardhat]},
        },
        required_items={"hardhat", "safety_vest"},
    )

    assert [status.track_id for status in statuses] == [1, 2]
    assert statuses[0].compliance_state == COMPLIANT
    assert statuses[0].missing_items == set()
    assert statuses[1].compliance_state == NON_COMPLIANT
    assert statuses[1].missing_items == {"safety_vest"}


def test_direct_missing_class_is_explicit_missing_evidence() -> None:
    person = _detection("Person", track_id=7)
    no_hardhat = _detection("NO-Hardhat")
    vest = _detection("Safety Vest")

    status = evaluate_person_compliance(
        person_detection=person,
        matched_items={"NO-Hardhat": [no_hardhat], "Safety Vest": [vest]},
        required_items={"hardhat", "safety_vest"},
    )

    assert status.compliance_state == NON_COMPLIANT
    assert status.missing_items == {"hardhat"}
    assert status.matched_items["missing:hardhat"] == [no_hardhat]
    assert status.matched_items["safety_vest"] == [vest]


def test_raw_model_class_names_normalize_to_internal_ppe_names() -> None:
    assert normalize_ppe_item_name("Hardhat") == "hardhat"
    assert normalize_ppe_item_name("Safety Vest") == "safety_vest"
    assert normalize_ppe_item_name("Mask") == "mask"
    assert normalize_ppe_item_name("NO-Hardhat") == "missing:hardhat"
    assert normalize_ppe_item_name("NO-Safety Vest") == "missing:safety_vest"
    assert normalize_ppe_item_name("NO-Mask") == "missing:mask"
