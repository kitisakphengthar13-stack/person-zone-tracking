from __future__ import annotations

import numpy as np

from compliance import COMPLIANT, NON_COMPLIANT, UNKNOWN, PersonPPEStatus
from detector import Detection
from dwell_time import DwellTimeTracker
from visualizer import Visualizer, build_ppe_status_by_track, format_ppe_status_label


def _detection(track_id: int | None = 7) -> Detection:
    return Detection(
        bbox=(10, 10, 80, 160),
        class_id=5,
        class_name="Person",
        confidence=0.90,
        track_id=track_id,
    )


def _status(
    state: str,
    missing_items: set[str] | None = None,
    track_id: int | None = 7,
) -> PersonPPEStatus:
    return PersonPPEStatus(
        track_id=track_id,
        person_detection=_detection(track_id),
        matched_items={},
        required_items={"hardhat", "safety_vest"},
        missing_items=missing_items or set(),
        compliance_state=state,
    )


def test_format_ppe_status_label_for_compliant_status() -> None:
    assert format_ppe_status_label(_status(COMPLIANT)) == "compliant"


def test_format_ppe_status_label_for_one_missing_item() -> None:
    assert format_ppe_status_label(_status(NON_COMPLIANT, {"hardhat"})) == (
        "missing: hardhat"
    )


def test_format_ppe_status_label_for_multiple_missing_items() -> None:
    assert format_ppe_status_label(
        _status(NON_COMPLIANT, {"safety_vest", "hardhat"})
    ) == "missing: hardhat, safety_vest"


def test_format_ppe_status_label_for_unknown_status() -> None:
    assert format_ppe_status_label(_status(UNKNOWN)) == "unknown PPE"


def test_build_ppe_status_by_track_skips_unknown_track_ids() -> None:
    status = _status(COMPLIANT, track_id=7)
    unknown_status = _status(UNKNOWN, track_id=None)

    assert build_ppe_status_by_track([status, unknown_status]) == {7: status}


def test_visualizer_draw_remains_backward_compatible_without_ppe_statuses() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    visualizer = Visualizer()

    output = visualizer.draw(
        frame=frame,
        zones=[],
        detections=[_detection()],
        zone_matches=[],
        dwell_tracker=DwellTimeTracker(),
    )

    assert output.shape == frame.shape
    assert np.count_nonzero(output) > 0
