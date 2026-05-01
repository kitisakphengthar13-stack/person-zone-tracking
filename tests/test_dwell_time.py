from __future__ import annotations

from types import SimpleNamespace

from dwell_time import DwellTimeTracker


def _match(zone_id: str = "zone_1", class_name: str = "person", track_id: int = 7) -> object:
    detection = SimpleNamespace(class_name=class_name, track_id=track_id)
    return SimpleNamespace(zone_id=zone_id, detection=detection)


def test_dwell_time_accumulates_leave_and_reentry() -> None:
    tracker = DwellTimeTracker()
    match = _match()

    tracker.update([match], 0.0)
    tracker.update([match], 2.0)
    tracker.update([], 5.0)
    tracker.update([match], 7.0)
    tracker.update([match], 9.0)

    assert tracker.get_current_dwell("zone_1", "person", 7) == 7.0
    summary = tracker.get_zone_summary("zone_1")
    assert summary["zone_1"]["person"][7]["inside"] is True


def test_dwell_time_reset_clears_records() -> None:
    tracker = DwellTimeTracker()
    tracker.update([_match()], 0.0)
    tracker.update([_match()], 3.0)

    tracker.reset()

    assert tracker.get_zone_summary() == {}
