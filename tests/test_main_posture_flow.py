from __future__ import annotations

from types import SimpleNamespace

from detector import Detection
from main import (
    _create_posture_violation_tracker,
    _maybe_log_posture_events,
    _process_frame_analytics,
)
from pose_types import Keypoint, PosePerson, PostureState
from posture_events import ACTIVE, SITTING_VIOLATION_ACTIVE, PostureViolationEvent
from zone_manager import ZoneMatch


class FakeObjectDetector:
    def __init__(self, detections: list[object]) -> None:
        self.detections = detections
        self.calls = 0

    def detect(self, frame: object) -> list[object]:
        self.calls += 1
        return self.detections


class FakePoseDetector:
    def __init__(self, people: list[PosePerson]) -> None:
        self.people = people
        self.calls = 0

    def detect(self, frame: object) -> list[PosePerson]:
        self.calls += 1
        return self.people


class FakeZoneManager:
    zones: list[object] = []

    def __init__(self) -> None:
        self.received_detections: list[object] | None = None

    def match_detections(self, detections: list[object]) -> list[ZoneMatch]:
        self.received_detections = detections
        return [
            ZoneMatch(
                zone_id="zone_1",
                zone_name="Zone 1",
                detection=detection,
                anchor_point=(0.0, 0.0),
            )
            for detection in detections
        ]


class FakeDwellTracker:
    def __init__(self) -> None:
        self.updated_matches: list[ZoneMatch] | None = None
        self.updated_timestamp: float | None = None

    def update(self, zone_matches: list[ZoneMatch], timestamp: float) -> None:
        self.updated_matches = zone_matches
        self.updated_timestamp = timestamp


class FakePostureViolationTracker:
    def __init__(self) -> None:
        self.received_matches: list[ZoneMatch] | None = None
        self.received_statuses: list[object] | None = None
        self.received_timestamp: float | None = None

    def update(
        self,
        zone_matches: list[ZoneMatch],
        posture_statuses: list[object],
        timestamp: float,
    ) -> list[object]:
        self.received_matches = zone_matches
        self.received_statuses = posture_statuses
        self.received_timestamp = timestamp
        return [SimpleNamespace(event_type="fake")]


def _config(posture_enabled: bool) -> object:
    return SimpleNamespace(
        posture=SimpleNamespace(
            enabled=posture_enabled,
            min_violation_seconds=30.0,
            clear_after_seconds=2.0,
            unknown_grace_seconds=3.0,
            max_missing_track_seconds=2.0,
            zones=SimpleNamespace(ignore_outside_zones=True),
            events=SimpleNamespace(
                enabled=True,
                log_events=False,
                event_log_path=None,
            ),
        )
    )


def _standing_pose_person(track_id: int = 7) -> PosePerson:
    keypoints = {
        "left_shoulder": Keypoint(40, 40, 0.90),
        "right_shoulder": Keypoint(60, 40, 0.90),
        "left_hip": Keypoint(42, 105, 0.90),
        "right_hip": Keypoint(58, 105, 0.90),
        "left_knee": Keypoint(43, 165, 0.90),
        "right_knee": Keypoint(57, 165, 0.90),
        "left_ankle": Keypoint(44, 215, 0.90),
        "right_ankle": Keypoint(56, 215, 0.90),
    }
    return PosePerson(
        bbox=(10.0, 20.0, 100.0, 220.0),
        confidence=0.90,
        track_id=track_id,
        keypoints=keypoints,
    )


def test_posture_disabled_uses_original_detector_path() -> None:
    detection = Detection(
        bbox=(10.0, 20.0, 100.0, 220.0),
        class_id=0,
        class_name="person",
        confidence=0.90,
        track_id=7,
    )
    detector = FakeObjectDetector([detection])
    pose_detector = FakePoseDetector([_standing_pose_person()])
    zone_manager = FakeZoneManager()
    dwell_tracker = FakeDwellTracker()

    analytics = _process_frame_analytics(
        config=_config(posture_enabled=False),
        frame=object(),
        detector=detector,
        pose_detector=pose_detector,
        zone_manager=zone_manager,
        dwell_tracker=dwell_tracker,
        posture_violation_tracker=None,
        timestamp=10.0,
    )

    assert detector.calls == 1
    assert pose_detector.calls == 0
    assert zone_manager.received_detections == [detection]
    assert analytics.detections_for_display == [detection]
    assert analytics.posture_statuses == []
    assert analytics.posture_events == []
    assert dwell_tracker.updated_timestamp == 10.0


def test_posture_enabled_uses_pose_detector_path() -> None:
    pose_person = _standing_pose_person()
    detector = FakeObjectDetector([])
    pose_detector = FakePoseDetector([pose_person])
    zone_manager = FakeZoneManager()
    dwell_tracker = FakeDwellTracker()
    violation_tracker = FakePostureViolationTracker()

    analytics = _process_frame_analytics(
        config=_config(posture_enabled=True),
        frame=object(),
        detector=detector,
        pose_detector=pose_detector,
        zone_manager=zone_manager,
        dwell_tracker=dwell_tracker,
        posture_violation_tracker=violation_tracker,
        timestamp=10.0,
    )

    assert detector.calls == 0
    assert pose_detector.calls == 1
    assert analytics.detections_for_display == [pose_person]
    assert zone_manager.received_detections == [pose_person]
    assert analytics.posture_statuses[0].state == PostureState.STANDING


def test_posture_enabled_updates_violation_tracker() -> None:
    pose_person = _standing_pose_person()
    zone_manager = FakeZoneManager()
    violation_tracker = FakePostureViolationTracker()

    analytics = _process_frame_analytics(
        config=_config(posture_enabled=True),
        frame=object(),
        detector=None,
        pose_detector=FakePoseDetector([pose_person]),
        zone_manager=zone_manager,
        dwell_tracker=FakeDwellTracker(),
        posture_violation_tracker=violation_tracker,
        timestamp=12.0,
    )

    assert violation_tracker.received_matches == analytics.zone_matches
    assert violation_tracker.received_statuses == analytics.posture_statuses
    assert violation_tracker.received_timestamp == 12.0
    assert len(analytics.posture_events) == 1


def test_create_posture_violation_tracker_uses_config_values() -> None:
    tracker = _create_posture_violation_tracker(_config(posture_enabled=True))

    assert tracker.min_violation_seconds == 30.0
    assert tracker.clear_after_seconds == 2.0
    assert tracker.unknown_grace_seconds == 3.0
    assert tracker.max_missing_track_seconds == 2.0
    assert tracker.ignore_outside_zones is True


def test_maybe_log_posture_events_is_gated_by_config(tmp_path) -> None:
    config = _config(posture_enabled=True)
    config.posture.events.log_events = True
    config.posture.events.event_log_path = tmp_path / "posture_events.jsonl"
    event = PostureViolationEvent(
        event_type=SITTING_VIOLATION_ACTIVE,
        zone_id="zone_1",
        zone_name="Zone 1",
        track_id=7,
        state=ACTIVE,
        posture_state=PostureState.SITTING,
        started_at=0.0,
        ended_at=None,
        duration_seconds=31.0,
    )

    _maybe_log_posture_events(config, [event], 31.0)

    assert config.posture.events.event_log_path.exists()


def test_maybe_log_posture_events_does_nothing_when_disabled(tmp_path) -> None:
    config = _config(posture_enabled=False)
    config.posture.events.log_events = True
    config.posture.events.event_log_path = tmp_path / "posture_events.jsonl"

    _maybe_log_posture_events(config, [], 31.0)

    assert not config.posture.events.event_log_path.exists()
