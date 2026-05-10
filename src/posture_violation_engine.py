from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from pose_types import PostureState, PostureStatus
from posture_events import (
    ACTIVE,
    CLEARED,
    PENDING,
    SITTING_VIOLATION_ACTIVE,
    SITTING_VIOLATION_CLEARED,
    PostureViolationEvent,
    PostureViolationKey,
    PostureViolationRecord,
)


@dataclass(frozen=True)
class PostureObservation:
    zone_name: str
    posture_state: PostureState


class PostureViolationTracker:
    def __init__(
        self,
        min_violation_seconds: float = 30.0,
        clear_after_seconds: float = 2.0,
        unknown_grace_seconds: float = 3.0,
        max_missing_track_seconds: float = 2.0,
        ignore_outside_zones: bool = True,
    ) -> None:
        self.min_violation_seconds = max(0.0, float(min_violation_seconds))
        self.clear_after_seconds = max(0.0, float(clear_after_seconds))
        self.unknown_grace_seconds = max(0.0, float(unknown_grace_seconds))
        self.max_missing_track_seconds = max(0.0, float(max_missing_track_seconds))
        self.ignore_outside_zones = ignore_outside_zones
        self.records: dict[PostureViolationKey, PostureViolationRecord] = {}

    def update(
        self,
        zone_matches: Iterable[object],
        posture_statuses: Iterable[PostureStatus],
        timestamp: float,
    ) -> list[PostureViolationEvent]:
        statuses_by_track = {
            int(status.track_id): status
            for status in posture_statuses
            if status.track_id is not None
        }
        observations = self._zone_observations(zone_matches, statuses_by_track)
        events: list[PostureViolationEvent] = []

        for key, observation in observations.items():
            event = self._update_observed_record(
                key=key,
                zone_name=observation.zone_name,
                posture_state=observation.posture_state,
                timestamp=timestamp,
            )
            if event is not None:
                events.append(event)

        observed_keys = set(observations)
        for key in list(self.records):
            if key in observed_keys:
                continue
            event = self._update_missing_record(key, timestamp)
            if event is not None:
                events.append(event)

        return events

    def reset(self) -> None:
        self.records.clear()

    def _zone_observations(
        self,
        zone_matches: Iterable[object],
        statuses_by_track: dict[int, PostureStatus],
    ) -> dict[PostureViolationKey, PostureObservation]:
        observations: dict[PostureViolationKey, PostureObservation] = {}
        if self.ignore_outside_zones:
            matches = zone_matches
        else:
            matches = zone_matches

        for match in matches:
            track_id = _track_id_from_match(match)
            if track_id is None:
                continue
            status = statuses_by_track.get(track_id)
            if status is None:
                continue

            key = PostureViolationKey(zone_id=str(match.zone_id), track_id=track_id)
            observations[key] = PostureObservation(
                zone_name=str(getattr(match, "zone_name", match.zone_id)),
                posture_state=status.state,
            )
        return observations

    def _update_observed_record(
        self,
        key: PostureViolationKey,
        zone_name: str,
        posture_state: PostureState,
        timestamp: float,
    ) -> PostureViolationEvent | None:
        record = self.records.get(key)

        if posture_state == PostureState.SITTING:
            if record is None or record.state == CLEARED:
                record = PostureViolationRecord(
                    state=PENDING,
                    first_sitting_seen=timestamp,
                    last_seen=timestamp,
                    zone_name=zone_name,
                )
                self.records[key] = record
                return None

            if record.unknown_started_at is not None:
                unknown_seconds = timestamp - record.unknown_started_at
                if unknown_seconds > self.unknown_grace_seconds:
                    record.first_sitting_seen = timestamp
                    record.emitted_active = False
                    record.state = PENDING

            record.last_seen = timestamp
            record.zone_name = zone_name
            record.clear_started_at = None
            record.unknown_started_at = None
            record.missing_started_at = None

            if record.state == PENDING:
                duration = timestamp - record.first_sitting_seen
                if duration > self.min_violation_seconds and not record.emitted_active:
                    record.state = ACTIVE
                    record.active_since = record.first_sitting_seen
                    record.emitted_active = True
                    return _active_event(key, record, timestamp)
            return None

        if record is None:
            return None

        record.last_seen = timestamp
        record.zone_name = zone_name
        record.missing_started_at = None

        if posture_state == PostureState.UNKNOWN:
            if record.unknown_started_at is None:
                record.unknown_started_at = timestamp
            if timestamp - record.unknown_started_at <= self.unknown_grace_seconds:
                return None
            if record.state == PENDING:
                record.first_sitting_seen = timestamp
                record.emitted_active = False
            return None

        if posture_state == PostureState.STANDING:
            record.unknown_started_at = None
            if record.clear_started_at is None:
                record.clear_started_at = timestamp
            if timestamp - record.clear_started_at < self.clear_after_seconds:
                return None

            if record.state == ACTIVE:
                record.state = CLEARED
                return _cleared_event(key, record, timestamp)
            if record.state == PENDING:
                record.state = CLEARED
            return None

        return None

    def _update_missing_record(
        self,
        key: PostureViolationKey,
        timestamp: float,
    ) -> PostureViolationEvent | None:
        record = self.records[key]
        if record.missing_started_at is None:
            record.missing_started_at = timestamp
        if timestamp - record.missing_started_at <= self.max_missing_track_seconds:
            return None

        if record.state == ACTIVE:
            record.state = CLEARED
            return _cleared_event(key, record, timestamp)
        record.state = CLEARED
        return None


def _active_event(
    key: PostureViolationKey,
    record: PostureViolationRecord,
    timestamp: float,
) -> PostureViolationEvent:
    started_at = record.active_since if record.active_since is not None else record.first_sitting_seen
    return PostureViolationEvent(
        event_type=SITTING_VIOLATION_ACTIVE,
        zone_id=key.zone_id,
        zone_name=record.zone_name,
        track_id=key.track_id,
        state=ACTIVE,
        posture_state=PostureState.SITTING,
        started_at=started_at,
        ended_at=None,
        duration_seconds=max(0.0, timestamp - started_at),
    )


def _cleared_event(
    key: PostureViolationKey,
    record: PostureViolationRecord,
    timestamp: float,
) -> PostureViolationEvent:
    started_at = record.active_since if record.active_since is not None else record.first_sitting_seen
    return PostureViolationEvent(
        event_type=SITTING_VIOLATION_CLEARED,
        zone_id=key.zone_id,
        zone_name=record.zone_name,
        track_id=key.track_id,
        state=CLEARED,
        posture_state=PostureState.STANDING,
        started_at=started_at,
        ended_at=timestamp,
        duration_seconds=max(0.0, timestamp - started_at),
    )


def _track_id_from_match(match: object) -> int | None:
    detection = getattr(match, "detection", None)
    track_id = getattr(detection, "track_id", None)
    if track_id is None:
        return None
    try:
        return int(track_id)
    except (TypeError, ValueError):
        return None
