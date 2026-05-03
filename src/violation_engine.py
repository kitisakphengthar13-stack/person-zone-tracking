from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from compliance import NON_COMPLIANT, UNKNOWN, PersonPPEStatus
from events import (
    ACTIVE,
    CLEARED,
    PENDING,
    PPE_VIOLATION,
    UNKNOWN_PPE,
    ViolationEvent,
    ViolationKey,
    ViolationRecord,
    ViolationType,
)


@dataclass(frozen=True)
class ViolationObservation:
    key: ViolationKey
    zone_name: str
    missing_items: set[str]


class ViolationStateTracker:
    def __init__(
        self,
        min_violation_seconds: float = 2.0,
        clear_after_seconds: float = 1.0,
        max_missing_track_seconds: float = 1.5,
        emit_unknown_ppe: bool = True,
    ) -> None:
        self.min_violation_seconds = max(0.0, float(min_violation_seconds))
        self.clear_after_seconds = max(0.0, float(clear_after_seconds))
        self.max_missing_track_seconds = max(0.0, float(max_missing_track_seconds))
        self.emit_unknown_ppe = emit_unknown_ppe
        self.records: dict[ViolationKey, ViolationRecord] = {}

    def update(
        self,
        zone_matches: Iterable[object],
        ppe_statuses: Iterable[PersonPPEStatus],
        timestamp: float,
    ) -> list[ViolationEvent]:
        observations = self._build_observations(zone_matches, ppe_statuses)
        events: list[ViolationEvent] = []

        for observation in observations.values():
            event = self._update_observed_record(observation, timestamp)
            if event is not None:
                events.append(event)

        observed_keys = set(observations)
        for key in list(self.records):
            if key in observed_keys:
                continue
            event = self._update_unobserved_record(key, timestamp)
            if event is not None:
                events.append(event)

        return events

    def reset(self) -> None:
        self.records.clear()

    def _build_observations(
        self,
        zone_matches: Iterable[object],
        ppe_statuses: Iterable[PersonPPEStatus],
    ) -> dict[ViolationKey, ViolationObservation]:
        statuses_by_track = {
            int(status.track_id): status
            for status in ppe_statuses
            if status.track_id is not None
        }
        observations: dict[ViolationKey, ViolationObservation] = {}

        for zone_match in zone_matches:
            track_id = _safe_track_id(getattr(zone_match.detection, "track_id", None))
            if track_id is None:
                continue

            status = statuses_by_track.get(track_id)
            if status is None:
                continue

            violation_type = self._violation_type_for_status(status)
            if violation_type is None:
                continue

            key = ViolationKey(
                zone_id=str(zone_match.zone_id),
                track_id=track_id,
                violation_type=violation_type,
            )
            missing_items = set(status.missing_items)
            existing = observations.get(key)
            if existing is None:
                observations[key] = ViolationObservation(
                    key=key,
                    zone_name=str(zone_match.zone_name),
                    missing_items=missing_items,
                )
            else:
                existing.missing_items.update(missing_items)

        return observations

    def _violation_type_for_status(
        self,
        status: PersonPPEStatus,
    ) -> ViolationType | None:
        if status.compliance_state == NON_COMPLIANT and status.missing_items:
            return PPE_VIOLATION
        if self.emit_unknown_ppe and status.compliance_state == UNKNOWN:
            return UNKNOWN_PPE
        return None

    def _update_observed_record(
        self,
        observation: ViolationObservation,
        timestamp: float,
    ) -> ViolationEvent | None:
        record = self.records.get(observation.key)
        if record is None or record.state == CLEARED:
            self.records[observation.key] = ViolationRecord(
                state=PENDING,
                first_seen=timestamp,
                last_seen=timestamp,
                missing_items=set(observation.missing_items),
                zone_name=observation.zone_name,
            )
            return None

        if (
            record.state == PENDING
            and record.cleared_since is not None
            and timestamp - record.cleared_since >= self.clear_after_seconds
        ):
            record.first_seen = timestamp
            record.emitted = False

        record.last_seen = timestamp
        record.cleared_since = None
        record.missing_items = set(observation.missing_items)
        record.zone_name = observation.zone_name

        if record.state == PENDING:
            duration = timestamp - record.first_seen
            if duration >= self.min_violation_seconds:
                record.state = ACTIVE
                record.active_since = record.first_seen
                record.emitted = True
                return _active_event(observation.key, record, timestamp)

        return None

    def _update_unobserved_record(
        self,
        key: ViolationKey,
        timestamp: float,
    ) -> ViolationEvent | None:
        record = self.records[key]

        if record.state == PENDING:
            if record.cleared_since is None:
                record.cleared_since = timestamp
            if timestamp - record.cleared_since >= self.clear_after_seconds:
                record.state = CLEARED
                record.cleared_since = timestamp
                record.emitted = False
            return None

        if record.state != ACTIVE:
            return None

        if record.cleared_since is None:
            record.cleared_since = timestamp

        disappearance_seconds = timestamp - record.last_seen
        if disappearance_seconds < self.clear_after_seconds:
            return None

        if disappearance_seconds > self.max_missing_track_seconds:
            record.cleared_since = timestamp

        record.state = CLEARED
        return _cleared_event(key, record, timestamp)


def _active_event(
    key: ViolationKey,
    record: ViolationRecord,
    timestamp: float,
) -> ViolationEvent:
    started_at = record.active_since if record.active_since is not None else record.first_seen
    return ViolationEvent(
        event_type="active",
        zone_id=key.zone_id,
        zone_name=record.zone_name,
        track_id=key.track_id,
        violation_type=key.violation_type,
        state=ACTIVE,
        missing_items=set(record.missing_items),
        started_at=started_at,
        ended_at=None,
        duration_seconds=max(0.0, timestamp - started_at),
    )


def _cleared_event(
    key: ViolationKey,
    record: ViolationRecord,
    timestamp: float,
) -> ViolationEvent:
    started_at = record.active_since if record.active_since is not None else record.first_seen
    return ViolationEvent(
        event_type="cleared",
        zone_id=key.zone_id,
        zone_name=record.zone_name,
        track_id=key.track_id,
        violation_type=key.violation_type,
        state=CLEARED,
        missing_items=set(record.missing_items),
        started_at=started_at,
        ended_at=timestamp,
        duration_seconds=max(0.0, timestamp - started_at),
    )


def _safe_track_id(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
