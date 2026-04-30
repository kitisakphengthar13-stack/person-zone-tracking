from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass
class DwellRecord:
    total_seconds: float = 0.0
    inside: bool = False
    last_timestamp: float | None = None


class DwellTimeTracker:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, dict[int, DwellRecord]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        self._inside_keys: set[tuple[str, str, int]] = set()

    def update(self, zone_matches: list[object], timestamp: float) -> None:
        active_keys = self._active_keys_from_matches(zone_matches)

        for key in active_keys:
            zone_id, class_name, track_id = key
            record = self._get_record(zone_id, class_name, track_id)
            if record.inside and record.last_timestamp is not None:
                record.total_seconds += max(0.0, timestamp - record.last_timestamp)
            record.inside = True
            record.last_timestamp = timestamp

        for key in list(self._inside_keys):
            if key in active_keys:
                continue
            zone_id, class_name, track_id = key
            record = self._get_record(zone_id, class_name, track_id)
            if record.inside and record.last_timestamp is not None:
                record.total_seconds += max(0.0, timestamp - record.last_timestamp)
            record.inside = False
            record.last_timestamp = None

        self._inside_keys = active_keys

    def get_current_dwell(self, zone_id: str, class_name: str, track_id: int) -> float:
        return self._get_record(zone_id, class_name, track_id).total_seconds

    def get_zone_summary(
        self, zone_id: str | None = None, active_only: bool = False
    ) -> dict[str, dict[str, dict[int, dict[str, float | bool | None]]]]:
        summary: dict[str, dict[str, dict[int, dict[str, float | bool | None]]]] = {}

        for current_zone_id, class_map in self.records.items():
            if zone_id is not None and current_zone_id != zone_id:
                continue
            summary[current_zone_id] = {}
            for class_name, track_map in class_map.items():
                summary[current_zone_id][class_name] = {}
                for track_id, record in track_map.items():
                    if active_only and not record.inside:
                        continue
                    summary[current_zone_id][class_name][track_id] = {
                        "total_seconds": record.total_seconds,
                        "inside": record.inside,
                        "last_timestamp": record.last_timestamp,
                    }

        return summary

    def reset(self) -> None:
        self.records.clear()
        self._inside_keys.clear()

    def _get_record(self, zone_id: str, class_name: str, track_id: int) -> DwellRecord:
        class_records = self.records[zone_id][class_name]
        if track_id not in class_records:
            class_records[track_id] = DwellRecord()
        return class_records[track_id]

    @staticmethod
    def _active_keys_from_matches(zone_matches: list[object]) -> set[tuple[str, str, int]]:
        active_keys: set[tuple[str, str, int]] = set()
        for match in zone_matches:
            detection = match.detection
            if detection.track_id is None:
                continue
            active_keys.add(
                (
                    str(match.zone_id),
                    str(detection.class_name),
                    int(detection.track_id),
                )
            )
        return active_keys
