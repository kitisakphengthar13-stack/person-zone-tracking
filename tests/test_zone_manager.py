from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from zone_manager import ZoneManager


def _write_zones(path: Path, target_classes: object) -> None:
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "zones": [
                    {
                        "id": "zone_1",
                        "name": "Zone 1",
                        "points": [[0, 0], [100, 0], [100, 100], [0, 100]],
                        "target_classes": target_classes,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _detection(class_name: str, bbox: tuple[float, float, float, float]) -> object:
    return SimpleNamespace(class_name=class_name, bbox=bbox, track_id=1)


def test_zone_loading_keeps_list_target_classes(tmp_path: Path) -> None:
    zones_path = tmp_path / "zones.json"
    _write_zones(zones_path, ["person", "worker"])

    manager = ZoneManager.from_file(zones_path, ["person"])

    assert manager.zones[0].target_classes == ["person", "worker"]


def test_zone_loading_normalizes_string_target_class(tmp_path: Path) -> None:
    zones_path = tmp_path / "zones.json"
    _write_zones(zones_path, "person")

    manager = ZoneManager.from_file(zones_path, ["worker"])

    assert manager.zones[0].target_classes == ["person"]
    matches = manager.match_detections([_detection("person", (10, 10, 30, 30))])
    assert len(matches) == 1
    assert matches[0].zone_id == "zone_1"
    assert matches[0].match_method == "center"


def test_zone_matching_respects_allowed_classes(tmp_path: Path) -> None:
    zones_path = tmp_path / "zones.json"
    _write_zones(zones_path, ["person"])
    manager = ZoneManager.from_file(zones_path, ["person"])

    matches = manager.match_detections(
        [
            _detection("person", (10, 10, 30, 30)),
            _detection("car", (10, 10, 30, 30)),
        ]
    )

    assert [match.detection.class_name for match in matches] == ["person"]
