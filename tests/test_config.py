from __future__ import annotations

from pathlib import Path

import pytest

from config import load_config
from utils import parse_bool


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("true", True),
        ("yes", True),
        ("1", True),
        ("false", False),
        ("no", False),
        ("0", False),
        (True, True),
        (False, False),
    ],
)
def test_parse_bool_accepts_common_cli_values(value: object, expected: bool) -> None:
    assert parse_bool(value) is expected


def test_parse_bool_rejects_unknown_value() -> None:
    with pytest.raises(ValueError):
        parse_bool("maybe")


def test_load_config_merges_yaml_and_cli_overrides(tmp_path: Path) -> None:
    config_file = tmp_path / "app.yaml"
    config_file.write_text(
        "\n".join(
            [
                "model_path: models/best.pt",
                "conf: 0.25",
                "iou: 0.40",
                "target_classes:",
                "  - person",
                "source_type: webcam",
                "camera_id: 0",
                "source_path: data/videos/input.mp4",
                "zones_path: assets/sample_zones.json",
                "draw_zones: false",
                "save_output: false",
                "output_path: data/outputs/result.mp4",
                "display: true",
                "device: auto",
                "imgsz: 640",
                "tracker_config: bytetrack.yaml",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(
        [
            "--config",
            str(config_file),
            "--conf",
            "0.5",
            "--classes",
            "person",
            "worker",
            "--display",
            "false",
            "--save-output",
            "true",
        ]
    )

    assert config.config_file == config_file.resolve()
    assert config.conf == 0.5
    assert config.iou == 0.40
    assert config.target_classes == ["person", "worker"]
    assert config.display is False
    assert config.save_output is True
    assert config.output_path == (Path.cwd() / "data/outputs/result.mp4").resolve()
    assert config.posture.enabled is False
    assert config.posture.model_path == (Path.cwd() / "models/yolo26n-pose.pt").resolve()
    assert config.posture.min_violation_seconds == 30.0


def test_load_config_parses_posture_settings(tmp_path: Path) -> None:
    config_file = tmp_path / "app.yaml"
    config_file.write_text(
        "\n".join(
            [
                "model_path: models/best.pt",
                "conf: 0.25",
                "iou: 0.40",
                "target_classes:",
                "  - person",
                "source_type: webcam",
                "camera_id: 0",
                "source_path: data/videos/input.mp4",
                "zones_path: assets/sample_zones.json",
                "draw_zones: false",
                "save_output: false",
                "output_path: data/outputs/result.mp4",
                "display: true",
                "device: auto",
                "imgsz: 640",
                "tracker_config: bytetrack.yaml",
                "posture:",
                "  enabled: true",
                "  model_path: models/custom-pose.pt",
                "  model_type: yolo_pose",
                "  person_class_name: person",
                "  required_state: standing",
                "  violation_state: sitting",
                "  min_violation_seconds: 45.0",
                "  clear_after_seconds: 3.0",
                "  unknown_grace_seconds: 4.0",
                "  max_missing_track_seconds: 5.0",
                "  keypoints:",
                "    min_confidence: 0.50",
                "    min_required_points: 8",
                "  zones:",
                "    ignore_outside_zones: true",
                "    posture_required_zone_ids:",
                "      - zone_1",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(["--config", str(config_file)])

    assert config.posture.enabled is True
    assert config.posture.model_path == (Path.cwd() / "models/custom-pose.pt").resolve()
    assert config.posture.model_type == "yolo_pose"
    assert config.posture.person_class_name == "person"
    assert config.posture.required_state == "standing"
    assert config.posture.violation_state == "sitting"
    assert config.posture.min_violation_seconds == 45.0
    assert config.posture.clear_after_seconds == 3.0
    assert config.posture.unknown_grace_seconds == 4.0
    assert config.posture.max_missing_track_seconds == 5.0
    assert config.posture.keypoints.min_confidence == 0.50
    assert config.posture.keypoints.min_required_points == 8
    assert config.posture.zones.ignore_outside_zones is True
    assert config.posture.zones.posture_required_zone_ids == ["zone_1"]
    assert config.posture.events.enabled is True
    assert config.posture.events.log_events is False
    assert config.posture.events.event_log_path == (
        Path.cwd() / "data/outputs/posture_events.jsonl"
    ).resolve()


def test_load_config_rejects_negative_posture_duration(tmp_path: Path) -> None:
    config_file = tmp_path / "app.yaml"
    config_file.write_text(
        "\n".join(
            [
                "model_path: models/best.pt",
                "conf: 0.25",
                "iou: 0.40",
                "target_classes:",
                "  - person",
                "source_type: webcam",
                "camera_id: 0",
                "source_path: data/videos/input.mp4",
                "zones_path: assets/sample_zones.json",
                "draw_zones: false",
                "save_output: false",
                "output_path: data/outputs/result.mp4",
                "display: true",
                "device: auto",
                "imgsz: 640",
                "tracker_config: bytetrack.yaml",
                "posture:",
                "  min_violation_seconds: -1.0",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="posture.min_violation_seconds"):
        load_config(["--config", str(config_file)])
