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
    assert config.ppe.enabled is False
    assert config.ppe.person_classes == ["Person"]
    assert config.violations.enabled is True
    assert config.violations.log_events is False
    assert config.violations.event_log_path == (
        Path.cwd() / "data/outputs/violations.jsonl"
    ).resolve()
    assert config.violations.min_violation_seconds == 2.0
    assert config.violations.clear_after_seconds == 1.0
    assert config.violations.max_missing_track_seconds == 1.5
    assert config.violations.emit_unknown_ppe is True


def test_load_config_parses_ppe_settings(tmp_path: Path) -> None:
    config_file = tmp_path / "app.yaml"
    config_file.write_text(
        "\n".join(
            [
                "model_path: models/best.pt",
                "conf: 0.25",
                "iou: 0.40",
                "target_classes:",
                "  - person",
                "  - helmet",
                "  - vest",
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
                "ppe:",
                "  enabled: true",
                "  person_classes:",
                "    - person",
                "  ppe_classes:",
                "    - helmet",
                "    - vest",
                "  required_items:",
                "    - hardhat",
                "    - safety_vest",
                "  matching:",
                "    center_inside_person: true",
                "    min_person_overlap_ratio: 0.03",
                "    min_region_overlap_ratio: 0.08",
                "    max_center_distance_ratio: 0.50",
                "    ppe_regions:",
                "      helmet: head",
                "      vest: torso",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(["--config", str(config_file)])

    assert config.ppe.enabled is True
    assert config.ppe.person_classes == ["person"]
    assert config.ppe.ppe_classes == ["helmet", "vest"]
    assert config.ppe.required_items == ["hardhat", "safety_vest"]
    assert config.ppe.matching.min_person_overlap_ratio == 0.03
    assert config.ppe.matching.min_region_overlap_ratio == 0.08
    assert config.ppe.matching.max_center_distance_ratio == 0.50
    assert config.ppe.matching.ppe_regions == {"helmet": "head", "vest": "torso"}


def test_load_config_accepts_actual_ppe_model_classes_when_ppe_enabled(tmp_path: Path) -> None:
    config_file = tmp_path / "app.yaml"
    config_file.write_text(
        "\n".join(
            [
                "model_path: models/ppe.pt",
                "conf: 0.25",
                "iou: 0.40",
                "target_classes:",
                "  - Person",
                "  - Hardhat",
                "  - Safety Vest",
                "  - Mask",
                "  - NO-Hardhat",
                "  - NO-Safety Vest",
                "  - NO-Mask",
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
                "ppe:",
                "  enabled: true",
                "  required_items:",
                "    - hardhat",
                "    - safety_vest",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(["--config", str(config_file)])

    assert config.ppe.enabled is True
    assert config.target_classes == [
        "Person",
        "Hardhat",
        "Safety Vest",
        "Mask",
        "NO-Hardhat",
        "NO-Safety Vest",
        "NO-Mask",
    ]


def test_ppe_enabled_rejects_missing_target_classes(tmp_path: Path) -> None:
    config_file = tmp_path / "app.yaml"
    config_file.write_text(
        "\n".join(
            [
                "model_path: models/ppe.pt",
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
                "ppe:",
                "  enabled: true",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="target_classes is missing PPE model class"):
        load_config(["--config", str(config_file)])


def test_load_config_parses_violation_thresholds(tmp_path: Path) -> None:
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
                "violations:",
                "  enabled: true",
                "  log_events: false",
                "  event_log_path: data/outputs/violations.jsonl",
                "  min_violation_seconds: 4.5",
                "  clear_after_seconds: 2.5",
                "  max_missing_track_seconds: 3.5",
                "  emit_unknown_ppe: false",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(["--config", str(config_file)])

    assert config.violations.min_violation_seconds == 4.5
    assert config.violations.clear_after_seconds == 2.5
    assert config.violations.max_missing_track_seconds == 3.5
    assert config.violations.emit_unknown_ppe is False


def test_negative_violation_thresholds_are_rejected(tmp_path: Path) -> None:
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
                "violations:",
                "  min_violation_seconds: -1.0",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="violations.min_violation_seconds"):
        load_config(["--config", str(config_file)])
