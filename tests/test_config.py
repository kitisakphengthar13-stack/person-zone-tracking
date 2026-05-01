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
