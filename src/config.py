from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - handled at runtime with a clear message.
    yaml = None

from utils import PROJECT_ROOT, log_info, parse_bool, resolve_path


DEFAULT_CONFIG: dict[str, Any] = {
    "model_path": "models/best.pt",
    "conf": 0.30,
    "iou": 0.50,
    "target_classes": ["person"],
    "source_type": "webcam",
    "source_path": "data/videos/input.mp4",
    "camera_id": 0,
    "zones_path": "assets/sample_zones.json",
    "draw_zones": False,
    "save_output": False,
    "output_path": "data/outputs/result.mp4",
    "display": True,
    "device": "auto",
    "imgsz": 640,
    "tracker_config": "bytetrack.yaml",
    "posture": {
        "enabled": False,
        "model_path": "models/yolo26n-pose.pt",
        "model_type": "yolo_pose",
        "person_class_name": "person",
        "required_state": "standing",
        "violation_state": "sitting",
        "min_violation_seconds": 30.0,
        "clear_after_seconds": 2.0,
        "unknown_grace_seconds": 3.0,
        "max_missing_track_seconds": 2.0,
        "keypoints": {
            "min_confidence": 0.35,
            "min_required_points": 6,
        },
        "zones": {
            "ignore_outside_zones": True,
            "posture_required_zone_ids": [],
        },
        "events": {
            "enabled": True,
            "log_events": False,
            "event_log_path": "data/outputs/posture_events.jsonl",
        },
    },
}

DEFAULT_CONFIG_PATH = "configs/app.yaml"

CONFIG_ALIASES = {
    "confidence": "conf",
    "confidence_threshold": "conf",
    "iou_threshold": "iou",
    "classes": "target_classes",
    "target_classes": "target_classes",
    "video_path": "source_path",
    "display_window": "display",
    "save_output_video": "save_output",
}


@dataclass(frozen=True)
class PostureKeypointConfig:
    min_confidence: float
    min_required_points: int


@dataclass(frozen=True)
class PostureZoneConfig:
    ignore_outside_zones: bool
    posture_required_zone_ids: list[str]


@dataclass(frozen=True)
class PostureEventConfig:
    enabled: bool
    log_events: bool
    event_log_path: Path


@dataclass(frozen=True)
class PostureConfig:
    enabled: bool
    model_path: Path
    model_type: str
    person_class_name: str
    required_state: str
    violation_state: str
    min_violation_seconds: float
    clear_after_seconds: float
    unknown_grace_seconds: float
    max_missing_track_seconds: float
    keypoints: PostureKeypointConfig
    zones: PostureZoneConfig
    events: PostureEventConfig


@dataclass(frozen=True)
class AppConfig:
    config_file: Path | None
    model_path: Path
    conf: float
    iou: float
    target_classes: list[str]
    source_type: str
    source_path: Path | None
    camera_id: int
    zones_path: Path
    draw_zones: bool
    save_output: bool
    output_path: Path
    display: bool
    device: str
    imgsz: int | None
    tracker_config: str | None
    posture: PostureConfig


def load_config(argv: list[str] | None = None) -> AppConfig:
    config_arg = _parse_config_path(argv)
    config_file = resolve_path(config_arg or DEFAULT_CONFIG_PATH)
    yaml_config = _load_yaml_config(config_file, explicit=config_arg is not None)
    cli_config = _parse_cli_overrides(argv)

    merged = dict(DEFAULT_CONFIG)
    merged.update(_normalize_config_keys(yaml_config))
    merged.update(cli_config)

    app_config = _build_app_config(merged, config_file)
    validate_config(app_config)
    return app_config


def validate_config(config: AppConfig) -> None:
    if config.source_type not in {"webcam", "video"}:
        raise ValueError("source_type must be either 'webcam' or 'video'.")
    if not 0.0 <= config.conf <= 1.0:
        raise ValueError("conf must be between 0.0 and 1.0.")
    if not 0.0 <= config.iou <= 1.0:
        raise ValueError("iou must be between 0.0 and 1.0.")
    if not config.target_classes:
        raise ValueError("At least one target class must be configured.")
    if not config.model_path:
        raise ValueError("model_path is required.")
    if config.source_type == "video" and config.source_path is None:
        raise ValueError("source_path is required when source_type is 'video'.")
    if config.camera_id < 0:
        raise ValueError("camera_id must be zero or a positive integer.")
    if not config.zones_path:
        raise ValueError("zones_path is required.")
    if config.save_output and not config.output_path:
        raise ValueError("output_path is required when save_output is enabled.")
    if config.imgsz is not None and config.imgsz <= 0:
        raise ValueError("imgsz must be a positive integer when provided.")
    _validate_posture_config(config.posture)


def print_resolved_config(config: AppConfig) -> None:
    log_info("Resolved configuration:")
    data = asdict(config)
    for key, value in data.items():
        if isinstance(value, Path):
            value = str(value)
        log_info(f"  {key}: {value}")


def _parse_config_path(argv: list[str] | None) -> str | None:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--config", default=None)
    args, _ = parser.parse_known_args(argv)
    return args.config


def _parse_cli_overrides(argv: list[str] | None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Multi-zone object tracking and dwell time analytics.",
        allow_abbrev=False,
    )
    parser.add_argument("--config", default=None, help="Path to YAML config file.")
    parser.add_argument("--model-path", dest="model_path", default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--classes", dest="target_classes", nargs="+", default=None)
    parser.add_argument("--source-type", choices=["webcam", "video"], default=None)
    parser.add_argument("--source-path", dest="source_path", default=None)
    parser.add_argument("--camera-id", dest="camera_id", type=int, default=None)
    parser.add_argument("--zones-path", dest="zones_path", default=None)
    parser.add_argument(
        "--draw-zones",
        dest="draw_zones",
        nargs="?",
        const=True,
        type=_bool_arg,
        default=None,
    )
    parser.add_argument(
        "--save-output",
        dest="save_output",
        nargs="?",
        const=True,
        type=_bool_arg,
        default=None,
    )
    parser.add_argument("--output-path", dest="output_path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument(
        "--display",
        nargs="?",
        const=True,
        type=_bool_arg,
        default=None,
    )
    parser.add_argument("--tracker-config", dest="tracker_config", default=None)

    args = parser.parse_args(argv)
    return {
        key: value
        for key, value in vars(args).items()
        if key != "config" and value is not None
    }


def _bool_arg(value: Any) -> bool:
    try:
        return parse_bool(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _load_yaml_config(config_file: Path | None, explicit: bool) -> dict[str, Any]:
    if config_file is None:
        return {}
    if not config_file.exists():
        if explicit:
            raise FileNotFoundError(f"Config file not found: {config_file}")
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required to load YAML config files.")

    with config_file.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_file}")
    return loaded


def _normalize_config_keys(config: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in config.items():
        normalized_key = CONFIG_ALIASES.get(key, key)
        normalized[normalized_key] = value
    return normalized


def _build_app_config(raw: dict[str, Any], config_file: Path | None) -> AppConfig:
    target_classes = _normalize_classes(raw.get("target_classes"))

    source_path = resolve_path(raw.get("source_path"))
    tracker_config = raw.get("tracker_config")
    if tracker_config is not None:
        tracker_config = str(tracker_config).strip() or None

    return AppConfig(
        config_file=config_file,
        model_path=_required_path(raw.get("model_path"), "model_path"),
        conf=float(raw.get("conf")),
        iou=float(raw.get("iou")),
        target_classes=target_classes,
        source_type=str(raw.get("source_type")).strip().lower(),
        source_path=source_path,
        camera_id=int(raw.get("camera_id")),
        zones_path=_required_path(raw.get("zones_path"), "zones_path"),
        draw_zones=parse_bool(raw.get("draw_zones")),
        save_output=parse_bool(raw.get("save_output")),
        output_path=_required_path(raw.get("output_path"), "output_path"),
        display=parse_bool(raw.get("display")),
        device=str(raw.get("device", "auto")).strip().lower(),
        imgsz=None if raw.get("imgsz") in (None, "") else int(raw.get("imgsz")),
        tracker_config=tracker_config,
        posture=_build_posture_config(raw.get("posture")),
    )


def _required_path(value: Any, field_name: str) -> Path:
    path = resolve_path(value)
    if path is None:
        raise ValueError(f"{field_name} is required.")
    return path


def _normalize_classes(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_classes = value.replace(",", " ").split()
    elif isinstance(value, (list, tuple, set)):
        raw_classes = []
        for item in value:
            if isinstance(item, str) and "," in item:
                raw_classes.extend(part.strip() for part in item.split(","))
            else:
                raw_classes.append(str(item).strip())
    else:
        raw_classes = [str(value).strip()]

    classes: list[str] = []
    seen: set[str] = set()
    for class_name in raw_classes:
        cleaned = class_name.strip()
        key = cleaned.lower()
        if cleaned and key not in seen:
            classes.append(cleaned)
            seen.add(key)
    return classes


def _build_posture_config(raw: Any) -> PostureConfig:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("posture config must be a YAML mapping.")

    default_posture = DEFAULT_CONFIG["posture"]
    merged = dict(default_posture)
    merged.update(raw)

    raw_keypoints = merged.get("keypoints") or {}
    if not isinstance(raw_keypoints, dict):
        raise ValueError("posture.keypoints config must be a YAML mapping.")
    keypoints = dict(default_posture["keypoints"])
    keypoints.update(raw_keypoints)

    raw_zones = merged.get("zones") or {}
    if not isinstance(raw_zones, dict):
        raise ValueError("posture.zones config must be a YAML mapping.")
    zones = dict(default_posture["zones"])
    zones.update(raw_zones)

    raw_events = merged.get("events") or {}
    if not isinstance(raw_events, dict):
        raise ValueError("posture.events config must be a YAML mapping.")
    events = dict(default_posture["events"])
    events.update(raw_events)

    return PostureConfig(
        enabled=parse_bool(merged.get("enabled")),
        model_path=_required_path(merged.get("model_path"), "posture.model_path"),
        model_type=str(merged.get("model_type")).strip(),
        person_class_name=str(merged.get("person_class_name")).strip(),
        required_state=str(merged.get("required_state")).strip().lower(),
        violation_state=str(merged.get("violation_state")).strip().lower(),
        min_violation_seconds=float(merged.get("min_violation_seconds")),
        clear_after_seconds=float(merged.get("clear_after_seconds")),
        unknown_grace_seconds=float(merged.get("unknown_grace_seconds")),
        max_missing_track_seconds=float(merged.get("max_missing_track_seconds")),
        keypoints=PostureKeypointConfig(
            min_confidence=float(keypoints.get("min_confidence")),
            min_required_points=int(keypoints.get("min_required_points")),
        ),
        zones=PostureZoneConfig(
            ignore_outside_zones=parse_bool(zones.get("ignore_outside_zones")),
            posture_required_zone_ids=[
                str(zone_id).strip()
                for zone_id in (zones.get("posture_required_zone_ids") or [])
                if str(zone_id).strip()
            ],
        ),
        events=PostureEventConfig(
            enabled=parse_bool(events.get("enabled")),
            log_events=parse_bool(events.get("log_events")),
            event_log_path=_required_path(
                events.get("event_log_path"),
                "posture.events.event_log_path",
            ),
        ),
    )


def _validate_posture_config(config: PostureConfig) -> None:
    if not config.model_path:
        raise ValueError("posture.model_path is required.")
    if not config.model_type:
        raise ValueError("posture.model_type is required.")
    if not config.person_class_name:
        raise ValueError("posture.person_class_name is required.")

    valid_states = {"standing", "sitting", "unknown"}
    if config.required_state not in valid_states:
        raise ValueError("posture.required_state must be standing, sitting, or unknown.")
    if config.violation_state not in valid_states:
        raise ValueError("posture.violation_state must be standing, sitting, or unknown.")

    duration_fields = {
        "posture.min_violation_seconds": config.min_violation_seconds,
        "posture.clear_after_seconds": config.clear_after_seconds,
        "posture.unknown_grace_seconds": config.unknown_grace_seconds,
        "posture.max_missing_track_seconds": config.max_missing_track_seconds,
    }
    for field_name, value in duration_fields.items():
        if value < 0.0:
            raise ValueError(f"{field_name} must be zero or greater.")

    if not 0.0 <= config.keypoints.min_confidence <= 1.0:
        raise ValueError("posture.keypoints.min_confidence must be between 0.0 and 1.0.")
    if config.keypoints.min_required_points < 0:
        raise ValueError("posture.keypoints.min_required_points must be zero or greater.")
