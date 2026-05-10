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
    "ppe": {
        "enabled": False,
        "person_classes": ["Person"],
        "ppe_classes": [
            "Hardhat",
            "Safety Vest",
            "Mask",
            "NO-Hardhat",
            "NO-Safety Vest",
            "NO-Mask",
        ],
        "required_items": [],
        "matching": {
            "center_inside_person": True,
            "min_person_overlap_ratio": 0.02,
            "min_region_overlap_ratio": 0.05,
            "max_center_distance_ratio": 0.60,
            "ppe_regions": {},
        },
    },
    "violations": {
        "enabled": True,
        "log_events": False,
        "event_log_path": "data/outputs/violations.jsonl",
        "min_violation_seconds": 2.0,
        "clear_after_seconds": 1.0,
        "max_missing_track_seconds": 1.5,
        "emit_unknown_ppe": True,
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
class PPEMatchingConfig:
    center_inside_person: bool
    min_person_overlap_ratio: float
    min_region_overlap_ratio: float
    max_center_distance_ratio: float
    ppe_regions: dict[str, str]


@dataclass(frozen=True)
class PPEConfig:
    enabled: bool
    person_classes: list[str]
    ppe_classes: list[str]
    required_items: list[str]
    matching: PPEMatchingConfig


@dataclass(frozen=True)
class ViolationsConfig:
    enabled: bool
    log_events: bool
    event_log_path: Path
    min_violation_seconds: float
    clear_after_seconds: float
    max_missing_track_seconds: float
    emit_unknown_ppe: bool


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
    ppe: PPEConfig
    violations: ViolationsConfig


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
    _validate_ppe_config(config.ppe)
    _validate_ppe_target_classes(config)
    _validate_violations_config(config.violations)


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
        ppe=_build_ppe_config(raw.get("ppe")),
        violations=_build_violations_config(raw.get("violations")),
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


def _build_ppe_config(raw: Any) -> PPEConfig:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("ppe config must be a YAML mapping.")

    default_ppe = DEFAULT_CONFIG["ppe"]
    merged = dict(default_ppe)
    merged.update(raw)

    raw_matching = merged.get("matching") or {}
    if not isinstance(raw_matching, dict):
        raise ValueError("ppe.matching config must be a YAML mapping.")

    default_matching = default_ppe["matching"]
    matching = dict(default_matching)
    matching.update(raw_matching)

    raw_regions = matching.get("ppe_regions") or {}
    if not isinstance(raw_regions, dict):
        raise ValueError("ppe.matching.ppe_regions must be a YAML mapping.")

    return PPEConfig(
        enabled=parse_bool(merged.get("enabled")),
        person_classes=_normalize_classes(merged.get("person_classes")),
        ppe_classes=_normalize_classes(merged.get("ppe_classes")),
        required_items=_normalize_classes(merged.get("required_items")),
        matching=PPEMatchingConfig(
            center_inside_person=parse_bool(matching.get("center_inside_person")),
            min_person_overlap_ratio=float(matching.get("min_person_overlap_ratio")),
            min_region_overlap_ratio=float(matching.get("min_region_overlap_ratio")),
            max_center_distance_ratio=float(matching.get("max_center_distance_ratio")),
            ppe_regions={
                str(class_name).strip().lower(): str(region).strip().lower()
                for class_name, region in raw_regions.items()
                if str(class_name).strip() and str(region).strip()
            },
        ),
    )


def _validate_ppe_config(config: PPEConfig) -> None:
    if not config.person_classes:
        raise ValueError("ppe.person_classes must contain at least one class.")
    for value_name, value in (
        ("ppe.matching.min_person_overlap_ratio", config.matching.min_person_overlap_ratio),
        ("ppe.matching.min_region_overlap_ratio", config.matching.min_region_overlap_ratio),
        ("ppe.matching.max_center_distance_ratio", config.matching.max_center_distance_ratio),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{value_name} must be between 0.0 and 1.0.")

    valid_regions = {"head", "torso", "lower_body", "full_body"}
    invalid_regions = sorted(
        region
        for region in config.matching.ppe_regions.values()
        if region not in valid_regions
    )
    if invalid_regions:
        raise ValueError(
            "ppe.matching.ppe_regions contains unsupported region(s): "
            + ", ".join(invalid_regions)
        )


def _validate_ppe_target_classes(config: AppConfig) -> None:
    if not config.ppe.enabled:
        return

    configured_targets = {class_name.lower() for class_name in config.target_classes}
    required_targets = config.ppe.person_classes + config.ppe.ppe_classes
    missing_targets = [
        class_name
        for class_name in required_targets
        if class_name.lower() not in configured_targets
    ]
    if missing_targets:
        raise ValueError(
            "ppe.enabled is true, but target_classes is missing PPE model class(es): "
            + ", ".join(missing_targets)
            + ". Add the PPE raw model classes to target_classes so detections are "
            "not filtered before PPE analysis."
        )


def _validate_violations_config(config: ViolationsConfig) -> None:
    for value_name, value in (
        ("violations.min_violation_seconds", config.min_violation_seconds),
        ("violations.clear_after_seconds", config.clear_after_seconds),
        ("violations.max_missing_track_seconds", config.max_missing_track_seconds),
    ):
        if value < 0.0:
            raise ValueError(f"{value_name} must be zero or greater.")


def _build_violations_config(raw: Any) -> ViolationsConfig:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("violations config must be a YAML mapping.")

    default_violations = DEFAULT_CONFIG["violations"]
    merged = dict(default_violations)
    merged.update(raw)

    return ViolationsConfig(
        enabled=parse_bool(merged.get("enabled")),
        log_events=parse_bool(merged.get("log_events")),
        event_log_path=_required_path(
            merged.get("event_log_path"),
            "violations.event_log_path",
        ),
        min_violation_seconds=float(merged.get("min_violation_seconds")),
        clear_after_seconds=float(merged.get("clear_after_seconds")),
        max_missing_track_seconds=float(merged.get("max_missing_track_seconds")),
        emit_unknown_ppe=parse_bool(merged.get("emit_unknown_ppe")),
    )
