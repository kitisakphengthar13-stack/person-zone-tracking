from __future__ import annotations

import sys
from dataclasses import dataclass

import cv2

from compliance import PersonPPEStatus, evaluate_all_compliance
from config import AppConfig, load_config, print_resolved_config
from detector import Detector
from dwell_time import DwellTimeTracker
from event_logger import append_violation_events_jsonl
from events import ViolationEvent
from ppe_matcher import (
    PPEMatchConfig,
    match_ppe_to_persons,
    split_person_and_ppe_detections,
)
from utils import ensure_parent_dir, log_error, log_info
from violation_engine import ViolationStateTracker
from video_source import VideoSource
from visualizer import Visualizer
from zone_editor import ZoneEditor
from zone_manager import ZoneManager, ZoneMatch


@dataclass(frozen=True)
class FrameAnalytics:
    detections_for_display: list[object]
    zone_matches: list[ZoneMatch]
    ppe_statuses: list[PersonPPEStatus]
    violation_events: list[ViolationEvent]


def main(argv: list[str] | None = None) -> int:
    source: VideoSource | None = None
    writer: cv2.VideoWriter | None = None

    try:
        config = load_config(argv)
        print_resolved_config(config)

        source = _open_source(config)
        _maybe_draw_zones(config, source)

        if source is not None:
            source.release()
        source = _open_source(config)

        zone_manager = ZoneManager.from_file(config.zones_path, config.target_classes)
        detector = Detector(config)
        dwell_tracker = DwellTimeTracker()
        violation_tracker = _create_violation_tracker(config)
        visualizer = Visualizer()

        writer = _create_writer(config, source)
        _run_loop(
            config,
            source,
            zone_manager,
            detector,
            dwell_tracker,
            violation_tracker,
            visualizer,
            writer,
        )
        return 0
    except KeyboardInterrupt:
        log_info("Stopped by user.")
        return 130
    except Exception as exc:
        log_error(str(exc))
        return 1
    finally:
        if writer is not None:
            writer.release()
        if source is not None:
            source.release()
        if "config" in locals() and config.display:
            cv2.destroyAllWindows()


def _open_source(config: AppConfig) -> VideoSource:
    source = VideoSource(config)
    source.open()

    if config.source_type == "video":
        log_info(f"Opened video source: {config.source_path}")
    else:
        log_info(f"Opened webcam source: camera {config.camera_id}")
    log_info(f"Source size: {source.width}x{source.height}, fps: {source.fps:.2f}")
    return source


def _maybe_draw_zones(config: AppConfig, source: VideoSource) -> None:
    should_draw = config.draw_zones or not config.zones_path.exists()
    if not should_draw:
        return

    if not config.display:
        raise RuntimeError(
            "Zone drawing requires display=true. Provide an existing zones file "
            "or run with --display true --draw-zones true."
        )

    if not config.zones_path.exists():
        log_info(f"Zones file not found: {config.zones_path}")
    log_info("Opening zone editor.")

    packet = source.read()
    if packet is None:
        raise RuntimeError("Unable to read a frame for zone drawing.")

    editor = ZoneEditor(
        frame=packet.frame,
        zones_path=config.zones_path,
        target_classes=config.target_classes,
    )
    saved = editor.run()
    if not saved and not config.zones_path.exists():
        raise RuntimeError(
            f"No zones were saved to {config.zones_path}. Press S in the zone editor "
            "to save zones before running analytics."
        )


def _create_writer(config: AppConfig, source: VideoSource) -> cv2.VideoWriter | None:
    if not config.save_output:
        return None

    ensure_parent_dir(config.output_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(config.output_path),
        fourcc,
        source.fps,
        (source.width, source.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create output video: {config.output_path}")

    log_info(f"Saving output video to: {config.output_path}")
    return writer


def _run_loop(
    config: AppConfig,
    source: VideoSource,
    zone_manager: ZoneManager,
    detector: Detector,
    dwell_tracker: DwellTimeTracker,
    violation_tracker: ViolationStateTracker | None,
    visualizer: Visualizer,
    writer: cv2.VideoWriter | None,
) -> None:
    log_info(f"Confidence threshold: {config.conf}")
    log_info(f"IoU threshold: {config.iou}")
    log_info(f"Target classes: {', '.join(config.target_classes)}")
    log_info(f"Zones path: {config.zones_path}")

    if config.display:
        cv2.namedWindow("person-zone-tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("person-zone-tracking", source.width, source.height)

    while True:
        packet = source.read()
        if packet is None:
            break

        detections = detector.detect(packet.frame)
        analytics = _process_frame_analytics(
            config=config,
            detections=detections,
            zone_manager=zone_manager,
            dwell_tracker=dwell_tracker,
            violation_tracker=violation_tracker,
            timestamp=packet.timestamp,
        )
        _maybe_log_violation_events(config, analytics.violation_events, packet.timestamp)

        output_frame = visualizer.draw(
            frame=packet.frame,
            zones=zone_manager.zones,
            detections=analytics.detections_for_display,
            zone_matches=analytics.zone_matches,
            dwell_tracker=dwell_tracker,
            ppe_statuses=analytics.ppe_statuses,
            violation_events=analytics.violation_events,
        )

        if writer is not None:
            writer.write(output_frame)

        if config.display:
            cv2.imshow("person-zone-tracking", output_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    log_info("Processing complete.")


def _process_frame_analytics(
    config: AppConfig,
    detections: list[object],
    zone_manager: ZoneManager,
    dwell_tracker: DwellTimeTracker,
    violation_tracker: ViolationStateTracker | None,
    timestamp: float,
) -> FrameAnalytics:
    if not config.ppe.enabled:
        zone_matches = zone_manager.match_detections(detections)
        dwell_tracker.update(zone_matches, timestamp)
        return FrameAnalytics(
            detections_for_display=detections,
            zone_matches=zone_matches,
            ppe_statuses=[],
            violation_events=[],
        )

    person_detections, ppe_detections = split_person_and_ppe_detections(
        detections=detections,
        person_classes=config.ppe.person_classes,
        ppe_classes=config.ppe.ppe_classes,
    )
    ppe_matches = match_ppe_to_persons(
        persons=person_detections,
        ppe_items=ppe_detections,
        config=_ppe_match_config(config),
    )
    ppe_statuses = evaluate_all_compliance(
        person_detections=person_detections,
        matched_items_by_track=ppe_matches,
        required_items=config.ppe.required_items,
    )
    zone_matches = zone_manager.match_detections(person_detections)
    violation_events = (
        violation_tracker.update(zone_matches, ppe_statuses, timestamp)
        if violation_tracker is not None
        else []
    )
    dwell_tracker.update(zone_matches, timestamp)

    return FrameAnalytics(
        detections_for_display=detections,
        zone_matches=zone_matches,
        ppe_statuses=ppe_statuses,
        violation_events=violation_events,
    )


def _ppe_match_config(config: AppConfig) -> PPEMatchConfig:
    return PPEMatchConfig(
        center_inside_person=config.ppe.matching.center_inside_person,
        min_person_overlap_ratio=config.ppe.matching.min_person_overlap_ratio,
        min_region_overlap_ratio=config.ppe.matching.min_region_overlap_ratio,
        max_center_distance_ratio=config.ppe.matching.max_center_distance_ratio,
        ppe_regions=dict(config.ppe.matching.ppe_regions),
    )


def _create_violation_tracker(config: AppConfig) -> ViolationStateTracker | None:
    if not config.ppe.enabled or not config.violations.enabled:
        return None
    return ViolationStateTracker(
        min_violation_seconds=config.violations.min_violation_seconds,
        clear_after_seconds=config.violations.clear_after_seconds,
        max_missing_track_seconds=config.violations.max_missing_track_seconds,
        emit_unknown_ppe=config.violations.emit_unknown_ppe,
    )


def _maybe_log_violation_events(
    config: AppConfig,
    violation_events: list[ViolationEvent],
    timestamp: float,
) -> None:
    if not config.ppe.enabled:
        return
    if not config.violations.enabled or not config.violations.log_events:
        return
    append_violation_events_jsonl(
        events=violation_events,
        path=config.violations.event_log_path,
        emitted_at=timestamp,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
