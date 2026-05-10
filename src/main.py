from __future__ import annotations

import sys
from dataclasses import dataclass

import cv2

from config import AppConfig, load_config, print_resolved_config
from detector import Detector
from dwell_time import DwellTimeTracker
from pose_detector import YOLOPoseDetector
from pose_types import PostureStatus
from posture_classifier import classify_posture
from posture_event_logger import append_posture_events_jsonl
from posture_events import PostureViolationEvent
from posture_violation_engine import PostureViolationTracker
from utils import ensure_parent_dir, log_error, log_info
from video_source import VideoSource
from visualizer import Visualizer
from zone_editor import ZoneEditor
from zone_manager import ZoneManager, ZoneMatch


@dataclass(frozen=True)
class FrameAnalytics:
    detections_for_display: list[object]
    zone_matches: list[ZoneMatch]
    posture_statuses: list[PostureStatus]
    posture_events: list[PostureViolationEvent]


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
        detector = None if config.posture.enabled else Detector(config)
        pose_detector = YOLOPoseDetector(config) if config.posture.enabled else None
        posture_violation_tracker = (
            _create_posture_violation_tracker(config)
            if config.posture.enabled
            else None
        )
        dwell_tracker = DwellTimeTracker()
        visualizer = Visualizer()

        writer = _create_writer(config, source)
        _run_loop(
            config,
            source,
            zone_manager,
            detector,
            pose_detector,
            posture_violation_tracker,
            dwell_tracker,
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
    detector: Detector | None,
    pose_detector: YOLOPoseDetector | None,
    posture_violation_tracker: PostureViolationTracker | None,
    dwell_tracker: DwellTimeTracker,
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

        analytics = _process_frame_analytics(
            config=config,
            frame=packet.frame,
            detector=detector,
            pose_detector=pose_detector,
            zone_manager=zone_manager,
            dwell_tracker=dwell_tracker,
            posture_violation_tracker=posture_violation_tracker,
            timestamp=packet.timestamp,
        )
        _maybe_log_posture_events(config, analytics.posture_events, packet.timestamp)

        output_frame = visualizer.draw(
            frame=packet.frame,
            zones=zone_manager.zones,
            detections=analytics.detections_for_display,
            zone_matches=analytics.zone_matches,
            dwell_tracker=dwell_tracker,
            posture_statuses=analytics.posture_statuses,
            posture_events=analytics.posture_events,
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
    frame: object,
    detector: object | None,
    pose_detector: object | None,
    zone_manager: ZoneManager,
    dwell_tracker: DwellTimeTracker,
    posture_violation_tracker: PostureViolationTracker | None,
    timestamp: float,
) -> FrameAnalytics:
    if not config.posture.enabled:
        if detector is None:
            raise RuntimeError("Object detector is required when posture is disabled.")
        detections = detector.detect(frame)
        zone_matches = zone_manager.match_detections(detections)
        dwell_tracker.update(zone_matches, timestamp)
        return FrameAnalytics(
            detections_for_display=detections,
            zone_matches=zone_matches,
            posture_statuses=[],
            posture_events=[],
        )

    if pose_detector is None:
        raise RuntimeError("Pose detector is required when posture is enabled.")
    pose_people = pose_detector.detect(frame)
    posture_statuses = [classify_posture(person) for person in pose_people]
    zone_matches = zone_manager.match_detections(pose_people)
    posture_events = (
        posture_violation_tracker.update(zone_matches, posture_statuses, timestamp)
        if posture_violation_tracker is not None
        else []
    )
    dwell_tracker.update(zone_matches, timestamp)

    return FrameAnalytics(
        detections_for_display=pose_people,
        zone_matches=zone_matches,
        posture_statuses=posture_statuses,
        posture_events=posture_events,
    )


def _create_posture_violation_tracker(config: AppConfig) -> PostureViolationTracker:
    return PostureViolationTracker(
        min_violation_seconds=config.posture.min_violation_seconds,
        clear_after_seconds=config.posture.clear_after_seconds,
        unknown_grace_seconds=config.posture.unknown_grace_seconds,
        max_missing_track_seconds=config.posture.max_missing_track_seconds,
        ignore_outside_zones=config.posture.zones.ignore_outside_zones,
    )


def _maybe_log_posture_events(
    config: AppConfig,
    posture_events: list[PostureViolationEvent],
    timestamp: float,
) -> None:
    if not config.posture.enabled:
        return
    if not config.posture.events.enabled or not config.posture.events.log_events:
        return
    append_posture_events_jsonl(
        events=posture_events,
        path=config.posture.events.event_log_path,
        emitted_at=timestamp,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
