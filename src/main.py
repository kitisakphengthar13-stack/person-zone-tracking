from __future__ import annotations

import sys

import cv2

from config import AppConfig, load_config, print_resolved_config
from detector import Detector
from dwell_time import DwellTimeTracker
from utils import ensure_parent_dir, log_error, log_info
from video_source import VideoSource
from visualizer import Visualizer
from zone_editor import ZoneEditor
from zone_manager import ZoneManager


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
        visualizer = Visualizer()

        writer = _create_writer(config, source)
        _run_loop(config, source, zone_manager, detector, dwell_tracker, visualizer, writer)
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
        zone_matches = zone_manager.match_detections(detections)
        dwell_tracker.update(zone_matches, packet.timestamp)

        output_frame = visualizer.draw(
            frame=packet.frame,
            zones=zone_manager.zones,
            detections=detections,
            zone_matches=zone_matches,
            dwell_tracker=dwell_tracker,
        )

        if writer is not None:
            writer.write(output_frame)

        if config.display:
            cv2.imshow("person-zone-tracking", output_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    log_info("Processing complete.")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
