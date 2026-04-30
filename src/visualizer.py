from __future__ import annotations

from collections import defaultdict
from typing import Any

import cv2
import numpy as np

from dwell_time import DwellTimeTracker
from utils import format_seconds
from zone_manager import Zone, ZoneMatch


COLOR_PALETTE = [
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
]


class Visualizer:
    def __init__(self, max_summary_lines_per_zone: int = 7) -> None:
        self.max_summary_lines_per_zone = max_summary_lines_per_zone

    def draw(
        self,
        frame: np.ndarray,
        zones: list[Zone],
        detections: list[Any],
        zone_matches: list[ZoneMatch],
        dwell_tracker: DwellTimeTracker,
    ) -> np.ndarray:
        output = frame.copy()

        self._draw_zones(output, zones)
        matches_by_detection = self._group_matches_by_detection(zone_matches)

        for detection in detections:
            self._draw_detection(
                output,
                detection,
                matches_by_detection.get(id(detection), []),
                dwell_tracker,
            )

        self._draw_zone_summaries(output, zones, zone_matches, dwell_tracker)
        return output

    @staticmethod
    def _ui(frame: np.ndarray) -> dict[str, int | float]:
        """
        Compute overlay sizes from the actual video frame resolution.

        This does not resize the frame.
        It only scales UI elements such as text, boxes, panels, and lines
        according to the received video size.
        """
        height, width = frame.shape[:2]
        base = min(width, height)

        return {
            "font_scale": max(0.45, base * 0.00075),
            "font_thickness": max(1, int(round(base * 0.0015))),
            "box_thickness": max(2, int(round(base * 0.0025))),
            "zone_thickness": max(2, int(round(base * 0.0025))),
            "point_radius": max(4, int(round(base * 0.006))),
            "margin_x": max(10, int(round(width * 0.008))),
            "margin_y": max(10, int(round(height * 0.012))),
            "label_pad_x": max(4, int(round(width * 0.004))),
            "label_pad_y": max(4, int(round(height * 0.006))),
            "line_height": max(20, int(round(height * 0.025))),
            "panel_width": max(260, int(round(width * 0.28))),
        }

    def _draw_zones(self, frame: np.ndarray, zones: list[Zone]) -> None:
        ui = self._ui(frame)

        for index, zone in enumerate(zones):
            color = COLOR_PALETTE[index % len(COLOR_PALETTE)]
            points = np.array(zone.points, dtype=np.int32)

            cv2.polylines(
                frame,
                [points],
                isClosed=True,
                color=color,
                thickness=int(ui["zone_thickness"]),
            )

            for x, y in zone.points:
                cv2.circle(
                    frame,
                    (int(x), int(y)),
                    int(ui["point_radius"]),
                    color,
                    -1,
                )

            centroid = points.mean(axis=0).astype(int)
            self._draw_label(
                frame,
                zone.name,
                int(centroid[0]),
                int(centroid[1]),
                color,
            )

    def _draw_detection(
        self,
        frame: np.ndarray,
        detection: Any,
        matches: list[ZoneMatch],
        dwell_tracker: DwellTimeTracker,
    ) -> None:
        ui = self._ui(frame)

        x1, y1, x2, y2 = [int(round(value)) for value in detection.bbox]
        color = _color_for_text(detection.class_name)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            int(ui["box_thickness"]),
        )

        track_text = "?" if detection.track_id is None else str(detection.track_id)
        label = f"{detection.class_name} #{track_text} {detection.confidence:.2f}"

        dwell_parts: list[str] = []
        if detection.track_id is not None:
            for match in matches[:2]:
                seconds = dwell_tracker.get_current_dwell(
                    match.zone_id,
                    detection.class_name,
                    int(detection.track_id),
                )
                dwell_parts.append(f"{match.zone_id} {format_seconds(seconds)}")

        if dwell_parts:
            if len(matches) > 2:
                dwell_parts.append(f"+{len(matches) - 2}")
            label = f"{label} | {' | '.join(dwell_parts)}"

        label_offset = int(ui["margin_y"])
        self._draw_label(frame, label, x1, max(0, y1 - label_offset), color)

    def _draw_zone_summaries(
        self,
        frame: np.ndarray,
        zones: list[Zone],
        zone_matches: list[ZoneMatch],
        dwell_tracker: DwellTimeTracker,
    ) -> None:
        ui = self._ui(frame)
        active_entries = self._active_entries(zone_matches, dwell_tracker)

        x = int(ui["margin_x"])
        y = int(ui["margin_y"])
        panel_width = min(int(ui["panel_width"]), frame.shape[1] - x * 2)
        line_height = int(ui["line_height"])
        font_scale = float(ui["font_scale"])
        thickness = int(ui["font_thickness"])
        pad_x = int(ui["label_pad_x"]) * 2
        header_y = int(ui["margin_y"]) * 2

        for zone in zones:
            entries = active_entries.get(zone.id, [])
            lines = [zone.name]

            if entries:
                lines.extend(entries[: self.max_summary_lines_per_zone])
                if len(entries) > self.max_summary_lines_per_zone:
                    lines.append(f"+{len(entries) - self.max_summary_lines_per_zone} more")
            else:
                lines.append("No active tracks")

            panel_height = header_y + line_height * len(lines)
            self._draw_panel(frame, x, y, panel_width, panel_height)

            for index, line in enumerate(lines):
                color = (255, 255, 255) if index == 0 else (230, 230, 230)
                cv2.putText(
                    frame,
                    line,
                    (x + pad_x, y + header_y + index * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    thickness,
                    cv2.LINE_AA,
                )

            y += panel_height + int(ui["margin_y"])
            if y > frame.shape[0] - int(ui["margin_y"] * 4):
                break

    @staticmethod
    def _active_entries(
        zone_matches: list[ZoneMatch],
        dwell_tracker: DwellTimeTracker,
    ) -> dict[str, list[str]]:
        entries: dict[str, list[str]] = defaultdict(list)
        seen: set[tuple[str, str, int]] = set()

        for match in zone_matches:
            detection = match.detection
            if detection.track_id is None:
                continue

            key = (match.zone_id, detection.class_name, int(detection.track_id))
            if key in seen:
                continue
            seen.add(key)

            seconds = dwell_tracker.get_current_dwell(
                match.zone_id,
                detection.class_name,
                int(detection.track_id),
            )
            entries[match.zone_id].append(
                f"{detection.class_name} #{detection.track_id}: {format_seconds(seconds)}"
            )

        return entries

    @staticmethod
    def _group_matches_by_detection(
        zone_matches: list[ZoneMatch],
    ) -> dict[int, list[ZoneMatch]]:
        grouped: dict[int, list[ZoneMatch]] = defaultdict(list)

        for match in zone_matches:
            grouped[id(match.detection)].append(match)

        return grouped

    @staticmethod
    def _draw_panel(frame: np.ndarray, x: int, y: int, width: int, height: int) -> None:
        base = min(frame.shape[1], frame.shape[0])
        thickness = max(1, int(round(base * 0.0015)))

        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (80, 80, 80), thickness)

    @staticmethod
    def _draw_label(
        frame: np.ndarray,
        text: str,
        x: int,
        y: int,
        color: tuple[int, int, int],
    ) -> None:
        height, width = frame.shape[:2]
        base = min(width, height)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.45, base * 0.00075)
        thickness = max(1, int(round(base * 0.0015)))
        pad_x = max(4, int(round(width * 0.004)))
        pad_y = max(4, int(round(height * 0.006)))

        text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
        text_width, text_height = text_size

        x = max(0, min(x, width - text_width - pad_x * 2))
        y = max(text_height + pad_y * 2, min(y, height - baseline - pad_y * 2))

        cv2.rectangle(
            frame,
            (x, y - text_height - pad_y * 2),
            (x + text_width + pad_x * 2, y + baseline + pad_y * 2),
            color,
            -1,
        )

        cv2.putText(
            frame,
            text,
            (x + pad_x, y - pad_y // 2),
            font,
            scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )


def _color_for_text(text: str) -> tuple[int, int, int]:
    index = sum(ord(char) for char in text) % len(COLOR_PALETTE)
    return COLOR_PALETTE[index]