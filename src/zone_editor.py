from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from utils import log_info, log_warn
from zone_manager import Zone, ZoneManager


class ZoneEditor:
    def __init__(
        self,
        frame: np.ndarray,
        zones_path: Path,
        target_classes: list[str],
        window_name: str = "Zone Editor",
    ) -> None:
        self.frame = frame.copy()
        self.zones_path = zones_path
        self.target_classes = list(target_classes)
        self.window_name = window_name
        self.zones: list[Zone] = []
        self.current_points: list[tuple[int, int]] = []
        self.saved = False

    def run(self) -> bool:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.frame.shape[1], self.frame.shape[0])
        cv2.setMouseCallback(self.window_name, self._on_mouse)

        while True:
            canvas = self._render()
            cv2.imshow(self.window_name, canvas)
            key = cv2.waitKey(20) & 0xFF

            if key in (13, 10):
                self._finish_current_zone()
            elif key == ord("n"):
                self._start_new_zone()
            elif key == ord("s"):
                self._save()
            elif key == ord("r"):
                self._reset()
            elif key in (ord("q"), 27):
                break

        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass
        return self.saved

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append((int(x), int(y)))
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._finish_current_zone()

    def _finish_current_zone(self) -> bool:
        if len(self.current_points) < 3:
            if self.current_points:
                log_warn("A valid zone needs at least 3 points.")
            return False

        zone_number = len(self.zones) + 1
        zone = Zone(
            id=f"zone_{zone_number}",
            name=f"Zone {zone_number}",
            points=list(self.current_points),
            target_classes=list(self.target_classes),
        )
        self.zones.append(zone)
        self.current_points.clear()
        log_info(f"Added {zone.name} with {len(zone.points)} point(s).")
        return True

    def _start_new_zone(self) -> None:
        if len(self.current_points) >= 3:
            self._finish_current_zone()
        elif self.current_points:
            log_warn("Discarding incomplete zone with fewer than 3 points.")
            self.current_points.clear()

    def _save(self) -> None:
        if len(self.current_points) >= 3:
            self._finish_current_zone()
        elif self.current_points:
            log_warn("Incomplete current zone was not saved.")

        if not self.zones:
            log_warn("No valid zones to save.")
            return

        ZoneManager.save_zones(self.zones, self.zones_path)
        self.saved = True

    def _reset(self) -> None:
        self.zones.clear()
        self.current_points.clear()
        self.saved = False
        log_info("Reset all zones in the editor.")

    def _render(self) -> np.ndarray:
        canvas = self.frame.copy()
        self._draw_existing_zones(canvas)
        self._draw_current_zone(canvas)
        self._draw_controls(canvas)
        return canvas

    def _draw_existing_zones(self, canvas: np.ndarray) -> None:
        for zone in self.zones:
            points = np.array(zone.points, dtype=np.int32)
            cv2.polylines(canvas, [points], isClosed=True, color=(0, 220, 0), thickness=2)
            centroid = points.mean(axis=0).astype(int)
            cv2.putText(
                canvas,
                zone.name,
                (int(centroid[0]), int(centroid[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 220, 0),
                2,
                cv2.LINE_AA,
            )

    def _draw_current_zone(self, canvas: np.ndarray) -> None:
        if not self.current_points:
            return

        points = np.array(self.current_points, dtype=np.int32)
        for point in self.current_points:
            cv2.circle(canvas, point, 5, (0, 255, 255), -1)
        if len(points) >= 2:
            cv2.polylines(canvas, [points], isClosed=False, color=(0, 255, 255), thickness=2)

    @staticmethod
    def _draw_controls(canvas: np.ndarray) -> None:
        text = "Left click: point | Right click/Enter: finish | N: new | S: save | R: reset | Q: quit"
        height, width = canvas.shape[:2]
        cv2.rectangle(canvas, (0, height - 34), (width, height), (20, 20, 20), -1)
        cv2.putText(
            canvas,
            text,
            (10, height - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
