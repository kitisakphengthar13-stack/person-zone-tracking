from __future__ import annotations

from typing import Any


class YOLOBuiltInTracker:
    """Small wrapper around Ultralytics YOLO tracking with persistent IDs."""

    def __init__(
        self,
        model: Any,
        conf: float,
        iou: float,
        device: str = "auto",
        imgsz: int | None = None,
        tracker_config: str | None = "bytetrack.yaml",
    ) -> None:
        self.model = model
        self.conf = conf
        self.iou = iou
        self.device = device
        self.imgsz = imgsz
        self.tracker_config = tracker_config

    def track(self, frame: object) -> list[Any]:
        kwargs: dict[str, Any] = {
            "persist": True,
            "conf": self.conf,
            "iou": self.iou,
            "verbose": False,
        }

        if self.device and self.device != "auto":
            kwargs["device"] = self.device
        if self.imgsz is not None:
            kwargs["imgsz"] = self.imgsz
        if self.tracker_config:
            kwargs["tracker"] = self.tracker_config

        return self.model.track(frame, **kwargs)
