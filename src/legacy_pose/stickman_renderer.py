"""Status-aware transparent stickman rendering."""

from __future__ import annotations

import cv2
import numpy as np

from .skeleton_model import SkeletonPoint, SkeletonPose


def hex_to_bgr(value: str) -> tuple[int, int, int]:
    text = value.strip().lstrip("#")
    if len(text) != 6:
        raise ValueError("Color must use #RRGGBB")
    red, green, blue = (int(text[index:index + 2], 16) for index in (0, 2, 4))
    return blue, green, red


def _point(point: SkeletonPoint) -> tuple[int, int]:
    return int(round(point.x)), int(round(point.y))


def _dashed_line(image: np.ndarray, start: tuple[int, int], end: tuple[int, int], color: tuple[int, ...], width: int) -> None:
    vector = np.array(end, dtype=np.float32) - np.array(start, dtype=np.float32)
    length = float(np.linalg.norm(vector))
    if length < 1:
        return
    direction = vector / length
    dash = max(5, width * 2)
    for offset in range(0, int(length), dash * 2):
        a = np.array(start) + direction * offset
        b = np.array(start) + direction * min(length, offset + dash)
        cv2.line(image, tuple(a.astype(int)), tuple(b.astype(int)), color, width, cv2.LINE_AA)


class StickmanRenderer:
    def __init__(
        self,
        color: str = "#22d3ee",
        line_width: int = 4,
        show_keypoints: bool = True,
        show_bbox: bool = True,
        keypoint_size: int = 5,
        show_confidence: bool = True,
        show_status: bool = True,
    ) -> None:
        self.detected_color = (*hex_to_bgr(color), 245)
        self.interpolated_color = (32, 190, 255, 220)
        self.predicted_color = (70, 165, 255, 145)
        self.derived_color = (220, 90, 220, 220)
        self.uncertain_color = (160, 160, 160, 150)
        self.line_width = max(1, int(line_width))
        self.show_keypoints = bool(show_keypoints)
        self.show_bbox = bool(show_bbox)
        self.keypoint_size = max(1, int(keypoint_size))
        self.show_confidence = bool(show_confidence)
        self.show_status = bool(show_status)

    def render_overlay(
        self,
        shape: tuple[int, int] | tuple[int, int, int],
        skeleton: SkeletonPose,
        bbox: np.ndarray | None = None,
        person_confidence: float = 0.0,
        track_state: str = "lost",
    ) -> np.ndarray:
        height, width = shape[:2]
        overlay = np.zeros((height, width, 4), dtype=np.uint8)
        for start_name, end_name in skeleton.segments:
            start, end = skeleton.points[start_name], skeleton.points[end_name]
            statuses = {start.status, end.status}
            if "uncertain" in statuses or "missing" in statuses or "rejected" in statuses:
                continue
            if "predicted" in statuses:
                _dashed_line(overlay, _point(start), _point(end), self.predicted_color, max(1, self.line_width - 1))
            elif "interpolated" in statuses:
                _dashed_line(overlay, _point(start), _point(end), self.interpolated_color, self.line_width)
            elif "derived" in statuses:
                _dashed_line(overlay, _point(start), _point(end), self.derived_color, max(1, self.line_width - 1))
            else:
                cv2.line(overlay, _point(start), _point(end), self.detected_color, self.line_width, cv2.LINE_AA)

        head = skeleton.points.get("head_center")
        if head is not None and np.isfinite([head.x, head.y]).all():
            cv2.circle(overlay, _point(head), max(3, int(round(skeleton.head_radius))), self.derived_color, max(1, self.line_width - 1), cv2.LINE_AA)

        if self.show_keypoints:
            for point in skeleton.points.values():
                if not np.isfinite([point.x, point.y]).all() or point.status in ("missing", "uncertain", "rejected"):
                    continue
                color = self.derived_color if point.status == "derived" else (
                    self.predicted_color if point.status == "predicted" else (
                        self.interpolated_color if point.status == "interpolated" else self.detected_color
                    )
                )
                radius = self.keypoint_size + (1 if point.status == "detected" else 0)
                thickness = -1 if point.status == "detected" else 2
                cv2.circle(overlay, _point(point), radius, color, thickness, cv2.LINE_AA)

        if self.show_bbox and bbox is not None and np.isfinite(bbox).all():
            x1, y1, x2, y2 = np.asarray(bbox, dtype=int)
            box_color = self.detected_color if track_state == "tracked" else self.uncertain_color
            cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, 1, cv2.LINE_AA)
            parts: list[str] = []
            if self.show_status:
                parts.append(f"track {track_state}")
            if self.show_confidence:
                parts.append(f"{person_confidence:.2f}")
            if parts:
                cv2.putText(overlay, " ".join(parts), (x1, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA)
        return overlay
