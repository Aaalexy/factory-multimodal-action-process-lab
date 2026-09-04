"""Body-guided hand crops and coordinate transforms.

The body model remains COCO-17.  Its anatomical left/right wrist, elbow, and
shoulder indices are used only to locate a high-resolution crop.  No hand
landmarks are invented here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Sequence

import numpy as np


COCO_HAND_GUIDE_INDICES: dict[str, tuple[int, int, int]] = {
    "left": (9, 7, 5),
    "right": (10, 8, 6),
}
_UNUSABLE_BODY_STATES = {"missing", "lost", "unavailable"}


@dataclass(frozen=True)
class CropTransform:
    """Map normalized crop coordinates back to source-frame pixels."""

    x_offset: int
    y_offset: int
    x_scale: int
    y_scale: int
    source_width: int
    source_height: int

    @property
    def bbox(self) -> list[int]:
        return [
            self.x_offset,
            self.y_offset,
            self.x_offset + self.x_scale,
            self.y_offset + self.y_scale,
        ]

    def normalized_to_source(
        self,
        normalized_x: float,
        normalized_y: float,
        *,
        clip: bool = True,
    ) -> tuple[float, float]:
        x = self.x_offset + float(normalized_x) * self.x_scale
        y = self.y_offset + float(normalized_y) * self.y_scale
        if clip:
            x = min(max(x, 0.0), float(max(0, self.source_width - 1)))
            y = min(max(y, 0.0), float(max(0, self.source_height - 1)))
        return x, y

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "normalized_roi_to_source_pixels",
            **asdict(self),
        }


def _status_is_usable(statuses: Sequence[str] | None, index: int) -> bool:
    if statuses is None:
        return True
    if index >= len(statuses):
        return False
    return str(statuses[index]).lower() not in _UNUSABLE_BODY_STATES


def _point(
    keypoints: np.ndarray,
    statuses: Sequence[str] | None,
    index: int,
) -> np.ndarray | None:
    if index >= keypoints.shape[0] or not _status_is_usable(statuses, index):
        return None
    value = np.asarray(keypoints[index, :2], dtype=np.float64)
    if value.shape != (2,) or not np.all(np.isfinite(value)):
        return None
    return value


def build_hand_crop_transform(
    frame_shape: Sequence[int],
    body_keypoints: Sequence[Sequence[float]] | np.ndarray | None,
    body_keypoint_statuses: Sequence[str] | None,
    anatomical_side: str,
    *,
    minimum_crop_pixels: int = 96,
    forearm_scale: float = 2.4,
    wrist_extension_ratio: float = 0.18,
) -> CropTransform | None:
    """Build a square source-resolution ROI from a COCO wrist and forearm.

    The crop is centred slightly beyond the wrist in the elbow-to-wrist
    direction.  Shoulder distance is used only as a conservative scale hint.
    Missing/lost guide joints yield no crop instead of a guessed hand box.
    """

    if anatomical_side not in COCO_HAND_GUIDE_INDICES:
        raise ValueError(f"Unsupported anatomical side: {anatomical_side!r}")
    if len(frame_shape) < 2:
        raise ValueError("frame_shape must contain height and width")
    height, width = int(frame_shape[0]), int(frame_shape[1])
    if height <= 0 or width <= 0 or body_keypoints is None:
        return None

    values = np.asarray(body_keypoints, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        return None
    wrist_index, elbow_index, shoulder_index = COCO_HAND_GUIDE_INDICES[
        anatomical_side
    ]
    wrist = _point(values, body_keypoint_statuses, wrist_index)
    elbow = _point(values, body_keypoint_statuses, elbow_index)
    shoulder = _point(values, body_keypoint_statuses, shoulder_index)
    if wrist is None or elbow is None:
        return None

    forearm_vector = wrist - elbow
    forearm_length = float(np.linalg.norm(forearm_vector))
    if not math.isfinite(forearm_length) or forearm_length < 2.0:
        return None
    direction = forearm_vector / forearm_length
    centre = wrist + direction * forearm_length * wrist_extension_ratio

    scale_candidates = [
        float(minimum_crop_pixels),
        forearm_length * forearm_scale,
        min(height, width) * 0.055,
    ]
    if shoulder is not None:
        upper_arm_length = float(np.linalg.norm(elbow - shoulder))
        if math.isfinite(upper_arm_length):
            scale_candidates.append(upper_arm_length * 1.05)
    side_length = int(math.ceil(max(scale_candidates)))
    side_length = min(side_length, height, width)
    if side_length < 2:
        return None

    x_min = int(round(float(centre[0]) - side_length / 2.0))
    y_min = int(round(float(centre[1]) - side_length / 2.0))
    x_min = min(max(x_min, 0), width - side_length)
    y_min = min(max(y_min, 0), height - side_length)
    return CropTransform(
        x_offset=x_min,
        y_offset=y_min,
        x_scale=side_length,
        y_scale=side_length,
        source_width=width,
        source_height=height,
    )


def body_wrist_point(
    body_keypoints: Sequence[Sequence[float]] | np.ndarray | None,
    anatomical_side: str,
) -> tuple[float, float] | None:
    """Return the COCO anatomical wrist coordinate when finite."""

    if anatomical_side not in COCO_HAND_GUIDE_INDICES or body_keypoints is None:
        return None
    values = np.asarray(body_keypoints, dtype=np.float64)
    wrist_index = COCO_HAND_GUIDE_INDICES[anatomical_side][0]
    if values.ndim != 2 or values.shape[1] < 2 or wrist_index >= values.shape[0]:
        return None
    wrist = values[wrist_index, :2]
    if not np.all(np.isfinite(wrist)):
        return None
    return float(wrist[0]), float(wrist[1])
