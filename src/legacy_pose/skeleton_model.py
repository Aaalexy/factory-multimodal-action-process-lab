"""COCO-17 to a labelled stickman skeleton, including derived geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


COCO_KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)

STICKMAN_SEGMENTS = (
    ("head_center", "neck"),
    ("left_shoulder", "right_shoulder"),
    ("neck", "hip_center"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("left_wrist", "left_palm"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("right_wrist", "right_palm"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("left_ankle", "left_foot"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("right_ankle", "right_foot"),
)


@dataclass(frozen=True)
class SkeletonPoint:
    x: float
    y: float
    confidence: float
    status: str
    sources: tuple[str, ...] = ()

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)


@dataclass
class SkeletonPose:
    points: dict[str, SkeletonPoint]
    segments: tuple[tuple[str, str], ...] = STICKMAN_SEGMENTS
    head_radius: float = 0.0


def _derived(
    xy: np.ndarray, confidence: float, sources: tuple[str, ...]
) -> SkeletonPoint:
    return SkeletonPoint(
        x=float(xy[0]),
        y=float(xy[1]),
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        status="derived",
        sources=sources,
    )


def _usable(point: SkeletonPoint) -> bool:
    return point.status not in ("missing", "uncertain", "rejected") and np.isfinite([point.x, point.y]).all()


def build_skeleton(
    keypoints: np.ndarray,
    statuses: np.ndarray | list[str] | None = None,
    confidence_threshold: float = 0.25,
) -> SkeletonPose:
    """Create detected/interpolated points and explicitly derived body parts."""

    array = np.asarray(keypoints, dtype=np.float32).reshape(17, 3)
    if statuses is None:
        state_array = np.where(array[:, 2] >= confidence_threshold, "detected", "missing")
    else:
        state_array = np.asarray(statuses, dtype="<U12").reshape(17)
    points: dict[str, SkeletonPoint] = {}
    for index, name in enumerate(COCO_KEYPOINT_NAMES):
        state = str(state_array[index])
        finite = np.isfinite(array[index, :2]).all()
        if not finite:
            state = "missing"
        points[name] = SkeletonPoint(
            float(array[index, 0]) if finite else float("nan"),
            float(array[index, 1]) if finite else float("nan"),
            float(np.clip(array[index, 2], 0.0, 1.0)) if np.isfinite(array[index, 2]) else 0.0,
            state,
        )

    left_shoulder, right_shoulder = points["left_shoulder"], points["right_shoulder"]
    left_hip, right_hip = points["left_hip"], points["right_hip"]
    shoulder_width = 0.0
    if _usable(left_shoulder) and _usable(right_shoulder):
        shoulder_width = float(np.linalg.norm(left_shoulder.xy - right_shoulder.xy))
        neck_xy = (left_shoulder.xy + right_shoulder.xy) * 0.5
        points["neck"] = _derived(
            neck_xy,
            min(left_shoulder.confidence, right_shoulder.confidence),
            ("left_shoulder", "right_shoulder"),
        )
    if _usable(left_hip) and _usable(right_hip):
        points["hip_center"] = _derived(
            (left_hip.xy + right_hip.xy) * 0.5,
            min(left_hip.confidence, right_hip.confidence),
            ("left_hip", "right_hip"),
        )

    face_names = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")
    visible_face = [points[name] for name in face_names if _usable(points[name])]
    head_radius = max(4.0, shoulder_width * 0.18) if shoulder_width > 0 else 8.0
    if visible_face:
        face_xy = np.stack([point.xy for point in visible_face])
        head_center = np.mean(face_xy, axis=0)
        if len(visible_face) >= 2:
            spread = float(np.max(np.linalg.norm(face_xy - head_center, axis=1)))
            head_radius = max(head_radius, spread * 1.35)
        points["head_center"] = _derived(
            head_center,
            float(np.mean([point.confidence for point in visible_face])),
            tuple(name for name in face_names if _usable(points[name])),
        )
    elif "neck" in points:
        points["head_center"] = _derived(
            points["neck"].xy + np.array([0.0, -max(8.0, shoulder_width * 0.38)]),
            points["neck"].confidence * 0.6,
            ("neck", "shoulder_width"),
        )

    scale = shoulder_width if shoulder_width > 0 else 50.0
    for side in ("left", "right"):
        elbow = points[f"{side}_elbow"]
        wrist = points[f"{side}_wrist"]
        if _usable(elbow) and _usable(wrist):
            direction = wrist.xy - elbow.xy
            length = float(np.linalg.norm(direction))
            if length > 1e-6:
                palm_xy = wrist.xy + direction / length * min(scale * 0.16, length * 0.28)
                points[f"{side}_palm"] = _derived(
                    palm_xy,
                    min(elbow.confidence, wrist.confidence) * 0.8,
                    (f"{side}_elbow", f"{side}_wrist"),
                )
        knee = points[f"{side}_knee"]
        ankle = points[f"{side}_ankle"]
        if _usable(knee) and _usable(ankle):
            shin = ankle.xy - knee.xy
            shin_length = float(np.linalg.norm(shin))
            if shin_length > 1e-6:
                # A modest horizontal bias makes the derived foot legible while
                # retaining the downward component of the lower-leg direction.
                lateral_sign = -1.0 if side == "left" else 1.0
                direction = shin / shin_length + np.array([0.45 * lateral_sign, 0.05])
                direction /= max(float(np.linalg.norm(direction)), 1e-6)
                foot_xy = ankle.xy + direction * min(scale * 0.24, shin_length * 0.30)
                points[f"{side}_foot"] = _derived(
                    foot_xy,
                    min(knee.confidence, ankle.confidence) * 0.8,
                    (f"{side}_knee", f"{side}_ankle"),
                )

    available_segments = tuple(
        (start, end)
        for start, end in STICKMAN_SEGMENTS
        if start in points and end in points and _usable(points[start]) and _usable(points[end])
    )
    return SkeletonPose(points=points, segments=available_segments, head_radius=head_radius)


class SkeletonModel:
    """Small state-free wrapper convenient for application dependency injection."""

    def __init__(self, confidence_threshold: float = 0.25) -> None:
        self.confidence_threshold = confidence_threshold

    def build(
        self, keypoints: np.ndarray, statuses: np.ndarray | list[str] | None = None
    ) -> SkeletonPose:
        return build_skeleton(keypoints, statuses, self.confidence_threshold)
