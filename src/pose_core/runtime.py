"""Thin adapters over the migrated YOLOv8 pose, skeleton, and renderer code."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from src.legacy_pose.pose_detector import PoseDetector
from src.legacy_pose.skeleton_model import SkeletonModel
from src.legacy_pose.stickman_renderer import StickmanRenderer


class PoseRuntime:
    """Load a real ONNX model; no mock detections or preset keypoints exist."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        confidence_threshold: float = 0.25,
        keypoint_threshold: float = 0.25,
        providers: Sequence[str] | None = None,
        provider_policy: str = "prefer_cuda",
    ) -> None:
        self.detector = PoseDetector(
            model_path,
            confidence_threshold=confidence_threshold,
            keypoint_threshold=keypoint_threshold,
            providers=providers,
            provider_policy=provider_policy,
        )
        self.model_path = str(Path(model_path).resolve())

    @property
    def providers(self) -> list[str]:
        return self.detector.providers

    @property
    def provider_status(self) -> dict[str, object]:
        return self.detector.provider_status.to_dict()

    def detect(self, frame: np.ndarray) -> list[Any]:
        return self.detector.detect(frame)

    def warmup(self, frame: np.ndarray, *, iterations: int = 1) -> list[float]:
        return self.detector.warmup(frame, iterations=iterations)


def _alpha_composite(frame: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
    foreground = overlay[:, :, :3].astype(np.float32)
    background = frame.astype(np.float32)
    return np.clip(foreground * alpha + background * (1.0 - alpha), 0, 255).astype(
        np.uint8
    )


def overlay_pose(
    frame: np.ndarray,
    track_result: Any,
    *,
    person_ref: str,
    lock_epoch: int,
    coarse_action: str,
    geometry_allowed: bool = True,
) -> np.ndarray:
    """Render only traceable pose geometry; lost retains the real video frame."""

    rendered = frame.copy()
    if (
        geometry_allowed
        and
        track_result.state == "tracked"
        and track_result.detection is not None
        and track_result.smoothed_pose is not None
    ):
        pose = track_result.smoothed_pose
        skeleton = SkeletonModel().build(pose.keypoints, pose.statuses)
        overlay = StickmanRenderer(color="#34d399", line_width=3).render_overlay(
            rendered.shape,
            skeleton,
            bbox=track_result.detection.bbox,
            person_confidence=track_result.detection.confidence,
            track_state=track_result.state,
        )
        rendered = _alpha_composite(rendered, overlay)

    state_color = (
        (76, 220, 154)
        if track_result.state == "tracked"
        else (58, 166, 255)
        if track_result.state == "uncertain"
        else (90, 90, 235)
    )
    labels = [
        f"{person_ref} / epoch {lock_epoch}",
        f"lock: {track_result.lock_state}",
        f"action: {coarse_action}",
    ]
    for index, label in enumerate(labels):
        cv2.putText(
            rendered,
            label,
            (18, 30 + index * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            state_color,
            2,
            cv2.LINE_AA,
        )
    return rendered
