"""Configuration and public state contracts for local USB Camera analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class CameraState(str, Enum):
    UNAVAILABLE = "unavailable"
    NO_DEVICE = "no_device"
    PERMISSION_DENIED = "permission_denied"
    BUSY = "busy"
    OPENING = "opening"
    LIVE = "live"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass(frozen=True)
class CameraConfig:
    enabled: bool = True
    device_index: int = 0
    backend: str = "auto"
    requested_width: int = 640
    requested_height: int = 480
    requested_fps: float = 15.0
    pose_display_fps: float = 12.0
    analysis_fps: float = 8.0
    open_timeout_seconds: float = 4.0
    read_timeout_seconds: float = 2.0
    stop_timeout_seconds: float = 5.0
    latest_frame_buffer_size: int = 2
    jpeg_quality: int = 82
    mirror_horizontal: bool = False
    persist_recording: bool = False
    body_model_path: str = "models/yolov8n-pose.onnx"
    body_provider_policy: str = "prefer_cuda"
    hand_model_path: str = "models/hand_pose/hand_landmarker.task"
    hand_enabled: bool = True
    candidate_token_expiry_seconds: float = 4.0
    candidate_sequence_tolerance: int = 24

    @classmethod
    def load(cls, path: str | Path) -> "CameraConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        forbidden_true = [
            name
            for name in (
                "factory_camera_validated",
                "production_action_model_ready",
                "external_factory_validated",
                "production_process_model_ready",
            )
            if payload.get(name) is not False
        ]
        if forbidden_true:
            raise ValueError(
                "Camera technical config must keep validation flags false: "
                + ", ".join(forbidden_true)
            )
        values: dict[str, Any] = {
            name: payload[name]
            for name in cls.__dataclass_fields__
            if name in payload
        }
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        if self.device_index < 0:
            raise ValueError("device_index cannot be negative")
        if self.backend not in {"auto", "dshow", "msmf"}:
            raise ValueError("USB backend must be auto, dshow or msmf")
        if self.body_provider_policy not in {
            "auto",
            "prefer_cuda",
            "require_cuda",
            "cpu",
        }:
            raise ValueError(
                "body_provider_policy must be auto, prefer_cuda, require_cuda or cpu"
            )
        if self.requested_width < 0 or self.requested_height < 0:
            raise ValueError("requested dimensions cannot be negative")
        if (
            self.requested_fps <= 0
            or self.pose_display_fps <= 0
            or self.analysis_fps <= 0
        ):
            raise ValueError("Camera FPS values must be positive")
        if self.pose_display_fps > self.requested_fps:
            raise ValueError(
                "pose_display_fps cannot exceed requested_fps"
            )
        if self.analysis_fps > self.pose_display_fps:
            raise ValueError(
                "analysis_fps cannot exceed pose_display_fps"
            )
        if self.open_timeout_seconds <= 0 or self.read_timeout_seconds <= 0:
            raise ValueError("Camera timeouts must be positive")
        if self.stop_timeout_seconds <= 0:
            raise ValueError("stop_timeout_seconds must be positive")
        if not 1.0 <= self.candidate_token_expiry_seconds <= 10.0:
            raise ValueError(
                "candidate_token_expiry_seconds must be between 1 and 10"
            )
        if not 1 <= self.candidate_sequence_tolerance <= 60:
            raise ValueError(
                "candidate_sequence_tolerance must be between 1 and 60"
            )
        if not 1 <= self.latest_frame_buffer_size <= 8:
            raise ValueError("latest_frame_buffer_size must be between 1 and 8")
        if not 40 <= self.jpeg_quality <= 95:
            raise ValueError("jpeg_quality must be between 40 and 95")
        if self.persist_recording:
            raise ValueError("Camera recording is not authorized in this stage")
