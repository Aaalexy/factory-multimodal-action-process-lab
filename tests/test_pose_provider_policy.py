from __future__ import annotations

import json
from pathlib import Path

import cv2
import pytest

from src.camera.contracts import CameraConfig
from src.pose_core import (
    BodyProviderUnavailableError,
    PoseRuntime,
    select_body_provider_request,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "yolov8n-pose.onnx"


def test_prefer_cuda_orders_cuda_before_cpu() -> None:
    selected, reason = select_body_provider_request(
        "prefer_cuda",
        available_providers=[
            "CPUExecutionProvider",
            "CUDAExecutionProvider",
        ],
    )
    assert selected == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert reason is None


def test_prefer_cuda_fallback_is_explicit() -> None:
    selected, reason = select_body_provider_request(
        "prefer_cuda",
        available_providers=["CPUExecutionProvider"],
    )
    assert selected == ["CPUExecutionProvider"]
    assert reason == "cuda_execution_provider_unavailable"


def test_require_cuda_fails_closed_when_unavailable() -> None:
    with pytest.raises(
        BodyProviderUnavailableError,
        match="required but unavailable",
    ):
        select_body_provider_request(
            "require_cuda",
            available_providers=["CPUExecutionProvider"],
        )


@pytest.mark.private_artifacts
def test_cpu_policy_remains_available_with_gpu_distribution() -> None:
    runtime = PoseRuntime(MODEL, provider_policy="cpu")
    assert runtime.providers[0] == "CPUExecutionProvider"
    status = runtime.provider_status
    assert status["active_provider"] == "CPUExecutionProvider"
    assert status["fallback_active"] is False


@pytest.mark.private_artifacts
def test_project_require_cuda_session_is_actually_cuda() -> None:
    runtime = PoseRuntime(MODEL, provider_policy="require_cuda")
    status = runtime.provider_status
    assert status["active_provider"] == "CUDAExecutionProvider"
    assert runtime.providers[0] == "CUDAExecutionProvider"
    assert status["fallback_active"] is False


@pytest.mark.private_artifacts
def test_real_frame_warmup_is_not_counted_as_evidence_inference() -> None:
    video = (
        ROOT
        / "outputs"
        / "private_regression"
        / "replay"
        / "sample_video_C"
        / "candidate"
        / "source_video.mp4"
    )
    capture = cv2.VideoCapture(str(video))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    assert ok is True
    runtime = PoseRuntime(MODEL, provider_policy="require_cuda")
    warmup = runtime.warmup(frame)
    assert len(warmup) == 1
    assert runtime.detector.warmup_call_count == 1
    assert runtime.detector.inference_call_count == 0
    assert runtime.detector.inference_times_ms == []
    runtime.detect(frame)
    assert runtime.detector.inference_call_count == 1


def test_camera_config_accepts_and_rejects_provider_policies(tmp_path: Path) -> None:
    config = json.loads(
        (ROOT / "configs" / "camera.json").read_text(encoding="utf-8")
    )
    assert CameraConfig(**{
        key: value
        for key, value in config.items()
        if key in CameraConfig.__dataclass_fields__
    }).body_provider_policy == "prefer_cuda"
    with pytest.raises(ValueError, match="body_provider_policy"):
        CameraConfig(body_provider_policy="silent_gpu").validate()


def test_camera_path_has_no_body_cpu_provider_hardcode() -> None:
    source = (ROOT / "src" / "camera" / "live_analysis.py").read_text(
        encoding="utf-8"
    )
    assert 'providers=["CPUExecutionProvider"]' not in source
    assert "provider_policy=body_provider_policy" in source


def test_four_validation_flags_remain_false() -> None:
    config = json.loads(
        (ROOT / "configs" / "camera.json").read_text(encoding="utf-8")
    )
    for name in (
        "factory_camera_validated",
        "production_action_model_ready",
        "external_factory_validated",
        "production_process_model_ready",
    ):
        assert config[name] is False
