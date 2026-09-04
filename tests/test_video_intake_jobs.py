from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.multimodal_pipeline import BaselineConfig
from src.web.resource_coordinator import AnalysisResourceCoordinator
from src.web.video_intake import (
    PreviewRegistry,
    VideoIntakeConfig,
    VideoIntakeManager,
    sanitize_original_filename,
)


ROOT = Path(__file__).resolve().parents[1]
UPLOADED_HAND_ANALYSIS = (
    ROOT
    / "outputs"
    / "analyses"
    / "analysis_2b5a5a6c2e254bf5a789b29a06636ec5"
    / "analysis.json"
)


def good_probe(_path: str | Path) -> SimpleNamespace:
    return SimpleNamespace(
        duration_seconds=3.0,
        fps=25.0,
        width=1280,
        height=720,
        frame_count=75,
        codec="h264",
        decodable=True,
    )


class BoundedReadStream(io.BytesIO):
    def __init__(self, value: bytes, maximum_read: int) -> None:
        super().__init__(value)
        self.maximum_read = maximum_read
        self.requests: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self.requests.append(size)
        assert 0 <= size <= self.maximum_read
        return super().read(size)


def manager(
    tmp_path: Path,
    *,
    maximum: int = 32 * 1024,
    chunk: int = 4096,
) -> VideoIntakeManager:
    return VideoIntakeManager(
        tmp_path,
        config=VideoIntakeConfig(
            maximum_upload_bytes=maximum,
            upload_chunk_bytes=chunk,
        ),
        probe=good_probe,
    )


@pytest.mark.parametrize(
    "name",
    [
        "../escape.mp4",
        r"..\escape.mp4",
        r"C:\escape.mp4",
        r"\\server\share\escape.mp4",
        "/absolute/escape.mp4",
        "wrong.mov",
    ],
)
def test_upload_filename_rejects_paths_and_non_mp4(name: str) -> None:
    with pytest.raises(ValueError):
        sanitize_original_filename(name)


def test_unicode_and_space_mp4_display_name_is_allowed() -> None:
    assert sanitize_original_filename("中文 工位 01.mp4") == "中文 工位 01.mp4"


def test_upload_is_chunked_probed_and_atomically_promoted(tmp_path: Path) -> None:
    content = b"real-video-bytes-for-streaming" * 400
    stream = BoundedReadStream(content, maximum_read=4096)
    intake = manager(tmp_path)
    record = intake.receive(
        stream,
        content_length=len(content),
        original_filename="中文 工位.mp4",
    )
    video = tmp_path / record["relative_path"]
    assert record["state"] == "ready"
    assert record["original_filename"] == "中文 工位.mp4"
    assert record["storage"]["streamed"] is True
    assert record["storage"]["atomic_part_rename"] is True
    assert video.read_bytes() == content
    assert not video.with_suffix(".mp4.part").exists()
    assert len(stream.requests) >= 2
    assert max(stream.requests) <= 4096


def test_same_original_filename_never_overwrites(tmp_path: Path) -> None:
    intake = manager(tmp_path)
    first = intake.receive(
        io.BytesIO(b"first"),
        content_length=5,
        original_filename="same.mp4",
    )
    second = intake.receive(
        io.BytesIO(b"second"),
        content_length=6,
        original_filename="same.mp4",
    )
    assert first["upload_id"] != second["upload_id"]
    assert first["relative_path"] != second["relative_path"]
    assert (tmp_path / first["relative_path"]).read_bytes() == b"first"
    assert (tmp_path / second["relative_path"]).read_bytes() == b"second"


def test_empty_and_oversized_uploads_are_rejected(tmp_path: Path) -> None:
    intake = manager(tmp_path, maximum=10)
    with pytest.raises(ValueError, match="empty"):
        intake.receive(
            io.BytesIO(),
            content_length=0,
            original_filename="empty.mp4",
        )
    with pytest.raises(ValueError, match="maximum"):
        intake.receive(
            io.BytesIO(b"x" * 11),
            content_length=11,
            original_filename="large.mp4",
        )


def test_interrupted_upload_retains_only_part_and_failure_record(tmp_path: Path) -> None:
    intake = manager(tmp_path)
    with pytest.raises(ConnectionError):
        intake.receive(
            io.BytesIO(b"short"),
            content_length=20,
            original_filename="broken.mp4",
        )
    intake_dirs = list((tmp_path / "outputs" / "intake").iterdir())
    assert len(intake_dirs) == 1
    directory = intake_dirs[0]
    assert (directory / "source.mp4.part").is_file()
    assert not (directory / "source.mp4").exists()
    failure = json.loads((directory / "failure.json").read_text("utf-8"))
    assert failure["complete_video_present"] is False
    assert failure["part_present"] is True


def test_undecodable_upload_is_never_promoted(tmp_path: Path) -> None:
    def bad_probe(_path: str | Path) -> dict[str, object]:
        return {
            "duration_seconds": 0,
            "fps": 0,
            "width": 0,
            "height": 0,
            "frame_count": 0,
            "codec": "unknown",
            "decodable": False,
        }

    intake = VideoIntakeManager(
        tmp_path,
        config=VideoIntakeConfig(upload_chunk_bytes=4096),
        probe=bad_probe,
    )
    with pytest.raises(ValueError, match="not decodable"):
        intake.receive(
            io.BytesIO(b"not-a-video"),
            content_length=11,
            original_filename="bad.mp4",
        )
    directory = next((tmp_path / "outputs" / "intake").iterdir())
    assert not (directory / "source.mp4").exists()


def candidate() -> dict[str, object]:
    return {
        "candidate_id": "C1",
        "video_path": r"C:\controlled\source.mp4",
        "selection_timestamp": 1.0,
        "selection_frame_index": 25,
        "bbox": [10.0, 20.0, 100.0, 220.0],
        "center": [55.0, 120.0],
        "size": [90.0, 200.0],
        "torso_keypoints": [[20.0, 30.0, 0.9]] * 4,
        "person_confidence": 0.9,
        "selection_source": "manual",
    }


def test_preview_registry_uses_opaque_upload_bound_tokens() -> None:
    registry = PreviewRegistry(expiry_seconds=30)
    public = registry.register(
        upload_id="intake_" + "a" * 32,
        result={
            "frame_index": 25,
            "timestamp": 1.0,
            "candidates": [candidate()],
        },
    )
    exposed = public["candidates"][0]
    assert exposed["candidate_token"] != "C1"
    resolved = registry.resolve(
        upload_id="intake_" + "a" * 32,
        preview_id=public["preview_id"],
        candidate_token=exposed["candidate_token"],
    )
    assert resolved["candidate_id"] == "C1"
    with pytest.raises(ValueError, match="another upload"):
        registry.resolve(
            upload_id="intake_" + "b" * 32,
            preview_id=public["preview_id"],
            candidate_token=exposed["candidate_token"],
        )


def test_uploaded_job_defaults_enable_hand_and_prefer_cuda() -> None:
    config = BaselineConfig(
        project_root=str(ROOT),
        source_video="outputs/intake/example/source.mp4",
    )
    assert config.hand_enabled is True
    assert config.body_provider_policy == "prefer_cuda"
    assert config.recording_group_id == "recording_group_unassigned"


def test_video_intake_config_keeps_all_validation_flags_false() -> None:
    config = VideoIntakeConfig.load(ROOT / "configs" / "video_intake.json")
    assert config.maximum_upload_bytes > 0
    payload = json.loads((ROOT / "configs" / "video_intake.json").read_text("utf-8"))
    assert all(value is False for value in payload["validation_flags"].values())


def test_camera_and_uploaded_analysis_share_one_resource_owner() -> None:
    coordinator = AnalysisResourceCoordinator()
    lease = coordinator.acquire("video_analysis")
    with pytest.raises(RuntimeError):
        coordinator.acquire("camera")
    coordinator.release(lease)
    camera = coordinator.acquire("camera")
    assert coordinator.active_mode == "camera"
    coordinator.release(camera)


def test_upload_ui_has_file_input_progress_cancel_and_hot_load_contract() -> None:
    html = (ROOT / "src" / "web" / "static" / "index.html").read_text("utf-8")
    script = (ROOT / "src" / "web" / "static" / "app.js").read_text("utf-8")
    assert 'type="file"' in html
    assert 'accept="video/mp4,.mp4"' in html
    assert "upload-progress-bar" in html
    assert "video-analysis-cancel-button" in html
    assert "/api/video/upload" in script
    assert "/api/video/jobs/cancel" in script
    assert "loadCurrentAnalysis({reloadVideo: true})" in script
    assert "XMLHttpRequest" in script


@pytest.mark.private_artifacts
def test_real_uploaded_window_reports_cuda_body_and_cpu_hand() -> None:
    payload = json.loads(UPLOADED_HAND_ANALYSIS.read_text("utf-8"))
    body = payload["runtime"]["pose_provider_status"]
    assert body["active_provider"] == "CUDAExecutionProvider"
    assert body["fallback_active"] is False
    assert payload["hand_model"]["provider"] == "CPU"
    assert payload["runtime"]["hand_provider"] == "CPU"
    assert (
        payload["hand_model"]["hand_gpu_status"]
        == "unsupported_current_backend_on_windows"
    )
    assert payload["runtime"]["hand_inference_calls"] > 0
    assert payload["runtime"]["mean_hand_inference_ms"] > 0
    assert payload["runtime"]["hand_inference_p50_ms"] > 0
    assert payload["runtime"]["hand_inference_p95_ms"] > 0


@pytest.mark.private_artifacts
def test_real_uploaded_hand_geometry_is_never_fabricated() -> None:
    payload = json.loads(UPLOADED_HAND_ANALYSIS.read_text("utf-8"))
    frames = payload["hand_pose_frames"]
    real_geometry = [frame for frame in frames if frame["landmark_count"] == 21]
    assert real_geometry
    assert all(len(frame["landmarks"]) == 21 for frame in real_geometry)
    assert all(
        frame["observation_state"] in {"detected", "uncertain"}
        for frame in real_geometry
    )
    assert all(frame["training_eligible"] is False for frame in frames)
    assert all(
        frame["landmarks"] == []
        for frame in frames
        if frame["observation_state"] in {"missing", "lost"}
    )


def test_hand_runtime_ui_exposes_provider_quality_and_percentiles() -> None:
    html = (ROOT / "src" / "web" / "static" / "index.html").read_text("utf-8")
    script = (ROOT / "src" / "web" / "static" / "app.js").read_text("utf-8")
    for identifier in (
        "hand-runtime-provider",
        "hand-gpu-status",
        "hand-inference-percentiles",
        "hand-drawable-geometry",
        "hand-association-warning",
    ):
        assert identifier in html
        assert identifier in script
