from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.validate_schemas import validate_all
from src.pose_core import PoseRuntime
from src.video_io import probe_video


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "outputs" / "baseline_run" / "analysis.json"
pytestmark = pytest.mark.private_artifacts


def _analysis() -> dict[str, object]:
    assert ANALYSIS.is_file(), "Run scripts/run_baseline.py before this test"
    return json.loads(ANALYSIS.read_text("utf-8"))


def test_real_pose_model_loads_with_cpu_provider():
    runtime = PoseRuntime(
        ROOT / "models" / "yolov8n-pose.onnx",
        providers=["CPUExecutionProvider"],
    )
    assert "CPUExecutionProvider" in runtime.providers


def test_real_baseline_used_model_inference_without_mock_or_presets():
    data = _analysis()
    runtime = data["runtime"]
    assert runtime["pose_inference_calls"] > 0
    assert runtime["mock_keypoints_used"] is False
    assert runtime["preset_actions_used"] is False
    assert runtime["model_training_performed"] is False
    assert runtime["deepseek_called"] is False


def test_real_baseline_video_is_decodable_and_sha_traceable():
    data = _analysis()
    local = ROOT / data["source_video"]["path"]
    probe = probe_video(local)
    assert probe.decodable is True
    assert probe.sha256 == data["source_video"]["sha256"]


def test_real_baseline_lost_frames_have_no_keypoint_geometry():
    data = _analysis()
    for frame in data["pose_frames"]:
        if frame["track_state"] == "lost":
            assert frame["keypoints"] == []
            assert frame["bbox"] is None


def test_real_baseline_action_events_do_not_cross_identity_boundaries():
    data = _analysis()
    segments = {item["segment_id"]: item for item in data["pose_segments"]}
    for event in data["action_events"]:
        identities = {
            (
                segments[source]["person_ref"],
                segments[source]["lock_epoch"],
            )
            for source in event["source_segment_ids"]
        }
        assert len(identities) == 1


def test_real_baseline_generated_artifacts_pass_schemas():
    result = validate_all(ANALYSIS)
    assert result["status"] == "passed"


def test_real_baseline_missing_models_propagate_unavailable():
    data = _analysis()
    states = {item["layer"]: item["status"] for item in data["layer_states"]}
    assert states["object_perception"] == "unavailable"
    assert states["interaction_fusion"] == "unavailable"
    assert states["temporal_action_model"] == "unavailable"
    assert states["process_reasoning"] == "unavailable"
    assert data["object_tracks"] == []
    assert data["interaction_events"] == []
    assert data["process_steps"] == []
