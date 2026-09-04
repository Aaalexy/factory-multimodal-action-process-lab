from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from scripts.compare_phase_b import compare_validation_root, extract_metrics
from src.action_segmentation import build_stable_action_events
from src.contracts import (
    HandPoseFrame,
    MultimodalResult,
    ValidationFlags,
)
from src.multimodal_pipeline import BaselineConfig, _load_phase_b_action_config


ROOT = Path(__file__).resolve().parents[1]


def _analysis(
    *,
    profile: str,
    frames: int,
    pose_calls: int,
    hand_calls: int,
    events: int,
    sub_1s: int,
    detected_hands: int,
    missing_hands: int,
) -> dict[str, object]:
    action_events = [
        {
            "action_event_id": f"event-{index}",
            "action": "reach",
            "start_time": float(index * 2),
            "end_time": float(index * 2 + 1.2),
            "source_segment_ids": [f"segment-{index}"],
            "status": "proposed",
            "training_eligible": False,
        }
        for index in range(events)
    ]
    hand_pose_frames = [
        {
            "frame_index": index,
            "anatomical_side": "left",
            "observation_state": "detected",
            "landmarks": [{"index": point} for point in range(21)],
            "association_checks": {"warnings": []},
            "training_eligible": False,
        }
        for index in range(detected_hands)
    ]
    hand_pose_frames.extend(
        {
            "frame_index": detected_hands + index,
            "anatomical_side": "right",
            "observation_state": "missing",
            "landmarks": [],
            "association_checks": {"warnings": []},
            "training_eligible": False,
        }
        for index in range(missing_hands)
    )
    return {
        "source_video": {
            "sha256": "a" * 64,
            "analysis_window": {
                "start_time": 0.0,
                "end_time": 12.0,
            },
        },
        "action_profile": profile,
        "pose_segments": [{"segment_id": f"segment-{index}"} for index in range(5)],
        "action_events": action_events,
        "hand_pose_frames": hand_pose_frames,
        "stabilization_metrics": {
            "suppressed_fragment_count": 3,
            "merged_fragment_count": 1,
            "unknown_transition_duration_seconds": 0.5,
        },
        "runtime": {
            "processed_frame_count": frames,
            "pose_inference_calls": pose_calls,
            "hand_inference_calls": hand_calls,
            "hand_detected_frame_count": detected_hands,
            "hand_uncertain_frame_count": 0,
            "hand_missing_frame_count": missing_hands,
            "hand_detected_observation_count": detected_hands,
            "hand_uncertain_observation_count": 0,
            "hand_missing_observation_count": missing_hands,
            "left_right_association_error_count": 0,
            "pose_segment_count": 5,
            "stable_action_event_count": events,
            "sub_1s_stable_event_count": sub_1s,
            "events_per_minute": events * 5.0,
            "suppressed_fragment_count": 3,
            "merged_fragment_count": 1,
            "unknown_transition_duration_seconds": 0.5,
            "lost_normal_action_false_positive_count": 0,
            "cross_identity_or_epoch_merge_count": 0,
            "mean_pose_inference_ms": 11.0,
            "mean_hand_inference_ms": 7.5 if hand_calls else None,
            "processing_seconds": 2.0,
            "end_to_end_seconds": 2.5,
            "end_to_end_frames_per_second": frames / 2.5,
        },
        "evaluation": {
            "status": "not_evaluable",
            "reason": "No human ground truth",
        },
    }


def test_comparison_reports_each_clip_and_aggregate_without_accuracy(
    tmp_path: Path,
) -> None:
    validation = tmp_path / "validation"
    before_path = validation / "clip-01" / "before" / "analysis.json"
    after_path = validation / "clip-01" / "after" / "analysis.json"
    before_path.parent.mkdir(parents=True)
    after_path.parent.mkdir(parents=True)
    before_path.write_text(
        json.dumps(
            _analysis(
                profile="phase_a_parameter_replay_with_crash_guard",
                frames=24,
                pose_calls=24,
                hand_calls=0,
                events=4,
                sub_1s=1,
                detected_hands=0,
                missing_hands=0,
            )
        ),
        encoding="utf-8",
    )
    after_path.write_text(
        json.dumps(
            _analysis(
                profile="phase_b",
                frames=96,
                pose_calls=96,
                hand_calls=160,
                events=3,
                sub_1s=0,
                detected_hands=4,
                missing_hands=92,
            )
        ),
        encoding="utf-8",
    )

    comparison = compare_validation_root(validation)
    clip = comparison["clips"][0]
    summary = comparison["summary"]
    assert comparison["evaluation"]["status"] == "not_evaluable"
    assert comparison["evaluation"]["accuracy_metrics_computed"] is False
    assert clip["evaluation"]["status"] == "not_evaluable"
    assert clip["profiles"]["before"]["name"].startswith("phase_a")
    assert clip["profiles"]["after"]["name"] == "phase_b"
    assert clip["before"]["processed_frame_count"] == 24
    assert clip["after"]["hand_inference_calls"] == 160
    assert clip["after"]["hand_detected_observation_count"] == 4
    assert clip["delta_after_minus_before"]["sub_1s_stable_event_count"] == -1
    assert summary["before"]["stable_action_event_count"] == 4
    assert summary["after"]["stable_action_event_count"] == 3
    assert summary["after"]["sub_1s_stable_event_count"] == 0


def test_metric_fallback_counts_hand_observations_and_empty_missing() -> None:
    payload = _analysis(
        profile="phase_b",
        frames=2,
        pose_calls=2,
        hand_calls=2,
        events=1,
        sub_1s=0,
        detected_hands=1,
        missing_hands=1,
    )
    payload["runtime"] = {}
    metrics = extract_metrics(payload)
    assert metrics["hand_detected_observation_count"] == 1
    assert metrics["hand_missing_observation_count"] == 1
    missing = payload["hand_pose_frames"][1]
    assert missing["observation_state"] == "missing"
    assert missing["landmarks"] == []


def test_phase_b_pipeline_and_project_config_contract() -> None:
    config = BaselineConfig(
        project_root=str(ROOT),
        source_video="read-only-input.mp4",
    )
    project = json.loads((ROOT / "configs" / "project.json").read_text("utf-8"))
    assert config.sample_fps == 8.0
    assert config.hand_enabled is True
    assert config.action_profile == "phase_b"
    assert config.hand_model_path == "models/hand_pose/hand_landmarker.task"
    assert project["pose_model"]["sample_fps"] == 8.0
    assert project["hand_pose"]["landmarks_per_hand"] == 21
    assert project["hand_pose"]["missing_geometry_policy"] == "empty_landmarks"
    quality_gate = project["hand_pose"]["quality_gate"]
    assert quality_gate["version"] == "hand_quality_gate_v1"
    assert quality_gate["maximum_own_wrist_distance_roi_ratio"] == 0.3
    assert quality_gate["required_landmark_count"] == 21
    assert quality_gate["require_unique_landmark_indices_0_to_20"] is True
    assert quality_gate["reject_association_warnings"] is True
    assert quality_gate["reject_duplicate_across_sides"] is True
    assert quality_gate["require_direct_body_wrist_and_elbow"] is True
    assert project["action_stability"]["stable_event_minimum_seconds"] >= 1.0
    loaded = _load_phase_b_action_config(ROOT, analysis_fps=config.sample_fps)
    assert loaded.stable_event_minimum_seconds == 1.2
    assert loaded.short_directional_event_minimum_seconds == 1.0
    assert loaded.short_gap_merge_seconds == 0.4
    assert loaded.start_confirmation_seconds == 0.5
    assert loaded.stop_confirmation_seconds == 0.5
    assert loaded.temporal_context_seconds == 2.5
    assert loaded.bounded_uncertain_gap_seconds == 0.375
    assert (
        project["legacy_configuration"]["action_analysis_yaml_role"]
        == "migrated_phase_a_reference_not_loaded_by_phase_b_runtime"
    )
    assert all(value is False for value in project["validation_flags"].values())


def test_forbidden_project_path_is_absent_from_runtime_inputs_and_commands() -> None:
    forbidden = r"<forbidden-unrelated-workspace>"
    inspected = [
        ROOT / "SOURCE_IMPORT_MANIFEST.json",
        ROOT / "HAND_MODEL_MANIFEST.json",
        *sorted((ROOT / "configs").glob("*.json")),
        *sorted((ROOT / "scripts").glob("*.py")),
        *sorted((ROOT / "scripts").glob("*.ps1")),
    ]
    assert inspected
    for path in inspected:
        assert forbidden not in path.read_text("utf-8"), path


def test_automatic_hand_and_action_contracts_remain_unconfirmed() -> None:
    hand = HandPoseFrame(
        hand_pose_id="hand-001",
        person_ref="person-001",
        lock_epoch=1,
        anatomical_side="left",
        frame_index=3,
        timestamp=0.375,
        crop_bbox=None,
        crop_transform=None,
        landmarks=[],
        landmark_count=0,
        confidence=None,
        observation_state="missing",
        occlusion="unknown",
        source_video_sha256="a" * 64,
        recording_group_id="recording-group-test",
        source_model_version="hand-model-test",
    )
    hand_payload = asdict(hand)
    stable = build_stable_action_events(
        [
            {
                "segment_id": "segment-001",
                "action": "reach",
                "action_name": "reach",
                "person_ref": "person-001",
                "lock_epoch": 1,
                "side": "left",
                "anatomical_side": "left",
                "start_time": 0.0,
                "end_time": 1.2,
                "duration_seconds": 1.2,
                "source_video_sha256": "a" * 64,
                "track_state": "tracked",
                "lock_state": "tracking",
                "observation_state": "detected",
                "detected_ratio": 1.0,
                "predicted_ratio": 0.0,
                "interpolated_ratio": 0.0,
                "missing_ratio": 0.0,
                "required_joints_reliable": True,
                "direction_clear": True,
                "raw_lost": False,
                "training_eligible": False,
                "source_segment_ids": ["segment-001"],
            }
        ]
    )["stable_events"]
    assert hand_payload["landmarks"] == []
    assert hand_payload["landmark_count"] == 0
    assert hand_payload["status"] == "uncertain"
    assert hand_payload["training_eligible"] is False
    assert hand_payload["backend_state"] == "unavailable"
    assert hand_payload["backend_mode"] == "image"
    assert hand_payload["quality_state"] == "not_observed"
    assert hand_payload["validation_state"] == "not_evaluable"
    assert hand_payload["action_feature_eligible"] is False
    assert hand_payload["quality_gate_version"] == "hand_quality_gate_v1"
    assert stable[0]["source_segment_ids"] == ["segment-001"]
    assert stable[0]["training_eligible"] is False

    result = MultimodalResult(
        schema_version="test",
        project="Factory Multimodal Action Process Lab",
        source_video={},
        validation_flags=ValidationFlags(),
        hand_pose_frames=[hand_payload],
        action_events=stable,
        evidence_timeline=[
            {
                "evidence_interval_id": "evidence-00001",
                "evidence_state": "normal",
                "action": "reach",
                "status": "proposed",
                "training_eligible": False,
            }
        ],
    ).to_dict()
    assert all(value is False for value in result["validation_flags"].values())
    assert result["hand_pose_frames"][0]["training_eligible"] is False
    assert result["action_events"][0]["training_eligible"] is False
    assert result["evidence_timeline"][0]["training_eligible"] is False
