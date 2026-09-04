"""Focused duration, visibility, identity-boundary, and evidence-state checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.action_analysis import (
    ActionAnalysisConfig,
    extract_motion_features,
    segment_actions,
    stabilize_action_events,
)


def _config(**values) -> ActionAnalysisConfig:
    defaults = {
        "movement_start_threshold": 0.4,
        "movement_stop_threshold": 0.2,
        "start_confirmation_seconds": 0.3,
        "stop_confirmation_seconds": 0.4,
        "minimum_segment_seconds": 0.2,
        "short_gap_merge_seconds": 0.5,
        "stable_event_minimum_seconds": 1.0,
        "short_event_minimum_seconds": 0.5,
        "minimum_detected_evidence_ratio": 0.65,
        "short_event_minimum_detected_ratio": 0.75,
        "maximum_prediction_ratio": 0.35,
    }
    defaults.update(values)
    return ActionAnalysisConfig(**defaults)


def _event(identifier: str, start: float, end: float, action: str = "reach", **values):
    row = {
        "event_id": identifier,
        "clip_id": "VTEST_01",
        "person_ref": "P1",
        "lock_epoch": "1",
        "side": "left",
        "start_time": start,
        "end_time": end,
        "duration_seconds": end - start,
        "action": action,
        "status": "proposed",
        "confirmation_status": "unconfirmed",
        "training_eligible": False,
        "training_approval": "pending",
        "observation_state": "detected",
        "detected_ratio": 0.95,
        "predicted_ratio": 0.0,
        "interpolated_ratio": 0.0,
        "missing_ratio": 0.05,
        "required_joints_reliable": True,
        "direction_clear": action in {"reach", "retract", "lift", "lower", "push", "pull"},
        "track_state_summary": "tracked",
        "occlusion": "none",
        "raw_multi": False,
        "raw_edge": False,
    }
    row.update(values)
    return row


def _feature(index: int, activity: float, **values):
    row = {
        "frame_index": index,
        "timestamp": index * 0.1,
        "activity_intensity": activity,
        "valid_keypoint_ratio": 0.9,
        "detected_keypoint_ratio": 0.9,
        "uncertain_keypoint_ratio": 0.0,
        "interpolation_ratio": 0.0,
        "lost": False,
        "uncertain": False,
        "locked_track_id": 1,
        "person_ref": "P1",
        "lock_epoch": 1,
        "side": "bilateral",
    }
    row.update(values)
    return row


def _pose_record(index: int, x: float, status: str = "detected", **values):
    points = np.zeros((17, 3), dtype=np.float32)
    points[:, 0] = x
    points[:, 1] = np.arange(17, dtype=np.float32) + 20
    points[:, 2] = 0.9
    row = {
        "frame_index": index,
        "timestamp": index * 0.1,
        "smoothed": points,
        "statuses": [status] * 17,
        "bbox": [0, 0, 100, 200],
        "locked_track_id": 1,
        "person_ref": "P1",
        "lock_epoch": 1,
        "track_state": "tracked",
        "detection_present": True,
    }
    row.update(values)
    return row


def test_01_700ms_lost_is_never_retract():
    row = _event("HAS-042", 5.2, 5.9, "retract", human_hard_boundary="lost;off_frame")
    result = stabilize_action_events([row], _config())
    assert [item["action"] for item in result["stable_events"]] == ["lost"]


def test_02_off_frame_is_hard_boundary_and_blocks_merge():
    rows = [
        _event("a", 0.0, 1.1),
        _event("lost", 1.1, 1.3, "reach", occlusion="off_frame"),
        _event("b", 1.3, 2.4),
    ]
    result = stabilize_action_events(rows, _config())
    assert [item["action"] for item in result["stable_events"]] == ["reach", "lost", "reach"]
    assert result["metrics"]["merge_count"] == 0


def test_03_temporarily_lost_suppresses_normal_action():
    row = _event("temp", 0.0, 0.7, "retract", track_state_summary="temporarily_lost;tracked")
    assert stabilize_action_events([row], _config())["stable_events"][0]["action"] == "lost"


def test_04_severe_or_ambiguous_never_forms_stable_action():
    rows = [
        _event("severe", 0.0, 1.2, "move", occlusion="severe"),
        _event("ambiguous", 2.0, 3.2, "move", track_state_summary="ambiguous;tracked"),
    ]
    actions = [item["action"] for item in stabilize_action_events(rows, _config())["stable_events"]]
    assert actions == ["unknown", "unknown"]


def test_05_same_person_epoch_side_action_fragments_can_merge():
    rows = [_event("a", 0.0, 1.1), _event("b", 1.3, 2.4)]
    result = stabilize_action_events(rows, _config())
    assert len(result["stable_events"]) == 1
    assert result["stable_events"][0]["source_event_ids"] == "a;b"


def test_06_person_change_blocks_fragment_merge():
    rows = [_event("a", 0.0, 1.1), _event("b", 1.2, 2.3, person_ref="P2")]
    assert len(stabilize_action_events(rows, _config())["stable_events"]) == 2


def test_07_lock_epoch_change_blocks_fragment_merge():
    rows = [_event("a", 0.0, 1.1), _event("b", 1.2, 2.3, lock_epoch="2")]
    assert len(stabilize_action_events(rows, _config())["stable_events"]) == 2


def test_08_anatomical_left_and_right_never_merge():
    rows = [_event("a", 0.0, 1.1), _event("b", 1.2, 2.3, side="right")]
    assert len(stabilize_action_events(rows, _config())["stable_events"]) == 2


def test_09_low_quality_short_move_is_suppressed():
    row = _event("move", 0.0, 0.8, "move", detected_ratio=0.6, predicted_ratio=0.4)
    result = stabilize_action_events([row], _config())
    assert not result["stable_events"] and result["suppressed_events"][0]["action"] == "transition"


def test_10_short_idle_blip_does_not_switch_display_name():
    rows = [
        _event("a", 0.0, 1.1),
        _event("idle", 1.1, 1.3, "idle", direction_clear=False),
        _event("b", 1.3, 2.4),
    ]
    result = stabilize_action_events(rows, _config())
    assert [item["action"] for item in result["stable_events"]] == ["reach"]


def test_11_reliable_directional_short_reach_and_retract_are_preserved():
    rows = [_event("reach", 0.0, 0.7), _event("retract", 1.0, 1.8, "retract")]
    actions = [item["action"] for item in stabilize_action_events(rows, _config())["stable_events"]]
    assert actions == ["reach", "retract"]


def test_12_predicted_and_interpolated_never_become_detected():
    row = _event(
        "pred", 0.0, 1.2, "move", observation_state="predicted",
        detected_ratio=0.1, predicted_ratio=0.8, interpolated_ratio=0.1,
    )
    result = stabilize_action_events([row], _config())
    assert result["pose_evidence"][0]["observation_state"] == "predicted"
    assert result["stable_events"][0]["observation_state"] == "uncertain"


def test_13_prediction_dominant_event_becomes_uncertain():
    row = _event("pred", 0.0, 1.2, "move", detected_ratio=0.3, predicted_ratio=0.7)
    stable = stabilize_action_events([row], _config())["stable_events"][0]
    assert stable["action"] == "unknown" and stable["status"] == "uncertain"


def test_14_detected_recovery_clears_prediction_and_velocity_history():
    rows = [_pose_record(0, 0), _pose_record(1, 20, "predicted"), _pose_record(2, 40)]
    features = extract_motion_features(rows)
    assert features[2]["activity_intensity"] == 0.0
    assert features[2]["detected_keypoint_ratio"] == 1.0


def test_15_action_start_waits_for_confirmation_window():
    features = [_feature(i, 0.8 if i < 3 else 0.05) for i in range(6)]
    segments, _ = segment_actions(features, _config(start_confirmation_seconds=0.3), video_fingerprint="x", locked_track_id=1, analysis_end_time=0.6)
    assert all(item.segment_type != "movement" for item in segments)


def test_16_action_end_uses_stop_hysteresis():
    values = [0.8] * 5 + [0.05] * 2 + [0.8] * 5
    features = [_feature(i, value) for i, value in enumerate(values)]
    segments, _ = segment_actions(features, _config(start_confirmation_seconds=0, stop_confirmation_seconds=0.4), video_fingerprint="x", locked_track_id=1, analysis_end_time=1.2)
    assert sum(item.segment_type == "movement" for item in segments) == 1


def test_17_subsecond_action_name_flapping_emits_no_stable_switches():
    rows = [
        _event("a", 0.0, 1.1),
        _event("blip", 1.1, 1.3, "retract"),
        _event("b", 1.3, 2.4),
    ]
    result = stabilize_action_events(rows, _config())
    assert len(result["stable_events"]) == 1 and result["stable_events"][0]["action"] == "reach"


def test_18_suppressed_or_merged_events_preserve_pose_phases():
    rows = [_event("a", 0.0, 1.1), _event("idle", 1.1, 1.3, "idle"), _event("b", 1.3, 2.4)]
    result = stabilize_action_events(rows, _config())
    assert result["pose_evidence"] == rows and len(result["pose_evidence"]) == 3


def test_19_partial_back_view_never_emits_grasp_or_take():
    row = _event("HAS-056", 18.2, 18.9, "grasp", human_semantic_defer=True, candidate_action_options="reach;lift;retract;unknown")
    result = stabilize_action_events([row], _config())
    serialized = str(result).lower()
    assert not result["stable_events"] and "candidate_action_options" in serialized
    assert all(item["action"] != "grasp" for item in result["suppressed_events"])


def test_20_action_config_loads_from_unicode_space_path(tmp_path: Path):
    location = tmp_path / "中文 空格" / "动作 配置.yaml"
    location.parent.mkdir(parents=True)
    location.write_text("stable_event_minimum_seconds: 1.0\nshort_event_minimum_seconds: 0.5\n", encoding="utf-8")
    config = ActionAnalysisConfig.from_yaml(location)
    assert config.stable_event_minimum_seconds == 1.0
