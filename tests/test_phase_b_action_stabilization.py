from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from src.action_segmentation import (
    CausalCoarseActionClassifier,
    FrameActionStabilityConfig,
    PhaseBActionStabilityConfig,
    build_pose_segments,
    build_evidence_timeline,
    build_stable_action_events,
    build_stable_action_events_from_frames,
    stabilize_coarse_frames,
)
from src.action_segmentation.coarse import CoarseFrame
from src.schema_validation import validate_instance


ROOT = Path(__file__).resolve().parents[1]


def _points(*, left_wrist_x: float = 20.0) -> np.ndarray:
    points = np.zeros((17, 3), dtype=np.float32)
    points[:, 2] = 0.9
    points[5, :2] = (0.0, 0.0)
    points[6, :2] = (100.0, 0.0)
    points[7, :2] = (5.0, 30.0)
    points[8, :2] = (95.0, 30.0)
    points[9, :2] = (left_wrist_x, 30.0)
    points[10, :2] = (90.0, 30.0)
    points[11, :2] = (0.0, 100.0)
    points[12, :2] = (100.0, 100.0)
    return points


def _classify(
    classifier: CausalCoarseActionClassifier,
    *,
    timestamp: float,
    points: np.ndarray | None,
    track_state: str = "tracked",
) -> tuple[str, str, dict[str, object]]:
    statuses = None if points is None else np.asarray(["detected"] * 17)
    return classifier.classify(
        timestamp=timestamp,
        person_ref="person-001",
        lock_epoch=1,
        track_state=track_state,
        lock_state="tracking" if track_state == "tracked" else "temporarily_lost",
        keypoints=points,
        statuses=statuses,
    )


def _frame(
    timestamp: float,
    action: str,
    *,
    side: str = "left",
    track_state: str = "tracked",
    observation_state: str = "detected",
    reliable: bool = True,
    person: str = "person-001",
    epoch: int = 1,
) -> CoarseFrame:
    return CoarseFrame(
        timestamp=timestamp,
        source_frame_index=round(timestamp * 100),
        person_ref=person,
        lock_epoch=epoch,
        track_state=track_state,
        lock_state="tracking" if track_state == "tracked" else "temporarily_lost",
        candidate_person_count=1,
        action=action,
        anatomical_side=side,
        observation_state=observation_state,
        detected_ratio=1.0 if observation_state == "detected" else 0.0,
        predicted_ratio=0.0,
        interpolated_ratio=0.0,
        missing_ratio=0.0 if observation_state == "detected" else 1.0,
        direction_clear=True,
        required_joints_reliable=reliable,
        keypoints=[],
        keypoint_statuses=[],
    )


def _segment(
    identifier: str,
    action: str,
    start: float,
    end: float,
    *,
    side: str = "left",
    person: str = "person-001",
    epoch: int = 1,
    lost: bool = False,
) -> dict[str, object]:
    return {
        "segment_id": identifier,
        "action": action,
        "action_name": action,
        "person_ref": person,
        "lock_epoch": epoch,
        "side": side,
        "anatomical_side": side,
        "start_time": start,
        "end_time": end,
        "duration_seconds": end - start,
        "source_video_sha256": "video-sha",
        "track_state": "lost" if lost else "tracked",
        "lock_state": "temporarily_lost" if lost else "tracking",
        "observation_state": "lost" if lost else "detected",
        "detected_ratio": 0.0 if lost else 0.95,
        "predicted_ratio": 0.0,
        "interpolated_ratio": 0.0,
        "missing_ratio": 1.0 if lost else 0.05,
        "required_joints_reliable": not lost,
        "direction_clear": action
        in {"reach", "retract", "lift", "lower", "push", "pull"},
        "raw_lost": lost,
        "training_eligible": False,
        "source_segment_ids": [identifier],
    }


def test_phase_b_defaults_are_real_runtime_targets():
    config = PhaseBActionStabilityConfig()
    project = json.loads((ROOT / "configs" / "project.json").read_text("utf-8"))
    action = project["action_stability"]
    assert project["pose_model"]["sample_fps"] == config.analysis_fps == 8.0
    assert action["analysis_fps"] == 8.0
    assert action["stable_event_minimum_seconds"] == 1.2
    assert action["short_directional_event_minimum_seconds"] == 1.0
    assert action["short_gap_merge_seconds"] == 0.4
    assert action["start_confirmation_seconds"] == 0.5
    assert action["stop_confirmation_seconds"] == 0.5
    assert action["temporal_context_seconds"] == 2.5
    assert action["bounded_uncertain_gap_seconds"] == 0.375


def test_all_nan_wrist_velocity_returns_unknown_without_crashing():
    classifier = CausalCoarseActionClassifier()
    points = _points()
    points[[9, 10], :2] = np.nan
    assert _classify(classifier, timestamp=0.0, points=points)[0] == "unknown"
    assert _classify(classifier, timestamp=0.125, points=points)[0] == "unknown"


def test_recovered_frame_does_not_calculate_velocity_across_lost():
    classifier = CausalCoarseActionClassifier()
    assert _classify(
        classifier, timestamp=0.0, points=_points(left_wrist_x=10.0)
    )[0] == "transition"
    assert _classify(
        classifier, timestamp=0.5, points=None, track_state="lost"
    )[0] == "lost"
    recovered = _classify(
        classifier, timestamp=1.0, points=_points(left_wrist_x=80.0)
    )
    assert recovered[0] == "transition"


def test_unreliable_arm_evidence_clears_velocity_history():
    classifier = CausalCoarseActionClassifier()
    assert _classify(classifier, timestamp=0.0, points=_points())[0] == "transition"
    unreliable = _points()
    unreliable[[5, 6, 7, 8, 9, 10], :2] = np.nan
    assert _classify(classifier, timestamp=0.5, points=unreliable)[0] == "unknown"
    recovered = _classify(
        classifier, timestamp=1.0, points=_points(left_wrist_x=90.0)
    )
    assert recovered[0] == "transition"


def test_pose_segment_boundaries_use_next_sample_and_analysis_end():
    frames = [
        _frame(0.0, "idle", side="bilateral"),
        _frame(0.13, "reach"),
        _frame(0.27, "reach"),
    ]
    segments = build_pose_segments(
        frames,
        source_video_sha256="video-sha",
        sample_interval_seconds=0.125,
        analysis_end_time=0.4,
    )
    assert len(segments) == 2
    assert segments[0]["end_time"] == segments[1]["start_time"] == 0.13
    assert segments[-1]["end_time"] == 0.4
    assert sum(item["duration_seconds"] for item in segments) == 0.4


def test_frame_confirmation_absorbs_brief_label_noise_without_mutating_raw():
    raw = [
        _frame(0.000, "reach"),
        _frame(0.125, "reach"),
        _frame(0.250, "reach"),
        _frame(0.375, "reach"),
        _frame(0.500, "retract"),
        _frame(0.625, "reach"),
        _frame(0.750, "reach"),
    ]
    result = stabilize_coarse_frames(
        raw,
        FrameActionStabilityConfig(
            start_confirmation_seconds=0.25,
            stop_confirmation_seconds=0.25,
            temporal_context_seconds=0.75,
        ),
    )
    assert raw[4].action == "retract"
    assert {item.action for item in result["frames"]} == {"reach"}


def test_frame_hard_boundary_resets_confirmation_context():
    raw = [
        _frame(0.000, "reach"),
        _frame(0.125, "reach"),
        _frame(0.250, "reach"),
        _frame(0.375, "lost", side="bilateral", track_state="lost"),
        _frame(0.500, "reach"),
        _frame(0.625, "reach"),
    ]
    result = stabilize_coarse_frames(
        raw,
        FrameActionStabilityConfig(
            start_confirmation_seconds=0.25,
            stop_confirmation_seconds=0.25,
            temporal_context_seconds=0.75,
        ),
    )
    actions = [item.action for item in result["frames"]]
    assert actions[3] == "lost"
    assert actions[4:] == ["transition", "transition"]


def test_brief_missing_and_recovery_are_explicit_bounded_gap_without_lane_reset():
    raw = [
        _frame(0.000, "reach"),
        _frame(0.125, "reach"),
        _frame(0.250, "reach"),
        _frame(
            0.375,
            "unknown",
            observation_state="missing",
            reliable=False,
        ),
        _frame(0.500, "transition"),
        _frame(0.625, "reach"),
    ]
    before = deepcopy(raw)
    result = stabilize_coarse_frames(
        raw,
        FrameActionStabilityConfig(
            start_confirmation_seconds=0.25,
            stop_confirmation_seconds=0.25,
            temporal_context_seconds=1.0,
            bounded_uncertain_gap_seconds=0.375,
        ),
    )

    assert raw == before
    assert [item.action for item in result["frames"]] == [
        "reach",
        "reach",
        "reach",
        "unknown",
        "unknown",
        "reach",
    ]
    assert result["metrics"]["hard_boundary_frame_count"] == 0
    assert result["metrics"]["bounded_uncertain_gap_frame_count"] == 2
    assert all(
        item.evidence_state == "uncertain"
        for item in result["frames"][3:5]
    )


def test_long_missing_reclassifies_gap_as_hard_boundary_and_resets_lanes():
    raw = [
        _frame(0.000, "reach"),
        _frame(0.125, "reach"),
        _frame(0.250, "reach"),
        *[
            _frame(
                timestamp,
                "unknown",
                observation_state="missing",
                reliable=False,
            )
            for timestamp in (0.375, 0.500, 0.625, 0.750)
        ],
        _frame(0.875, "reach"),
    ]
    result = stabilize_coarse_frames(
        raw,
        FrameActionStabilityConfig(
            start_confirmation_seconds=0.25,
            stop_confirmation_seconds=0.25,
            temporal_context_seconds=1.0,
            bounded_uncertain_gap_seconds=0.375,
        ),
    )

    assert result["metrics"]["hard_boundary_frame_count"] == 4
    assert result["metrics"]["long_gap_hard_boundary_frame_count"] == 4
    assert result["frames"][-1].action == "transition"
    assert all(item.hard_boundary for item in result["frames"][3:7])


def test_anatomical_side_switch_uses_independent_lanes_without_global_reset():
    raw = [
        _frame(0.000, "reach", side="left"),
        _frame(0.125, "reach", side="left"),
        _frame(0.250, "reach", side="left"),
        _frame(0.375, "lift", side="right"),
        _frame(0.500, "lift", side="right"),
        _frame(0.625, "lift", side="right"),
        _frame(0.750, "reach", side="left"),
    ]
    result = stabilize_coarse_frames(
        raw,
        FrameActionStabilityConfig(
            start_confirmation_seconds=0.25,
            stop_confirmation_seconds=0.25,
            temporal_context_seconds=1.0,
            bounded_uncertain_gap_seconds=0.375,
        ),
    )

    assert [item.action for item in result["frames"][:3]] == ["reach"] * 3
    assert [item.action for item in result["frames"][3:6]] == ["lift"] * 3
    assert result["frames"][-1].action == "reach"
    assert result["metrics"]["hard_boundary_frame_count"] == 0
    assert result["metrics"]["anatomical_side_switch_count"] == 2


def test_frame_to_event_helper_preserves_raw_pose_segment_lineage():
    frames = [_frame(index * 0.125, "reach") for index in range(12)]
    raw_segments = build_pose_segments(
        frames,
        source_video_sha256="video-sha",
        sample_interval_seconds=0.125,
        analysis_end_time=1.5,
    )
    result = build_stable_action_events_from_frames(
        frames,
        raw_segments,
        source_video_sha256="video-sha",
        sample_interval_seconds=0.125,
        analysis_end_time=1.5,
        frame_config=FrameActionStabilityConfig(
            start_confirmation_seconds=0.5,
            stop_confirmation_seconds=0.5,
            temporal_context_seconds=2.5,
        ),
    )
    assert len(result["stable_events"]) == 1
    assert result["stable_events"][0]["source_segment_ids"] == [
        raw_segments[0]["segment_id"]
    ]
    assert result["pose_evidence"] == raw_segments
    assert result["metrics"]["frame_stabilization"]["confirmed_switch_count"] == 1


def test_evidence_timeline_covers_window_and_preserves_raw_lineage() -> None:
    frames = [
        _frame(0.000, "reach"),
        _frame(0.125, "reach"),
        _frame(0.250, "reach"),
        _frame(
            0.375,
            "unknown",
            observation_state="missing",
            reliable=False,
        ),
        _frame(0.500, "transition"),
        _frame(0.625, "reach"),
        _frame(
            0.750,
            "lost",
            side="bilateral",
            track_state="lost",
            observation_state="lost",
            reliable=False,
        ),
    ]
    raw_segments = build_pose_segments(
        frames,
        source_video_sha256="a" * 64,
        sample_interval_seconds=0.125,
        analysis_end_time=0.875,
    )
    raw_before = deepcopy(raw_segments)
    result = build_stable_action_events_from_frames(
        frames,
        raw_segments,
        source_video_sha256="a" * 64,
        sample_interval_seconds=0.125,
        analysis_start_time=0.0,
        analysis_end_time=0.875,
        frame_config=FrameActionStabilityConfig(
            start_confirmation_seconds=0.25,
            stop_confirmation_seconds=0.25,
            temporal_context_seconds=1.0,
            bounded_uncertain_gap_seconds=0.375,
        ),
    )

    timeline = result["evidence_timeline"]
    metrics = result["evidence_timeline_metrics"]
    schema = json.loads(
        (ROOT / "schemas" / "evidence_timeline.schema.json").read_text("utf-8")
    )
    assert raw_segments == raw_before
    assert timeline[0]["start_time"] == 0.0
    assert timeline[-1]["end_time"] == 0.875
    assert all(
        left["end_time"] == right["start_time"]
        for left, right in zip(timeline, timeline[1:])
    )
    assert metrics["coverage_ratio"] == 1.0
    assert metrics["uncovered_seconds"] == 0.0
    assert metrics["pose_fragment_count"] == len(raw_segments)
    assert any(
        item["continuity_kind"] == "bounded_uncertain_gap"
        for item in timeline
    )
    assert any(item["evidence_state"] == "lost" for item in timeline)
    assert all(isinstance(item["source_segment_ids"], list) for item in timeline)
    for item in timeline:
        validate_instance(item, schema)


def test_default_stable_layer_suppresses_all_sub_one_second_normal_actions():
    result = build_stable_action_events(
        [
            _segment("short-lower", "lower", 0.0, 0.999),
            _segment("one-second-lift", "lift", 1.1, 2.1),
        ]
    )
    normal = [
        item
        for item in result["stable_events"]
        if item["event_kind"] == "stable_action"
    ]
    assert [item["action"] for item in normal] == ["lift"]
    assert result["metrics"]["sub_1s_stable_event_count"] == 0
    assert result["metrics"]["short_directional_action_preserved_count"] == 1


def test_bilateral_lost_blocks_left_lane_merge_and_overlap():
    result = build_stable_action_events(
        [
            _segment("left-a", "reach", 0.0, 1.2),
            _segment(
                "lost",
                "lost",
                1.2,
                1.3,
                side="bilateral",
                lost=True,
            ),
            _segment("left-b", "reach", 1.3, 2.5),
        ]
    )
    reaches = [
        item for item in result["stable_events"] if item["action"] == "reach"
    ]
    assert len(reaches) == 2
    assert result["metrics"]["merge_count"] == 0
    assert result["metrics"]["lost_normal_action_overlap_count"] == 0


def test_explicit_other_side_observation_blocks_merge():
    result = build_stable_action_events(
        [
            _segment("left-a", "reach", 0.0, 1.2),
            _segment("right-noise", "move", 1.2, 1.4, side="right"),
            _segment("left-b", "reach", 1.4, 2.6),
        ]
    )
    reaches = [
        item for item in result["stable_events"] if item["action"] == "reach"
    ]
    assert len(reaches) == 2
    assert result["metrics"]["merge_count"] == 0


def test_same_side_short_noise_is_absorbed_with_traceable_source_ids():
    result = build_stable_action_events(
        [
            _segment("reach-a", "reach", 0.0, 1.2),
            _segment("idle-noise", "idle", 1.2, 1.4),
            _segment("reach-b", "reach", 1.4, 2.6),
        ]
    )
    reaches = [
        item for item in result["stable_events"] if item["action"] == "reach"
    ]
    assert len(reaches) == 1
    assert reaches[0]["source_segment_ids"] == ["reach-a", "reach-b"]
    assert reaches[0]["bounded_gap_source_segment_ids"] == ["idle-noise"]
    assert reaches[0]["absorbed_segment_ids"] == ["idle-noise"]
    assert reaches[0]["observed_support_seconds"] == 2.4
    assert reaches[0]["bounded_gap_seconds"] == 0.2
    assert result["metrics"]["merge_count"] == 1
    assert result["metrics"]["pre_gate_merged_fragment_count"] == 1
    assert result["metrics"]["absorbed_segment_count"] == 1


def test_pre_gate_aggregation_recovers_real_support_without_counting_gap():
    result = build_stable_action_events(
        [
            _segment("move-a", "move", 0.0, 0.6),
            _segment("idle-noise", "idle", 0.6, 0.8),
            _segment("move-b", "move", 0.8, 1.4),
        ]
    )
    moves = [
        item
        for item in result["stable_events"]
        if item["event_kind"] == "stable_action" and item["action"] == "move"
    ]

    assert len(moves) == 1
    assert moves[0]["duration_seconds"] == 1.4
    assert moves[0]["observed_support_seconds"] == 1.2
    assert moves[0]["bounded_gap_seconds"] == 0.2
    assert moves[0]["maximum_bounded_gap_seconds"] == 0.2
    assert moves[0]["bounded_uncertain_gaps"] == [
        {
            "start_time": 0.6,
            "end_time": 0.8,
            "duration_seconds": 0.2,
            "source_segment_ids": ["idle-noise"],
            "reason": "short_same_lane_noise_gap",
        }
    ]
    assert moves[0]["source_segment_ids"] == ["move-a", "move-b"]
    assert moves[0]["bounded_gap_source_segment_ids"] == ["idle-noise"]


def test_span_above_threshold_does_not_pass_when_observed_support_is_short():
    result = build_stable_action_events(
        [
            _segment("move-a", "move", 0.0, 0.5),
            _segment("idle-noise", "idle", 0.5, 0.7),
            _segment("move-b", "move", 0.7, 1.2),
        ]
    )

    assert not [
        item
        for item in result["stable_events"]
        if item["event_kind"] == "stable_action" and item["action"] == "move"
    ]
    suppressed_moves = [
        item
        for item in result["suppressed_events"]
        if item.get("support_group_id") and item["source_segment_ids"] == [
            "move-a",
            "move-b",
        ]
    ]
    assert len(suppressed_moves) == 1
    assert suppressed_moves[0]["duration_seconds"] == 1.2
    assert suppressed_moves[0]["observed_support_seconds"] == 1.0
    assert suppressed_moves[0]["stabilization_reason"] == (
        "below_phase_b_observed_support_gate"
    )
    assert result["metrics"]["pre_gate_merged_fragment_count"] == 1
    assert result["metrics"]["suppressed_pre_gate_merged_fragment_count"] == 1
    assert result["metrics"]["merged_fragment_count"] == 0


def test_person_and_epoch_remain_hard_merge_partitions():
    result = build_stable_action_events(
        [
            _segment("p1-e1", "reach", 0.0, 1.2),
            _segment("p1-e2", "reach", 1.3, 2.5, epoch=2),
            _segment("p2-e1", "reach", 2.6, 3.8, person="person-002"),
        ]
    )
    assert len(
        [item for item in result["stable_events"] if item["action"] == "reach"]
    ) == 3
    assert result["metrics"]["cross_identity_or_epoch_merge_count"] == 0
