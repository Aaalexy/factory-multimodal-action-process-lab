from __future__ import annotations

import json
from pathlib import Path

from scripts.compare_phase_b1 import compare_roots, extract_phase_b1_metrics


def _analysis(*, candidate: bool) -> dict:
    timeline = (
        [
            {
                "start_time": 0.0,
                "end_time": 2.0,
                "evidence_state": "normal",
            }
        ]
        if candidate
        else []
    )
    return {
        "source_video": {
            "sha256": "a" * 64,
            "analysis_window": {"start_time": 0.0, "end_time": 2.0},
        },
        "validation_flags": {
            "factory_camera_validated": False,
            "production_action_model_ready": False,
            "external_factory_validated": False,
            "production_process_model_ready": False,
        },
        "pose_frames": [
            {"track_state": "tracked"},
            {"track_state": "tracked"},
        ],
        "pose_segments": [
            {
                "segment_id": "segment-1",
                "source_video_sha256": "a" * 64,
                "person_ref": "person-1",
                "lock_epoch": 1,
                "anatomical_side": "left",
                "action": "reach",
                "start_time": 0.0,
                "end_time": 2.0,
            }
        ],
        "action_events": [
            {
                "event_kind": "stable_action",
                "action": "reach",
                "start_time": 0.0,
                "end_time": 2.0,
                "duration_seconds": 2.0,
                "observed_support_seconds": 2.0,
                "maximum_bounded_gap_seconds": 0.0,
                "source_segment_ids": ["segment-1"],
                "bounded_gap_source_segment_ids": [],
                "person_ref": "person-1",
                "lock_epoch": 1,
                "anatomical_side": "left",
                "direction_clear": False,
                "display_eligible": True,
            }
        ],
        "evidence_timeline": timeline,
        "evidence_timeline_metrics": {
            "coverage_ratio": 1.0 if candidate else 0.0,
            "uncovered_seconds": 0.0 if candidate else 2.0,
            "normal_action_coverage_ratio": 1.0 if candidate else 0.0,
        },
        "stabilization_metrics": {
            "frame_stabilization": {
                "hard_boundary_frame_count": 0,
                "bounded_uncertain_gap_frame_count": 0,
            }
        },
        "runtime": {
            "processed_frame_count": 2,
            "pose_segment_count": 1,
            "stable_action_event_count": 1,
            "stable_normal_action_count": 1,
            "sub_1s_stable_event_count": 0,
            "suppressed_fragment_count": 0,
            "merged_fragment_count": 0,
            "lost_normal_action_false_positive_count": 0,
            "cross_identity_or_epoch_merge_count": 0,
            "end_to_end_seconds": 1.0,
        },
        "evaluation": {"status": "not_evaluable"},
        "hand_pose_frames": [],
    }


def test_phase_b1_metrics_use_analysis_window_and_union_coverage() -> None:
    metrics = extract_phase_b1_metrics(_analysis(candidate=True))
    assert metrics["stable_normal_action_seconds"] == 2.0
    assert metrics["stable_normal_action_coverage_ratio"] == 1.0
    assert metrics["evidence_timeline_coverage_ratio"] == 1.0
    assert metrics["raw_action_switch_count"] == 0


def test_comparison_requires_same_raw_evidence_and_all_safety_gates(
    tmp_path: Path,
) -> None:
    baseline_root = tmp_path / "baseline"
    candidate_root = tmp_path / "candidate"
    baseline_path = baseline_root / "clip-1" / "after" / "analysis.json"
    candidate_path = candidate_root / "clip-1" / "candidate" / "analysis.json"
    baseline_path.parent.mkdir(parents=True)
    candidate_path.parent.mkdir(parents=True)
    baseline_path.write_text(
        json.dumps(_analysis(candidate=False)),
        encoding="utf-8",
    )
    candidate = _analysis(candidate=True)
    candidate["pose_segments"][0]["evidence_state"] = "normal"
    candidate["pose_segments"][0]["temporal_reason"] = "derived_annotation"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    comparison = compare_roots(baseline_root, candidate_root)

    assert comparison["all_gates_passed"] is True
    assert comparison["clips"][0]["gates"]["raw_pose_segments_unchanged"] is True
    assert comparison["summary"]["candidate"][
        "evidence_timeline_coverage_ratio"
    ] == 1.0
    assert comparison["evaluation"]["status"] == "not_evaluable"


def test_comparison_rejects_changed_raw_pose_action(tmp_path: Path) -> None:
    baseline_root = tmp_path / "baseline"
    candidate_root = tmp_path / "candidate"
    baseline_path = baseline_root / "clip-1" / "after" / "analysis.json"
    candidate_path = candidate_root / "clip-1" / "candidate" / "analysis.json"
    baseline_path.parent.mkdir(parents=True)
    candidate_path.parent.mkdir(parents=True)
    baseline_path.write_text(
        json.dumps(_analysis(candidate=False)),
        encoding="utf-8",
    )
    candidate = _analysis(candidate=True)
    candidate["pose_segments"][0]["action"] = "lower"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    comparison = compare_roots(baseline_root, candidate_root)

    assert comparison["all_gates_passed"] is False
    assert comparison["clips"][0]["gates"]["raw_pose_segments_unchanged"] is False


def test_comparison_rejects_gap_promoted_to_action_support(
    tmp_path: Path,
) -> None:
    baseline_root = tmp_path / "baseline"
    candidate_root = tmp_path / "candidate"
    baseline_path = baseline_root / "clip-1" / "after" / "analysis.json"
    candidate_path = candidate_root / "clip-1" / "candidate" / "analysis.json"
    baseline_path.parent.mkdir(parents=True)
    candidate_path.parent.mkdir(parents=True)
    baseline_path.write_text(
        json.dumps(_analysis(candidate=False)),
        encoding="utf-8",
    )
    candidate = _analysis(candidate=True)
    candidate["action_events"][0]["bounded_gap_source_segment_ids"] = [
        "segment-1"
    ]
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    comparison = compare_roots(baseline_root, candidate_root)

    assert comparison["all_gates_passed"] is False
    assert comparison["clips"][0]["gates"][
        "normal_event_support_and_lineage_are_consistent"
    ] is False
    violations = comparison["clips"][0]["candidate"][
        "normal_event_integrity_violations"
    ]
    assert violations == [
        {
            "action_event_id": None,
            "reasons": ["gap_source_promoted_to_action_support"],
        }
    ]
