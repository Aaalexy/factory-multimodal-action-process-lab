from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from scripts.compare_phase_b2_quality import compare_roots


FLAGS = {
    "factory_camera_validated": False,
    "production_action_model_ready": False,
    "external_factory_validated": False,
    "production_process_model_ready": False,
}
EXPECTED_COUNTS = {"detected": 1, "uncertain": 1, "missing": 1}


def _hand(
    identifier: str,
    observation: str,
    *,
    frame_index: int,
    warnings: list[str] | None = None,
) -> dict[str, object]:
    detected = observation == "detected"
    return {
        "hand_pose_id": identifier,
        "person_ref": "person-001",
        "lock_epoch": 1,
        "anatomical_side": "left",
        "frame_index": frame_index,
        "timestamp": frame_index / 8.0,
        "crop_bbox": [10.0, 20.0, 110.0, 120.0],
        "crop_transform": {"offset_x": 10.0, "offset_y": 20.0},
        "landmarks": (
            [
                {
                    "index": index,
                    "x": float(index),
                    "y": float(index + 1),
                    "observation_state": "detected",
                }
                for index in range(21)
            ]
            if detected
            else []
        ),
        "landmark_count": 21 if detected else 0,
        "confidence": None,
        "detection_confidence": None,
        "presence_confidence": None,
        "tracking_confidence": None,
        "raw_confidence_availability": {},
        "observation_state": observation,
        "occlusion": "none" if detected else "unknown",
        "source_video_sha256": "a" * 64,
        "recording_group_id": "recording-group-test",
        "source_model_version": "hand-model-test",
        "runtime_version": "runtime-test",
        "status": "proposed" if detected else "uncertain",
        "reviewer": None,
        "reviewed_at": None,
        "training_approval": "pending",
        "training_eligible": False,
        "model_handedness_label": "Left" if detected else None,
        "model_handedness_score": 0.9 if detected else None,
        "inference_time_ms": 4.0,
        "reason": "fixture",
        "evidence_type": "real_hand_landmarks" if detected else "no_hand_geometry",
        "association_checks": {
            "warnings": list(warnings or []),
            "duplicate_across_sides": False,
        },
    }


def _baseline_payload() -> dict[str, object]:
    hands = [
        _hand("hand-detected", "detected", frame_index=1),
        _hand(
            "hand-warning",
            "uncertain",
            frame_index=2,
            warnings=["model_wrist_too_far_from_own_body_wrist"],
        ),
        _hand("hand-missing", "missing", frame_index=3),
    ]
    return {
        "source_video": {
            "sha256": "a" * 64,
            "analysis_window": {
                "start_time": 0.0,
                "end_time": 1.0,
                "sample_fps": 8.0,
            },
        },
        "hand_pose_frames": hands,
        "action_events": [
            {
                "action_event_id": "action-001",
                "action": "move",
                "person_ref": "person-001",
                "lock_epoch": 1,
                "start_time": 0.0,
                "end_time": 1.0,
                "source_segment_ids": ["segment-001"],
                "training_eligible": False,
            }
        ],
        "runtime": {
            "processed_frame_count": 8,
            "hand_inference_calls": 3,
            "hand_detected_observation_count": 1,
            "hand_uncertain_observation_count": 1,
            "hand_missing_observation_count": 1,
            "association_warning_count": 1,
        },
        "hand_model": {
            "backend": "fixture",
            "version": "fixture",
        },
        "validation_flags": dict(FLAGS),
        "evaluation": {
            "status": "not_evaluable",
            "reason": "No human truth",
        },
    }


def _candidate_payload() -> dict[str, object]:
    payload = deepcopy(_baseline_payload())
    quality = {
        "hand-detected": ("qualified", "not_reviewed", True),
        "hand-warning": (
            "association_uncertain",
            "review_required",
            False,
        ),
        "hand-missing": ("not_observed", "not_evaluable", False),
    }
    for record in payload["hand_pose_frames"]:
        state, validation, eligible = quality[record["hand_pose_id"]]
        record["backend_state"] = "available"
        record["quality_state"] = state
        record["validation_state"] = validation
        record["action_feature_eligible"] = eligible
        record["inference_time_ms"] = 8.0
    payload["runtime"].update(
        {
            "hand_backend_state_counts": {
                "available": 3,
                "error": 0,
                "unavailable": 0,
                "unknown": 0,
            },
            "hand_quality_state_counts": {
                "association_uncertain": 1,
                "insufficient_geometry": 0,
                "lost": 0,
                "not_observed": 1,
                "qualified": 1,
                "unknown": 0,
            },
            "hand_validation_state_counts": {
                "not_evaluable": 1,
                "not_reviewed": 1,
                "review_required": 1,
                "unknown": 0,
            },
            "hand_action_feature_eligible_observation_count": 1,
            "hand_action_feature_eligible_frame_count": 1,
        }
    )
    payload["hand_model"].update(
        {
            "backend_state": "available",
            "backend_mode": "image",
            "quality_gate_version": "hand_quality_gate_v1",
        }
    )
    return payload


def _write_analysis(
    root: Path,
    payload: dict[str, object],
    *,
    clip_id: str = "clip-01",
) -> None:
    path = root / clip_id / "candidate" / "analysis.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _compare(
    tmp_path: Path,
    candidate: dict[str, object],
) -> dict[str, object]:
    baseline_root = tmp_path / "baseline"
    candidate_root = tmp_path / "candidate"
    _write_analysis(baseline_root, _baseline_payload())
    _write_analysis(candidate_root, candidate)
    return compare_roots(
        baseline_root,
        candidate_root,
        expected_observation_counts=EXPECTED_COUNTS,
        expected_warning_count=1,
    )


def test_phase_b2_quality_positive_preserves_core_and_counts(
    tmp_path: Path,
) -> None:
    result = _compare(tmp_path, _candidate_payload())
    assert result["status"] == "passed"
    assert result["evaluation"]["status"] == "not_evaluable"
    clip = result["clips"][0]
    assert clip["hand_core"]["match"] is True
    assert clip["action_event_core"]["match"] is True
    assert clip["quality_metrics"][
        "action_feature_eligible_observation_count"
    ] == 1
    assert all(clip["runtime_quality_checks"].values())
    assert result["summary"]["candidate_hand_counts"][
        "observation_counts"
    ] == EXPECTED_COUNTS


def test_warning_record_cannot_be_marked_action_feature_eligible(
    tmp_path: Path,
) -> None:
    candidate = _candidate_payload()
    warning = next(
        item
        for item in candidate["hand_pose_frames"]
        if item["hand_pose_id"] == "hand-warning"
    )
    warning["action_feature_eligible"] = True
    candidate["runtime"][
        "hand_action_feature_eligible_observation_count"
    ] = 2
    candidate["runtime"]["hand_action_feature_eligible_frame_count"] = 2
    result = _compare(tmp_path, candidate)
    assert result["status"] == "failed"
    categories = {
        item["category"]
        for item in result["clips"][0]["quality_contract_issues"]["items"]
    }
    assert "warning_record_marked_eligible" in categories
    assert "eligible_does_not_match_quality_gate" in categories


def test_original_hand_core_change_fails_sha_gate(tmp_path: Path) -> None:
    candidate = _candidate_payload()
    candidate["hand_pose_frames"][0]["landmarks"][0]["x"] = 999.0
    result = _compare(tmp_path, candidate)
    assert result["status"] == "failed"
    clip = result["clips"][0]
    assert clip["hand_core"]["match"] is False
    assert clip["gates"]["hand_core_sha256_match"] is False
    assert clip["hand_core"]["mismatches"][0]["fields"] == ["landmarks"]


def test_action_event_core_change_fails_sha_gate(tmp_path: Path) -> None:
    candidate = _candidate_payload()
    candidate["action_events"][0]["action"] = "reach"
    result = _compare(tmp_path, candidate)
    assert result["status"] == "failed"
    clip = result["clips"][0]
    assert clip["action_event_core"]["match"] is False
    assert clip["gates"]["action_event_core_sha256_match"] is False
    assert clip["action_event_core"]["mismatches"][0]["fields"] == ["action"]
