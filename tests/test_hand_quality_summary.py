from __future__ import annotations

from src.multimodal_pipeline import _hand_summary


def test_hand_summary_keeps_observation_quality_and_validation_axes_separate() -> None:
    records = [
        {
            "frame_index": 10,
            "anatomical_side": "left",
            "backend_state": "available",
            "observation_state": "detected",
            "quality_state": "qualified",
            "validation_state": "not_reviewed",
            "action_feature_eligible": True,
            "association_checks": {"warnings": []},
        },
        {
            "frame_index": 10,
            "anatomical_side": "right",
            "backend_state": "available",
            "observation_state": "uncertain",
            "quality_state": "association_uncertain",
            "validation_state": "review_required",
            "action_feature_eligible": False,
            "association_checks": {
                "warnings": ["duplicate_hand_candidate_across_sides"],
            },
        },
        {
            "frame_index": 11,
            "anatomical_side": "left",
            "backend_state": "available",
            "observation_state": "missing",
            "quality_state": "not_observed",
            "validation_state": "not_evaluable",
            "action_feature_eligible": False,
            "association_checks": {"warnings": []},
        },
    ]

    summary = _hand_summary(records, 2, backend_enabled=True)

    assert summary["hand_detected_frame_count"] == 1
    assert summary["hand_uncertain_frame_count"] == 0
    assert summary["hand_missing_frame_count"] == 1
    assert summary["hand_backend_state_counts"]["available"] == 3
    assert summary["hand_quality_state_counts"] == {
        "qualified": 1,
        "association_uncertain": 1,
        "insufficient_geometry": 0,
        "not_observed": 1,
        "lost": 0,
        "unknown": 0,
    }
    assert summary["hand_validation_state_counts"] == {
        "not_reviewed": 1,
        "review_required": 1,
        "not_evaluable": 1,
        "unknown": 0,
    }
    assert summary["hand_action_feature_eligible_observation_count"] == 1
    assert summary["hand_action_feature_eligible_frame_count"] == 1


def test_warning_record_never_becomes_eligible_in_summary_fixture() -> None:
    record = {
        "frame_index": 3,
        "anatomical_side": "right",
        "backend_state": "available",
        "observation_state": "uncertain",
        "quality_state": "association_uncertain",
        "validation_state": "review_required",
        "action_feature_eligible": False,
        "association_checks": {
            "warnings": ["model_hand_is_closer_to_opposite_body_wrist"],
        },
    }

    summary = _hand_summary([record], 1, backend_enabled=True)

    assert summary["hand_action_feature_eligible_observation_count"] == 0
    assert summary["hand_action_feature_eligible_frame_count"] == 0
    assert summary["association_warning_count"] == 1
