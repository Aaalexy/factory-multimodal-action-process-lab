from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.contracts import LayerState
from src.schema_validation import SchemaValidationError, validate_instance
from src.temporal_actions import (
    NotConfiguredTemporalActionModel,
    TEMPORAL_ACTION_VOCABULARY,
    TemporalActionCandidate,
    TemporalActionOutput,
    TemporalActionProvider,
    TemporalFeatureFrame,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "a" * 64
PROFILE_SHA = "b" * 64
FROZEN_ANALYSES = {
    "sample_video_A": (
        ROOT
        / "outputs/private_regression/replay"
        / "sample_video_A/candidate/analysis.json"
    ),
    "sample_video_B": (
        ROOT
        / "outputs/private_regression/replay"
        / "sample_video_B/candidate/analysis.json"
    ),
    "sample_video_C": (
        ROOT
        / "outputs/private_regression/replay"
        / "sample_video_C/candidate/analysis.json"
    ),
}


def _feature(**overrides: object) -> TemporalFeatureFrame:
    values: dict[str, object] = {
        "temporal_feature_id": "tf-person-001-e1-left-f10",
        "coarse_action": "move",
        "evidence_state": "normal",
        "person_ref": "person-001",
        "lock_epoch": 1,
        "anatomical_side": "left",
        "frame_index": 10,
        "timestamp": 1.25,
        "source_video_sha256": SOURCE_SHA,
        "recording_group_id": "recording-group-1",
        "rule_version": "temporal_action_rules_v3.0.0",
        "parameter_profile_id": "temporal_action_v3_default_2p5s",
        "parameter_profile_sha256": PROFILE_SHA,
        "body_feature_state": "available",
        "hand_feature_state": "not_observed",
        "object_feature_state": "unavailable",
        "body_features_used": True,
        "hand_features_used": False,
        "object_features_used": False,
        "body_motion_features": {
            "wrist_speed_body_scales_per_second": 0.4,
            "wrist_velocity_xy_body_scales_per_second": [0.8, -0.2],
            "motion_direction": "lateral",
            "detected_ratio": 0.9,
            "required_joints_reliable": True,
        },
        "hand_motion_features": None,
        "object_motion_features": None,
        "source_frame_indices": [10],
        "source_segment_ids": ["pose-segment-1"],
        "source_hand_pose_ids": [],
        "source_object_track_ids": [],
        "source_interaction_ids": [],
        "hard_boundary": False,
        "boundary_reasons": [],
        "status": "proposed",
        "reviewer": None,
        "reviewed_at": None,
        "training_approval": "pending",
        "training_eligible": False,
    }
    values.update(overrides)
    return TemporalFeatureFrame(**values)


def _candidate(**overrides: object) -> TemporalActionCandidate:
    values: dict[str, object] = {
        "temporal_action_candidate_id": "ta-person-001-e1-left-10-18",
        "action": "move",
        "evidence_state": "normal",
        "person_ref": "person-001",
        "lock_epoch": 1,
        "anatomical_side": "left",
        "start_frame_index": 10,
        "end_frame_index": 18,
        "start_time": 1.25,
        "end_time": 2.25,
        "duration_seconds": 1.0,
        "source_video_sha256": SOURCE_SHA,
        "recording_group_id": "recording-group-1",
        "engine_kind": "interpretable_rules",
        "shadow_mode": True,
        "rule_version": "temporal_action_rules_v3.0.0",
        "parameter_profile_id": "temporal_action_v3_default_2p5s",
        "parameter_profile_sha256": PROFILE_SHA,
        "temporal_context_seconds": 2.5,
        "body_feature_state": "available",
        "hand_feature_state": "not_observed",
        "object_feature_state": "unavailable",
        "body_features_used": True,
        "hand_features_used": False,
        "object_features_used": False,
        "source_temporal_feature_ids": ["tf-10", "tf-18"],
        "source_segment_ids": ["pose-segment-1"],
        "source_frame_indices": [10, 18],
        "source_hand_pose_ids": [],
        "source_object_track_ids": [],
        "source_interaction_ids": [],
        "change_point_evidence": [
            {
                "change_type": "start_confirmation",
                "timestamp": 1.25,
                "reason": "body_motion_support_started",
                "source_temporal_feature_ids": ["tf-10"],
            }
        ],
        "hard_boundary": False,
        "boundary_reasons": [],
        "status": "proposed",
        "reviewer": None,
        "reviewed_at": None,
        "training_approval": "pending",
        "training_eligible": False,
    }
    values.update(overrides)
    return TemporalActionCandidate(**values)


def _schema(name: str) -> dict[str, object]:
    return json.loads(
        (ROOT / "schemas" / f"{name}.schema.json").read_text(
            encoding="utf-8"
        )
    )


def test_temporal_feature_contract_accepts_traceable_body_feature() -> None:
    payload = _feature().to_dict()
    assert payload["temporal_feature_id"].startswith("tf-")
    assert payload["body_features_used"] is True
    assert payload["training_eligible"] is False


def test_temporal_candidate_contract_accepts_shadow_rule_candidate() -> None:
    payload = _candidate().to_dict()
    assert payload["shadow_mode"] is True
    assert payload["source_temporal_feature_ids"] == ["tf-10", "tf-18"]
    assert payload["training_eligible"] is False


@pytest.mark.parametrize("action", ["assembly_completed", "", "MOVE"])
def test_temporal_contract_rejects_unknown_action(action: str) -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        _candidate(action=action).validate()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("person_ref", ""),
        ("lock_epoch", -1),
        ("anatomical_side", "model_left"),
        ("source_video_sha256", "not-a-sha"),
        ("parameter_profile_sha256", "A" * 64),
    ],
)
def test_feature_rejects_invalid_identity_or_provenance(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError):
        _feature(**{field: value}).validate()


@pytest.mark.parametrize("timestamp", [-0.1, float("nan"), float("inf")])
def test_feature_rejects_invalid_timestamp(timestamp: float) -> None:
    with pytest.raises(ValueError):
        _feature(timestamp=timestamp).validate()


def test_proposed_feature_requires_actual_body_feature_use() -> None:
    with pytest.raises(ValueError, match="Proposed features"):
        _feature(body_features_used=False).validate()


def test_feature_rejects_unqualified_hand_as_used() -> None:
    with pytest.raises(
        ValueError,
        match="Unqualified Hand|qualified real Hand",
    ):
        _feature(
            hand_features_used=True,
            hand_feature_state="association_uncertain",
            hand_motion_features={"hand_motion_feature": {"speed": 0.1}},
            source_hand_pose_ids=["hand-1"],
        ).validate()


def test_feature_rejects_lost_hand_geometry_even_when_not_used() -> None:
    with pytest.raises(ValueError, match="Unqualified Hand state"):
        _feature(
            hand_feature_state="lost",
            hand_features_used=False,
            hand_motion_features={"hand_motion_feature": {"speed": 0.1}},
        ).validate()


def test_feature_accepts_only_qualified_traceable_hand_use() -> None:
    payload = _feature(
        hand_features_used=True,
        hand_feature_state="qualified",
        hand_motion_features={"hand_motion_feature": {"speed": 0.1}},
        source_hand_pose_ids=["hand-1"],
    ).to_dict()
    assert payload["hand_feature_state"] == "qualified"
    assert payload["source_hand_pose_ids"] == ["hand-1"]


def test_feature_rejects_object_semantic_without_real_lineage() -> None:
    with pytest.raises(ValueError, match="real Object"):
        _feature(coarse_action="carry").validate()


def test_candidate_rejects_object_semantic_without_real_lineage() -> None:
    with pytest.raises(ValueError, match="object-track"):
        _candidate(action="place").validate()


def test_candidate_accepts_object_semantic_only_with_real_lineage() -> None:
    payload = _candidate(
        action="carry",
        object_feature_state="available",
        object_features_used=True,
        source_object_track_ids=["object-track-1"],
        source_interaction_ids=["interaction-1"],
    ).to_dict()
    assert payload["object_features_used"] is True


def test_lost_feature_is_uncertain_hard_boundary_without_geometry() -> None:
    payload = _feature(
        coarse_action="lost",
        evidence_state="lost",
        body_feature_state="lost",
        body_features_used=False,
        body_motion_features=None,
        hard_boundary=True,
        boundary_reasons=["track_state_lost"],
        status="uncertain",
    ).to_dict()
    assert payload["hard_boundary"] is True
    assert payload["hand_features_used"] is False


def test_lost_feature_cannot_cross_without_hard_boundary() -> None:
    with pytest.raises(ValueError, match="lost features"):
        _feature(
            coarse_action="lost",
            evidence_state="lost",
            body_feature_state="lost",
            body_features_used=False,
            body_motion_features=None,
            status="uncertain",
        ).validate()


def test_normal_action_cannot_claim_lost_evidence_without_boundary() -> None:
    with pytest.raises(ValueError, match="requires coarse_action=lost"):
        _feature(
            coarse_action="move",
            evidence_state="lost",
            status="uncertain",
        ).validate()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reviewer", "ai"),
        ("reviewed_at", "2026-07-27T00:00:00Z"),
        ("training_approval", "approved"),
        ("training_eligible", True),
    ],
)
def test_automatic_feature_cannot_claim_human_truth(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match="Automatic temporal"):
        _feature(**{field: value}).validate()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("end_time", 1.0),
        ("duration_seconds", 9.0),
        ("start_frame_index", 19),
        ("temporal_context_seconds", 0.0),
    ],
)
def test_candidate_rejects_invalid_time_or_frame_span(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError):
        _candidate(**{field: value}).validate()


def test_candidate_frame_bounds_cannot_expand_beyond_source_endpoints() -> None:
    with pytest.raises(ValueError, match="equal source frame endpoints"):
        _candidate(
            start_frame_index=0,
            end_frame_index=100,
            source_frame_indices=[10, 18],
        ).validate()


def test_candidate_change_point_must_reference_own_features() -> None:
    with pytest.raises(ValueError, match="lineage"):
        _candidate(
            change_point_evidence=[
                {
                    "change_type": "start_confirmation",
                    "timestamp": 1.25,
                    "reason": "bad_reference",
                    "source_temporal_feature_ids": ["tf-other"],
                }
            ]
        ).validate()


def test_candidate_rejects_unknown_change_point_type() -> None:
    with pytest.raises(ValueError, match="Unsupported change-point"):
        _candidate(
            change_point_evidence=[
                {
                    "change_type": "semantic_guess",
                    "timestamp": 1.25,
                    "reason": "not_a_supported_change_point",
                    "source_temporal_feature_ids": ["tf-10"],
                }
            ]
        ).validate()


def test_hard_boundary_candidate_cannot_use_hand_or_object() -> None:
    with pytest.raises(ValueError, match="Hard-boundary"):
        _candidate(
            action="unknown",
            evidence_state="uncertain",
            status="uncertain",
            hard_boundary=True,
            boundary_reasons=["long_missing"],
            hand_features_used=True,
            hand_feature_state="qualified",
            source_hand_pose_ids=["hand-1"],
        ).validate()


def test_not_configured_provider_is_empty_and_unavailable() -> None:
    provider = NotConfiguredTemporalActionModel()
    output = provider.analyze(
        [{"action_event_id": "existing-body-event"}],
        [{"interaction_id": "must-not-be-consumed"}],
        feature_frames=[_feature()],
    )
    payload = output.to_dict()
    assert payload["state"]["status"] == "unavailable"
    assert payload["action_candidates"] == []
    assert payload["feature_frames"] == []
    assert payload["change_point_candidates"] == []
    assert payload["diagnostics"]["accuracy_status"] == "not_evaluable"


def test_not_configured_provider_satisfies_runtime_protocol() -> None:
    assert isinstance(NotConfiguredTemporalActionModel(), TemporalActionProvider)


def test_unavailable_output_cannot_emit_shadow_evidence() -> None:
    output = TemporalActionOutput(
        state=LayerState(
            layer="temporal_action_model",
            status="unavailable",
            reason="not_configured",
        ),
        feature_frames=[_feature()],
    )
    with pytest.raises(ValueError, match="Unavailable"):
        output.validate()


def test_output_validates_canonical_dict_records() -> None:
    first = _feature(temporal_feature_id="tf-10")
    second = _feature(
        temporal_feature_id="tf-18",
        frame_index=18,
        timestamp=2.25,
        source_frame_indices=[18],
    )
    output = TemporalActionOutput(
        state=LayerState(
            layer="temporal_action_model",
            status="available",
            reason="shadow_contract_test",
            evidence_count=3,
        ),
        feature_frames=[first.to_dict(), second.to_dict()],
        action_candidates=[_candidate().to_dict()],
        known_pose_segment_ids=frozenset({"pose-segment-1"}),
    )
    payload = output.to_dict()
    assert len(payload["feature_frames"]) == 2
    assert len(payload["action_candidates"]) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("person_ref", "person-999"),
        ("lock_epoch", 7),
        ("anatomical_side", "right"),
        ("source_video_sha256", "c" * 64),
        ("parameter_profile_id", "other-profile"),
    ],
)
def test_output_rejects_candidate_crossing_feature_boundary(
    field: str,
    value: object,
) -> None:
    features = [
        _feature(temporal_feature_id="tf-10"),
        _feature(
            temporal_feature_id="tf-18",
            frame_index=18,
            timestamp=2.25,
            source_frame_indices=[18],
        ),
    ]
    output = TemporalActionOutput(
        state=LayerState(
            layer="temporal_action_model",
            status="available",
            reason="boundary_negative_test",
            evidence_count=3,
        ),
        feature_frames=features,
        action_candidates=[_candidate(**{field: value})],
        known_pose_segment_ids=frozenset({"pose-segment-1"}),
    )
    with pytest.raises(ValueError, match="boundary"):
        output.validate()


def test_output_rejects_unknown_feature_lineage() -> None:
    output = TemporalActionOutput(
        state=LayerState(
            layer="temporal_action_model",
            status="available",
            reason="unknown_lineage_test",
            evidence_count=1,
        ),
        feature_frames=[_feature()],
        known_pose_segment_ids=frozenset(),
    )
    with pytest.raises(ValueError, match="unknown Pose segment"):
        output.validate()


def test_output_rejects_candidate_time_extension_without_feature_support() -> None:
    output = TemporalActionOutput(
        state=LayerState(
            layer="temporal_action_model",
            status="available",
            reason="time_extension_negative_test",
            evidence_count=3,
        ),
        feature_frames=[
            _feature(temporal_feature_id="tf-10"),
            _feature(
                temporal_feature_id="tf-18",
                frame_index=18,
                timestamp=2.25,
                source_frame_indices=[18],
            ),
        ],
        action_candidates=[
            _candidate(
                start_time=0.25,
                end_time=2.25,
                duration_seconds=2.0,
            )
        ],
        known_pose_segment_ids=frozenset({"pose-segment-1"}),
    )
    with pytest.raises(ValueError, match="time bounds"):
        output.validate()


@pytest.mark.parametrize(
    "diagnostics",
    [
        {"training_eligible": True},
        {"reviewer": "ai"},
        {"nested": {"reviewed_at": "2026-07-27T00:00:00Z"}},
        {"status": "confirmed"},
        {"review_state": "confirmed"},
    ],
)
def test_output_diagnostics_cannot_hide_truth_claims(
    diagnostics: dict[str, object],
) -> None:
    output = NotConfiguredTemporalActionModel().analyze([], [])
    output.diagnostics = diagnostics
    with pytest.raises(ValueError, match="cannot claim|cannot authorize"):
        output.validate()


def test_output_change_points_cannot_hide_training_truth() -> None:
    output = TemporalActionOutput(
        state=LayerState(
            layer="temporal_action_model",
            status="available",
            reason="truth_claim_negative_test",
            evidence_count=0,
        ),
        change_point_candidates=[
            {"reason": "shadow", "training_eligible": True}
        ],
    )
    with pytest.raises(ValueError, match="cannot authorize training"):
        output.validate()


def test_output_requires_object_and_interaction_ids_to_exist_upstream() -> None:
    feature_values = {
        "coarse_action": "carry",
        "object_feature_state": "available",
        "object_features_used": True,
        "object_motion_features": {"track_motion": {"speed": 0.1}},
        "source_object_track_ids": ["object-track-1"],
        "source_interaction_ids": ["interaction-1"],
    }
    features = [
        _feature(temporal_feature_id="tf-10", **feature_values),
        _feature(
            temporal_feature_id="tf-18",
            frame_index=18,
            timestamp=2.25,
            source_frame_indices=[18],
            **feature_values,
        ),
    ]
    candidate = _candidate(
        action="carry",
        object_feature_state="available",
        object_features_used=True,
        source_object_track_ids=["object-track-1"],
        source_interaction_ids=["interaction-1"],
    )
    output = TemporalActionOutput(
        state=LayerState(
            layer="temporal_action_model",
            status="available",
            reason="object_lineage_negative_test",
            evidence_count=3,
        ),
        feature_frames=features,
        action_candidates=[candidate],
        known_pose_segment_ids=frozenset({"pose-segment-1"}),
    )
    with pytest.raises(ValueError, match="unknown Object track"):
        output.validate()


def test_output_accepts_object_semantic_only_with_known_upstream_ids() -> None:
    feature_values = {
        "coarse_action": "carry",
        "object_feature_state": "available",
        "object_features_used": True,
        "object_motion_features": {"track_motion": {"speed": 0.1}},
        "source_object_track_ids": ["object-track-1"],
        "source_interaction_ids": ["interaction-1"],
    }
    output = TemporalActionOutput(
        state=LayerState(
            layer="temporal_action_model",
            status="available",
            reason="object_lineage_positive_test",
            evidence_count=3,
        ),
        feature_frames=[
            _feature(temporal_feature_id="tf-10", **feature_values),
            _feature(
                temporal_feature_id="tf-18",
                frame_index=18,
                timestamp=2.25,
                source_frame_indices=[18],
                **feature_values,
            ),
        ],
        action_candidates=[
            _candidate(
                action="carry",
                object_feature_state="available",
                object_features_used=True,
                source_object_track_ids=["object-track-1"],
                source_interaction_ids=["interaction-1"],
            )
        ],
        known_pose_segment_ids=frozenset({"pose-segment-1"}),
        known_object_track_ids=frozenset({"object-track-1"}),
        known_interaction_ids=frozenset({"interaction-1"}),
    )
    output.validate()


def test_feature_schema_accepts_typed_positive_example() -> None:
    validate_instance(
        _feature().to_dict(),
        _schema("temporal_feature_frames"),
    )


def test_candidate_schema_accepts_typed_positive_example() -> None:
    validate_instance(
        _candidate().to_dict(),
        _schema("temporal_action_candidates"),
    )


def test_feature_schema_rejects_unqualified_hand_use() -> None:
    payload = _feature().to_dict()
    payload.update(
        {
            "hand_features_used": True,
            "hand_feature_state": "association_uncertain",
            "hand_motion_features": {"hand_motion_feature": {"speed": 0.1}},
            "source_hand_pose_ids": ["hand-1"],
        }
    )
    with pytest.raises(SchemaValidationError):
        validate_instance(payload, _schema("temporal_feature_frames"))


def test_candidate_schema_rejects_carry_without_object_evidence() -> None:
    payload = _candidate().to_dict()
    payload["action"] = "carry"
    with pytest.raises(SchemaValidationError):
        validate_instance(payload, _schema("temporal_action_candidates"))


def test_feature_schema_rejects_carry_without_object_evidence() -> None:
    payload = _feature().to_dict()
    payload["coarse_action"] = "carry"
    with pytest.raises(SchemaValidationError):
        validate_instance(payload, _schema("temporal_feature_frames"))


def test_feature_schema_rejects_lost_hand_geometry() -> None:
    payload = _feature().to_dict()
    payload.update(
        {
            "hand_feature_state": "lost",
            "hand_features_used": False,
            "hand_motion_features": {
                "hand_motion_feature": {"speed": 0.1}
            },
        }
    )
    with pytest.raises(SchemaValidationError):
        validate_instance(payload, _schema("temporal_feature_frames"))


def test_candidate_schema_rejects_normal_action_with_lost_evidence() -> None:
    payload = _candidate().to_dict()
    payload.update(
        {
            "evidence_state": "lost",
            "status": "uncertain",
            "hard_boundary": False,
        }
    )
    with pytest.raises(SchemaValidationError):
        validate_instance(payload, _schema("temporal_action_candidates"))


def test_temporal_config_hash_and_fail_closed_defaults() -> None:
    config = json.loads(
        (ROOT / "configs/temporal_action_v3.json").read_text(
            encoding="utf-8"
        )
    )
    canonical = json.dumps(
        config["parameter_payload"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == (
        config["parameter_profile"]["parameter_profile_sha256"]
    )
    assert config["engine"]["shadow_mode"] is True
    assert config["engine"]["publishes_primary_action_events"] is False
    assert (
        config["parameter_payload"]["hand_feature_policy"]["eligibility"]
        == "qualified_only"
    )
    assert (
        config["parameter_payload"]["object_feature_policy"]["default_state"]
        == "unavailable"
    )
    assert not any(config["validation_flags"].values())


def test_temporal_vocabulary_is_exactly_bounded() -> None:
    assert TEMPORAL_ACTION_VOCABULARY == {
        "idle",
        "reach",
        "retract",
        "lift",
        "lower",
        "move",
        "carry",
        "place",
        "hold",
        "release",
        "rotate",
        "push",
        "pull",
        "transition",
        "unknown",
        "lost",
    }


@pytest.mark.private_artifacts
def test_frozen_three_window_analysis_hashes_and_core_counts_are_unchanged() -> None:
    manifest = json.loads(
        (ROOT / "outputs/private_regression/fixture_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    totals = {
        "pose_frames": 0,
        "hand_pose_frames": 0,
        "pose_segments": 0,
        "action_events": 0,
        "evidence_timeline": 0,
    }
    for clip_id, path in FROZEN_ANALYSES.items():
        expected_hash = manifest["analyses"][clip_id]["sha256"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key in totals:
            totals[key] += len(payload.get(key, []))
        assert all(
            event.get("training_eligible") is False
            for event in payload.get("action_events", [])
        )
    assert totals == {
        "pose_frames": 288,
        "hand_pose_frames": 576,
        "pose_segments": 120,
        "action_events": 13,
        "evidence_timeline": 94,
    }
