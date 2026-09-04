from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.temporal_actions import TemporalActionEngineV3


ROOT = Path(__file__).resolve().parents[1]
FROZEN = {
    "sample_video_A": (
        ROOT
        / "outputs/private_regression"
        / "replay/sample_video_A/candidate/analysis.json"
    ),
    "sample_video_B": (
        ROOT
        / "outputs/private_regression"
        / "replay/sample_video_B/candidate/analysis.json"
    ),
    "sample_video_C": (
        ROOT
        / "outputs/private_regression"
        / "replay/sample_video_C/candidate/analysis.json"
    ),
}


def _payload(name: str = "sample_video_C") -> dict:
    return json.loads(FROZEN[name].read_text(encoding="utf-8"))


@pytest.mark.private_artifacts
def test_v3_builds_three_independent_lanes_per_real_pose_frame() -> None:
    analysis = _payload()
    output = TemporalActionEngineV3().analyze_analysis(analysis)
    assert len(output.feature_frames) == len(analysis["pose_frames"]) * 3
    first_frame = int(analysis["pose_frames"][0]["source_frame_index"])
    first = [
        item
        for item in output.feature_frames
        if item.frame_index == first_frame
    ]
    assert {item.anatomical_side for item in first} == {
        "left",
        "right",
        "bilateral",
    }
    assert len({item.temporal_feature_id for item in first}) == 3


@pytest.mark.private_artifacts
def test_v3_uses_only_qualified_real_hand_records() -> None:
    analysis = _payload()
    output = TemporalActionEngineV3().analyze_analysis(analysis)
    known = {
        hand["hand_pose_id"]: hand
        for hand in analysis["hand_pose_frames"]
    }
    used = [
        feature
        for feature in output.feature_frames
        if feature.hand_features_used
    ]
    assert used
    for feature in used:
        assert feature.hand_feature_state == "qualified"
        assert feature.hand_motion_features
        for hand_id in feature.source_hand_pose_ids:
            hand = known[hand_id]
            assert hand["landmark_count"] == 21
            assert len(hand["landmarks"]) == 21
            assert hand["observation_state"] == "detected"
            assert hand["quality_state"] == "qualified"
            assert hand["action_feature_eligible"] is True
            assert not hand["association_checks"].get("warnings")
            assert not hand["association_checks"].get(
                "duplicate_across_sides"
            )


@pytest.mark.private_artifacts
def test_v3_missing_and_lost_never_carry_hand_geometry() -> None:
    output = TemporalActionEngineV3().analyze_analysis(_payload())
    for feature in output.feature_frames:
        if feature.hand_feature_state in {"not_observed", "lost"}:
            assert feature.hand_features_used is False
            assert feature.hand_motion_features is None
            assert feature.source_hand_pose_ids == []
        if feature.hard_boundary:
            assert feature.hand_features_used is False
            assert feature.object_features_used is False


@pytest.mark.private_artifacts
def test_v3_object_layer_is_unavailable_and_never_invented() -> None:
    output = TemporalActionEngineV3().analyze_analysis(_payload())
    assert output.diagnostics["object_feature_state"] == "unavailable"
    assert output.diagnostics["object_feature_use_count"] == 0
    assert all(
        feature.object_feature_state == "unavailable"
        and feature.object_features_used is False
        and feature.source_object_track_ids == []
        for feature in output.feature_frames
    )
    assert all(
        candidate.object_feature_state == "unavailable"
        and candidate.object_features_used is False
        and candidate.source_object_track_ids == []
        for candidate in output.action_candidates
    )


@pytest.mark.private_artifacts
def test_v3_candidates_do_not_cross_person_epoch_or_lane() -> None:
    output = TemporalActionEngineV3().analyze_analysis(_payload())
    features = {
        feature.temporal_feature_id: feature
        for feature in output.feature_frames
    }
    for candidate in output.action_candidates:
        sources = [
            features[source_id]
            for source_id in candidate.source_temporal_feature_ids
        ]
        assert {
            (
                source.person_ref,
                source.lock_epoch,
                source.anatomical_side,
            )
            for source in sources
        } == {
            (
                candidate.person_ref,
                candidate.lock_epoch,
                candidate.anatomical_side,
            )
        }
        assert candidate.source_segment_ids


@pytest.mark.private_artifacts
def test_v3_normal_candidates_are_at_least_1p2_seconds() -> None:
    engine = TemporalActionEngineV3()
    normal = []
    for name in FROZEN:
        output = engine.analyze_analysis(_payload(name))
        normal.extend(
            item
            for item in output.action_candidates
            if item.evidence_state == "normal"
        )
    assert len(normal) == 4
    assert all(item.duration_seconds >= 1.2 for item in normal)
    assert all(item.status == "proposed" for item in normal)


@pytest.mark.private_artifacts
def test_v3_shadow_keeps_primary_action_events_byte_equivalent() -> None:
    analysis = _payload()
    before = json.dumps(
        analysis["action_events"],
        sort_keys=True,
        separators=(",", ":"),
    )
    TemporalActionEngineV3().analyze_analysis(analysis)
    after = json.dumps(
        analysis["action_events"],
        sort_keys=True,
        separators=(",", ":"),
    )
    assert before == after


def test_v3_safe_error_falls_back_without_shadow_evidence() -> None:
    output = TemporalActionEngineV3().analyze_analysis_safe({})
    assert output.state.status == "unavailable"
    assert output.feature_frames == []
    assert output.action_candidates == []
    assert output.diagnostics["fallback_primary_action_events"] is True
    assert output.diagnostics["accuracy_status"] == "not_evaluable"
    assert output.diagnostics["training_eligible"] is False


@pytest.mark.private_artifacts
def test_v3_frozen_real_inputs_remain_hash_identical() -> None:
    engine = TemporalActionEngineV3()
    manifest = json.loads(
        (ROOT / "outputs/private_regression/fixture_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    for name, path in FROZEN.items():
        expected_hash = manifest["analyses"][name]["sha256"]
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        assert before == expected_hash
        engine.analyze_analysis(_payload(name))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_v3_four_validation_flags_remain_false() -> None:
    config = json.loads(
        (ROOT / "configs/temporal_action_v3.json").read_text(
            encoding="utf-8"
        )
    )
    assert not any(config["validation_flags"].values())
