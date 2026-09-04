from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.action_segmentation import build_stable_action_events
from src.contracts import ProcessStep
from src.interaction_fusion import InteractionFusionEngine
from src.object_perception import NotConfiguredObjectPerception
from src.process_reasoning import ProcessReasoner
from src.tracking import AnonymousPersonLock


def _segment(
    segment_id: str,
    action: str,
    start: float,
    end: float,
    *,
    person: str = "person-001",
    epoch: int = 1,
    track_state: str = "tracked",
    lock_state: str = "tracked",
) -> dict[str, object]:
    lost = action == "lost"
    return {
        "segment_id": segment_id,
        "action": action,
        "action_name": action,
        "person_ref": person,
        "lock_epoch": epoch,
        "side": "left",
        "anatomical_side": "left",
        "start_time": start,
        "end_time": end,
        "duration_seconds": end - start,
        "status": "lost" if lost else "proposed",
        "observation_state": "lost" if lost else "detected",
        "track_state": "lost" if lost else track_state,
        "lock_state": "lost" if lost else lock_state,
        "detected_ratio": 0.0 if lost else 1.0,
        "predicted_ratio": 0.0,
        "interpolated_ratio": 0.0,
        "missing_ratio": 1.0 if lost else 0.0,
        "required_joints_reliable": not lost,
        "direction_clear": True,
        "raw_lost": lost,
        "training_eligible": False,
        "source_segment_ids": [segment_id],
    }


def test_lost_is_a_stable_hard_boundary_not_a_normal_action():
    result = build_stable_action_events(
        [
            _segment("s1", "move", 0.0, 1.2),
            _segment("lost", "lost", 1.2, 1.5),
            _segment("s2", "move", 1.5, 2.8),
        ]
    )
    actions = [event["action"] for event in result["stable_events"]]
    assert actions == ["move", "lost", "move"]
    assert result["stable_events"][1]["status"] == "lost"


def test_short_fragment_is_suppressed_from_stable_events():
    result = build_stable_action_events(
        [
            _segment("tiny", "move", 0.0, 0.2),
            _segment("stable", "idle", 0.2, 1.5),
        ]
    )
    assert [event["action"] for event in result["stable_events"]] == ["idle"]
    assert any(
        item["source_segment_ids"] == ["tiny"]
        for item in result["suppressed_events"]
    )


def test_same_action_merges_only_inside_same_person_and_epoch():
    result = build_stable_action_events(
        [
            _segment("a", "reach", 0.0, 1.0),
            _segment("b", "reach", 1.1, 2.1),
            _segment("c", "reach", 2.2, 3.3, epoch=2),
            _segment("d", "reach", 3.4, 4.5, person="person-002", epoch=1),
        ]
    )
    events = result["stable_events"]
    assert len(events) == 3
    assert events[0]["source_segment_ids"] == ["a", "b"]
    assert events[1]["lock_epoch"] == 2
    assert events[2]["person_ref"] == "person-002"


def test_action_event_always_references_pose_segments():
    event = build_stable_action_events(
        [_segment("source-pose-segment", "lift", 0.0, 1.2)]
    )["stable_events"][0]
    assert event["source_segment_ids"] == ["source-pose-segment"]
    assert event["training_eligible"] is False
    assert event["confirmation_status"] == "unconfirmed"


def _track_result(track_id: int, *, state: str = "tracked"):
    return SimpleNamespace(
        track_id=track_id,
        state=state,
        lock_state=state,
        smoothed_pose=object() if state == "tracked" else None,
        detection=object() if state == "tracked" else None,
        candidate_scores=[{"track_id": track_id}],
    )


def test_person_switch_is_exposed_and_requires_manual_relock():
    lock = AnonymousPersonLock()
    initial = lock.consume_result(_track_result(1))
    changed = lock.consume_result(_track_result(2))
    assert initial.person_ref == "person-001"
    assert initial.lock_epoch == 1
    assert changed.switch_exposed is True
    assert changed.lock_state == "awaiting_manual_relock"
    assert changed.track_state == "lost"
    assert changed.usable_pose is False
    assert changed.person_ref == initial.person_ref


def test_manual_relock_starts_new_person_ref_and_epoch():
    lock = AnonymousPersonLock()
    lock.consume_result(_track_result(1))
    lock.consume_result(_track_result(2))
    lock.confirm_relock()
    confirmed = lock.consume_result(_track_result(2))
    assert confirmed.person_ref == "person-002"
    assert confirmed.lock_epoch == 2
    assert confirmed.usable_pose is True


def test_object_model_not_configured_returns_unavailable_without_tracks():
    output = NotConfiguredObjectPerception().analyze([])
    assert output.state.status == "unavailable"
    assert "not_configured" in output.state.reason
    assert output.object_tracks == []


def test_interaction_without_real_object_evidence_generates_nothing():
    output = InteractionFusionEngine().derive(
        pose_frames=[{"keypoints": []}],
        object_tracks=[],
        object_layer_status="unavailable",
    )
    assert output.state.status == "unavailable"
    assert output.interaction_events == []


def test_process_missing_evidence_stays_unavailable_and_empty():
    output = ProcessReasoner().infer(
        pose_action_events=[{"action": "reach"}],
        interaction_events=[],
        temporal_action_candidates=[],
        required_layers_available=False,
    )
    assert output.state.status == "unavailable"
    assert output.process_steps == []


def test_process_step_without_evidence_cannot_be_proposed():
    step = ProcessStep(
        process_step_id="p1",
        process_name="candidate step",
        involved_object_classes=[],
        source_action_event_ids=[],
        source_interaction_ids=[],
        start_time=0.0,
        end_time=1.0,
        predecessor_step_ids=[],
        completion_evidence=[],
        confidence=0.2,
        status="proposed",
        review_state="pending",
    )
    with pytest.raises(ValueError, match="traceable evidence"):
        step.validate()


def test_model_prediction_cannot_become_training_truth_automatically():
    step = ProcessStep(
        process_step_id="p1",
        process_name="candidate step",
        involved_object_classes=["part"],
        source_action_event_ids=["a1"],
        source_interaction_ids=[],
        start_time=0.0,
        end_time=1.0,
        predecessor_step_ids=[],
        completion_evidence=["a1"],
        confidence=0.4,
        status="proposed",
        review_state="pending",
        training_eligible=True,
    )
    with pytest.raises(ValueError, match="explicitly reviewed"):
        step.validate()
