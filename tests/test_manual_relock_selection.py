"""Selectable Camera relock contract tests using synthetic state fixtures only."""

from __future__ import annotations

import queue
import time
from pathlib import Path

import numpy as np
import pytest

from src.action_segmentation import CausalCoarseActionClassifier
from src.camera.controller import CameraController
from src.camera.contracts import CameraState
from src.camera.live_analysis import LiveFrameAnalyzer
from src.legacy_pose.manual_selection import (
    ManualSelectionSeed,
    candidate_at_point,
)
from src.legacy_pose.pose_postprocess import PoseDetection
from src.tracking.anonymous_lock import AnonymousPersonLock
from src.web.resource_coordinator import AnalysisResourceCoordinator


ROOT = Path.cwd()


def detection(x1: float = 10, x2: float = 50) -> PoseDetection:
    points = np.zeros((17, 3), dtype=np.float32)
    points[:, 0] = np.linspace(x1 + 4, x2 - 4, 17)
    points[:, 1] = np.linspace(12, 74, 17)
    points[:, 2] = 0.9
    return PoseDetection(
        bbox=np.asarray([x1, 8, x2, 80], dtype=np.float32),
        confidence=0.91,
        keypoints=points,
    )


def candidate_record(frame_index: int = 12) -> dict:
    item = detection()
    torso = item.keypoints[[5, 6, 11, 12]].astype(float)
    return {
        "candidate_id": "camera-candidate-001",
        "bbox": item.bbox.astype(float).tolist(),
        "center": item.center.astype(float).tolist(),
        "size": (item.bbox[2:] - item.bbox[:2]).astype(float).tolist(),
        "torso_keypoints": torso.tolist(),
        "normalized_bbox": [0.1, 0.1, 0.5, 0.9],
        "normalized_torso_keypoints": (
            torso / np.asarray([100, 100, 1], dtype=np.float32)
        ).tolist(),
        "confidence": item.confidence,
        "source_frame_index": frame_index,
        "timestamp": 1.5,
        "source_width": 100,
        "source_height": 100,
        "mirror_horizontal": False,
        "candidate_fingerprint": "a" * 64,
    }


def bare_controller() -> CameraController:
    value = CameraController(ROOT, AnalysisResourceCoordinator())
    value._session_id = "usb_test"
    value._state = CameraState.LIVE
    value._command_queue = queue.Queue(maxsize=4)
    return value


def decorated_candidate(value: CameraController) -> dict:
    frame_item = {
        "kind": "frame",
        "state": "live",
        "session_id": value._session_id,
        "sequence": 7,
        "width": 100,
        "height": 100,
        "timestamp": 1.5,
        "evidence": {
            "frame": {"anonymous_candidates": [candidate_record()]},
            "hand_pose_frames": [],
        },
        "metrics": {},
    }
    result = value._decorate_frame_candidates(frame_item)
    value._latest = result
    return result["evidence"]["frame"]["anonymous_candidates"][0]


def test_public_candidate_protocol_is_opaque_and_frame_bound():
    value = bare_controller()
    candidate = decorated_candidate(value)
    assert set(candidate) >= {
        "session_id",
        "frame_sequence",
        "candidate_id",
        "candidate_token",
        "bbox",
        "confidence",
        "expiry",
        "source_width",
        "source_height",
        "mirror_horizontal",
    }
    assert "track_id" not in candidate
    assert candidate["session_id"] == "usb_test"
    assert candidate["frame_sequence"] == 7
    assert len(candidate["candidate_token"]) >= 24


def test_confirm_requires_exact_session_sequence_and_opaque_token():
    value = bare_controller()
    candidate = decorated_candidate(value)
    with pytest.raises(RuntimeError, match="current Camera session"):
        value.confirm_relock(
            session_id="usb_wrong",
            frame_sequence=7,
            candidate_token=candidate["candidate_token"],
        )
    with pytest.raises(RuntimeError, match="requested frame"):
        value.confirm_relock(
            session_id="usb_test",
            frame_sequence=8,
            candidate_token=candidate["candidate_token"],
        )
    result = value.confirm_relock(
        session_id="usb_test",
        frame_sequence=7,
        candidate_token=candidate["candidate_token"],
    )
    command = value._command_queue.get_nowait()
    assert result["selection_queued"] is True
    assert command["command"] == "confirm_relock"
    assert command["selection"]["candidate_fingerprint"] == "a" * 64
    assert "track_id" not in command["selection"]


def test_expired_or_disappeared_candidate_fails_closed():
    value = bare_controller()
    candidate = decorated_candidate(value)
    value._candidate_tokens[candidate["candidate_token"]][
        "expires_at_monotonic"
    ] = time.monotonic() - 1
    with pytest.raises(RuntimeError, match="expired or unknown"):
        value.confirm_relock(
            session_id="usb_test",
            frame_sequence=7,
            candidate_token=candidate["candidate_token"],
        )


def test_cancel_is_explicit_and_does_not_change_or_queue_a_person():
    value = bare_controller()
    candidate = decorated_candidate(value)
    result = value.cancel_relock(
        session_id="usb_test",
        candidate_token=candidate["candidate_token"],
    )
    assert result["selection_cancelled"] is True
    assert result["person_changed"] is False
    assert value._command_queue.empty()


def test_overlapping_candidate_rule_chooses_smallest_hit_box():
    large = {"candidate_id": "large", "bbox": [0, 0, 100, 100]}
    small = {"candidate_id": "small", "bbox": [20, 20, 40, 40]}
    assert candidate_at_point([large, small], 30, 30)["candidate_id"] == "small"


def seed_for(item: PoseDetection) -> ManualSelectionSeed:
    torso = item.keypoints[[5, 6, 11, 12]].astype(float)
    return ManualSelectionSeed(
        candidate_id="camera-candidate-001",
        video_path="local-usb-session:test",
        selection_timestamp=1.5,
        selection_frame_index=12,
        bbox=tuple(float(value) for value in item.bbox),
        center=tuple(float(value) for value in item.center),
        size=tuple(float(value) for value in item.bbox[2:] - item.bbox[:2]),
        torso_keypoints=tuple(tuple(point) for point in torso),
        person_confidence=item.confidence,
        selection_source="manual",
        manual_reselection=True,
        source_width=100,
        source_height=100,
        normalized_bbox=(0.1, 0.08, 0.5, 0.8),
        normalized_torso_keypoints=tuple(
            tuple(point)
            for point in torso / np.asarray([100, 100, 1], dtype=np.float32)
        ),
        camera_backend="local_usb",
        selected_candidate_fingerprint="a" * 64,
    )


def test_exact_manual_candidate_creates_new_person_and_epoch_after_reset():
    item = detection()
    lock = AnonymousPersonLock()
    first = lock.update([item], (100, 100, 3))
    old_ref, old_epoch = first.person_ref, first.lock_epoch
    old_processors = lock.tracker.track_processor_ids
    lock.select_candidate(seed_for(item))
    assert lock.active_track_id is None
    assert lock.awaiting_manual_relock is True
    second = lock.update([item], (100, 100, 3))
    assert second.usable_pose is True
    assert second.person_ref != old_ref
    assert second.lock_epoch == old_epoch + 1
    assert lock.tracker.track_processor_ids != old_processors
    assert lock.switch_events[-1]["event"] == "manual_relock_confirmed"


class ResetProbe:
    def __init__(self) -> None:
        self.called = 0

    def reset(self, **_kwargs) -> None:
        self.called += 1


def test_live_relock_resets_body_hand_and_temporal_state_without_truth_upgrade():
    analyzer = object.__new__(LiveFrameAnalyzer)
    analyzer.session_id = "usb_test"
    analyzer.lock = AnonymousPersonLock()
    analyzer.lock.update([detection()], (100, 100, 3))
    analyzer.hand = ResetProbe()
    analyzer.classifier = CausalCoarseActionClassifier()
    analyzer.frames = [object()]
    analyzer._next_action_due = 2.0
    analyzer._last_action_boundary = ("person-001", 1, "tracked", "tracking")
    analyzer._last_action_payload = {"action": "move"}
    analyzer.manual_relock_events = []
    analyzer._candidate_history = {12: [candidate_record()]}
    event = analyzer.confirm_relock(
        {
            **candidate_record(),
            "session_id": "usb_test",
            "frame_sequence": 7,
        }
    )
    assert analyzer.hand.called == 1
    assert analyzer.frames == []
    assert analyzer._next_action_due is None
    assert analyzer._last_action_boundary is None
    assert analyzer._last_action_payload is None
    assert event["body_smoother_reset"] is True
    assert event["hand_state_reset"] is True
    assert event["temporal_action_state_reset"] is True
    assert event["status"] == "proposed"
    assert event["training_eligible"] is False


def test_original_web_exposes_select_confirm_cancel_without_identity_fields():
    html = (ROOT / "src/web/static/index.html").read_text(encoding="utf-8")
    javascript = (ROOT / "src/web/static/app.js").read_text(encoding="utf-8")
    assert 'id="camera-select-person-button"' in html
    assert 'id="camera-confirm-relock-button"' in html
    assert 'id="camera-cancel-relock-button"' in html
    assert "/api/camera/relock" in javascript
    assert "/api/camera/relock/cancel" in javascript
    assert "candidate_token" in javascript
    assert "track_id:" not in javascript
