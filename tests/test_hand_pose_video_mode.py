from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.hand_pose import (
    HAND_BACKEND_MODE,
    MediaPipeHandLandmarkerBackend,
    MediaPipeHandLandmarkerVideoBackend,
    RealHandPoseBackend,
)
from src.schema_validation import SchemaValidationError, validate_instance


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "hand_pose" / "hand_landmarker.task"
VIDEO_SHA = "a" * 64
FRAME = np.zeros((720, 1000, 3), dtype=np.uint8)
VIDEO_ONLY_RECORD_FIELDS = {
    "backend_timestamp_ms",
    "tracker_session_generation",
    "tracker_reset_reason",
    "roi_center_jump_ratio",
    "roi_scale_change_ratio",
    "input_timestamp_raw",
    "timestamp_validation_state",
}


def _body_keypoints(
    *,
    left_x_offset: float = 0.0,
    close_wrists: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Synthetic contract fixture; it is never emitted as runtime evidence."""

    points = np.full((17, 2), np.nan, dtype=np.float64)
    if close_wrists:
        points[5] = (450.0, 220.0)
        points[7] = (480.0, 300.0)
        points[9] = (500.0, 390.0)
        points[6] = (550.0, 220.0)
        points[8] = (520.0, 300.0)
        points[10] = (510.0, 390.0)
    else:
        points[5] = (220.0 + left_x_offset, 220.0)
        points[7] = (180.0 + left_x_offset, 300.0)
        points[9] = (150.0 + left_x_offset, 390.0)
        points[6] = (780.0, 220.0)
        points[8] = (820.0, 300.0)
        points[10] = (850.0, 390.0)
    statuses = ["missing"] * 17
    for index in (5, 7, 9, 6, 8, 10):
        statuses[index] = "detected"
    return points, statuses


def _landmarks(
    *,
    wrist_x: float = 0.5,
    wrist_y: float = 0.5,
) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            x=wrist_x,
            y=wrist_y,
            z=-index / 100.0,
            visibility=None,
            presence=None,
        )
        for index in range(21)
    ]


def _detected_result(
    *,
    model_handedness: str = "Right",
    wrist_x: float = 0.5,
    wrist_y: float = 0.5,
) -> SimpleNamespace:
    return SimpleNamespace(
        hand_landmarks=[
            _landmarks(wrist_x=wrist_x, wrist_y=wrist_y)
        ],
        handedness=[
            [
                SimpleNamespace(
                    category_name=model_handedness,
                    score=0.99,
                )
            ]
        ],
    )


def _missing_result() -> SimpleNamespace:
    return SimpleNamespace(hand_landmarks=[], handedness=[])


class _FakeVideoSession:
    def __init__(
        self,
        *,
        side: str,
        generation: int,
        owner: "_FakeLandmarkerFactory",
    ) -> None:
        self.side = side
        self.generation = generation
        self.owner = owner
        self.timestamps_ms: list[int] = []
        self.closed = False

    def detect_for_video(
        self,
        image: Any,
        timestamp_ms: int,
    ) -> SimpleNamespace:
        del image
        assert self.closed is False
        self.timestamps_ms.append(int(timestamp_ms))
        return self.owner.next_result(self.side)

    def close(self) -> None:
        self.closed = True


class _FakeLandmarkerFactory:
    def __init__(
        self,
        *,
        left: list[SimpleNamespace] | None = None,
        right: list[SimpleNamespace] | None = None,
    ) -> None:
        self.results = {
            "left": deque(left or []),
            "right": deque(right or []),
        }
        self.sessions: dict[str, list[_FakeVideoSession]] = defaultdict(list)

    def __call__(self, side: str) -> _FakeVideoSession:
        generation = len(self.sessions[side]) + 1
        session = _FakeVideoSession(
            side=side,
            generation=generation,
            owner=self,
        )
        self.sessions[side].append(session)
        return session

    def next_result(self, side: str) -> SimpleNamespace:
        queue = self.results[side]
        return queue.popleft() if queue else _missing_result()


class _ExceptionalVideoSession:
    def __init__(
        self,
        *,
        side: str,
        generation: int,
        detect_error: bool,
        close_error: bool,
    ) -> None:
        self.side = side
        self.generation = generation
        self.detect_error = detect_error
        self.close_error = close_error
        self.timestamps_ms: list[int] = []
        self.detect_calls = 0
        self.close_calls = 0
        self.closed = False

    def detect_for_video(
        self,
        image: Any,
        timestamp_ms: int,
    ) -> SimpleNamespace:
        del image
        assert self.closed is False
        self.detect_calls += 1
        self.timestamps_ms.append(int(timestamp_ms))
        if self.detect_error:
            raise RuntimeError("synthetic_detect_failure")
        return _detected_result()

    def close(self) -> None:
        self.close_calls += 1
        self.closed = True
        if self.close_error:
            raise RuntimeError("synthetic_close_failure")


class _ExceptionalLandmarkerFactory:
    def __init__(
        self,
        *,
        factory_error: bool = False,
        first_detect_error: bool = False,
        first_close_error: bool = False,
    ) -> None:
        self.factory_error = factory_error
        self.first_detect_error = first_detect_error
        self.first_close_error = first_close_error
        self.factory_calls: dict[str, int] = defaultdict(int)
        self.sessions: dict[
            str,
            list[_ExceptionalVideoSession],
        ] = defaultdict(list)

    def __call__(self, side: str) -> _ExceptionalVideoSession:
        self.factory_calls[side] += 1
        generation = self.factory_calls[side]
        if self.factory_error:
            raise RuntimeError("synthetic_factory_failure")
        session = _ExceptionalVideoSession(
            side=side,
            generation=generation,
            detect_error=self.first_detect_error and generation == 1,
            close_error=self.first_close_error and generation == 1,
        )
        self.sessions[side].append(session)
        return session


def _backend(
    factory: Any,
    **overrides: Any,
) -> MediaPipeHandLandmarkerVideoBackend:
    backend = MediaPipeHandLandmarkerVideoBackend(
        MODEL,
        landmarker_factory=factory,
        **overrides,
    )
    # The fake session inspects only timestamps and queued results.  Keeping
    # the crop as a NumPy array makes this suite independent of MediaPipe task
    # construction and any third-party runtime side effects.
    backend._mediapipe_image = lambda crop_rgb: crop_rgb  # type: ignore[method-assign]
    return backend


def _infer(
    backend: MediaPipeHandLandmarkerVideoBackend,
    *,
    frame_index: int,
    timestamp: float,
    points: np.ndarray | None = None,
    statuses: list[str] | None = None,
    person_ref: str = "person-001",
    lock_epoch: int = 1,
    track_state: str = "tracked",
    lock_state: str = "tracked",
) -> list[dict[str, Any]]:
    if points is None and statuses is None:
        points, statuses = _body_keypoints()
    return backend.infer_frame(
        FRAME,
        body_keypoints=points,
        body_keypoint_statuses=statuses,
        person_ref=person_ref,
        lock_epoch=lock_epoch,
        frame_index=frame_index,
        timestamp=timestamp,
        source_video_sha256=VIDEO_SHA,
        recording_group_id="recording-group-test",
        track_state=track_state,
        lock_state=lock_state,
    )


def _by_side(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    assert len(records) == 2
    assert {record["anatomical_side"] for record in records} == {
        "left",
        "right",
    }
    return {str(record["anatomical_side"]): record for record in records}


def _assert_no_geometry(record: dict[str, Any]) -> None:
    assert record["landmarks"] == []
    assert record["landmark_count"] == 0
    assert record["action_feature_eligible"] is False
    assert record["training_eligible"] is False


def test_video_mode_uses_independent_left_and_right_sessions() -> None:
    factory = _FakeLandmarkerFactory(
        left=[_detected_result(model_handedness="Right")],
        right=[_detected_result(model_handedness="Left")],
    )
    backend = _backend(factory)
    try:
        records = _by_side(_infer(backend, frame_index=0, timestamp=0.0))

        assert len(factory.sessions["left"]) == 1
        assert len(factory.sessions["right"]) == 1
        assert factory.sessions["left"][0] is not factory.sessions["right"][0]
        assert factory.sessions["left"][0].timestamps_ms == [0]
        assert factory.sessions["right"][0].timestamps_ms == [0]
        for side, record in records.items():
            assert record["backend_mode"] == "video"
            assert record["anatomical_side"] == side
            assert record["backend_timestamp_ms"] == 0
            assert record["tracker_session_generation"] == 1
            assert record["tracker_reset_reason"] == "initial_session"
            assert record["observation_state"] == "detected"
            assert record["landmark_count"] == 21
            assert (
                record["association_checks"][
                    "model_handedness_used_for_assignment"
                ]
                is False
            )
    finally:
        backend.close()


def test_video_timestamps_are_strictly_monotonic_per_side() -> None:
    factory = _FakeLandmarkerFactory(
        left=[_detected_result() for _ in range(3)],
        right=[_detected_result() for _ in range(3)],
    )
    backend = _backend(factory)
    try:
        _infer(backend, frame_index=0, timestamp=0.0)
        second = _by_side(
            _infer(backend, frame_index=1, timestamp=0.125)
        )
        assert all(
            record["backend_timestamp_ms"] == round(0.125 * 1000)
            for record in second.values()
        )
        invalid = _by_side(_infer(backend, frame_index=2, timestamp=0.120))

        for side, record in invalid.items():
            assert record["backend_timestamp_ms"] == 120
            assert record["observation_state"] == "uncertain"
            assert record["reason"] == "non_monotonic_backend_timestamp"
            assert (
                record["tracker_reset_reason"]
                == "non_monotonic_backend_timestamp"
            )
            _assert_no_geometry(record)
            assert factory.sessions[side][0].timestamps_ms == [0, 125]
            assert factory.sessions[side][0].closed is True

        recovered = _by_side(
            _infer(backend, frame_index=3, timestamp=0.250)
        )
        for side, record in recovered.items():
            assert len(factory.sessions[side]) == 2
            assert factory.sessions[side][1].timestamps_ms == [250]
            assert record["tracker_session_generation"] == 2
            assert record["backend_timestamp_ms"] == 250
            assert (
                record["tracker_reset_reason"]
                == "non_monotonic_backend_timestamp"
            )
    finally:
        backend.close()


def test_timestamp_regression_after_one_side_reset_fails_closed_for_both() -> None:
    factory = _FakeLandmarkerFactory(
        left=[_detected_result(), _detected_result()],
        right=[
            _detected_result(),
            _detected_result(),
            _detected_result(),
        ],
    )
    backend = _backend(factory)
    try:
        _infer(backend, frame_index=0, timestamp=0.0)
        points, statuses = _body_keypoints()
        statuses[9] = "missing"
        reset_frame = _by_side(
            _infer(
                backend,
                frame_index=1,
                timestamp=0.250,
                points=points,
                statuses=statuses,
            )
        )
        assert reset_frame["left"]["tracker_reset_reason"] == (
            "body_guided_roi_unavailable"
        )
        assert factory.sessions["left"][0].closed is True
        assert factory.sessions["right"][0].timestamps_ms == [0, 250]

        regressed = _by_side(
            _infer(backend, frame_index=2, timestamp=0.200)
        )
        for side, record in regressed.items():
            assert record["backend_state"] == "error"
            assert record["observation_state"] == "uncertain"
            assert record["status"] == "uncertain"
            assert record["timestamp"] == pytest.approx(0.200)
            assert record["backend_timestamp_ms"] == 200
            assert record["reason"] == "non_monotonic_backend_timestamp"
            assert record["tracker_reset_reason"] == (
                "non_monotonic_backend_timestamp"
            )
            _assert_no_geometry(record)

        # The reset left side must not silently start a new session at 200 ms.
        assert len(factory.sessions["left"]) == 1
        assert len(factory.sessions["right"]) == 1
        assert factory.sessions["right"][0].closed is True
    finally:
        backend.close()


@pytest.mark.parametrize(
    "timestamp",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive_infinity"),
        pytest.param(float("-inf"), id="negative_infinity"),
        pytest.param(-0.001, id="negative"),
        pytest.param("not-a-timestamp", id="non_numeric"),
    ],
)
def test_invalid_video_timestamp_returns_fail_closed_records(
    timestamp: Any,
) -> None:
    factory = _FakeLandmarkerFactory(
        left=[_detected_result()],
        right=[_detected_result()],
    )
    backend = _backend(factory)
    try:
        records = _by_side(
            _infer(
                backend,
                frame_index=0,
                timestamp=timestamp,
            )
        )
        for record in records.values():
            assert record["backend_mode"] == "video"
            assert record["backend_state"] == "error"
            assert record["observation_state"] == "uncertain"
            assert record["status"] == "uncertain"
            assert record["timestamp"] is None
            assert record["backend_timestamp_ms"] is None
            assert record["reason"] == "invalid_backend_timestamp"
            assert record["tracker_reset_reason"] == (
                "invalid_backend_timestamp"
            )
            assert record["timestamp_validation_state"] == "invalid"
            assert isinstance(record["input_timestamp_raw"], str)
            assert isinstance(record["tracker_session_generation"], int)
            assert record["tracker_session_generation"] >= 0
            _assert_no_geometry(record)
        assert not factory.sessions["left"]
        assert not factory.sessions["right"]
    finally:
        backend.close()


@pytest.mark.parametrize(
    ("person_ref", "lock_epoch"),
    [
        ("person-002", 1),
        ("person-001", 2),
    ],
)
def test_person_or_epoch_change_can_start_a_new_input_clock(
    person_ref: str,
    lock_epoch: int,
) -> None:
    factory = _FakeLandmarkerFactory(
        left=[_detected_result(), _detected_result()],
        right=[_detected_result(), _detected_result()],
    )
    backend = _backend(factory)
    try:
        _infer(backend, frame_index=0, timestamp=10.0)
        restarted = _by_side(
            _infer(
                backend,
                frame_index=1,
                timestamp=0.0,
                person_ref=person_ref,
                lock_epoch=lock_epoch,
            )
        )
        for side, record in restarted.items():
            assert record["backend_state"] == "available"
            assert record["observation_state"] == "detected"
            assert record["timestamp"] == pytest.approx(0.0)
            assert record["backend_timestamp_ms"] == 0
            assert record["tracker_session_generation"] == 2
            assert record["tracker_reset_reason"] == (
                "person_or_lock_epoch_changed"
            )
            assert record["person_ref"] == person_ref
            assert record["lock_epoch"] == lock_epoch
            assert len(factory.sessions[side]) == 2
            assert factory.sessions[side][0].closed is True
            assert factory.sessions[side][1].timestamps_ms == [0]
    finally:
        backend.close()


@pytest.mark.parametrize(
    ("person_ref", "lock_epoch"),
    [
        ("person-002", 1),
        ("person-001", 2),
    ],
)
def test_person_or_epoch_change_recreates_both_side_sessions(
    person_ref: str,
    lock_epoch: int,
) -> None:
    factory = _FakeLandmarkerFactory(
        left=[_detected_result(), _missing_result()],
        right=[_detected_result(), _missing_result()],
    )
    backend = _backend(factory)
    try:
        before = _by_side(_infer(backend, frame_index=0, timestamp=0.0))
        after = _by_side(
            _infer(
                backend,
                frame_index=1,
                timestamp=0.125,
                person_ref=person_ref,
                lock_epoch=lock_epoch,
            )
        )

        for side in ("left", "right"):
            assert factory.sessions[side][0].closed is True
            assert len(factory.sessions[side]) == 2
            assert after[side]["tracker_session_generation"] == 2
            assert (
                after[side]["tracker_reset_reason"]
                == "person_or_lock_epoch_changed"
            )
            assert after[side]["person_ref"] == person_ref
            assert after[side]["lock_epoch"] == lock_epoch
            assert before[side]["hand_pose_id"] != after[side]["hand_pose_id"]
            assert after[side]["observation_state"] == "missing"
            _assert_no_geometry(after[side])
    finally:
        backend.close()


@pytest.mark.parametrize(
    ("track_state", "lock_state"),
    [
        ("lost", "tracked"),
        ("off_frame", "tracked"),
        ("temporarily_lost", "tracked"),
        ("tracked", "lost"),
        ("tracked", "awaiting_manual_relock"),
        ("tracked", "unlocked"),
    ],
)
def test_lost_and_relock_boundaries_reset_and_never_carry_geometry(
    track_state: str,
    lock_state: str,
) -> None:
    factory = _FakeLandmarkerFactory(
        left=[_detected_result(), _missing_result()],
        right=[_detected_result(), _missing_result()],
    )
    backend = _backend(factory)
    try:
        _infer(backend, frame_index=0, timestamp=0.0)
        boundary = _by_side(
            _infer(
                backend,
                frame_index=1,
                timestamp=0.125,
                track_state=track_state,
                lock_state=lock_state,
            )
        )

        for side, record in boundary.items():
            expected_reset_reason = (
                f"hard_boundary:{track_state}:{lock_state}"
            )
            assert len(factory.sessions[side]) == 1
            assert factory.sessions[side][0].timestamps_ms == [0]
            assert factory.sessions[side][0].closed is True
            assert record["observation_state"] == "lost"
            assert record["tracker_reset_reason"] == expected_reset_reason
            _assert_no_geometry(record)

        recovered = _by_side(
            _infer(backend, frame_index=2, timestamp=0.250)
        )
        for side, record in recovered.items():
            assert len(factory.sessions[side]) == 2
            assert record["tracker_session_generation"] == 2
            assert record["tracker_reset_reason"] == expected_reset_reason
            assert record["observation_state"] == "missing"
            _assert_no_geometry(record)
    finally:
        backend.close()


def test_nontracked_frame_resets_video_sessions_fail_closed() -> None:
    factory = _FakeLandmarkerFactory(
        left=[_detected_result(), _missing_result()],
        right=[_detected_result(), _missing_result()],
    )
    backend = _backend(factory)
    try:
        _infer(backend, frame_index=0, timestamp=0.0)
        uncertain = _by_side(
            _infer(
                backend,
                frame_index=1,
                timestamp=0.125,
                track_state="uncertain",
            )
        )
        for side, record in uncertain.items():
            assert factory.sessions[side][0].closed is True
            assert record["observation_state"] == "uncertain"
            assert (
                record["tracker_reset_reason"]
                == "body_tracking_not_reliable:uncertain"
            )
            _assert_no_geometry(record)

        recovered = _by_side(
            _infer(backend, frame_index=2, timestamp=0.250)
        )
        for side, record in recovered.items():
            assert len(factory.sessions[side]) == 2
            assert record["tracker_session_generation"] == 2
            assert (
                record["tracker_reset_reason"]
                == "body_tracking_not_reliable:uncertain"
            )
            assert record["observation_state"] == "missing"
            _assert_no_geometry(record)
    finally:
        backend.close()


def test_missing_roi_resets_only_that_anatomical_side() -> None:
    factory = _FakeLandmarkerFactory(
        left=[_detected_result(), _detected_result()],
        right=[
            _detected_result(),
            _detected_result(),
            _detected_result(),
        ],
    )
    backend = _backend(factory)
    try:
        _infer(backend, frame_index=0, timestamp=0.0)
        points, statuses = _body_keypoints()
        statuses[9] = "missing"
        middle = _by_side(
            _infer(
                backend,
                frame_index=1,
                timestamp=0.125,
                points=points,
                statuses=statuses,
            )
        )

        assert middle["left"]["observation_state"] == "missing"
        assert (
            middle["left"]["tracker_reset_reason"]
            == "body_guided_roi_unavailable"
        )
        _assert_no_geometry(middle["left"])
        assert factory.sessions["left"][0].closed is True
        assert len(factory.sessions["left"]) == 1
        assert middle["right"]["tracker_session_generation"] == 1
        assert factory.sessions["right"][0].closed is False
        assert factory.sessions["right"][0].timestamps_ms == [0, 125]

        recovered = _by_side(
            _infer(backend, frame_index=2, timestamp=0.250)
        )
        assert len(factory.sessions["left"]) == 2
        assert recovered["left"]["tracker_session_generation"] == 2
        assert (
            recovered["left"]["tracker_reset_reason"]
            == "body_guided_roi_unavailable"
        )
        assert factory.sessions["left"][1].timestamps_ms == [250]
        assert len(factory.sessions["right"]) == 1
        assert recovered["right"]["tracker_session_generation"] == 1
        assert factory.sessions["right"][0].timestamps_ms == [0, 125, 250]
    finally:
        backend.close()


def test_model_missing_after_detection_never_reuses_old_points() -> None:
    factory = _FakeLandmarkerFactory(
        left=[_detected_result(), _missing_result()],
        right=[_missing_result(), _missing_result()],
    )
    backend = _backend(
        factory,
        maximum_consecutive_model_missing_frames=3,
    )
    try:
        before = _by_side(_infer(backend, frame_index=0, timestamp=0.0))
        after = _by_side(_infer(backend, frame_index=1, timestamp=0.125))

        assert before["left"]["landmark_count"] == 21
        assert after["left"]["observation_state"] == "missing"
        assert after["left"]["tracker_session_generation"] == 1
        assert after["left"]["backend_timestamp_ms"] == 125
        _assert_no_geometry(after["left"])
    finally:
        backend.close()


def test_dynamic_roi_maps_landmarks_with_the_current_transform() -> None:
    factory = _FakeLandmarkerFactory(
        left=[_detected_result(), _detected_result()],
        right=[_detected_result(), _detected_result()],
    )
    backend = _backend(factory)
    try:
        first = _by_side(_infer(backend, frame_index=0, timestamp=0.0))
        moved_points, moved_statuses = _body_keypoints(left_x_offset=120.0)
        second = _by_side(
            _infer(
                backend,
                frame_index=1,
                timestamp=0.125,
                points=moved_points,
                statuses=moved_statuses,
            )
        )

        first_transform = first["left"]["crop_transform"]
        second_transform = second["left"]["crop_transform"]
        first_expected_x = (
            first_transform["x_offset"] + 0.5 * first_transform["x_scale"]
        )
        second_expected_x = (
            second_transform["x_offset"] + 0.5 * second_transform["x_scale"]
        )
        assert first["left"]["landmarks"][0]["x"] == pytest.approx(
            first_expected_x
        )
        assert second["left"]["landmarks"][0]["x"] == pytest.approx(
            second_expected_x
        )
        assert second_expected_x > first_expected_x + 100.0
        assert first["left"]["crop_bbox"] != second["left"]["crop_bbox"]
    finally:
        backend.close()


def test_roi_transform_discontinuity_resets_only_the_affected_side() -> None:
    factory = _FakeLandmarkerFactory(
        left=[_detected_result() for _ in range(3)],
        right=[_detected_result() for _ in range(3)],
    )
    backend = _backend(
        factory,
        maximum_roi_center_jump_ratio=0.5,
        maximum_roi_scale_change_ratio=0.5,
    )
    try:
        first = _by_side(_infer(backend, frame_index=0, timestamp=0.0))
        for record in first.values():
            assert record["roi_center_jump_ratio"] is None
            assert record["roi_scale_change_ratio"] is None
            assert record["tracker_session_generation"] == 1

        small_points, small_statuses = _body_keypoints(left_x_offset=8.0)
        small = _by_side(
            _infer(
                backend,
                frame_index=1,
                timestamp=0.125,
                points=small_points,
                statuses=small_statuses,
            )
        )
        for record in small.values():
            assert isinstance(record["roi_center_jump_ratio"], float)
            assert isinstance(record["roi_scale_change_ratio"], float)
            assert record["roi_center_jump_ratio"] <= 0.5
            assert record["roi_scale_change_ratio"] <= 0.5
            assert record["tracker_session_generation"] == 1
            assert record["tracker_reset_reason"] is None
        assert len(factory.sessions["left"]) == 1
        assert len(factory.sessions["right"]) == 1

        large_points, large_statuses = _body_keypoints(left_x_offset=300.0)
        large = _by_side(
            _infer(
                backend,
                frame_index=2,
                timestamp=0.250,
                points=large_points,
                statuses=large_statuses,
            )
        )
        left = large["left"]
        right = large["right"]
        assert left["roi_center_jump_ratio"] > 0.5
        assert isinstance(left["roi_scale_change_ratio"], float)
        assert left["tracker_reset_reason"] == "roi_transform_discontinuity"
        assert left["tracker_session_generation"] == 2
        assert len(factory.sessions["left"]) == 2
        assert factory.sessions["left"][0].closed is True
        assert factory.sessions["left"][1].timestamps_ms == [250]

        assert right["roi_center_jump_ratio"] <= 0.5
        assert right["roi_scale_change_ratio"] <= 0.5
        assert right["tracker_reset_reason"] is None
        assert right["tracker_session_generation"] == 1
        assert len(factory.sessions["right"]) == 1
        assert factory.sessions["right"][0].closed is False
        assert factory.sessions["right"][0].timestamps_ms == [0, 125, 250]
    finally:
        backend.close()


def test_overlapping_side_rois_mark_duplicates_ineligible() -> None:
    points, statuses = _body_keypoints(close_wrists=True)
    factory = _FakeLandmarkerFactory(
        left=[_detected_result()],
        right=[_detected_result()],
    )
    backend = _backend(factory)
    try:
        records = _by_side(
            _infer(
                backend,
                frame_index=0,
                timestamp=0.0,
                points=points,
                statuses=statuses,
            )
        )

        for record in records.values():
            checks = record["association_checks"]
            assert checks["duplicate_across_sides"] is True
            assert "duplicate_hand_candidate_across_sides" in checks["warnings"]
            assert record["observation_state"] == "uncertain"
            assert record["quality_state"] == "association_uncertain"
            assert record["validation_state"] == "review_required"
            assert record["action_feature_eligible"] is False
            assert record["training_eligible"] is False
    finally:
        backend.close()


def test_factory_exception_returns_empty_error_records() -> None:
    factory = _ExceptionalLandmarkerFactory(factory_error=True)
    backend = _backend(factory)
    try:
        records = _by_side(_infer(backend, frame_index=0, timestamp=0.0))
        for side, record in records.items():
            assert factory.factory_calls[side] == 1
            assert not factory.sessions[side]
            assert record["backend_state"] == "error"
            assert record["observation_state"] == "uncertain"
            assert record["reason"] == "inference_error:RuntimeError"
            assert record["tracker_reset_reason"] == (
                "inference_error:RuntimeError"
            )
            _assert_no_geometry(record)
    finally:
        backend.close()


def test_detect_exception_closes_session_and_never_reuses_it() -> None:
    factory = _ExceptionalLandmarkerFactory(first_detect_error=True)
    backend = _backend(factory)
    try:
        failed = _by_side(_infer(backend, frame_index=0, timestamp=0.0))
        for side, record in failed.items():
            assert record["backend_state"] == "error"
            assert record["observation_state"] == "uncertain"
            assert record["reason"] == "inference_error:RuntimeError"
            _assert_no_geometry(record)
            assert len(factory.sessions[side]) == 1
            assert factory.sessions[side][0].detect_calls == 1
            assert factory.sessions[side][0].close_calls == 1
            assert factory.sessions[side][0].closed is True

        recovered = _by_side(
            _infer(backend, frame_index=1, timestamp=0.125)
        )
        for side, record in recovered.items():
            assert len(factory.sessions[side]) == 2
            assert factory.sessions[side][0] is not factory.sessions[side][1]
            assert factory.sessions[side][0].detect_calls == 1
            assert factory.sessions[side][1].detect_calls == 1
            assert record["backend_state"] == "available"
            assert record["observation_state"] == "detected"
            assert record["tracker_session_generation"] == 2
            assert record["tracker_reset_reason"] == (
                "inference_error:RuntimeError"
            )
    finally:
        backend.close()


def test_close_exception_never_reuses_old_session_or_old_context() -> None:
    factory = _ExceptionalLandmarkerFactory(first_close_error=True)
    backend = _backend(factory)
    try:
        before = _by_side(_infer(backend, frame_index=0, timestamp=1.0))
        assert all(
            record["observation_state"] == "detected"
            for record in before.values()
        )

        error_count_before = backend.inference_error_count
        changed_context = _by_side(
            _infer(
                backend,
                frame_index=1,
                timestamp=0.0,
                person_ref="person-002",
                lock_epoch=2,
            )
        )
        for side, record in changed_context.items():
            assert len(factory.sessions[side]) == 2
            assert factory.sessions[side][0].close_calls == 1
            assert factory.sessions[side][0].closed is True
            assert factory.sessions[side][0].detect_calls == 1
            assert factory.sessions[side][1].detect_calls == 1
            assert factory.sessions[side][0] is not factory.sessions[side][1]
            assert record["backend_state"] == "available"
            assert record["observation_state"] == "detected"
            assert record["person_ref"] == "person-002"
            assert record["lock_epoch"] == 2
            assert record["tracker_session_generation"] == 2
            assert record["tracker_reset_reason"] == (
                "person_or_lock_epoch_changed"
            )
        assert backend.inference_error_count == error_count_before + 2

        recovered = _by_side(
            _infer(
                backend,
                frame_index=2,
                timestamp=0.125,
                person_ref="person-002",
                lock_epoch=2,
            )
        )
        for side, record in recovered.items():
            assert len(factory.sessions[side]) == 2
            assert factory.sessions[side][0] is not factory.sessions[side][1]
            assert factory.sessions[side][0].detect_calls == 1
            assert factory.sessions[side][1].detect_calls == 2
            assert record["backend_state"] == "available"
            assert record["observation_state"] == "detected"
            assert record["tracker_session_generation"] == 2
    finally:
        backend.close()


def test_consecutive_model_missing_limit_resets_only_affected_side() -> None:
    factory = _FakeLandmarkerFactory(
        left=[
            _detected_result(),
            _missing_result(),
            _missing_result(),
            _detected_result(),
        ],
        right=[_detected_result() for _ in range(4)],
    )
    backend = _backend(
        factory,
        maximum_consecutive_model_missing_frames=2,
    )
    try:
        _infer(backend, frame_index=0, timestamp=0.0)
        _infer(backend, frame_index=1, timestamp=0.125)
        second_missing = _by_side(
            _infer(backend, frame_index=2, timestamp=0.250)
        )

        _assert_no_geometry(second_missing["left"])
        assert (
            second_missing["left"]["tracker_reset_reason"]
            == "consecutive_model_missing_limit_reached"
        )
        assert factory.sessions["left"][0].closed is True
        assert factory.sessions["right"][0].closed is False
        assert factory.sessions["right"][0].timestamps_ms == [0, 125, 250]

        recovered = _by_side(
            _infer(backend, frame_index=3, timestamp=0.375)
        )
        assert len(factory.sessions["left"]) == 2
        assert recovered["left"]["tracker_session_generation"] == 2
        assert (
            recovered["left"]["tracker_reset_reason"]
            == "consecutive_model_missing_limit_reached"
        )
        assert factory.sessions["left"][1].timestamps_ms == [375]
        assert len(factory.sessions["right"]) == 1
        assert recovered["right"]["tracker_session_generation"] == 1
        assert factory.sessions["right"][0].timestamps_ms == [
            0,
            125,
            250,
            375,
        ]
    finally:
        backend.close()


def test_explicit_reset_is_idempotent_and_clears_both_sides() -> None:
    factory = _FakeLandmarkerFactory(
        left=[_detected_result(), _missing_result()],
        right=[_detected_result(), _missing_result()],
    )
    backend = _backend(factory)
    try:
        _infer(backend, frame_index=0, timestamp=0.0)
        backend.reset(person_ref="person-001", lock_epoch=1)
        backend.reset(person_ref="person-001", lock_epoch=1)

        assert factory.sessions["left"][0].closed is True
        assert factory.sessions["right"][0].closed is True
        after = _by_side(_infer(backend, frame_index=1, timestamp=0.125))
        for side, record in after.items():
            assert len(factory.sessions[side]) == 2
            assert record["tracker_session_generation"] == 2
            assert record["tracker_reset_reason"] == "explicit_reset"
            assert record["observation_state"] == "missing"
            _assert_no_geometry(record)
    finally:
        backend.close()


def test_hand_schema_accepts_clean_image_and_video_records() -> None:
    schema = json.loads(
        (ROOT / "schemas" / "hand_pose_frames.schema.json").read_text("utf-8")
    )
    factory = _FakeLandmarkerFactory(
        left=[_detected_result()],
        right=[_detected_result()],
    )
    backend = _backend(factory)
    try:
        video_records = _infer(backend, frame_index=0, timestamp=0.0)
    finally:
        backend.close()

    for index, record in enumerate(video_records):
        assert record["backend_mode"] == "video"
        assert record["landmark_count"] == 21
        assert record["training_eligible"] is False
        validate_instance(record, schema, path=f"video[{index}]")

        image_record = deepcopy(record)
        image_record["backend_mode"] = "image"
        for field in VIDEO_ONLY_RECORD_FIELDS:
            image_record.pop(field, None)
        image_record["raw_confidence_availability"][
            "tracking_confidence"
        ] = "not_applicable_stateless_image_mode_and_not_exposed"
        validate_instance(image_record, schema, path=f"image[{index}]")

        invalid_missing = deepcopy(record)
        invalid_missing["observation_state"] = "missing"
        with pytest.raises(SchemaValidationError):
            validate_instance(
                invalid_missing,
                schema,
                path=f"invalid_missing[{index}]",
            )


def test_hand_schema_rejects_invalid_video_and_evidence_contracts() -> None:
    schema = json.loads(
        (ROOT / "schemas" / "hand_pose_frames.schema.json").read_text("utf-8")
    )
    factory = _FakeLandmarkerFactory(
        left=[_detected_result()],
        right=[_detected_result()],
    )
    backend = _backend(factory)
    try:
        qualified = _by_side(
            _infer(backend, frame_index=0, timestamp=0.0)
        )["left"]
    finally:
        backend.close()

    invalid_records: dict[str, dict[str, Any]] = {}

    null_timestamp = deepcopy(qualified)
    null_timestamp["backend_timestamp_ms"] = None
    invalid_records["normal_video_null_backend_timestamp"] = null_timestamp

    null_generation = deepcopy(qualified)
    null_generation["tracker_session_generation"] = None
    invalid_records["normal_video_null_session_generation"] = null_generation

    missing_proposed = deepcopy(qualified)
    missing_proposed.update(
        {
            "landmarks": [],
            "landmark_count": 0,
            "observation_state": "missing",
            "quality_state": "not_observed",
            "quality_reasons": ["observation_missing"],
            "validation_state": "not_evaluable",
            "action_feature_eligible": False,
            "feature_eligibility_reasons": ["observation_missing"],
            "status": "proposed",
            "evidence_type": "no_hand_geometry",
        }
    )
    invalid_records["missing_with_proposed_status"] = missing_proposed

    missing_fake_evidence = deepcopy(missing_proposed)
    missing_fake_evidence["status"] = "uncertain"
    missing_fake_evidence["evidence_type"] = "real_hand_landmarker"
    invalid_records["missing_with_real_landmarker_evidence"] = (
        missing_fake_evidence
    )

    qualified_without_crop = deepcopy(qualified)
    qualified_without_crop["crop_bbox"] = None
    qualified_without_crop["crop_transform"] = None
    invalid_records["qualified_without_crop"] = qualified_without_crop

    qualified_without_evidence = deepcopy(qualified)
    qualified_without_evidence.pop("evidence_type", None)
    invalid_records["qualified_without_evidence_type"] = (
        qualified_without_evidence
    )

    image_with_tracker_fields = deepcopy(qualified)
    image_with_tracker_fields["backend_mode"] = "image"
    image_with_tracker_fields["raw_confidence_availability"][
        "tracking_confidence"
    ] = "not_applicable_stateless_image_mode_and_not_exposed"
    invalid_records["image_with_video_tracker_fields"] = (
        image_with_tracker_fields
    )

    for name, record in invalid_records.items():
        with pytest.raises(SchemaValidationError):
            validate_instance(record, schema, path=name)


def test_hand_schema_accepts_explicit_invalid_timestamp_error_records() -> None:
    schema = json.loads(
        (ROOT / "schemas" / "hand_pose_frames.schema.json").read_text("utf-8")
    )
    factory = _FakeLandmarkerFactory(
        left=[_detected_result()],
        right=[_detected_result()],
    )
    backend = _backend(factory)
    try:
        records = _infer(
            backend,
            frame_index=0,
            timestamp=float("nan"),
        )
    finally:
        backend.close()

    for index, record in enumerate(records):
        assert record["backend_state"] == "error"
        assert record["observation_state"] == "uncertain"
        assert record["timestamp"] is None
        assert record["backend_timestamp_ms"] is None
        _assert_no_geometry(record)
        validate_instance(record, schema, path=f"invalid_timestamp[{index}]")


@pytest.mark.private_artifacts
def test_real_default_image_records_have_no_video_only_fields() -> None:
    assert HAND_BACKEND_MODE == "image"
    assert RealHandPoseBackend is MediaPipeHandLandmarkerBackend
    backend = RealHandPoseBackend(MODEL)
    try:
        records = _infer(backend, frame_index=0, timestamp=0.0)
    finally:
        backend.close()

    assert len(records) == 2
    for record in records:
        assert record["backend_mode"] == "image"
        assert VIDEO_ONLY_RECORD_FIELDS.isdisjoint(record)
        assert record["training_eligible"] is False


def test_default_image_backend_remains_the_accepted_stateless_path() -> None:
    assert HAND_BACKEND_MODE == "image"
    assert RealHandPoseBackend is MediaPipeHandLandmarkerBackend

    constructor_source = inspect.getsource(
        MediaPipeHandLandmarkerBackend.__init__
    )
    detect_source = inspect.getsource(
        MediaPipeHandLandmarkerBackend._detect_crop
    )
    assert "vision.RunningMode.IMAGE" in constructor_source
    assert "self._landmarker.detect(image)" in detect_source
    assert "detect_for_video" not in detect_source
