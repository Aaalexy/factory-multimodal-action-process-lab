from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.validate_schemas import validate_all
from src.hand_pose import (
    CropTransform,
    DisabledHandBackend,
    HAND_QUALITY_GATE_VERSION,
    MediaPipeHandLandmarkerBackend,
    build_hand_crop_transform,
    finalize_hand_pose_record,
    map_normalized_landmarks,
)
from src.hand_pose.backend import _check_cross_side_consistency
from src.schema_validation import SchemaValidationError, validate_instance


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "hand_pose" / "hand_landmarker.task"
VIDEO_SHA = "a" * 64


def _body_keypoints() -> tuple[np.ndarray, list[str]]:
    """Synthetic contract fixture; never emitted as runtime evidence."""

    points = np.full((17, 2), np.nan, dtype=np.float64)
    points[5] = (220.0, 220.0)  # anatomical left shoulder
    points[7] = (180.0, 300.0)  # anatomical left elbow
    points[9] = (150.0, 390.0)  # anatomical left wrist
    points[6] = (780.0, 220.0)  # anatomical right shoulder
    points[8] = (820.0, 300.0)  # anatomical right elbow
    points[10] = (850.0, 390.0)  # anatomical right wrist
    statuses = ["missing"] * 17
    for index in (5, 7, 9, 6, 8, 10):
        statuses[index] = "detected"
    return points, statuses


def _infer_kwargs(
    points: np.ndarray | None,
    statuses: list[str] | None,
    **overrides: object,
) -> dict[str, object]:
    values: dict[str, object] = {
        "body_keypoints": points,
        "body_keypoint_statuses": statuses,
        "person_ref": "person-001",
        "lock_epoch": 1,
        "frame_index": 3,
        "timestamp": 0.375,
        "source_video_sha256": VIDEO_SHA,
        "recording_group_id": "recording-group-test",
        "track_state": "tracked",
        "lock_state": "tracked",
    }
    values.update(overrides)
    return values


def _quality_gate_record(
    *,
    anatomical_side: str = "left",
    observation_state: str = "detected",
    backend_state: str = "available",
) -> dict[str, object]:
    landmarks = [
        {
            "index": index,
            "x": float(150 + index),
            "y": float(390 + index),
            "z_roi_normalized": 0.0,
            "visibility": None,
            "presence": None,
            "observation_state": "detected",
        }
        for index in range(21)
    ]
    return {
        "hand_pose_id": f"hand-test-{anatomical_side}",
        "person_ref": "person-001",
        "lock_epoch": 1,
        "anatomical_side": anatomical_side,
        "frame_index": 3,
        "timestamp": 0.375,
        "crop_bbox": [100, 300, 300, 500],
        "crop_transform": {
            "kind": "normalized_roi_to_source_pixels",
            "x_offset": 100,
            "y_offset": 300,
            "x_scale": 200,
            "y_scale": 200,
            "source_width": 1000,
            "source_height": 720,
        },
        "landmarks": landmarks,
        "landmark_count": 21,
        "confidence": None,
        "detection_confidence": None,
        "presence_confidence": None,
        "tracking_confidence": None,
        "raw_confidence_availability": {},
        "backend_state": backend_state,
        "observation_state": observation_state,
        "occlusion": "not_inferred",
        "source_video_sha256": VIDEO_SHA,
        "recording_group_id": "recording-group-test",
        "source_model_version": "hand-model-test",
        "runtime_version": "test-runtime",
        "status": "proposed",
        "reviewer": None,
        "reviewed_at": None,
        "training_approval": "pending",
        "training_eligible": False,
        "model_handedness_label": None,
        "model_handedness_score": None,
        "inference_time_ms": 1.0,
        "reason": "real_model_landmarks",
        "evidence_type": "real_hand_landmarker",
        "association_checks": {
            "body_side_source": f"coco17_anatomical_{anatomical_side}",
            "model_handedness_used_for_assignment": False,
            "closer_to_own_body_wrist": True,
            "duplicate_across_sides": False,
            "warnings": [],
            "own_wrist_distance_roi_ratio": 0.1,
            "maximum_own_wrist_distance_roi_ratio": 0.3,
            "body_guide_observation_states": {
                "wrist": "detected",
                "elbow": "detected",
                "shoulder": "detected",
            },
        },
    }


def _finalize(
    record: dict[str, object],
    **overrides: object,
) -> dict[str, object]:
    context: dict[str, object] = {
        "expected_person_ref": "person-001",
        "expected_lock_epoch": 1,
        "expected_anatomical_side": str(record["anatomical_side"]),
        "track_state": "tracked",
        "lock_state": "tracked",
    }
    context.update(overrides)
    return finalize_hand_pose_record(record, **context)


def test_disabled_backend_is_honest_and_emits_no_geometry() -> None:
    points, statuses = _body_keypoints()
    backend = DisabledHandBackend()
    records = backend.infer_frame(
        np.zeros((720, 1000, 3), dtype=np.uint8),
        **_infer_kwargs(points, statuses),
    )
    assert backend.enabled is False
    assert backend.availability_status == "unavailable"
    assert backend.inference_call_count == 0
    assert records == []


def test_dynamic_rois_use_coco_anatomical_sides() -> None:
    points, statuses = _body_keypoints()
    left = build_hand_crop_transform(
        (720, 1000, 3), points, statuses, "left"
    )
    right = build_hand_crop_transform(
        (720, 1000, 3), points, statuses, "right"
    )
    assert left is not None and right is not None
    assert left.x_offset + left.x_scale / 2 < 500
    assert right.x_offset + right.x_scale / 2 > 500
    assert left.bbox != right.bbox


def test_crop_coordinates_map_back_to_source_pixels() -> None:
    transform = CropTransform(
        x_offset=100,
        y_offset=200,
        x_scale=300,
        y_scale=300,
        source_width=1000,
        source_height=800,
    )
    x, y = transform.normalized_to_source(0.25, 0.5)
    assert x == 175.0
    assert y == 350.0


def test_21_landmark_mapping_has_real_output_contract() -> None:
    transform = CropTransform(100, 50, 200, 200, 640, 480)
    normalized = [
        SimpleNamespace(
            x=index / 20.0,
            y=0.5,
            z=-index / 100.0,
            visibility=None,
            presence=None,
        )
        for index in range(21)
    ]
    mapped = map_normalized_landmarks(normalized, transform)
    assert len(mapped) == 21
    assert [item["index"] for item in mapped] == list(range(21))
    assert all(item["observation_state"] == "detected" for item in mapped)
    assert mapped[0]["x"] == 100.0
    assert mapped[-1]["x"] == 300.0


@pytest.mark.private_artifacts
def test_real_cpu_backend_loads_and_runs_without_fabricating_blank_hands() -> None:
    assert MODEL.is_file()
    points, statuses = _body_keypoints()
    with MediaPipeHandLandmarkerBackend(MODEL) as backend:
        records = backend.infer_frame(
            np.zeros((720, 1000, 3), dtype=np.uint8),
            **_infer_kwargs(points, statuses),
        )
        assert backend.runtime_version == "0.10.35"
        assert backend.inference_call_count == 2
        assert len(records) == 2
        assert {item["anatomical_side"] for item in records} == {
            "left",
            "right",
        }
        for record in records:
            assert record["observation_state"] in {
                "detected",
                "uncertain",
                "missing",
            }
            if record["observation_state"] == "detected":
                assert record["landmark_count"] == 21
            else:
                assert record["landmark_count"] in {0, 21}
            assert record["training_eligible"] is False
            assert record["backend_state"] == "available"
            assert record["backend_mode"] == "image"
            assert record["quality_gate_version"] == HAND_QUALITY_GATE_VERSION
            assert record["action_feature_eligible"] is False
            assert record["confidence"] is None
            assert record["detection_confidence"] is None
            assert record["presence_confidence"] is None
            assert record["tracking_confidence"] is None
            assert (
                record["association_checks"][
                    "model_handedness_used_for_assignment"
                ]
                is False
            )
    expected_cache = ROOT / "outputs" / "runtime_cache" / "matplotlib"
    assert Path(os.environ["MPLCONFIGDIR"]).resolve() == expected_cache.resolve()


@pytest.mark.private_artifacts
def test_missing_guides_return_empty_landmarks_without_model_calls() -> None:
    points, statuses = _body_keypoints()
    statuses[9] = "missing"
    statuses[10] = "missing"
    with MediaPipeHandLandmarkerBackend(MODEL) as backend:
        records = backend.infer_frame(
            np.zeros((720, 1000, 3), dtype=np.uint8),
            **_infer_kwargs(points, statuses),
        )
        assert backend.inference_call_count == 0
        assert len(records) == 2
        assert all(item["observation_state"] == "missing" for item in records)
        assert all(item["landmarks"] == [] for item in records)
        assert all(item["landmark_count"] == 0 for item in records)
        assert all(item["quality_state"] == "not_observed" for item in records)
        assert all(
            item["action_feature_eligible"] is False for item in records
        )


@pytest.mark.private_artifacts
def test_lost_is_a_hard_boundary_and_never_emits_hand_geometry() -> None:
    points, statuses = _body_keypoints()
    with MediaPipeHandLandmarkerBackend(MODEL) as backend:
        records = backend.infer_frame(
            np.zeros((720, 1000, 3), dtype=np.uint8),
            **_infer_kwargs(
                points,
                statuses,
                track_state="lost",
                lock_state="lost",
            ),
        )
        assert backend.inference_call_count == 0
        assert all(item["observation_state"] == "lost" for item in records)
        assert all(item["landmarks"] == [] for item in records)
        assert all(item["training_eligible"] is False for item in records)
        assert all(item["quality_state"] == "lost" for item in records)
        assert all(
            item["action_feature_eligible"] is False for item in records
        )


@pytest.mark.private_artifacts
def test_uncertain_body_track_does_not_run_or_fake_hand_geometry() -> None:
    points, statuses = _body_keypoints()
    with MediaPipeHandLandmarkerBackend(MODEL) as backend:
        records = backend.infer_frame(
            np.zeros((720, 1000, 3), dtype=np.uint8),
            **_infer_kwargs(points, statuses, track_state="uncertain"),
        )
        assert backend.inference_call_count == 0
        assert all(item["observation_state"] == "uncertain" for item in records)
        assert all(item["landmarks"] == [] for item in records)


@pytest.mark.private_artifacts
def test_person_and_epoch_are_never_carried_across_calls() -> None:
    points, statuses = _body_keypoints()
    with MediaPipeHandLandmarkerBackend(MODEL) as backend:
        first = backend.infer_frame(
            np.zeros((720, 1000, 3), dtype=np.uint8),
            **_infer_kwargs(
                points,
                statuses,
                person_ref="person-001",
                lock_epoch=1,
                track_state="lost",
                lock_state="lost",
            ),
        )
        second = backend.infer_frame(
            np.zeros((720, 1000, 3), dtype=np.uint8),
            **_infer_kwargs(
                points,
                statuses,
                person_ref="person-002",
                lock_epoch=2,
                track_state="lost",
                lock_state="lost",
            ),
        )
        assert {item["person_ref"] for item in first} == {"person-001"}
        assert {item["lock_epoch"] for item in first} == {1}
        assert {item["person_ref"] for item in second} == {"person-002"}
        assert {item["lock_epoch"] for item in second} == {2}
        assert set(item["hand_pose_id"] for item in first).isdisjoint(
            item["hand_pose_id"] for item in second
        )


@pytest.mark.private_artifacts
def test_hand_schema_accepts_missing_real_backend_records() -> None:
    points, statuses = _body_keypoints()
    statuses[9] = "missing"
    statuses[10] = "missing"
    schema = json.loads(
        (ROOT / "schemas" / "hand_pose_frames.schema.json").read_text("utf-8")
    )
    with MediaPipeHandLandmarkerBackend(MODEL) as backend:
        records = backend.infer_frame(
            np.zeros((720, 1000, 3), dtype=np.uint8),
            **_infer_kwargs(points, statuses),
        )
    for index, record in enumerate(records):
        validate_instance(record, schema, path=f"hand_pose_frames[{index}]")


def test_model_wrist_far_from_anatomical_wrist_is_downgraded() -> None:
    points, _ = _body_keypoints()
    record = {
        "anatomical_side": "left",
        "observation_state": "detected",
        "status": "proposed",
        "occlusion": "not_inferred",
        "landmarks": [{"index": 0, "x": 230.0, "y": 390.0}],
        "crop_transform": {"x_scale": 200},
        "association_checks": {
            "closer_to_own_body_wrist": None,
            "duplicate_across_sides": False,
            "warnings": [],
        },
    }
    _check_cross_side_consistency([record], points)
    assert record["association_checks"]["own_wrist_distance_roi_ratio"] == 0.4
    assert record["observation_state"] == "uncertain"
    assert "model_wrist_too_far_from_own_body_wrist" in record[
        "association_checks"
    ]["warnings"]


def test_own_wrist_distance_is_checked_when_opposite_wrist_is_missing() -> None:
    points, _ = _body_keypoints()
    points[10, :2] = np.nan
    record = {
        "anatomical_side": "left",
        "observation_state": "detected",
        "status": "proposed",
        "occlusion": "not_inferred",
        "landmarks": [{"index": 0, "x": 160.0, "y": 390.0}],
        "crop_transform": {"x_scale": 200},
        "association_checks": {
            "closer_to_own_body_wrist": None,
            "duplicate_across_sides": False,
            "warnings": [],
        },
    }
    _check_cross_side_consistency([record], points)
    assert record["association_checks"]["own_wrist_distance_roi_ratio"] == 0.05
    assert record["association_checks"]["closer_to_own_body_wrist"] is None
    assert record["observation_state"] == "detected"


@pytest.mark.private_artifacts
def test_hand_schema_enforces_detected_and_missing_landmark_invariants() -> None:
    schema = json.loads(
        (ROOT / "schemas" / "hand_pose_frames.schema.json").read_text("utf-8")
    )
    points, statuses = _body_keypoints()
    statuses[9] = "missing"
    statuses[10] = "missing"
    with MediaPipeHandLandmarkerBackend(MODEL) as backend:
        missing = backend.infer_frame(
            np.zeros((720, 1000, 3), dtype=np.uint8),
            **_infer_kwargs(points, statuses),
        )[0]
    invalid_missing = dict(missing)
    invalid_missing["landmarks"] = [
        {
            "index": 0,
            "x": 1.0,
            "y": 1.0,
            "z_roi_normalized": None,
            "visibility": None,
            "presence": None,
            "observation_state": "detected",
        }
    ]
    invalid_missing["landmark_count"] = 1
    with pytest.raises(SchemaValidationError):
        validate_instance(invalid_missing, schema)

    invalid_detected = dict(missing)
    invalid_detected["observation_state"] = "detected"
    invalid_detected["status"] = "proposed"
    invalid_detected["landmarks"] = []
    invalid_detected["landmark_count"] = 0
    with pytest.raises(SchemaValidationError):
        validate_instance(invalid_detected, schema)


def test_schema_registry_includes_hand_pose_frames() -> None:
    result = validate_all()
    assert result["status"] == "passed"
    assert result["schema_file_count"] >= 10


def test_quality_gate_qualifies_only_complete_real_context_bound_geometry() -> None:
    record = _finalize(_quality_gate_record())
    assert record["backend_state"] == "available"
    assert record["backend_mode"] == "image"
    assert record["quality_state"] == "qualified"
    assert record["quality_reasons"] == ["quality_gate_passed"]
    assert record["validation_state"] == "not_reviewed"
    assert record["action_feature_eligible"] is True
    assert record["feature_eligibility_reasons"] == ["quality_gate_passed"]
    assert record["training_eligible"] is False
    assert record["training_approval"] == "pending"


def test_quality_gate_rejects_warnings_and_cross_side_duplicates() -> None:
    record = _quality_gate_record(observation_state="uncertain")
    checks = record["association_checks"]
    assert isinstance(checks, dict)
    checks["warnings"] = ["duplicate_hand_candidate_across_sides"]
    checks["duplicate_across_sides"] = True
    finalized = _finalize(record)
    assert finalized["quality_state"] == "association_uncertain"
    assert finalized["validation_state"] == "review_required"
    assert finalized["action_feature_eligible"] is False
    assert "duplicate_across_anatomical_sides" in finalized[
        "feature_eligibility_reasons"
    ]


def test_quality_gate_rejects_partial_or_duplicate_landmark_indices() -> None:
    partial = _quality_gate_record(observation_state="uncertain")
    partial["landmarks"] = list(partial["landmarks"])[:-1]
    partial["landmark_count"] = 20
    finalized_partial = _finalize(partial)
    assert finalized_partial["quality_state"] == "insufficient_geometry"
    assert finalized_partial["action_feature_eligible"] is False

    duplicate_index = _quality_gate_record(observation_state="uncertain")
    landmarks = list(duplicate_index["landmarks"])
    assert isinstance(landmarks[-1], dict)
    landmarks[-1] = dict(landmarks[-1], index=19)
    duplicate_index["landmarks"] = landmarks
    finalized_duplicate = _finalize(duplicate_index)
    assert finalized_duplicate["quality_state"] == "insufficient_geometry"
    assert finalized_duplicate["action_feature_eligible"] is False


@pytest.mark.parametrize(
    ("observation_state", "track_state", "lock_state", "quality_state"),
    [
        ("missing", "tracked", "tracked", "not_observed"),
        ("lost", "lost", "lost", "lost"),
    ],
)
def test_quality_gate_missing_and_lost_clear_geometry(
    observation_state: str,
    track_state: str,
    lock_state: str,
    quality_state: str,
) -> None:
    record = _quality_gate_record(observation_state=observation_state)
    finalized = _finalize(
        record,
        track_state=track_state,
        lock_state=lock_state,
    )
    assert finalized["landmarks"] == []
    assert finalized["landmark_count"] == 0
    assert finalized["evidence_type"] == "no_hand_geometry"
    assert finalized["quality_state"] == quality_state
    assert finalized["validation_state"] == "not_evaluable"
    assert finalized["action_feature_eligible"] is False


@pytest.mark.private_artifacts
def test_inference_error_is_fail_closed_at_backend_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points, statuses = _body_keypoints()
    with MediaPipeHandLandmarkerBackend(MODEL) as backend:
        monkeypatch.setattr(
            backend,
            "_detect_crop",
            lambda *_: (None, 0.0, "inference_error:RuntimeError"),
        )
        records = backend.infer_frame(
            np.zeros((720, 1000, 3), dtype=np.uint8),
            **_infer_kwargs(points, statuses),
        )
    assert all(item["backend_state"] == "error" for item in records)
    assert all(item["quality_state"] == "insufficient_geometry" for item in records)
    assert all(item["action_feature_eligible"] is False for item in records)
    assert all(item["training_eligible"] is False for item in records)


@pytest.mark.parametrize(
    ("context_key", "context_value", "reason"),
    [
        ("expected_person_ref", "person-002", "person_ref_context_mismatch"),
        ("expected_lock_epoch", 2, "lock_epoch_context_mismatch"),
        (
            "expected_anatomical_side",
            "right",
            "anatomical_side_context_mismatch",
        ),
    ],
)
def test_quality_gate_rejects_context_mismatch(
    context_key: str,
    context_value: object,
    reason: str,
) -> None:
    finalized = _finalize(
        _quality_gate_record(),
        **{context_key: context_value},
    )
    assert finalized["quality_state"] == "association_uncertain"
    assert finalized["action_feature_eligible"] is False
    assert reason in finalized["feature_eligibility_reasons"]


def test_quality_gate_schema_is_backward_compatible_and_fail_closed() -> None:
    schema = json.loads(
        (ROOT / "schemas" / "hand_pose_frames.schema.json").read_text("utf-8")
    )
    current = _finalize(_quality_gate_record())
    validate_instance(current, schema)

    legacy = dict(current)
    for key in (
        "backend_state",
        "backend_mode",
        "quality_state",
        "quality_reasons",
        "validation_state",
        "action_feature_eligible",
        "feature_eligibility_reasons",
        "quality_gate_version",
    ):
        legacy.pop(key)
    validate_instance(legacy, schema)

    invalid = _finalize(_quality_gate_record(observation_state="missing"))
    invalid["action_feature_eligible"] = True
    with pytest.raises(SchemaValidationError):
        validate_instance(invalid, schema)
