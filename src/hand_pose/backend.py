"""Pluggable, evidence-preserving hand-pose backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Sequence

import numpy as np

from .roi import (
    COCO_HAND_GUIDE_INDICES,
    CropTransform,
    body_wrist_point,
    build_hand_crop_transform,
)


ANATOMICAL_SIDES = ("left", "right")
HAND_LANDMARK_COUNT = 21
MAX_MODEL_WRIST_DISTANCE_ROI_RATIO = 0.30
HAND_QUALITY_GATE_VERSION = "hand_quality_gate_v1"
HAND_BACKEND_MODE = "image"
HAND_BACKEND_MODES = ("image", "video")
_VALID_LOCK_STATES = {"tracked", "tracking", "locked"}


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def map_normalized_landmarks(
    landmarks: Sequence[Any],
    transform: CropTransform,
) -> list[dict[str, Any]]:
    """Map real model output from normalized ROI space to source pixels."""

    mapped: list[dict[str, Any]] = []
    for index, landmark in enumerate(landmarks):
        normalized_x = _finite_or_none(getattr(landmark, "x", None))
        normalized_y = _finite_or_none(getattr(landmark, "y", None))
        if normalized_x is None or normalized_y is None:
            continue
        x, y = transform.normalized_to_source(normalized_x, normalized_y)
        mapped.append(
            {
                "index": index,
                "x": round(x, 3),
                "y": round(y, 3),
                "z_roi_normalized": _finite_or_none(
                    getattr(landmark, "z", None)
                ),
                "visibility": _finite_or_none(
                    getattr(landmark, "visibility", None)
                ),
                "presence": _finite_or_none(
                    getattr(landmark, "presence", None)
                ),
                "observation_state": "detected",
            }
        )
    return mapped


def _identity_slug(value: str) -> str:
    normalized = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in str(value)
    )
    return normalized or "unassigned"


def _hand_pose_id(
    source_video_sha256: str,
    person_ref: str,
    lock_epoch: int,
    frame_index: int,
    anatomical_side: str,
) -> str:
    return (
        f"hand-{source_video_sha256[:12]}-{_identity_slug(person_ref)}-"
        f"e{int(lock_epoch)}-f{int(frame_index):08d}-{anatomical_side}"
    )


def _base_record(
    *,
    person_ref: str,
    lock_epoch: int,
    anatomical_side: str,
    frame_index: int,
    timestamp: float,
    source_video_sha256: str,
    recording_group_id: str,
    source_model_version: str,
    runtime_version: str | None,
    transform: CropTransform | None,
    observation_state: str,
    reason: str,
    backend_state: str = "available",
    backend_mode: str = HAND_BACKEND_MODE,
    quality_gate_version: str = HAND_QUALITY_GATE_VERSION,
    backend_timestamp_ms: int | None = None,
    tracker_session_generation: int | None = None,
    tracker_reset_reason: str | None = None,
) -> dict[str, Any]:
    record = {
        "hand_pose_id": _hand_pose_id(
            source_video_sha256,
            person_ref,
            lock_epoch,
            frame_index,
            anatomical_side,
        ),
        "person_ref": str(person_ref),
        "lock_epoch": int(lock_epoch),
        "anatomical_side": anatomical_side,
        "frame_index": int(frame_index),
        "timestamp": round(float(timestamp), 6),
        "crop_bbox": transform.bbox if transform is not None else None,
        "crop_transform": transform.to_dict() if transform is not None else None,
        "landmarks": [],
        "landmark_count": 0,
        "confidence": None,
        "detection_confidence": None,
        "presence_confidence": None,
        "tracking_confidence": None,
        "raw_confidence_availability": {
            "detection_confidence": "not_exposed_by_mediapipe_python_result",
            "presence_confidence": "not_exposed_by_mediapipe_python_result",
            "tracking_confidence": (
                "not_applicable_stateless_image_mode_and_not_exposed"
            ),
        },
        "backend_state": backend_state,
        "backend_mode": backend_mode,
        "observation_state": observation_state,
        "quality_state": "not_observed",
        "quality_reasons": ["quality_gate_pending"],
        "validation_state": "not_evaluable",
        "action_feature_eligible": False,
        "feature_eligibility_reasons": ["quality_gate_pending"],
        "quality_gate_version": quality_gate_version,
        "occlusion": (
            "not_evaluable"
            if observation_state == "lost"
            else "unknown_or_out_of_roi"
            if observation_state in {"missing", "uncertain"}
            else "not_inferred"
        ),
        "source_video_sha256": source_video_sha256,
        "recording_group_id": recording_group_id,
        "source_model_version": source_model_version,
        "runtime_version": runtime_version,
        "status": "proposed" if observation_state == "detected" else "uncertain",
        "reviewer": None,
        "reviewed_at": None,
        "training_approval": "pending",
        "training_eligible": False,
        "model_handedness_label": None,
        "model_handedness_score": None,
        "inference_time_ms": None,
        "reason": reason,
        "evidence_type": "real_hand_landmarker"
        if observation_state in {"detected", "uncertain"}
        and transform is not None
        else "no_hand_geometry",
        "association_checks": {
            "body_side_source": (
                f"coco17_anatomical_{anatomical_side}_"
                f"wrist_{COCO_HAND_GUIDE_INDICES[anatomical_side][0]}"
            ),
            "model_handedness_used_for_assignment": False,
            "closer_to_own_body_wrist": None,
            "duplicate_across_sides": False,
            "warnings": [],
        },
    }
    if backend_mode == "video":
        record.update(
            {
                "backend_timestamp_ms": backend_timestamp_ms,
                "tracker_session_generation": tracker_session_generation,
                "tracker_reset_reason": tracker_reset_reason,
            }
        )
    return record


class HandPoseBackend(ABC):
    """Interface for frame-local hand evidence."""

    enabled: bool
    model_version: str | None
    inference_call_count: int
    inference_times_ms: list[float]

    @abstractmethod
    def infer_frame(
        self,
        frame: np.ndarray,
        *,
        body_keypoints: Sequence[Sequence[float]] | np.ndarray | None,
        body_keypoint_statuses: Sequence[str] | None,
        person_ref: str,
        lock_epoch: int,
        frame_index: int,
        timestamp: float,
        source_video_sha256: str,
        recording_group_id: str,
        track_state: str = "tracked",
        lock_state: str = "tracked",
    ) -> list[dict[str, Any]]:
        """Return traceable left/right hand observations for one person."""

    def reset(
        self,
        *,
        person_ref: str | None = None,
        lock_epoch: int | None = None,
    ) -> None:
        """Reset temporal state. Stateless backends intentionally do nothing."""

    def close(self) -> None:
        """Release backend resources."""


class DisabledHandBackend(HandPoseBackend):
    """Honest null backend: disabled means no hand records or geometry."""

    enabled = False
    model_version = None

    def __init__(self) -> None:
        self.inference_call_count = 0
        self.inference_times_ms: list[float] = []
        self.availability_status = "unavailable"
        self.reason = "not_configured: hand pose backend is disabled"

    def infer_frame(
        self,
        frame: np.ndarray,
        *,
        body_keypoints: Sequence[Sequence[float]] | np.ndarray | None,
        body_keypoint_statuses: Sequence[str] | None,
        person_ref: str,
        lock_epoch: int,
        frame_index: int,
        timestamp: float,
        source_video_sha256: str,
        recording_group_id: str,
        track_state: str = "tracked",
        lock_state: str = "tracked",
    ) -> list[dict[str, Any]]:
        del (
            frame,
            body_keypoints,
            body_keypoint_statuses,
            person_ref,
            lock_epoch,
            frame_index,
            timestamp,
            source_video_sha256,
            recording_group_id,
            track_state,
            lock_state,
        )
        return []


def _mark_uncertain(record: dict[str, Any], warning: str) -> None:
    record["observation_state"] = "uncertain"
    record["status"] = "uncertain"
    record["occlusion"] = "association_uncertain"
    warnings = record["association_checks"]["warnings"]
    if warning not in warnings:
        warnings.append(warning)


def _body_guide_states(
    body_keypoint_statuses: Sequence[str] | None,
    anatomical_side: str,
) -> dict[str, str]:
    names = ("wrist", "elbow", "shoulder")
    indices = COCO_HAND_GUIDE_INDICES[anatomical_side]
    if body_keypoint_statuses is None:
        return {name: "unknown" for name in names}
    return {
        name: (
            str(body_keypoint_statuses[index]).lower()
            if index < len(body_keypoint_statuses)
            else "missing"
        )
        for name, index in zip(names, indices, strict=True)
    }


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason and reason not in reasons:
        reasons.append(reason)


def _has_complete_real_landmarks(record: dict[str, Any]) -> bool:
    landmarks = list(record.get("landmarks") or [])
    if len(landmarks) != HAND_LANDMARK_COUNT:
        return False
    if int(record.get("landmark_count") or 0) != HAND_LANDMARK_COUNT:
        return False
    indices: list[int] = []
    for landmark in landmarks:
        index = landmark.get("index")
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            return False
        if _finite_or_none(landmark.get("x")) is None:
            return False
        if _finite_or_none(landmark.get("y")) is None:
            return False
        indices.append(int(index))
    return (
        len(set(indices)) == HAND_LANDMARK_COUNT
        and set(indices) == set(range(HAND_LANDMARK_COUNT))
    )


def finalize_hand_pose_record(
    record: dict[str, Any],
    *,
    expected_person_ref: str,
    expected_lock_epoch: int,
    expected_anatomical_side: str,
    track_state: str,
    lock_state: str,
    maximum_own_wrist_distance_roi_ratio: float = (
        MAX_MODEL_WRIST_DISTANCE_ROI_RATIO
    ),
    quality_gate_version: str = HAND_QUALITY_GATE_VERSION,
) -> dict[str, Any]:
    """Finalize one hand record with an explicit, fail-closed quality gate.

    The gate never promotes an observation or invents geometry.  Association-
    uncertain real landmarks remain available for review and dashed rendering,
    but only a directly detected, structurally complete, context-consistent
    record can become an input to future proposed action features.
    """

    observation_state = str(
        record.get("observation_state", "missing")
    ).lower()
    backend_state = str(record.get("backend_state", "available")).lower()
    if backend_state not in {"available", "unavailable", "error"}:
        backend_state = "error"
    backend_mode = str(
        record.get("backend_mode", HAND_BACKEND_MODE)
    ).lower()
    if backend_mode not in HAND_BACKEND_MODES:
        backend_mode = HAND_BACKEND_MODE
        backend_state = "error"
    record["backend_state"] = backend_state
    record["backend_mode"] = backend_mode
    record["quality_gate_version"] = str(quality_gate_version)

    landmarks = list(record.get("landmarks") or [])
    if observation_state in {"missing", "lost"} and landmarks:
        # Fail closed without creating replacement points.
        record["landmarks"] = []
        record["landmark_count"] = 0
        record["evidence_type"] = "no_hand_geometry"
        landmarks = []

    association = record.setdefault("association_checks", {})
    warnings = [str(item) for item in association.get("warnings", []) if item]
    association["warnings"] = warnings
    duplicate = bool(association.get("duplicate_across_sides", False))
    own_wrist_ratio = _finite_or_none(
        association.get("own_wrist_distance_roi_ratio")
    )
    body_guides = association.get("body_guide_observation_states") or {}

    reasons: list[str] = []
    if backend_state != "available":
        _append_reason(reasons, f"backend_{backend_state}")
    if observation_state == "lost":
        _append_reason(reasons, "observation_lost")
    elif observation_state == "missing":
        _append_reason(reasons, "observation_missing")
    elif observation_state != "detected":
        _append_reason(reasons, f"observation_{observation_state}")

    complete_landmarks = _has_complete_real_landmarks(record)
    if not complete_landmarks:
        _append_reason(reasons, "landmarks_not_complete_unique_0_to_20")
    if str(record.get("evidence_type", "")) != "real_hand_landmarker":
        _append_reason(reasons, "not_real_hand_landmarker_evidence")

    for warning in warnings:
        _append_reason(reasons, f"association_warning:{warning}")
    if duplicate:
        _append_reason(reasons, "duplicate_across_anatomical_sides")
    if own_wrist_ratio is None:
        _append_reason(reasons, "own_wrist_distance_unavailable")
    elif own_wrist_ratio > float(maximum_own_wrist_distance_roi_ratio):
        _append_reason(reasons, "own_wrist_distance_gate_failed")
    if str(body_guides.get("wrist", "")).lower() != "detected":
        _append_reason(reasons, "body_wrist_not_directly_detected")
    if str(body_guides.get("elbow", "")).lower() != "detected":
        _append_reason(reasons, "body_elbow_not_directly_detected")

    normalized_track_state = str(track_state).lower()
    normalized_lock_state = str(lock_state).lower()
    if normalized_track_state != "tracked":
        _append_reason(reasons, "track_not_tracked")
    if normalized_lock_state not in _VALID_LOCK_STATES:
        _append_reason(reasons, "lock_not_valid")
    if str(record.get("person_ref")) != str(expected_person_ref):
        _append_reason(reasons, "person_ref_context_mismatch")
    try:
        record_epoch = int(record.get("lock_epoch"))
    except (TypeError, ValueError):
        record_epoch = -1
    if record_epoch != int(expected_lock_epoch):
        _append_reason(reasons, "lock_epoch_context_mismatch")
    if str(record.get("anatomical_side")) != str(expected_anatomical_side):
        _append_reason(reasons, "anatomical_side_context_mismatch")

    qualified = (
        backend_state == "available"
        and observation_state == "detected"
        and complete_landmarks
        and str(record.get("evidence_type", "")) == "real_hand_landmarker"
        and not warnings
        and not duplicate
        and own_wrist_ratio is not None
        and own_wrist_ratio
        <= float(maximum_own_wrist_distance_roi_ratio)
        and str(body_guides.get("wrist", "")).lower() == "detected"
        and str(body_guides.get("elbow", "")).lower() == "detected"
        and normalized_track_state == "tracked"
        and normalized_lock_state in _VALID_LOCK_STATES
        and str(record.get("person_ref")) == str(expected_person_ref)
        and record_epoch == int(expected_lock_epoch)
        and str(record.get("anatomical_side"))
        == str(expected_anatomical_side)
    )

    if observation_state == "lost":
        quality_state = "lost"
    elif observation_state == "missing":
        quality_state = "not_observed"
    elif qualified:
        quality_state = "qualified"
    elif landmarks and (
        warnings
        or duplicate
        or "own_wrist_distance_gate_failed" in reasons
        or "own_wrist_distance_unavailable" in reasons
        or any(reason.endswith("_context_mismatch") for reason in reasons)
        or "body_wrist_not_directly_detected" in reasons
        or "body_elbow_not_directly_detected" in reasons
    ):
        quality_state = "association_uncertain"
    else:
        quality_state = "insufficient_geometry"

    validation_state = (
        "not_reviewed"
        if qualified
        else "review_required"
        if landmarks
        else "not_evaluable"
    )
    if qualified:
        reasons = ["quality_gate_passed"]

    record["quality_state"] = quality_state
    record["quality_reasons"] = reasons
    record["validation_state"] = validation_state
    record["action_feature_eligible"] = qualified
    record["feature_eligibility_reasons"] = list(reasons)
    record["status"] = "proposed" if qualified else "uncertain"
    record["training_eligible"] = False
    record["training_approval"] = "pending"
    return record


def _check_cross_side_consistency(
    records: list[dict[str, Any]],
    body_keypoints: Sequence[Sequence[float]] | np.ndarray | None,
    *,
    maximum_own_wrist_distance_roi_ratio: float = (
        MAX_MODEL_WRIST_DISTANCE_ROI_RATIO
    ),
) -> None:
    by_side = {record["anatomical_side"]: record for record in records}
    for side, record in by_side.items():
        if not record["landmarks"]:
            continue
        own_wrist = body_wrist_point(body_keypoints, side)
        other_side = "right" if side == "left" else "left"
        other_wrist = body_wrist_point(body_keypoints, other_side)
        if own_wrist is None:
            _mark_uncertain(
                record,
                "own_body_wrist_unavailable_for_consistency_check",
            )
            continue
        model_wrist = record["landmarks"][0]
        model_point = (float(model_wrist["x"]), float(model_wrist["y"]))
        own_distance = math.dist(model_point, own_wrist)
        record["association_checks"]["own_wrist_distance_pixels"] = round(
            own_distance, 3
        )
        crop_scale = float(
            (record.get("crop_transform") or {}).get("x_scale") or 0.0
        )
        distance_ratio = (
            own_distance / crop_scale if crop_scale > 0.0 else float("inf")
        )
        record["association_checks"]["own_wrist_distance_roi_ratio"] = round(
            distance_ratio,
            6,
        )
        record["association_checks"][
            "maximum_own_wrist_distance_roi_ratio"
        ] = float(maximum_own_wrist_distance_roi_ratio)
        if distance_ratio > float(maximum_own_wrist_distance_roi_ratio):
            _mark_uncertain(
                record,
                "model_wrist_too_far_from_own_body_wrist",
            )

        if other_wrist is None:
            record["association_checks"][
                "closer_to_own_body_wrist"
            ] = None
            continue
        other_distance = math.dist(model_point, other_wrist)
        closer_to_own = own_distance <= other_distance
        record["association_checks"]["closer_to_own_body_wrist"] = closer_to_own
        record["association_checks"]["opposite_wrist_distance_pixels"] = round(
            other_distance, 3
        )
        if not closer_to_own:
            _mark_uncertain(
                record,
                "model_hand_is_closer_to_opposite_body_wrist",
            )

    left = by_side.get("left")
    right = by_side.get("right")
    if not left or not right or not left["landmarks"] or not right["landmarks"]:
        return
    left_wrist = left["landmarks"][0]
    right_wrist = right["landmarks"][0]
    model_wrist_distance = math.dist(
        (float(left_wrist["x"]), float(left_wrist["y"])),
        (float(right_wrist["x"]), float(right_wrist["y"])),
    )
    crop_sizes = [
        float(record["crop_transform"]["x_scale"])
        for record in (left, right)
        if record.get("crop_transform")
    ]
    duplicate_threshold = max(
        12.0,
        (min(crop_sizes) * 0.18) if crop_sizes else 12.0,
    )
    if model_wrist_distance <= duplicate_threshold:
        for record in (left, right):
            record["association_checks"]["duplicate_across_sides"] = True
            record["association_checks"][
                "cross_side_model_wrist_distance_pixels"
            ] = round(model_wrist_distance, 3)
            _mark_uncertain(record, "duplicate_hand_candidate_across_sides")


class MediaPipeHandLandmarkerBackend(HandPoseBackend):
    """CPU Hand Landmarker using independent body-guided left/right crops.

    IMAGE running mode is deliberately stateless.  It cannot propagate hand
    geometry across person_ref, lock_epoch, lost, or missing observations.
    """

    enabled = True

    def __init__(
        self,
        model_path: str | Path,
        *,
        model_version: str = "mediapipe_hand_landmarker_float16_v1",
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        minimum_crop_pixels: int = 96,
        maximum_own_wrist_distance_roi_ratio: float = (
            MAX_MODEL_WRIST_DISTANCE_ROI_RATIO
        ),
        quality_gate_version: str = HAND_QUALITY_GATE_VERSION,
    ) -> None:
        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Hand model is missing: {path}")
        self.model_path = str(path)
        self.model_version = str(model_version)
        self.minimum_crop_pixels = int(minimum_crop_pixels)
        self.maximum_own_wrist_distance_roi_ratio = float(
            maximum_own_wrist_distance_roi_ratio
        )
        self.quality_gate_version = str(quality_gate_version)
        self.inference_call_count = 0
        self.inference_times_ms: list[float] = []
        self.inference_error_count = 0
        self.availability_status = "available"
        self.reason = "real MediaPipe Hand Landmarker CPU backend"
        self._lock = threading.Lock()
        self._closed = False

        project_root = Path(__file__).resolve().parents[2]
        matplotlib_cache = (
            project_root / "outputs" / "runtime_cache" / "matplotlib"
        )
        matplotlib_cache.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(matplotlib_cache)

        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise RuntimeError(
                "mediapipe is required in the project-local environment"
            ) from exc

        self.runtime_version = str(mp.__version__)
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=self.model_path,
                delegate=python.BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=float(
                min_hand_detection_confidence
            ),
            min_hand_presence_confidence=float(min_hand_presence_confidence),
        )
        self._mp = mp
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def __enter__(self) -> "MediaPipeHandLandmarkerBackend":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._landmarker.close()
                self._closed = True

    def reset(
        self,
        *,
        person_ref: str | None = None,
        lock_epoch: int | None = None,
    ) -> None:
        del person_ref, lock_epoch
        # IMAGE mode has no tracker state to carry over a hard boundary.

    def _empty_records(
        self,
        *,
        person_ref: str,
        lock_epoch: int,
        frame_index: int,
        timestamp: float,
        source_video_sha256: str,
        recording_group_id: str,
        observation_state: str,
        reason: str,
        track_state: str,
        lock_state: str,
    ) -> list[dict[str, Any]]:
        return [
            finalize_hand_pose_record(
                _base_record(
                    person_ref=person_ref,
                    lock_epoch=lock_epoch,
                    anatomical_side=side,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    source_video_sha256=source_video_sha256,
                    recording_group_id=recording_group_id,
                    source_model_version=self.model_version,
                    runtime_version=self.runtime_version,
                    transform=None,
                    observation_state=observation_state,
                    reason=reason,
                    quality_gate_version=self.quality_gate_version,
                ),
                expected_person_ref=person_ref,
                expected_lock_epoch=lock_epoch,
                expected_anatomical_side=side,
                track_state=track_state,
                lock_state=lock_state,
                maximum_own_wrist_distance_roi_ratio=(
                    self.maximum_own_wrist_distance_roi_ratio
                ),
                quality_gate_version=self.quality_gate_version,
            )
            for side in ANATOMICAL_SIDES
        ]

    def _detect_crop(
        self,
        frame: np.ndarray,
        transform: CropTransform,
    ) -> tuple[Any | None, float, str | None]:
        x_min, y_min, x_max, y_max = transform.bbox
        crop_bgr = frame[y_min:y_max, x_min:x_max]
        if crop_bgr.size == 0:
            return None, 0.0, "empty_source_crop"
        crop_rgb = np.ascontiguousarray(crop_bgr[:, :, ::-1])
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=crop_rgb,
        )
        started = time.perf_counter()
        try:
            with self._lock:
                if self._closed:
                    raise RuntimeError("Hand backend is closed")
                result = self._landmarker.detect(image)
        except Exception as exc:  # keep the body-pose pipeline available
            elapsed = (time.perf_counter() - started) * 1000.0
            self.inference_call_count += 1
            self.inference_times_ms.append(elapsed)
            self.inference_error_count += 1
            return None, elapsed, f"inference_error:{type(exc).__name__}"
        elapsed = (time.perf_counter() - started) * 1000.0
        self.inference_call_count += 1
        self.inference_times_ms.append(elapsed)
        return result, elapsed, None

    @staticmethod
    def _choose_hand(
        result: Any,
        transform: CropTransform,
        own_body_wrist: tuple[float, float] | None,
    ) -> int | None:
        candidates = list(getattr(result, "hand_landmarks", []) or [])
        if not candidates:
            return None
        if own_body_wrist is None or len(candidates) == 1:
            return 0
        distances: list[float] = []
        for candidate in candidates:
            if not candidate:
                distances.append(float("inf"))
                continue
            x, y = transform.normalized_to_source(
                float(candidate[0].x),
                float(candidate[0].y),
            )
            distances.append(math.dist((x, y), own_body_wrist))
        return int(np.argmin(np.asarray(distances)))

    def infer_frame(
        self,
        frame: np.ndarray,
        *,
        body_keypoints: Sequence[Sequence[float]] | np.ndarray | None,
        body_keypoint_statuses: Sequence[str] | None,
        person_ref: str,
        lock_epoch: int,
        frame_index: int,
        timestamp: float,
        source_video_sha256: str,
        recording_group_id: str,
        track_state: str = "tracked",
        lock_state: str = "tracked",
    ) -> list[dict[str, Any]]:
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError("frame must be a BGR image with three channels")
        normalized_track_state = str(track_state).lower()
        normalized_lock_state = str(lock_state).lower()
        hard_boundary = (
            normalized_track_state in {"lost", "off_frame", "temporarily_lost"}
            or normalized_lock_state
            in {"lost", "awaiting_manual_relock", "unlocked"}
            or str(person_ref) == "unlocked"
        )
        if hard_boundary:
            self.reset(person_ref=person_ref, lock_epoch=lock_epoch)
            return self._empty_records(
                person_ref=person_ref,
                lock_epoch=lock_epoch,
                frame_index=frame_index,
                timestamp=timestamp,
                source_video_sha256=source_video_sha256,
                recording_group_id=recording_group_id,
                observation_state="lost",
                reason="person_or_lock_hard_boundary",
                track_state=track_state,
                lock_state=lock_state,
            )
        if normalized_track_state != "tracked":
            return self._empty_records(
                person_ref=person_ref,
                lock_epoch=lock_epoch,
                frame_index=frame_index,
                timestamp=timestamp,
                source_video_sha256=source_video_sha256,
                recording_group_id=recording_group_id,
                observation_state="uncertain",
                reason="body_tracking_not_reliable_for_hand_roi",
                track_state=track_state,
                lock_state=lock_state,
            )

        records: list[dict[str, Any]] = []
        for side in ANATOMICAL_SIDES:
            transform = build_hand_crop_transform(
                frame.shape,
                body_keypoints,
                body_keypoint_statuses,
                side,
                minimum_crop_pixels=self.minimum_crop_pixels,
            )
            record = _base_record(
                person_ref=person_ref,
                lock_epoch=lock_epoch,
                anatomical_side=side,
                frame_index=frame_index,
                timestamp=timestamp,
                source_video_sha256=source_video_sha256,
                recording_group_id=recording_group_id,
                source_model_version=self.model_version,
                runtime_version=self.runtime_version,
                transform=transform,
                observation_state="missing",
                reason="body_guided_roi_unavailable"
                if transform is None
                else "no_hand_detected_in_body_guided_roi",
                quality_gate_version=self.quality_gate_version,
            )
            guide_states = _body_guide_states(body_keypoint_statuses, side)
            record["association_checks"]["body_guide_observation_states"] = (
                guide_states
            )
            if transform is None:
                records.append(record)
                continue

            result, elapsed_ms, error = self._detect_crop(frame, transform)
            record["inference_time_ms"] = round(elapsed_ms, 3)
            if error is not None:
                record["backend_state"] = "error"
                record["observation_state"] = "uncertain"
                record["status"] = "uncertain"
                record["reason"] = error
                records.append(record)
                continue
            hand_index = self._choose_hand(
                result,
                transform,
                body_wrist_point(body_keypoints, side),
            )
            if hand_index is None:
                records.append(record)
                continue

            raw_landmarks = result.hand_landmarks[hand_index]
            mapped = map_normalized_landmarks(raw_landmarks, transform)
            record["landmarks"] = mapped
            record["landmark_count"] = len(mapped)
            record["reason"] = "real_model_landmarks"
            record["evidence_type"] = "real_hand_landmarker"
            if len(mapped) == HAND_LANDMARK_COUNT:
                record["observation_state"] = "detected"
                record["status"] = "proposed"
                record["occlusion"] = "not_inferred"
            else:
                record["observation_state"] = "uncertain"
                record["status"] = "uncertain"
                record["occlusion"] = "partial_or_invalid_model_output"

            handedness = list(getattr(result, "handedness", []) or [])
            if hand_index < len(handedness) and handedness[hand_index]:
                category = handedness[hand_index][0]
                record["model_handedness_label"] = getattr(
                    category, "category_name", None
                )
                record["model_handedness_score"] = _finite_or_none(
                    getattr(category, "score", None)
                )
            if any(
                guide_states[name] != "detected" for name in ("wrist", "elbow")
            ):
                _mark_uncertain(
                    record,
                    "body_wrist_or_elbow_was_not_directly_detected",
                )
            records.append(record)

        _check_cross_side_consistency(
            records,
            body_keypoints,
            maximum_own_wrist_distance_roi_ratio=(
                self.maximum_own_wrist_distance_roi_ratio
            ),
        )
        return [
            finalize_hand_pose_record(
                record,
                expected_person_ref=person_ref,
                expected_lock_epoch=lock_epoch,
                expected_anatomical_side=str(record["anatomical_side"]),
                track_state=track_state,
                lock_state=lock_state,
                maximum_own_wrist_distance_roi_ratio=(
                    self.maximum_own_wrist_distance_roi_ratio
                ),
                quality_gate_version=self.quality_gate_version,
            )
            for record in records
        ]


RealHandPoseBackend = MediaPipeHandLandmarkerBackend
