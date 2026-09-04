"""Stateful MediaPipe VIDEO backend with anatomical-side isolation.

This backend is intentionally separate from the stable IMAGE implementation.
It never carries geometry across missing or hard-boundary observations and
never rewrites source timestamps to satisfy MediaPipe's ordering contract.
"""

from __future__ import annotations

from importlib import metadata
import math
from pathlib import Path
import threading
import time
from typing import Any, Callable, Sequence

import numpy as np

from .backend import (
    ANATOMICAL_SIDES,
    HAND_LANDMARK_COUNT,
    HAND_QUALITY_GATE_VERSION,
    MAX_MODEL_WRIST_DISTANCE_ROI_RATIO,
    HandPoseBackend,
    MediaPipeHandLandmarkerBackend,
    _base_record,
    _body_guide_states,
    _check_cross_side_consistency,
    _finite_or_none,
    _mark_uncertain,
    finalize_hand_pose_record,
    map_normalized_landmarks,
)
from .roi import (
    CropTransform,
    body_wrist_point,
    build_hand_crop_transform,
)


VIDEO_BACKEND_MODE = "video"
LandmarkerFactory = Callable[[str], Any]
_HARD_TRACK_STATES = {"lost", "off_frame", "temporarily_lost"}
_HARD_LOCK_STATES = {"lost", "awaiting_manual_relock", "unlocked"}
_VALID_LOCK_STATES = {"tracked", "tracking", "locked"}


class MediaPipeHandLandmarkerVideoBackend(HandPoseBackend):
    """CPU VIDEO-mode backend with one lazy tracker per anatomical side."""

    enabled = True

    def __init__(
        self,
        model_path: str | Path,
        *,
        model_version: str = "mediapipe_hand_landmarker_float16_v1",
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        minimum_crop_pixels: int = 96,
        maximum_own_wrist_distance_roi_ratio: float = (
            MAX_MODEL_WRIST_DISTANCE_ROI_RATIO
        ),
        maximum_consecutive_model_missing_frames: int = 2,
        maximum_roi_center_jump_ratio: float = 0.5,
        maximum_roi_scale_change_ratio: float = 0.5,
        quality_gate_version: str = HAND_QUALITY_GATE_VERSION,
        landmarker_factory: LandmarkerFactory | None = None,
    ) -> None:
        path = Path(model_path).expanduser().resolve()
        if landmarker_factory is None and not path.is_file():
            raise FileNotFoundError(f"Hand model is missing: {path}")
        if int(maximum_consecutive_model_missing_frames) < 1:
            raise ValueError(
                "maximum_consecutive_model_missing_frames must be at least 1"
            )

        self.model_path = str(path)
        self.model_version = str(model_version)
        self.minimum_crop_pixels = int(minimum_crop_pixels)
        self.maximum_own_wrist_distance_roi_ratio = float(
            maximum_own_wrist_distance_roi_ratio
        )
        self.maximum_consecutive_model_missing_frames = int(
            maximum_consecutive_model_missing_frames
        )
        self.maximum_roi_center_jump_ratio = float(
            maximum_roi_center_jump_ratio
        )
        self.maximum_roi_scale_change_ratio = float(
            maximum_roi_scale_change_ratio
        )
        if (
            self.maximum_roi_center_jump_ratio < 0.0
            or self.maximum_roi_scale_change_ratio < 0.0
        ):
            raise ValueError("ROI discontinuity thresholds cannot be negative")
        self.quality_gate_version = str(quality_gate_version)
        self.min_hand_detection_confidence = float(
            min_hand_detection_confidence
        )
        self.min_hand_presence_confidence = float(
            min_hand_presence_confidence
        )
        self.min_tracking_confidence = float(min_tracking_confidence)
        self._injected_landmarker_factory = landmarker_factory

        try:
            self.runtime_version = metadata.version("mediapipe")
        except metadata.PackageNotFoundError:
            self.runtime_version = None
        self.inference_call_count = 0
        self.inference_times_ms: list[float] = []
        self.inference_error_count = 0
        self.availability_status = "available"
        self.reason = "real MediaPipe Hand Landmarker VIDEO CPU backend"

        self._state_lock = threading.RLock()
        self._side_locks = {
            side: threading.RLock() for side in ANATOMICAL_SIDES
        }
        self._landmarkers: dict[str, Any | None] = {
            side: None for side in ANATOMICAL_SIDES
        }
        self._session_last_timestamp_ms: dict[str, int | None] = {
            side: None for side in ANATOMICAL_SIDES
        }
        self._session_generations = {
            side: 0 for side in ANATOMICAL_SIDES
        }
        self._consecutive_model_missing = {
            side: 0 for side in ANATOMICAL_SIDES
        }
        self._pending_reset_reasons: dict[str, str | None] = {
            side: "initial_session" for side in ANATOMICAL_SIDES
        }
        self._previous_transforms: dict[str, CropTransform | None] = {
            side: None for side in ANATOMICAL_SIDES
        }
        self._context_person_ref: str | None = None
        self._context_lock_epoch: int | None = None
        self._context_last_input_timestamp_ms: int | None = None
        self._mp: Any | None = None
        self._closed = False

    def __enter__(self) -> "MediaPipeHandLandmarkerVideoBackend":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _load_mediapipe(self) -> Any:
        if self._mp is not None:
            return self._mp
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "mediapipe is required in the project-local environment"
            ) from exc
        self._mp = mp
        if self.runtime_version is None:
            self.runtime_version = str(mp.__version__)
        return mp

    def _default_landmarker_factory(self, anatomical_side: str) -> Any:
        del anatomical_side
        mp = self._load_mediapipe()
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=self.model_path,
                delegate=python.BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=self.min_hand_detection_confidence,
            min_hand_presence_confidence=self.min_hand_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        return vision.HandLandmarker.create_from_options(options)

    def _create_landmarker(self, anatomical_side: str) -> Any:
        factory = (
            self._injected_landmarker_factory
            or self._default_landmarker_factory
        )
        landmarker = factory(anatomical_side)
        if not callable(getattr(landmarker, "detect_for_video", None)):
            raise TypeError(
                "landmarker_factory must return an object with "
                "detect_for_video(image, timestamp_ms)"
            )
        if not callable(getattr(landmarker, "close", None)):
            raise TypeError(
                "landmarker_factory must return an object with close()"
            )
        return landmarker

    def _ensure_landmarker(
        self,
        anatomical_side: str,
    ) -> tuple[Any, int, str | None]:
        if self._closed:
            raise RuntimeError("Hand VIDEO backend is closed")
        landmarker = self._landmarkers[anatomical_side]
        reset_reason = self._pending_reset_reasons[anatomical_side]
        if landmarker is None:
            landmarker = self._create_landmarker(anatomical_side)
            self._landmarkers[anatomical_side] = landmarker
            self._session_generations[anatomical_side] += 1
            self._pending_reset_reasons[anatomical_side] = None
        return (
            landmarker,
            self._session_generations[anatomical_side],
            reset_reason,
        )

    def _reset_side(
        self,
        anatomical_side: str,
        reason: str,
    ) -> str | None:
        close_error: str | None = None
        with self._side_locks[anatomical_side]:
            landmarker = self._landmarkers[anatomical_side]
            self._landmarkers[anatomical_side] = None
            self._session_last_timestamp_ms[anatomical_side] = None
            self._consecutive_model_missing[anatomical_side] = 0
            self._previous_transforms[anatomical_side] = None
            self._pending_reset_reasons[anatomical_side] = str(reason)
            if landmarker is not None:
                try:
                    landmarker.close()
                except Exception as exc:
                    # Reset remains fail-closed even if a third-party close
                    # reports an error; no tracker handle is reused.
                    self.inference_error_count += 1
                    close_error = (
                        f"tracker_close_error:{anatomical_side}:"
                        f"{type(exc).__name__}"
                    )
        return close_error

    def _reset_all(self, reason: str) -> list[str]:
        errors: list[str] = []
        for side in ANATOMICAL_SIDES:
            error = self._reset_side(side, reason)
            if error is not None:
                errors.append(error)
        return errors

    def reset(
        self,
        *,
        person_ref: str | None = None,
        lock_epoch: int | None = None,
    ) -> None:
        with self._state_lock:
            has_explicit_context = (
                person_ref is not None and lock_epoch is not None
            )
            context_changed = (
                has_explicit_context
                and (
                    self._context_person_ref != str(person_ref)
                    or self._context_lock_epoch != int(lock_epoch)
                )
            )
            self._reset_all("explicit_reset")
            if not has_explicit_context or context_changed:
                self._context_last_input_timestamp_ms = None
            if has_explicit_context:
                self._context_person_ref = str(person_ref)
                self._context_lock_epoch = int(lock_epoch)
            else:
                self._context_person_ref = None
                self._context_lock_epoch = None

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._reset_all("backend_closed")
            self._closed = True

    @staticmethod
    def _input_timestamp_raw(timestamp: Any) -> str:
        try:
            return str(timestamp)
        except Exception:
            return f"<unprintable:{type(timestamp).__name__}>"

    @classmethod
    def _parse_backend_timestamp_ms(
        cls,
        timestamp: Any,
    ) -> tuple[int | None, str | None, str]:
        raw_value = cls._input_timestamp_raw(timestamp)
        if isinstance(timestamp, bool):
            return None, "invalid_source_timestamp:not_numeric", raw_value
        try:
            numeric = float(timestamp)
        except (TypeError, ValueError, OverflowError):
            return None, "invalid_source_timestamp:not_numeric", raw_value
        if not math.isfinite(numeric):
            return None, "invalid_source_timestamp:not_finite", raw_value
        if numeric < 0.0:
            return None, "invalid_source_timestamp:negative", raw_value
        try:
            timestamp_ms = int(round(numeric * 1000.0))
        except (OverflowError, ValueError):
            return None, "invalid_source_timestamp:out_of_range", raw_value
        if timestamp_ms > 9_223_372_036_854_775_807:
            return None, "invalid_source_timestamp:out_of_range", raw_value
        return timestamp_ms, None, raw_value

    @staticmethod
    def _roi_change_ratios(
        previous: CropTransform | None,
        current: CropTransform,
    ) -> tuple[float | None, float | None]:
        if previous is None:
            return None, None
        previous_center = (
            previous.x_offset + previous.x_scale / 2.0,
            previous.y_offset + previous.y_scale / 2.0,
        )
        current_center = (
            current.x_offset + current.x_scale / 2.0,
            current.y_offset + current.y_scale / 2.0,
        )
        reference_scale = max(
            1.0,
            float(previous.x_scale),
        )
        center_jump_ratio = math.dist(
            previous_center,
            current_center,
        ) / reference_scale
        scale_change_ratio = abs(
            float(current.x_scale) - float(previous.x_scale)
        ) / reference_scale
        return center_jump_ratio, scale_change_ratio

    def _video_base_record(
        self,
        *,
        person_ref: str,
        lock_epoch: int,
        anatomical_side: str,
        frame_index: int,
        timestamp: Any,
        source_video_sha256: str,
        recording_group_id: str,
        transform: CropTransform | None,
        observation_state: str,
        reason: str,
        backend_timestamp_ms: int | None,
        tracker_reset_reason: str | None,
        input_timestamp_raw: str,
        timestamp_validation_state: str = "valid",
        roi_center_jump_ratio: float | None = None,
        roi_scale_change_ratio: float | None = None,
        backend_state: str = "available",
    ) -> dict[str, Any]:
        record = _base_record(
            person_ref=person_ref,
            lock_epoch=lock_epoch,
            anatomical_side=anatomical_side,
            frame_index=frame_index,
            timestamp=timestamp,
            source_video_sha256=source_video_sha256,
            recording_group_id=recording_group_id,
            source_model_version=self.model_version,
            runtime_version=self.runtime_version,
            transform=transform,
            observation_state=observation_state,
            reason=reason,
            backend_state=backend_state,
            backend_mode=VIDEO_BACKEND_MODE,
            quality_gate_version=self.quality_gate_version,
            backend_timestamp_ms=backend_timestamp_ms,
            tracker_session_generation=self._session_generations[
                anatomical_side
            ],
            tracker_reset_reason=tracker_reset_reason,
        )
        record["raw_confidence_availability"]["tracking_confidence"] = (
            "not_exposed_by_mediapipe_python_result_in_video_mode"
        )
        record["input_timestamp_raw"] = input_timestamp_raw
        record["timestamp_validation_state"] = timestamp_validation_state
        record["roi_center_jump_ratio"] = (
            None
            if roi_center_jump_ratio is None
            else round(float(roi_center_jump_ratio), 6)
        )
        record["roi_scale_change_ratio"] = (
            None
            if roi_scale_change_ratio is None
            else round(float(roi_scale_change_ratio), 6)
        )
        return record

    def _finalize(
        self,
        record: dict[str, Any],
        *,
        person_ref: str,
        lock_epoch: int,
        track_state: str,
        lock_state: str,
    ) -> dict[str, Any]:
        return finalize_hand_pose_record(
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
        tracker_reset_reason: str,
        backend_timestamp_ms: int,
        input_timestamp_raw: str,
        backend_state: str = "available",
    ) -> list[dict[str, Any]]:
        return [
            self._finalize(
                self._video_base_record(
                    person_ref=person_ref,
                    lock_epoch=lock_epoch,
                    anatomical_side=side,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    source_video_sha256=source_video_sha256,
                    recording_group_id=recording_group_id,
                    transform=None,
                    observation_state=observation_state,
                    reason=reason,
                    backend_timestamp_ms=backend_timestamp_ms,
                    tracker_reset_reason=tracker_reset_reason,
                    input_timestamp_raw=input_timestamp_raw,
                    backend_state=backend_state,
                ),
                person_ref=person_ref,
                lock_epoch=lock_epoch,
                track_state=track_state,
                lock_state=lock_state,
            )
            for side in ANATOMICAL_SIDES
        ]

    def _invalid_timestamp_records(
        self,
        *,
        person_ref: str,
        lock_epoch: int,
        frame_index: int,
        source_video_sha256: str,
        recording_group_id: str,
        track_state: str,
        lock_state: str,
        detail_reason: str,
        input_timestamp_raw: str,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for side in ANATOMICAL_SIDES:
            record = self._video_base_record(
                person_ref=person_ref,
                lock_epoch=lock_epoch,
                anatomical_side=side,
                frame_index=frame_index,
                timestamp=0.0,
                source_video_sha256=source_video_sha256,
                recording_group_id=recording_group_id,
                transform=None,
                observation_state="uncertain",
                reason="invalid_backend_timestamp",
                backend_timestamp_ms=None,
                tracker_reset_reason="invalid_backend_timestamp",
                input_timestamp_raw=input_timestamp_raw,
                timestamp_validation_state="invalid",
                backend_state="error",
            )
            record["timestamp"] = None
            finalized = self._finalize(
                record,
                person_ref=person_ref,
                lock_epoch=lock_epoch,
                track_state=track_state,
                lock_state=lock_state,
            )
            for field in (
                "quality_reasons",
                "feature_eligibility_reasons",
            ):
                if detail_reason not in finalized[field]:
                    finalized[field].append(detail_reason)
            records.append(finalized)
        return records

    def _mediapipe_image(self, crop_rgb: np.ndarray) -> Any:
        mp = self._load_mediapipe()
        return mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=crop_rgb,
        )

    def _detect_crop_for_video(
        self,
        anatomical_side: str,
        frame: np.ndarray,
        transform: CropTransform,
        backend_timestamp_ms: int,
    ) -> tuple[Any | None, float, str | None, int, str | None]:
        x_min, y_min, x_max, y_max = transform.bbox
        crop_bgr = frame[y_min:y_max, x_min:x_max]
        if crop_bgr.size == 0:
            return (
                None,
                0.0,
                "empty_source_crop",
                self._session_generations[anatomical_side],
                None,
            )
        crop_rgb = np.ascontiguousarray(crop_bgr[:, :, ::-1])
        started = time.perf_counter()
        try:
            with self._side_locks[anatomical_side]:
                landmarker, generation, reset_reason = (
                    self._ensure_landmarker(anatomical_side)
                )
                result = landmarker.detect_for_video(
                    self._mediapipe_image(crop_rgb),
                    backend_timestamp_ms,
                )
                self._session_last_timestamp_ms[anatomical_side] = (
                    backend_timestamp_ms
                )
        except Exception as exc:
            elapsed = (time.perf_counter() - started) * 1000.0
            self.inference_call_count += 1
            self.inference_times_ms.append(elapsed)
            self.inference_error_count += 1
            return (
                None,
                elapsed,
                f"inference_error:{type(exc).__name__}",
                self._session_generations[anatomical_side],
                None,
            )
        elapsed = (time.perf_counter() - started) * 1000.0
        self.inference_call_count += 1
        self.inference_times_ms.append(elapsed)
        return result, elapsed, None, generation, reset_reason

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
        person_ref = str(person_ref)
        lock_epoch = int(lock_epoch)
        (
            backend_timestamp_ms,
            invalid_timestamp_reason,
            input_timestamp_raw,
        ) = self._parse_backend_timestamp_ms(timestamp)

        with self._state_lock:
            if self._closed:
                raise RuntimeError("Hand VIDEO backend is closed")
            context_changed = (
                self._context_person_ref is not None
                and (
                    self._context_person_ref != person_ref
                    or self._context_lock_epoch != lock_epoch
                )
            )
            if context_changed:
                self._reset_all(
                    "person_or_lock_epoch_changed"
                )
                self._context_last_input_timestamp_ms = None
            self._context_person_ref = person_ref
            self._context_lock_epoch = lock_epoch
            if invalid_timestamp_reason is not None:
                self._reset_all("invalid_backend_timestamp")
                return self._invalid_timestamp_records(
                    person_ref=person_ref,
                    lock_epoch=lock_epoch,
                    frame_index=frame_index,
                    source_video_sha256=source_video_sha256,
                    recording_group_id=recording_group_id,
                    track_state=track_state,
                    lock_state=lock_state,
                    detail_reason=invalid_timestamp_reason,
                    input_timestamp_raw=input_timestamp_raw,
                )
            assert backend_timestamp_ms is not None
            if (
                self._context_last_input_timestamp_ms is not None
                and backend_timestamp_ms
                <= self._context_last_input_timestamp_ms
            ):
                reset_reason = "non_monotonic_backend_timestamp"
                self._reset_all(reset_reason)
                return self._empty_records(
                    person_ref=person_ref,
                    lock_epoch=lock_epoch,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    source_video_sha256=source_video_sha256,
                    recording_group_id=recording_group_id,
                    observation_state="uncertain",
                    reason=reset_reason,
                    track_state=track_state,
                    lock_state=lock_state,
                    tracker_reset_reason=reset_reason,
                    backend_timestamp_ms=backend_timestamp_ms,
                    input_timestamp_raw=input_timestamp_raw,
                    backend_state="error",
                )
            self._context_last_input_timestamp_ms = backend_timestamp_ms

            hard_boundary = (
                normalized_track_state in _HARD_TRACK_STATES
                or normalized_lock_state in _HARD_LOCK_STATES
                or person_ref == "unlocked"
            )
            if hard_boundary:
                reset_reason = (
                    f"hard_boundary:{normalized_track_state}:"
                    f"{normalized_lock_state}"
                )
                self._reset_all(reset_reason)
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
                    tracker_reset_reason=reset_reason,
                    backend_timestamp_ms=backend_timestamp_ms,
                    input_timestamp_raw=input_timestamp_raw,
                )
            if normalized_track_state != "tracked":
                reset_reason = (
                    f"body_tracking_not_reliable:{normalized_track_state}"
                )
                self._reset_all(reset_reason)
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
                    tracker_reset_reason=reset_reason,
                    backend_timestamp_ms=backend_timestamp_ms,
                    input_timestamp_raw=input_timestamp_raw,
                )
            if normalized_lock_state not in _VALID_LOCK_STATES:
                reset_reason = (
                    f"lock_state_not_reliable:{normalized_lock_state}"
                )
                self._reset_all(reset_reason)
                return self._empty_records(
                    person_ref=person_ref,
                    lock_epoch=lock_epoch,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    source_video_sha256=source_video_sha256,
                    recording_group_id=recording_group_id,
                    observation_state="uncertain",
                    reason="lock_state_not_reliable_for_hand_tracking",
                    track_state=track_state,
                    lock_state=lock_state,
                    tracker_reset_reason=reset_reason,
                    backend_timestamp_ms=backend_timestamp_ms,
                    input_timestamp_raw=input_timestamp_raw,
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
                guide_states = _body_guide_states(
                    body_keypoint_statuses,
                    side,
                )
                if transform is None:
                    reset_reason = "body_guided_roi_unavailable"
                    reset_error = self._reset_side(side, reset_reason)
                    record = self._video_base_record(
                        person_ref=person_ref,
                        lock_epoch=lock_epoch,
                        anatomical_side=side,
                        frame_index=frame_index,
                        timestamp=timestamp,
                        source_video_sha256=source_video_sha256,
                        recording_group_id=recording_group_id,
                        transform=None,
                        observation_state=(
                            "uncertain" if reset_error else "missing"
                        ),
                        reason=reset_error or reset_reason,
                        backend_timestamp_ms=backend_timestamp_ms,
                        tracker_reset_reason=reset_error or reset_reason,
                        input_timestamp_raw=input_timestamp_raw,
                        backend_state=(
                            "error" if reset_error else "available"
                        ),
                    )
                    if reset_error:
                        record["tracker_reset_errors"] = [reset_error]
                    record["association_checks"][
                        "body_guide_observation_states"
                    ] = guide_states
                    records.append(record)
                    continue

                (
                    roi_center_jump_ratio,
                    roi_scale_change_ratio,
                ) = self._roi_change_ratios(
                    self._previous_transforms[side],
                    transform,
                )
                roi_discontinuity = (
                    roi_center_jump_ratio is not None
                    and roi_center_jump_ratio
                    > self.maximum_roi_center_jump_ratio
                ) or (
                    roi_scale_change_ratio is not None
                    and roi_scale_change_ratio
                    > self.maximum_roi_scale_change_ratio
                )
                roi_reset_error: str | None = None
                if roi_discontinuity:
                    roi_reset_error = self._reset_side(
                        side,
                        "roi_transform_discontinuity",
                    )

                if roi_reset_error is not None:
                    record = self._video_base_record(
                        person_ref=person_ref,
                        lock_epoch=lock_epoch,
                        anatomical_side=side,
                        frame_index=frame_index,
                        timestamp=timestamp,
                        source_video_sha256=source_video_sha256,
                        recording_group_id=recording_group_id,
                        transform=transform,
                        observation_state="uncertain",
                        reason=roi_reset_error,
                        backend_timestamp_ms=backend_timestamp_ms,
                        tracker_reset_reason=roi_reset_error,
                        input_timestamp_raw=input_timestamp_raw,
                        roi_center_jump_ratio=roi_center_jump_ratio,
                        roi_scale_change_ratio=roi_scale_change_ratio,
                        backend_state="error",
                    )
                    record["tracker_reset_errors"] = [roi_reset_error]
                    record["association_checks"][
                        "body_guide_observation_states"
                    ] = guide_states
                    records.append(record)
                    continue

                record = self._video_base_record(
                    person_ref=person_ref,
                    lock_epoch=lock_epoch,
                    anatomical_side=side,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    source_video_sha256=source_video_sha256,
                    recording_group_id=recording_group_id,
                    transform=transform,
                    observation_state="missing",
                    reason="no_hand_detected_in_body_guided_roi",
                    backend_timestamp_ms=backend_timestamp_ms,
                    tracker_reset_reason=None,
                    input_timestamp_raw=input_timestamp_raw,
                    roi_center_jump_ratio=roi_center_jump_ratio,
                    roi_scale_change_ratio=roi_scale_change_ratio,
                )
                record["association_checks"][
                    "body_guide_observation_states"
                ] = guide_states
                (
                    result,
                    elapsed_ms,
                    error,
                    generation,
                    session_reset_reason,
                ) = self._detect_crop_for_video(
                    side,
                    frame,
                    transform,
                    backend_timestamp_ms,
                )
                record["inference_time_ms"] = round(elapsed_ms, 3)
                record["tracker_session_generation"] = generation
                record["tracker_reset_reason"] = session_reset_reason
                if error is not None:
                    reset_reason = error
                    reset_error = self._reset_side(side, reset_reason)
                    record["backend_state"] = "error"
                    record["observation_state"] = "uncertain"
                    record["status"] = "uncertain"
                    record["reason"] = error
                    record["tracker_reset_reason"] = (
                        reset_error or reset_reason
                    )
                    if reset_error:
                        record["tracker_reset_errors"] = [reset_error]
                    records.append(record)
                    continue

                self._previous_transforms[side] = transform
                hand_index = MediaPipeHandLandmarkerBackend._choose_hand(
                    result,
                    transform,
                    body_wrist_point(body_keypoints, side),
                )
                if hand_index is None:
                    self._consecutive_model_missing[side] += 1
                    if (
                        self._consecutive_model_missing[side]
                        >= self.maximum_consecutive_model_missing_frames
                    ):
                        reset_reason = (
                            "consecutive_model_missing_limit_reached"
                        )
                        reset_error = self._reset_side(side, reset_reason)
                        record["tracker_reset_reason"] = (
                            reset_error or reset_reason
                        )
                        if reset_error:
                            record["backend_state"] = "error"
                            record["observation_state"] = "uncertain"
                            record["status"] = "uncertain"
                            record["reason"] = reset_error
                            record["tracker_reset_errors"] = [reset_error]
                    records.append(record)
                    continue

                self._consecutive_model_missing[side] = 0
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
                        category,
                        "category_name",
                        None,
                    )
                    record["model_handedness_score"] = _finite_or_none(
                        getattr(category, "score", None)
                    )
                if any(
                    guide_states[name] != "detected"
                    for name in ("wrist", "elbow")
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
                self._finalize(
                    record,
                    person_ref=person_ref,
                    lock_epoch=lock_epoch,
                    track_state=track_state,
                    lock_state=lock_state,
                )
                for record in records
            ]
