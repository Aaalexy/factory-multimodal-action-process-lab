"""Identity-free global multi-target tracking with a locked primary track."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from math import hypot
from time import perf_counter
from typing import Any, Literal

import numpy as np

from .joint_observation_validator import (
    JointObservationValidator,
    JointValidationResult,
    JointValidatorConfig,
)
from .manual_selection import ManualSelectionSeed, choose_manual_seed_detection
from .pose_postprocess import PoseDetection, box_iou
from .pose_smoother import PoseSmoother, SmoothedPose, SmootherConfig


TrackState = Literal["tracked", "uncertain", "lost"]
InternalTrackState = Literal[
    "tentative", "confirmed", "uncertain", "lost", "retired"
]
TORSO_INDICES = np.array([5, 6, 11, 12], dtype=np.int64)
# Distal wrists/ankles are deliberately excluded from person association.
# Their raw observations are much less stable and must not decide identity.
ASSOCIATION_KEYPOINT_INDICES = np.array(
    [5, 6, 7, 8, 11, 12, 13, 14], dtype=np.int64
)


@dataclass(frozen=True)
class TrackerConfig:
    keypoint_threshold: float = 0.25
    max_lost_frames: int = 8
    minimum_match_score: float = 0.34
    maximum_center_distance: float = 0.90
    minimum_area_similarity: float = 0.50
    maximum_dimension_ratio: float = 1.90
    track_retention_frames: int = 12
    minimum_confirmed_hits: int = 1
    switch_confirm_frames: int = 3
    switch_score_margin: float = 0.12
    ambiguous_match_margin: float = 0.055
    selection_mode: Literal["automatic", "manual"] = "automatic"
    manual_selection_seed: ManualSelectionSeed | None = None
    manual_minimum_match_score: float = 0.52
    validator_config: JointValidatorConfig = field(
        default_factory=JointValidatorConfig
    )
    smoother_config: SmootherConfig = field(default_factory=SmootherConfig)


@dataclass
class TrackResult:
    track_id: int | None
    detection: PoseDetection | None
    state: TrackState
    match_score: float
    lost_frames: int
    possible_switch: bool = False
    switch_reason: str | None = None
    candidate_scores: list[dict[str, Any]] = field(default_factory=list)
    smoothed_pose: SmoothedPose | None = None
    validation: JointValidationResult | None = None
    association_scores: list[dict[str, Any]] = field(default_factory=list)
    track_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    ambiguous_association: bool = False
    reacquired: bool = False
    primary_switch: bool = False
    selection_mode: str = "automatic"
    selection_source: str = "automatic"
    selected_candidate_id: str | None = None
    locked_track_id: int | None = None
    lock_state: str = "lost"
    temporarily_lost: bool = False
    track_switch_attempted: bool = False
    track_switch_blocked: bool = False
    manual_reselection: bool = False

    @property
    def uncertain(self) -> bool:
        return self.state != "tracked"


@dataclass
class _MotionTrack:
    track_id: int
    detection: PoseDetection
    validator: JointObservationValidator
    smoother: PoseSmoother
    age: int = 1
    hits: int = 1
    lost_frames: int = 0
    internal_state: InternalTrackState = "tentative"
    center_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float32)
    )
    size_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float32)
    )
    current_detection: PoseDetection | None = None
    current_score: float = 0.0
    current_reacquired: bool = False
    smoothed_pose: SmoothedPose | None = None
    validation: JointValidationResult | None = None
    bbox_history: list[list[float]] = field(default_factory=list)
    torso_history: list[list[list[float]]] = field(default_factory=list)
    keypoint_history: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.current_detection = self.detection
        self.bbox_history.append(self.detection.bbox.astype(float).tolist())

    def predicted_bbox(self) -> np.ndarray:
        size = np.maximum(2.0, self.detection.bbox[2:] - self.detection.bbox[:2])
        predicted_center = self.detection.center + self.center_velocity
        predicted_size = np.maximum(2.0, size + self.size_velocity)
        return np.concatenate(
            (predicted_center - predicted_size * 0.5, predicted_center + predicted_size * 0.5)
        ).astype(np.float32)

    def associate(
        self, detection: PoseDetection, score: float, minimum_confirmed_hits: int
    ) -> None:
        old_center = self.detection.center
        old_size = self.detection.bbox[2:] - self.detection.bbox[:2]
        center_step = detection.center - old_center
        size_step = (detection.bbox[2:] - detection.bbox[:2]) - old_size
        self.center_velocity = (
            0.65 * center_step + 0.35 * self.center_velocity
        ).astype(np.float32)
        self.size_velocity = (
            0.55 * size_step + 0.45 * self.size_velocity
        ).astype(np.float32)
        self.current_reacquired = self.lost_frames > 0
        self.detection = detection
        self.current_detection = detection
        self.current_score = float(score)
        self.age += 1
        self.hits += 1
        self.lost_frames = 0
        self.internal_state = (
            "confirmed" if self.hits >= minimum_confirmed_hits else "tentative"
        )
        self.bbox_history.append(detection.bbox.astype(float).tolist())
        self.bbox_history[:] = self.bbox_history[-30:]

    def miss(self, max_lost_frames: int, retention_frames: int) -> None:
        self.current_detection = None
        self.current_score = 0.0
        self.current_reacquired = False
        self.age += 1
        self.lost_frames += 1
        self.center_velocity *= 0.82
        self.size_velocity *= 0.70
        if self.lost_frames > retention_frames:
            self.internal_state = "retired"
        elif self.lost_frames > max_lost_frames:
            self.internal_state = "lost"
        else:
            self.internal_state = "uncertain"

    def process_pose(
        self, frame_shape: tuple[int, int] | tuple[int, int, int] | None
    ) -> tuple[float, float, float]:
        started = perf_counter()
        if self.current_detection is None:
            limits = (
                self.validation.interpolation_limits
                if self.validation is not None else None
            )
            missing_observations = np.full(17, "missing", dtype="<U12")
            smoothing_started = perf_counter()
            self.smoothed_pose = self.smoother.update(
                None, limits, missing_observations
            )
            smoothing_ms = (perf_counter() - smoothing_started) * 1000.0
            return 0.0, smoothing_ms, (perf_counter() - started) * 1000.0
        validation_started = perf_counter()
        validation = self.validator.validate(
            self.current_detection.keypoints,
            self.current_detection.bbox,
            frame_shape,
        )
        validation_ms = (perf_counter() - validation_started) * 1000.0
        sanitized = PoseDetection(
            self.current_detection.bbox.copy(),
            self.current_detection.confidence,
            validation.validated_keypoints,
        )
        smoothing_started = perf_counter()
        smoothed = self.smoother.update(
            sanitized,
            validation.interpolation_limits,
            validation.observation_statuses,
        )
        smoothing_ms = (perf_counter() - smoothing_started) * 1000.0
        # Preserve actual model output separately from the sanitized input.
        smoothed.raw_keypoints = validation.raw_keypoints.copy()
        self.validation = validation
        self.smoothed_pose = smoothed
        reliable = np.isfinite(validation.validated_keypoints[:, :2]).all(axis=1)
        self.keypoint_history.append(validation.validated_keypoints.copy())
        self.keypoint_history[:] = self.keypoint_history[-30:]
        torso = validation.validated_keypoints[TORSO_INDICES, :2]
        self.torso_history.append(
            np.where(np.isfinite(torso), torso, np.nan).astype(float).tolist()
        )
        self.torso_history[:] = self.torso_history[-30:]
        return validation_ms, smoothing_ms, (perf_counter() - started) * 1000.0


def _bbox_iou(left: np.ndarray, right: np.ndarray) -> float:
    return float(box_iou(left, np.asarray(right, dtype=np.float32).reshape(1, 4))[0])


def _area_similarity_boxes(left: np.ndarray, right: np.ndarray) -> float:
    left_size = np.maximum(0.0, left[2:] - left[:2])
    right_size = np.maximum(0.0, right[2:] - right[:2])
    left_area = float(np.prod(left_size))
    right_area = float(np.prod(right_size))
    if min(left_area, right_area) <= 0:
        return 0.0
    return min(left_area, right_area) / max(left_area, right_area)


def _dimension_ratio(left: np.ndarray, right: np.ndarray) -> float:
    left_size = np.maximum(1.0, left[2:] - left[:2])
    right_size = np.maximum(1.0, right[2:] - right[:2])
    ratios = np.maximum(left_size / right_size, right_size / left_size)
    return float(np.max(ratios))


def _keypoint_similarity(
    previous: PoseDetection, current: PoseDetection, threshold: float
) -> tuple[float, float, int]:
    previous_kpts = previous.keypoints
    current_kpts = current.keypoints
    valid = (
        np.isfinite(previous_kpts[:, :2]).all(axis=1)
        & np.isfinite(current_kpts[:, :2]).all(axis=1)
        & (previous_kpts[:, 2] >= threshold)
        & (current_kpts[:, 2] >= threshold)
    )
    association_mask = np.zeros(17, dtype=bool)
    association_mask[ASSOCIATION_KEYPOINT_INDICES] = True
    valid &= association_mask
    if not np.any(valid):
        return 0.0, 0.0, 0
    scale = max(1.0, np.sqrt(previous.area))
    distances = np.linalg.norm(
        current_kpts[valid, :2] - previous_kpts[valid, :2], axis=1
    )
    similarity = float(np.exp(-np.median(distances) / scale))
    torso_indices = TORSO_INDICES[valid[TORSO_INDICES]]
    if len(torso_indices) < 2:
        torso_similarity = 0.0
    else:
        torso_distances = np.linalg.norm(
            current_kpts[torso_indices, :2]
            - previous_kpts[torso_indices, :2],
            axis=1,
        )
        torso_similarity = float(np.exp(-np.median(torso_distances) / scale))
    return similarity, torso_similarity, int(len(torso_indices))


class PrimaryPersonTracker:
    """Globally associate all anonymous tracks, then lock one primary."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        if self.config.max_lost_frames < 0:
            raise ValueError("max_lost_frames cannot be negative")
        if self.config.switch_confirm_frames < 1:
            raise ValueError("switch_confirm_frames must be positive")
        if self.config.selection_mode not in ("automatic", "manual"):
            raise ValueError("selection_mode must be automatic or manual")
        if self.config.selection_mode == "manual" and self.config.manual_selection_seed is None:
            raise ValueError("manual selection mode requires a confirmed selection seed")
        self.reset()

    def reset(self) -> None:
        self._tracks: dict[int, _MotionTrack] = {}
        self._primary_track_id: int | None = None
        self._last_primary_track_id: int | None = None
        self._next_track_id = 1
        self._challenger_track_id: int | None = None
        self._challenger_wins = 0
        self.possible_switch_count = 0
        self.primary_track_switch_count = 0
        self.reacquisition_count = 0
        self.maximum_primary_track_lost_frames = 0
        self.ambiguous_association_frame_count = 0
        self.switch_events: list[dict[str, Any]] = []
        self._association_scores: list[dict[str, Any]] = []
        self._ambiguous_tracks: set[int] = set()
        self._frame_number = 0
        self._manual_seed_pending = self.config.manual_selection_seed
        self._selected_candidate_id = (
            self._manual_seed_pending.candidate_id
            if self._manual_seed_pending is not None else None
        )
        self._manual_reselection = bool(
            self._manual_seed_pending and self._manual_seed_pending.manual_reselection
        )
        self._awaiting_manual_relock = False
        self.last_manual_match: dict[str, Any] = {
            "score": 0.0, "ambiguous": False, "reason": "not_attempted",
        }
        self.lock_events: list[dict[str, Any]] = []
        self.track_switch_blocked_count = 0
        self.primary_full_processing_count = 0
        self.non_primary_full_processing_count = 0
        self.primary_processing_times_ms: list[float] = []
        self.non_primary_processing_times_ms: list[float] = []
        self.joint_validation_times_ms: list[float] = []
        self.pose_smoothing_times_ms: list[float] = []
        self.association_times_ms: list[float] = []
        self.active_track_counts: list[int] = []
        self.confirmed_track_counts: list[int] = []

    def _new_track(self, detection: PoseDetection) -> _MotionTrack:
        validator_config = JointValidatorConfig(
            **{
                **self.config.validator_config.__dict__,
                "keypoint_threshold": self.config.keypoint_threshold,
            }
        )
        smoother_config = SmootherConfig(
            **{
                **self.config.smoother_config.__dict__,
                "keypoint_threshold": self.config.keypoint_threshold,
            }
        )
        track = _MotionTrack(
            self._next_track_id,
            detection,
            JointObservationValidator(validator_config),
            PoseSmoother(smoother_config),
        )
        if self.config.minimum_confirmed_hits <= 1:
            track.internal_state = "confirmed"
        track.current_score = 1.0
        self._tracks[track.track_id] = track
        self._next_track_id += 1
        return track

    def _association(
        self, track: _MotionTrack, detection: PoseDetection
    ) -> dict[str, float | bool]:
        predicted = track.predicted_bbox()
        predicted_center = (predicted[:2] + predicted[2:]) * 0.5
        diagonal = max(
            1.0,
            hypot(
                float(predicted[2] - predicted[0]),
                float(predicted[3] - predicted[1]),
            ),
        )
        center_distance = float(
            np.linalg.norm(detection.center - predicted_center) / diagonal
        )
        center_similarity = max(0.0, 1.0 - center_distance)
        iou = _bbox_iou(predicted, detection.bbox)
        area = _area_similarity_boxes(predicted, detection.bbox)
        dimension_ratio = _dimension_ratio(predicted, detection.bbox)
        pose, torso, torso_count = _keypoint_similarity(
            track.detection, detection, self.config.keypoint_threshold
        )
        observed_step = detection.center - track.detection.center
        motion_error = float(
            np.linalg.norm(observed_step - track.center_velocity) / diagonal
        )
        motion = float(np.exp(-2.0 * motion_error))
        confidence = float(np.clip(detection.confidence, 0.0, 1.0))
        continuity = 1.0 / (1.0 + track.lost_frames)
        score = (
            0.20 * iou
            + 0.16 * center_similarity
            + 0.13 * area
            + 0.10 * pose
            + 0.24 * torso
            + 0.08 * motion
            + 0.06 * confidence
            + 0.03 * continuity
        )
        allowed_distance = self.config.maximum_center_distance * (
            1.0 + 0.10 * min(track.lost_frames, self.config.max_lost_frames)
        )
        reliable = (
            score >= self.config.minimum_match_score
            and center_distance <= allowed_distance
            and area >= self.config.minimum_area_similarity
            and dimension_ratio <= self.config.maximum_dimension_ratio
            and (torso_count < 2 or torso >= 0.20)
        )
        manual_recovery_gate = bool(
            self.config.selection_mode == "manual"
            and track.track_id == self._primary_track_id
            and track.lost_frames > 0
        )
        if manual_recovery_gate:
            # Recovery is intentionally stricter than ordinary global
            # association.  Confidence or apparent size cannot compensate for
            # poor predicted-bbox, torso, scale or direction continuity.
            reliable = bool(
                reliable
                and score >= max(
                    self.config.minimum_match_score,
                    self.config.manual_minimum_match_score - 0.04,
                )
                and center_distance <= min(allowed_distance, 0.55)
                and area >= max(self.config.minimum_area_similarity, 0.62)
                and dimension_ratio <= min(self.config.maximum_dimension_ratio, 1.55)
                and torso_count >= 2
                and torso >= 0.32
                and motion >= 0.20
            )
        return {
            "score": float(score),
            "cost": float(1.0 - score),
            "iou": iou,
            "center_distance": center_distance,
            "area_similarity": area,
            "dimension_ratio": dimension_ratio,
            "pose_similarity": pose,
            "torso_similarity": torso,
            "motion_similarity": motion,
            "confidence": confidence,
            "track_age": track.age,
            "track_lost_frames": track.lost_frames,
            "manual_recovery_gate": manual_recovery_gate,
            "reliable": bool(reliable),
        }

    @staticmethod
    def _optimal_pairs(
        track_ids: list[int],
        detection_count: int,
        lookup: dict[tuple[int, int], dict[str, float | bool]],
        excluded_tracks: set[int],
        preused_detection_mask: int = 0,
    ) -> list[tuple[int, int]]:
        """Maximum-score one-to-one assignment using exact bitmask DP."""

        @lru_cache(maxsize=None)
        def solve(track_offset: int, used_mask: int) -> tuple[float, tuple[tuple[int, int], ...]]:
            if track_offset >= len(track_ids):
                return 0.0, ()
            track_id = track_ids[track_offset]
            best_score, best_pairs = solve(track_offset + 1, used_mask)
            if track_id in excluded_tracks:
                return best_score, best_pairs
            for detection_index in range(detection_count):
                if used_mask & (1 << detection_index):
                    continue
                details = lookup[(track_id, detection_index)]
                if not details["reliable"]:
                    continue
                remaining_score, remaining_pairs = solve(
                    track_offset + 1, used_mask | (1 << detection_index)
                )
                candidate_score = float(details["score"]) + remaining_score
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_pairs = ((track_id, detection_index),) + remaining_pairs
            return best_score, best_pairs

        return list(solve(0, preused_detection_mask)[1])

    def _assign(
        self,
        detections: list[PoseDetection],
        frame_shape: tuple[int, int] | tuple[int, int, int] | None,
    ) -> list[dict[str, Any]]:
        for track in self._tracks.values():
            track.current_detection = None
            track.current_score = 0.0
            track.current_reacquired = False
        existing_track_ids = [
            track_id for track_id, track in self._tracks.items()
            if track.internal_state != "retired"
        ]
        lookup: dict[tuple[int, int], dict[str, float | bool]] = {}
        association_scores: list[dict[str, Any]] = []
        for track_id in existing_track_ids:
            track = self._tracks[track_id]
            for detection_index, detection in enumerate(detections):
                details = self._association(track, detection)
                lookup[(track_id, detection_index)] = details
                association_scores.append(
                    {
                        "track_id": track_id,
                        "detection_index": detection_index,
                        **{
                            key: (
                                round(float(value), 6)
                                if not isinstance(value, bool) else value
                            )
                            for key, value in details.items()
                        },
                    }
                )
        ambiguous_tracks: set[int] = set()
        if self._primary_track_id in existing_track_ids:
            primary = self._tracks[self._primary_track_id]
            reliable_scores = sorted(
                [
                    float(lookup[(primary.track_id, index)]["score"])
                    for index in range(len(detections))
                    if lookup[(primary.track_id, index)]["reliable"]
                ],
                reverse=True,
            )
            if (
                len(reliable_scores) >= 2
                and reliable_scores[0] - reliable_scores[1]
                < self.config.ambiguous_match_margin
            ):
                ambiguous_tracks.add(primary.track_id)
        # A confirmed primary owns its strongest reliable association.  This
        # reservation prevents a nearby secondary track from winning the same
        # detection by a tiny global-score difference.  All remaining tracks
        # are still solved by one-to-one global assignment.
        reserved_pairs: list[tuple[int, int]] = []
        reserved_detection_mask = 0
        assignment_exclusions = set(ambiguous_tracks)
        primary_id = self._primary_track_id
        manual_primary_blocked = bool(
            self.config.selection_mode == "manual"
            and self._awaiting_manual_relock
            and primary_id in existing_track_ids
        )
        if manual_primary_blocked and primary_id is not None:
            assignment_exclusions.add(primary_id)
        if (
            primary_id in existing_track_ids
            and primary_id not in ambiguous_tracks
            and not manual_primary_blocked
        ):
            reliable_primary = [
                (index, lookup[(primary_id, index)])
                for index in range(len(detections))
                if lookup[(primary_id, index)]["reliable"]
            ]
            if reliable_primary:
                detection_index, _ = max(
                    reliable_primary, key=lambda item: float(item[1]["score"])
                )
                reserved_pairs.append((primary_id, detection_index))
                reserved_detection_mask |= 1 << detection_index
                assignment_exclusions.add(primary_id)
        pairs = reserved_pairs + self._optimal_pairs(
            existing_track_ids,
            len(detections),
            lookup,
            assignment_exclusions,
            reserved_detection_mask,
        )
        assigned_tracks = {track_id for track_id, _ in pairs}
        assigned_detections = {index for _, index in pairs}
        detection_tracks: dict[int, int] = {}
        for track_id, detection_index in pairs:
            details = lookup[(track_id, detection_index)]
            self._tracks[track_id].associate(
                detections[detection_index],
                float(details["score"]),
                self.config.minimum_confirmed_hits,
            )
            detection_tracks[detection_index] = track_id
        for track_id in existing_track_ids:
            if track_id not in assigned_tracks:
                self._tracks[track_id].miss(
                    self.config.max_lost_frames,
                    self.config.track_retention_frames,
                )
        for detection_index, detection in enumerate(detections):
            if detection_index not in assigned_detections:
                track = self._new_track(detection)
                detection_tracks[detection_index] = track.track_id
        active_tracks = [
            track for track in self._tracks.values()
            if track.internal_state != "retired"
        ]
        self.active_track_counts.append(len(active_tracks))
        self.confirmed_track_counts.append(sum(
            track.internal_state == "confirmed" for track in active_tracks
        ))

        candidates: list[dict[str, Any]] = []
        for detection_index, detection in enumerate(detections):
            track_id = detection_tracks[detection_index]
            assigned_details = lookup.get((track_id, detection_index))
            if assigned_details is None:
                prior = [
                    details
                    for (prior_track, index), details in lookup.items()
                    if index == detection_index
                ]
                assigned_details = (
                    max(prior, key=lambda item: float(item["score"]))
                    if prior else {
                        "score": 1.0, "cost": 0.0, "iou": 0.0,
                        "center_distance": 0.0, "area_similarity": 1.0,
                        "dimension_ratio": 1.0, "pose_similarity": 0.0,
                        "torso_similarity": 0.0, "motion_similarity": 0.0,
                        "confidence": detection.confidence, "track_age": 1,
                        "track_lost_frames": 0, "reliable": True,
                    }
                )
            candidates.append(
                {
                    "candidate_index": detection_index,
                    "track_id": track_id,
                    "bbox": [round(float(value), 3) for value in detection.bbox],
                    "confidence": round(float(detection.confidence), 6),
                    **{
                        key: (
                            round(float(value), 6)
                            if not isinstance(value, bool) else value
                        )
                        for key, value in assigned_details.items()
                    },
                    "is_primary": track_id == self._primary_track_id,
                    "track_state": self._tracks[track_id].internal_state,
                }
            )
        self._association_scores = association_scores
        self._ambiguous_tracks = ambiguous_tracks
        self.ambiguous_association_frame_count += int(
            self._primary_track_id in ambiguous_tracks
        )
        return candidates

    def _process_primary_pose(
        self,
        track: _MotionTrack | None,
        frame_shape: tuple[int, int] | tuple[int, int, int] | None,
    ) -> None:
        """Run joint-level processing only for the currently selected primary.

        Secondary tracks retain raw bbox, torso pose and motion state for global
        association.  They intentionally do not update validators, smoothers,
        bone histories, derived skeletons or joint-level exports.
        """

        if track is None or track.internal_state == "retired":
            return
        validation_ms, smoothing_ms, processing_ms = track.process_pose(frame_shape)
        self.joint_validation_times_ms.append(validation_ms)
        self.pose_smoothing_times_ms.append(smoothing_ms)
        self.primary_full_processing_count += 1
        self.primary_processing_times_ms.append(processing_ms)

    @staticmethod
    def _refresh_result_pose(result: TrackResult, track: _MotionTrack | None) -> TrackResult:
        if track is not None:
            result.smoothed_pose = track.smoothed_pose
            result.validation = track.validation
        return result

    @staticmethod
    def _initial_priority(track: _MotionTrack, largest_area: float) -> float:
        return 0.65 * float(track.detection.confidence) + 0.35 * min(
            1.0, track.detection.area / max(largest_area, 1.0)
        )

    def _track_diagnostics(self) -> list[dict[str, Any]]:
        diagnostics: list[dict[str, Any]] = []
        for track in self._tracks.values():
            validation = track.validation
            diagnostics.append(
                {
                    "track_id": track.track_id,
                    "state": track.internal_state,
                    "age": track.age,
                    "hits": track.hits,
                    "lost_frames": track.lost_frames,
                    "match_score": round(track.current_score, 6),
                    "bbox": track.detection.bbox.astype(float).tolist(),
                    "center_velocity": track.center_velocity.astype(float).tolist(),
                    "edge_state": bool(validation.edge_state) if validation else False,
                    "rejected_joints": validation.rejected_indices if validation else [],
                    "provisional_joints": validation.provisional_indices if validation else [],
                }
            )
        return diagnostics

    def _result(
        self,
        track: _MotionTrack | None,
        state: TrackState,
        candidates: list[dict[str, Any]],
        *,
        possible_switch: bool = False,
        switch_reason: str | None = None,
        primary_switch: bool = False,
        lock_state: str | None = None,
        track_switch_attempted: bool = False,
        track_switch_blocked: bool = False,
    ) -> TrackResult:
        ambiguous = bool(track and track.track_id in self._ambiguous_tracks)
        resolved_lock_state = lock_state or (
            "tracking" if state == "tracked" else (
                "ambiguous" if ambiguous else (
                    "temporarily_lost" if state == "uncertain" else "lost"
                )
            )
        )
        return TrackResult(
            track_id=track.track_id if track else None,
            detection=track.current_detection if track else None,
            state=state,
            match_score=track.current_score if track else 0.0,
            lost_frames=track.lost_frames if track else 0,
            possible_switch=possible_switch,
            switch_reason=switch_reason,
            candidate_scores=candidates,
            smoothed_pose=track.smoothed_pose if track else None,
            validation=track.validation if track else None,
            association_scores=self._association_scores,
            track_diagnostics=self._track_diagnostics(),
            ambiguous_association=ambiguous,
            reacquired=bool(track and track.current_reacquired),
            primary_switch=primary_switch,
            selection_mode=self.config.selection_mode,
            selection_source=("manual" if self.config.selection_mode == "manual" else "automatic"),
            selected_candidate_id=self._selected_candidate_id,
            locked_track_id=self._primary_track_id,
            lock_state=resolved_lock_state,
            temporarily_lost=resolved_lock_state == "temporarily_lost",
            track_switch_attempted=track_switch_attempted,
            track_switch_blocked=track_switch_blocked,
            manual_reselection=self._manual_reselection,
        )

    def _select_manual_seed(
        self, detections: list[PoseDetection], candidates: list[dict[str, Any]]
    ) -> TrackResult:
        seed = self._manual_seed_pending
        if seed is None:
            return self._result(None, "lost", candidates, lock_state="lost")
        detection_index, score, ambiguous = choose_manual_seed_detection(
            seed,
            detections,
            minimum_score=self.config.manual_minimum_match_score,
            ambiguity_margin=self.config.ambiguous_match_margin,
            keypoint_threshold=self.config.keypoint_threshold,
        )
        if detection_index is None:
            reason = "manual_seed_ambiguous" if ambiguous else "manual_seed_not_matched"
            self.last_manual_match = {"score": float(score), "ambiguous": bool(ambiguous), "reason": reason}
            self.lock_events.append({
                "frame": self._frame_number,
                "event": reason,
                "selected_candidate_id": seed.candidate_id,
                "score": round(score, 6),
            })
            return self._result(
                None, "lost", candidates,
                switch_reason=reason,
                lock_state="ambiguous" if ambiguous else "lost",
            )
        track_id = next(
            int(item["track_id"])
            for item in candidates
            if int(item["candidate_index"]) == detection_index
        )
        self._primary_track_id = track_id
        self.last_manual_match = {"score": float(score), "ambiguous": False, "reason": "manual_lock_established"}
        self._last_primary_track_id = track_id
        self._manual_seed_pending = None
        for candidate in candidates:
            candidate["is_primary"] = int(candidate["track_id"]) == track_id
        track = self._tracks[track_id]
        self.lock_events.append({
            "frame": self._frame_number,
            "event": "manual_lock_established",
            "selected_candidate_id": seed.candidate_id,
            "track_id": track_id,
            "score": round(score, 6),
            "selection_source": "manual",
            "selection_frame_index": seed.selection_frame_index,
            "selection_timestamp": seed.selection_timestamp,
        })
        return self._result(track, "tracked", candidates, lock_state="tracking")

    def _select_initial(self, candidates: list[dict[str, Any]]) -> TrackResult:
        visible = [
            track for track in self._tracks.values()
            if track.current_detection is not None and track.internal_state != "retired"
        ]
        if not visible:
            return self._result(None, "lost", candidates)
        largest = max(track.detection.area for track in visible)
        selected = max(
            visible, key=lambda track: self._initial_priority(track, largest)
        )
        previous = self._last_primary_track_id
        self._primary_track_id = selected.track_id
        self._last_primary_track_id = selected.track_id
        for candidate in candidates:
            candidate["is_primary"] = candidate["track_id"] == selected.track_id
        if previous is not None and previous != selected.track_id:
            reason = "previous_primary_retired; selected_new_anonymous_track"
            self.primary_track_switch_count += 1
            self.possible_switch_count += 1
            self.switch_events.append(
                {
                    "from_track_id": previous,
                    "to_track_id": selected.track_id,
                    "reason": reason,
                }
            )
            return self._result(
                selected,
                "tracked" if selected.internal_state == "confirmed" else "uncertain",
                candidates,
                possible_switch=True,
                switch_reason=reason,
                primary_switch=True,
            )
        return self._result(
            selected,
            "tracked" if selected.internal_state == "confirmed" else "uncertain",
            candidates,
        )

    def _best_challenger(self) -> _MotionTrack | None:
        visible = [
            track
            for track in self._tracks.values()
            if track.current_detection is not None
            and track.track_id != self._primary_track_id
            and track.internal_state in ("tentative", "confirmed")
        ]
        if not visible:
            return None
        return max(
            visible,
            key=lambda track: track.current_score + 0.02 * min(track.hits, 10),
        )

    def update(
        self,
        detections: list[PoseDetection],
        frame_shape: tuple[int, int] | tuple[int, int, int] | None = None,
    ) -> TrackResult:
        self._frame_number += 1
        detections = list(detections)
        association_started = perf_counter()
        candidates = self._assign(detections, frame_shape)
        self.association_times_ms.append(
            (perf_counter() - association_started) * 1000.0
        )
        if self._primary_track_id is None:
            if self.config.selection_mode == "manual":
                result = self._select_manual_seed(detections, candidates)
            else:
                result = self._select_initial(candidates)
            selected = self._tracks.get(self._primary_track_id)
            self._process_primary_pose(selected, frame_shape)
            return self._refresh_result_pose(result, selected)

        primary = self._tracks.get(self._primary_track_id)
        self._process_primary_pose(primary, frame_shape)
        if primary is not None and primary.current_detection is not None:
            if primary.current_reacquired:
                self.reacquisition_count += 1
                if self.config.selection_mode == "manual":
                    self.lock_events.append({
                        "frame": self._frame_number,
                        "event": "manual_track_reacquired",
                        "track_id": primary.track_id,
                    })
            self._challenger_track_id = None
            self._challenger_wins = 0
            self._last_primary_track_id = primary.track_id
            return self._result(
                primary,
                "tracked" if primary.internal_state == "confirmed" else "uncertain",
                candidates,
            )

        if primary is None:
            if self.config.selection_mode == "manual":
                return self._result(
                    None,
                    "lost",
                    candidates,
                    lock_state=(
                        "awaiting_manual_relock"
                        if self._awaiting_manual_relock else "lost"
                    ),
                )
            self._primary_track_id = None
            return self._select_initial(candidates)
        self.maximum_primary_track_lost_frames = max(
            self.maximum_primary_track_lost_frames, primary.lost_frames
        )
        challenger = self._best_challenger()
        if challenger is None:
            self._challenger_track_id = None
            self._challenger_wins = 0
        else:
            wins = challenger.current_score >= self.config.switch_score_margin
            if wins and challenger.track_id == self._challenger_track_id:
                self._challenger_wins += 1
            elif wins:
                self._challenger_track_id = challenger.track_id
                self._challenger_wins = 1
            else:
                self._challenger_track_id = None
                self._challenger_wins = 0

        beyond_recovery = primary.lost_frames > self.config.max_lost_frames
        if (
            self.config.selection_mode == "manual"
            and beyond_recovery
            and not self._awaiting_manual_relock
        ):
            self._awaiting_manual_relock = True
            self.lock_events.append({
                "frame": self._frame_number,
                "event": "awaiting_manual_relock",
                "track_id": primary.track_id,
                "lost_frames": primary.lost_frames,
            })
        switch_ready = (
            beyond_recovery
            and challenger is not None
            and challenger.track_id == self._challenger_track_id
            and self._challenger_wins >= self.config.switch_confirm_frames
            and challenger.internal_state == "confirmed"
        )
        if switch_ready and self.config.selection_mode == "manual":
            reason = (
                f"strict_manual_lock_blocked_candidate_track_{challenger.track_id}; "
                f"locked_track_{primary.track_id}_lost_{primary.lost_frames}_frames"
            )
            self.track_switch_blocked_count += 1
            self.lock_events.append({
                "frame": self._frame_number,
                "event": "automatic_switch_blocked",
                "locked_track_id": primary.track_id,
                "candidate_track_id": challenger.track_id,
                "reason": reason,
            })
            return self._result(
                primary,
                "lost",
                candidates,
                switch_reason=reason,
                lock_state="awaiting_manual_relock",
                track_switch_attempted=True,
                track_switch_blocked=True,
            )
        if switch_ready:
            old_id = primary.track_id
            self._primary_track_id = challenger.track_id
            self._last_primary_track_id = challenger.track_id
            reason = (
                f"primary_lost>{self.config.max_lost_frames}; challenger "
                f"won {self._challenger_wins} consecutive frames with margin "
                f">={self.config.switch_score_margin:.2f}"
            )
            self.switch_events.append(
                {
                    "from_track_id": old_id,
                    "to_track_id": challenger.track_id,
                    "reason": reason,
                }
            )
            self.primary_track_switch_count += 1
            self.possible_switch_count += 1
            self._challenger_track_id = None
            self._challenger_wins = 0
            self._process_primary_pose(challenger, frame_shape)
            for candidate in candidates:
                candidate["is_primary"] = (
                    candidate["track_id"] == challenger.track_id
                )
            return self._result(
                challenger,
                "uncertain",
                candidates,
                possible_switch=True,
                switch_reason=reason,
                primary_switch=True,
            )

        if beyond_recovery and not detections and self.config.selection_mode != "manual":
            self._primary_track_id = None
        ambiguous = primary.track_id in self._ambiguous_tracks
        lock_state = (
            "awaiting_manual_relock"
            if self.config.selection_mode == "manual" and self._awaiting_manual_relock
            else (
                "ambiguous" if ambiguous else (
                    "lost" if beyond_recovery else "temporarily_lost"
                )
            )
        )
        if self.config.selection_mode == "manual" and primary.lost_frames == 1:
            self.lock_events.append({
                "frame": self._frame_number,
                "event": "manual_track_occlusion_started",
                "track_id": primary.track_id,
            })
        if self.config.selection_mode == "manual" and ambiguous:
            self.lock_events.append({
                "frame": self._frame_number,
                "event": "manual_track_association_ambiguous",
                "track_id": primary.track_id,
            })
        if (
            self.config.selection_mode == "manual"
            and beyond_recovery
            and primary.lost_frames == self.config.max_lost_frames + 1
        ):
            self.lock_events.append({
                "frame": self._frame_number,
                "event": "manual_track_lost",
                "track_id": primary.track_id,
                "lost_frames": primary.lost_frames,
            })
        return self._result(
            primary,
            "lost" if beyond_recovery else "uncertain",
            candidates,
            lock_state=lock_state,
        )

    def reselect_manual(self, seed: ManualSelectionSeed) -> None:
        """Start a new anonymous lock segment without cross-person interpolation."""

        if self.config.selection_mode != "manual":
            raise RuntimeError("manual reselection requires manual selection mode")
        next_track_id = self._next_track_id
        self.reset()
        self._next_track_id = next_track_id
        self._manual_seed_pending = seed
        self._selected_candidate_id = seed.candidate_id
        self._manual_reselection = True
        self.lock_events.append({
            "frame": 0,
            "event": "manual_reselection_requested",
            "selected_candidate_id": seed.candidate_id,
            "selection_timestamp": seed.selection_timestamp,
        })

    @property
    def track_processor_ids(self) -> dict[int, tuple[int, int]]:
        """Object IDs prove validators/smoothers are isolated per track."""

        return {
            track_id: (id(track.validator), id(track.smoother))
            for track_id, track in self._tracks.items()
        }


PersonTracker = PrimaryPersonTracker
