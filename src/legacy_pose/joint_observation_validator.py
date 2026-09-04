"""Causal per-track validation of raw COCO joint observations.

The validator runs before interpolation, smoothing and bone constraints.  A
rejected point is replaced by a missing observation so it cannot update filter,
velocity or bone-length history.  It uses pose geometry only; it does not
detect the damper or perform segmentation/identity recognition.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np


TORSO_INDICES = (5, 6, 11, 12)
ELBOWS = (7, 8)
WRISTS = (9, 10)
KNEES = (13, 14)
ANKLES = (15, 16)
ARM_PARENTS = {7: 5, 8: 6, 9: 7, 10: 8}
CHAIN_SHOULDER = {9: 5, 10: 6}


@dataclass(frozen=True)
class JointValidatorConfig:
    keypoint_threshold: float = 0.25
    history_size: int = 21
    minimum_bone_samples: int = 4
    wrist_max_speed_ratio: float = 0.70
    elbow_max_speed_ratio: float = 0.55
    wrist_max_acceleration_ratio: float = 0.34
    elbow_max_acceleration_ratio: float = 0.28
    large_motion_ratio: float = 0.22
    large_motion_confirmation_frames: int = 2
    large_motion_consistency_ratio: float = 0.45
    fast_chain_confidence: float = 0.55
    fast_chain_direction_cosine: float = 0.15
    fast_chain_min_companion_motion_ratio: float = 0.035
    wrist_extreme_motion_ratio: float = 1.15
    elbow_extreme_motion_ratio: float = 0.95
    recovery_bone_length_min_ratio: float = 0.38
    recovery_bone_length_max_ratio: float = 2.00
    torso_stable_ratio: float = 0.065
    bone_length_min_ratio: float = 0.52
    bone_length_max_ratio: float = 1.65
    chain_angle_change_degrees: float = 105.0
    # The normal forearm continuity budget is time based.  Frame limits are
    # retained as lower/upper compatibility controls for callers that already
    # expose them, while the default 0.2 s gives two causal frames at 10 FPS.
    output_fps: float = 10.0
    forearm_prediction_seconds: float = 0.20
    edge_prediction_seconds: float = 0.10
    wrist_prediction_seconds: float | None = None
    elbow_prediction_seconds: float | None = None
    knee_prediction_seconds: float = 0.20
    ankle_prediction_seconds: float = 0.15
    edge_prediction_scale: float = 0.50
    wrist_interpolation_frames: int | None = None
    elbow_interpolation_frames: int | None = None
    edge_wrist_interpolation_frames: int = 1
    edge_elbow_interpolation_frames: int = 1
    edge_margin_ratio: float = 0.035


@dataclass
class JointValidationResult:
    raw_keypoints: np.ndarray
    validated_keypoints: np.ndarray
    observation_statuses: np.ndarray
    rejection_reasons: dict[int, list[str]] = field(default_factory=dict)
    suspected_object_induced: list[int] = field(default_factory=list)
    diagnostics: dict[int, dict[str, Any]] = field(default_factory=dict)
    body_scale: float = 100.0
    torso_motion_ratio: float = 0.0
    edge_state: bool = False
    interpolation_limits: np.ndarray = field(
        default_factory=lambda: np.full(17, 2, dtype=np.int32)
    )

    @property
    def rejected_indices(self) -> list[int]:
        return [
            int(index) for index, status in enumerate(self.observation_statuses)
            if status == "rejected"
        ]

    @property
    def provisional_indices(self) -> list[int]:
        return [
            int(index) for index, status in enumerate(self.observation_statuses)
            if status == "provisional"
        ]

    @property
    def invalid_indices(self) -> list[int]:
        """All observations withheld from downstream filter histories."""

        return sorted(self.rejection_reasons)


def _angle_degrees(first: np.ndarray, middle: np.ndarray, last: np.ndarray) -> float | None:
    left = first - middle
    right = last - middle
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-6:
        return None
    cosine = float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


class JointObservationValidator:
    """Validate one anonymous track's observations and own its histories."""

    def __init__(self, config: JointValidatorConfig | None = None) -> None:
        self.config = config or JointValidatorConfig()
        if self.config.large_motion_confirmation_frames < 1:
            raise ValueError("large_motion_confirmation_frames must be positive")
        self.reset()

    def reset(self) -> None:
        self._last_reliable = np.full((17, 2), np.nan, dtype=np.float32)
        self._velocities = np.zeros((17, 2), dtype=np.float32)
        self._last_torso = np.full((4, 2), np.nan, dtype=np.float32)
        self._last_body_scale = 100.0
        self._bone_history: dict[tuple[int, int], deque[float]] = {
            bone: deque(maxlen=self.config.history_size) for bone in ARM_PARENTS.items()
        }
        self._chain_history: dict[int, deque[float]] = {
            index: deque(maxlen=self.config.history_size) for index in WRISTS
        }
        self._pending: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}

    def _body_scale(self, raw: np.ndarray, bbox: np.ndarray) -> float:
        candidates: list[float] = []
        threshold = self.config.keypoint_threshold
        if raw[5, 2] >= threshold and raw[6, 2] >= threshold:
            candidates.append(float(np.linalg.norm(raw[5, :2] - raw[6, :2])))
        shoulders = [index for index in (5, 6) if raw[index, 2] >= threshold]
        hips = [index for index in (11, 12) if raw[index, 2] >= threshold]
        if shoulders and hips:
            shoulder_center = np.mean(raw[shoulders, :2], axis=0)
            hip_center = np.mean(raw[hips, :2], axis=0)
            candidates.append(float(np.linalg.norm(shoulder_center - hip_center)))
        box_diagonal = float(np.linalg.norm(bbox[2:] - bbox[:2]))
        if box_diagonal > 1:
            candidates.append(box_diagonal * 0.28)
        usable = [value for value in candidates if np.isfinite(value) and value > 8.0]
        if usable:
            scale = float(np.median(usable))
            self._last_body_scale = 0.8 * self._last_body_scale + 0.2 * scale
        return max(20.0, self._last_body_scale)

    def _edge_state(
        self, bbox: np.ndarray, frame_shape: tuple[int, int] | tuple[int, int, int] | None
    ) -> bool:
        if frame_shape is None:
            return False
        height, width = frame_shape[:2]
        margin_x = width * self.config.edge_margin_ratio
        margin_y = height * self.config.edge_margin_ratio
        return bool(
            bbox[0] <= margin_x
            or bbox[1] <= margin_y
            or bbox[2] >= width - 1 - margin_x
            or bbox[3] >= height - 1 - margin_y
        )

    def _torso_motion(self, raw: np.ndarray, scale: float) -> tuple[float, np.ndarray]:
        current = raw[list(TORSO_INDICES), :2]
        confidence = raw[list(TORSO_INDICES), 2]
        valid = (
            np.isfinite(current).all(axis=1)
            & np.isfinite(self._last_torso).all(axis=1)
            & (confidence >= self.config.keypoint_threshold)
        )
        torso_delta = (
            np.median(current[valid] - self._last_torso[valid], axis=0).astype(np.float32)
            if np.any(valid) else np.zeros(2, dtype=np.float32)
        )
        motion = float(np.linalg.norm(torso_delta) / scale)
        update = np.isfinite(current).all(axis=1) & (confidence >= self.config.keypoint_threshold)
        self._last_torso[update] = current[update]
        return motion, torso_delta

    def _interpolation_limits(self, edge_state: bool) -> np.ndarray:
        limits = np.full(17, 2, dtype=np.int32)
        wrist_seconds = self.config.wrist_prediction_seconds
        if wrist_seconds is None:
            wrist_seconds = self.config.forearm_prediction_seconds
        elbow_seconds = self.config.elbow_prediction_seconds
        if elbow_seconds is None:
            elbow_seconds = self.config.forearm_prediction_seconds

        def duration_frames(seconds: float) -> int:
            normal = max(0, int(round(self.config.output_fps * seconds)))
            if not edge_state:
                return normal
            return min(normal, int(round(normal * self.config.edge_prediction_scale)))

        wrist_frames = duration_frames(wrist_seconds)
        elbow_frames = duration_frames(elbow_seconds)
        knee_frames = duration_frames(self.config.knee_prediction_seconds)
        ankle_frames = duration_frames(self.config.ankle_prediction_seconds)
        edge_time_frames = max(
            0, int(round(self.config.output_fps * self.config.edge_prediction_seconds))
        )
        limits[list(WRISTS)] = (
            min(self.config.edge_wrist_interpolation_frames, edge_time_frames, wrist_frames)
            if edge_state else (
                wrist_frames
                if self.config.wrist_interpolation_frames is None
                else self.config.wrist_interpolation_frames
            )
        )
        limits[list(ELBOWS)] = (
            min(self.config.edge_elbow_interpolation_frames, edge_time_frames, elbow_frames)
            if edge_state else (
                elbow_frames
                if self.config.elbow_interpolation_frames is None
                else self.config.elbow_interpolation_frames
            )
        )
        limits[list(KNEES)] = knee_frames
        limits[list(ANKLES)] = ankle_frames
        return limits

    def validate(
        self,
        keypoints: np.ndarray,
        bbox: np.ndarray,
        frame_shape: tuple[int, int] | tuple[int, int, int] | None = None,
    ) -> JointValidationResult:
        raw = np.asarray(keypoints, dtype=np.float32).reshape(17, 3).copy()
        validated = raw.copy()
        statuses = np.full(17, "uncertain", dtype="<U12")
        bbox_array = np.asarray(bbox, dtype=np.float32).reshape(4)
        scale = self._body_scale(raw, bbox_array)
        torso_motion, torso_delta = self._torso_motion(raw, scale)
        torso_stable = torso_motion <= self.config.torso_stable_ratio
        edge_state = self._edge_state(bbox_array, frame_shape)
        reasons_by_joint: dict[int, list[str]] = {}
        suspected: list[int] = []
        diagnostics: dict[int, dict[str, Any]] = {}
        # Parent/companion motion must be measured against the state entering
        # this frame.  Parents are visited before distal joints and may update
        # the live history below, so a snapshot prevents artificial zero
        # motion and the resulting fixed confirmation delay.
        previous_reliable = self._last_reliable.copy()

        for index in range(17):
            confidence = float(raw[index, 2]) if np.isfinite(raw[index, 2]) else 0.0
            valid = np.isfinite(raw[index, :2]).all() and confidence >= self.config.keypoint_threshold
            if not valid:
                validated[index, :2] = np.nan
                validated[index, 2] = 0.0
                statuses[index] = "uncertain" if confidence > 0 else "missing"
                continue

            reasons: list[str] = []
            last_valid = np.isfinite(self._last_reliable[index]).all()
            displacement = raw[index, :2] - self._last_reliable[index] if last_valid else np.zeros(2, np.float32)
            # Articulated-joint gates operate in the torso's moving frame.  A
            # worker translating through the image must not look like eight
            # independent limb teleports.
            motion_displacement = (
                displacement - torso_delta if index in ARM_PARENTS else displacement
            )
            speed_ratio = float(np.linalg.norm(motion_displacement) / scale) if last_valid else 0.0
            acceleration_ratio = (
                float(np.linalg.norm(motion_displacement - self._velocities[index]) / scale)
                if last_valid else 0.0
            )
            bone_ratio: float | None = None
            chain_angle: float | None = None
            chain_change: float | None = None

            if index in ARM_PARENTS:
                parent = ARM_PARENTS[index]
                parent_valid = (
                    np.isfinite(raw[parent, :2]).all()
                    and raw[parent, 2] >= self.config.keypoint_threshold
                    and statuses[parent] not in ("rejected", "provisional")
                )
                history = self._bone_history[(index, parent)]
                if parent_valid:
                    length = float(np.linalg.norm(raw[index, :2] - raw[parent, :2]))
                    if len(history) >= self.config.minimum_bone_samples:
                        median_length = float(np.median(history))
                        bone_ratio = length / max(median_length, 1e-6)
                        if not self.config.bone_length_min_ratio <= bone_ratio <= self.config.bone_length_max_ratio:
                            reasons.append("bone_length")
                else:
                    reasons.append("parent_unreliable")

                speed_limit = (
                    self.config.wrist_max_speed_ratio if index in WRISTS
                    else self.config.elbow_max_speed_ratio
                )
                acceleration_limit = (
                    self.config.wrist_max_acceleration_ratio if index in WRISTS
                    else self.config.elbow_max_acceleration_ratio
                )
                if last_valid and speed_ratio > speed_limit:
                    reasons.append("speed")
                if (
                    last_valid and torso_stable
                    and speed_ratio > self.config.large_motion_ratio
                    and acceleration_ratio > acceleration_limit
                ):
                    reasons.append("acceleration")
                if last_valid and torso_stable and speed_ratio > self.config.large_motion_ratio:
                    reasons.append("torso_decoupled")

                if index in WRISTS and parent_valid:
                    shoulder = CHAIN_SHOULDER[index]
                    if raw[shoulder, 2] >= self.config.keypoint_threshold:
                        chain_angle = _angle_degrees(
                            raw[shoulder, :2], raw[parent, :2], raw[index, :2]
                        )
                        angle_history = self._chain_history[index]
                        if chain_angle is not None and len(angle_history) >= self.config.minimum_bone_samples:
                            chain_change = abs(chain_angle - float(np.median(angle_history)))
                            if (
                                chain_change > self.config.chain_angle_change_degrees
                                and torso_stable and speed_ratio > 0.15
                            ):
                                reasons.append("motion_chain")

                extreme_limit = (
                    self.config.wrist_extreme_motion_ratio
                    if index in WRISTS else self.config.elbow_extreme_motion_ratio
                )
                recovery_geometry_plausible = (
                    bone_ratio is None
                    or self.config.recovery_bone_length_min_ratio
                    <= bone_ratio
                    <= self.config.recovery_bone_length_max_ratio
                )
                hard_reasons = set(reasons) & {"parent_unreliable"}
                if speed_ratio > extreme_limit or not recovery_geometry_plausible:
                    hard_reasons.add("nonrecoverable_geometry")
                companion = parent
                if index in ELBOWS:
                    companion = 9 if index == 7 else 10
                companion_previous_valid = np.isfinite(
                    previous_reliable[companion]
                ).all()
                companion_current_valid = (
                    np.isfinite(raw[companion, :2]).all()
                    and raw[companion, 2] >= self.config.fast_chain_confidence
                )
                companion_motion = np.zeros(2, dtype=np.float32)
                direction_agreement: float | None = None
                companion_motion_ratio = 0.0
                if companion_previous_valid and companion_current_valid:
                    companion_motion = (
                        raw[companion, :2]
                        - previous_reliable[companion]
                        - torso_delta
                    )
                    companion_motion_ratio = float(
                        np.linalg.norm(companion_motion) / scale
                    )
                    denominator = float(
                        np.linalg.norm(motion_displacement)
                        * np.linalg.norm(companion_motion)
                    )
                    if denominator > 1e-6:
                        direction_agreement = float(
                            np.dot(motion_displacement, companion_motion) / denominator
                        )
                conflicting_geometry = bool(
                    set(reasons)
                    & {"bone_length", "motion_chain", "parent_unreliable"}
                )
                fast_chain_accepted = bool(
                    last_valid
                    and confidence >= self.config.fast_chain_confidence
                    and parent_valid
                    and recovery_geometry_plausible
                    and speed_ratio <= extreme_limit
                    and not conflicting_geometry
                    and companion_motion_ratio
                    >= self.config.fast_chain_min_companion_motion_ratio
                    and direction_agreement is not None
                    and direction_agreement
                    >= self.config.fast_chain_direction_cosine
                )
                if fast_chain_accepted:
                    reasons = [
                        item
                        for item in reasons
                        if item not in ("speed", "acceleration", "torso_decoupled")
                    ]
                needs_confirmation = (
                    last_valid and torso_stable
                    and speed_ratio > self.config.large_motion_ratio
                    and not fast_chain_accepted
                    and not hard_reasons
                )
                if needs_confirmation:
                    direction = motion_displacement / max(float(np.linalg.norm(motion_displacement)), 1e-6)
                    pending = self._pending.get(index)
                    consistent = False
                    count = 1
                    if pending is not None:
                        pending_xy, pending_direction, previous_count = pending
                        position_delta = float(np.linalg.norm(raw[index, :2] - pending_xy) / scale)
                        pending_direction_agreement = float(
                            np.dot(direction, pending_direction)
                        )
                        consistent = (
                            position_delta <= self.config.large_motion_consistency_ratio
                            and pending_direction_agreement >= 0.0
                        )
                        count = previous_count + 1 if consistent else 1
                    if count < self.config.large_motion_confirmation_frames:
                        reasons.append("large_motion_unconfirmed")
                        self._pending[index] = (raw[index, :2].copy(), direction, count)
                    else:
                        self._pending.pop(index, None)
                        reasons = [
                            item for item in reasons
                            if item not in (
                                "speed", "torso_decoupled", "acceleration",
                                "motion_chain", "bone_length",
                            )
                        ]
                elif hard_reasons:
                    self._pending.pop(index, None)
                else:
                    self._pending.pop(index, None)
            else:
                fast_chain_accepted = False
                direction_agreement = None
                companion_motion_ratio = 0.0

            diagnostics[index] = {
                "speed_ratio": round(speed_ratio, 6),
                "acceleration_ratio": round(acceleration_ratio, 6),
                "bone_ratio": round(bone_ratio, 6) if bone_ratio is not None else None,
                "chain_angle": round(chain_angle, 3) if chain_angle is not None else None,
                "chain_angle_change": round(chain_change, 3) if chain_change is not None else None,
                "torso_motion_ratio": round(torso_motion, 6),
                "fast_chain_accepted": bool(fast_chain_accepted),
                "chain_direction_agreement": (
                    round(direction_agreement, 6)
                    if direction_agreement is not None else None
                ),
                "companion_motion_ratio": round(companion_motion_ratio, 6),
            }
            if reasons:
                unique_reasons = list(dict.fromkeys(reasons))
                reasons_by_joint[index] = unique_reasons
                # A mild, single-frame conflict is withheld from the filter but
                # may use bounded rendering prediction.  Multiple independent
                # geometry conflicts or non-recoverable motion remain hard
                # rejection, preserving the damper-object guard.
                independent = set(unique_reasons) - {"large_motion_unconfirmed"}
                hard = (
                    "nonrecoverable_geometry" in unique_reasons
                    or len(independent & {
                        "speed", "acceleration", "torso_decoupled",
                        "bone_length", "motion_chain", "parent_unreliable",
                    }) >= 3
                    or (
                        "bone_length" in independent
                        and "motion_chain" in independent
                    )
                )
                statuses[index] = "rejected" if hard else "provisional"
                diagnostics[index]["observation_state"] = str(statuses[index])
                validated[index, :2] = np.nan
                validated[index, 2] = 0.0
                if (
                    confidence >= 0.5 and torso_stable
                    and set(unique_reasons) & {"bone_length", "motion_chain", "torso_decoupled"}
                ):
                    suspected.append(index)
                continue

            statuses[index] = "detected"
            diagnostics[index]["observation_state"] = "detected"
            if last_valid:
                self._velocities[index] = motion_displacement
            else:
                self._velocities[index] = 0.0
            self._last_reliable[index] = raw[index, :2]
            if index in ARM_PARENTS:
                parent = ARM_PARENTS[index]
                if statuses[parent] == "detected":
                    length = float(np.linalg.norm(raw[index, :2] - raw[parent, :2]))
                    if length > 1.0:
                        self._bone_history[(index, parent)].append(length)
                if index in WRISTS and chain_angle is not None:
                    self._chain_history[index].append(chain_angle)

        return JointValidationResult(
            raw_keypoints=raw,
            validated_keypoints=validated,
            observation_statuses=statuses,
            rejection_reasons=reasons_by_joint,
            suspected_object_induced=sorted(set(suspected)),
            diagnostics=diagnostics,
            body_scale=scale,
            torso_motion_ratio=torso_motion,
            edge_state=edge_state,
            interpolation_limits=self._interpolation_limits(edge_state),
        )

    @property
    def bone_history_snapshot(self) -> dict[tuple[int, int], tuple[float, ...]]:
        """Read-only snapshot used by deterministic isolation tests."""

        return {key: tuple(values) for key, values in self._bone_history.items()}

    @property
    def last_reliable_snapshot(self) -> np.ndarray:
        return self._last_reliable.copy()

    @property
    def velocity_snapshot(self) -> np.ndarray:
        return self._velocities.copy()
