"""Causal adaptive pose smoothing, bounded prediction and bone constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .pose_postprocess import PoseDetection


KeypointStatus = Literal[
    "detected", "predicted", "interpolated", "uncertain", "missing"
]
WRISTS_AND_ANKLES = frozenset((9, 10, 15, 16))
ELBOWS = frozenset((7, 8))
WRISTS = frozenset((9, 10))
RESPONSIVE_JOINTS = frozenset((7, 8, 9, 10, 13, 14, 15, 16))
TORSO = (5, 6, 11, 12)
TORSO_SET = frozenset(TORSO)
BONES = (
    (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (5, 6), (11, 12), (5, 11), (6, 12),
)


@dataclass(frozen=True)
class SmootherConfig:
    keypoint_threshold: float = 0.25
    # Legacy UI control retained as a response multiplier.  At 0.45 the
    # group-specific ranges below are used unchanged.
    ema_alpha: float = 0.45
    max_interpolation_frames: int = 3
    extremity_interpolation_frames: int = 1
    major_joint_interpolation_frames: int = 2
    extremity_max_speed_ratio: float = 0.45
    general_max_speed_ratio: float = 0.95
    torso_alpha_min: float = 0.58
    torso_alpha_max: float = 0.70
    responsive_alpha_min: float = 0.70
    responsive_alpha_max: float = 0.85
    elbow_alpha_min: float = 0.74
    elbow_alpha_max: float = 0.90
    wrist_alpha_min: float = 0.80
    wrist_alpha_max: float = 0.94
    elbow_recovery_alpha: float = 0.92
    wrist_recovery_alpha: float = 0.96
    other_alpha_min: float = 0.55
    other_alpha_max: float = 0.72
    speed_response_ratio: float = 0.18
    elbow_speed_response_ratio: float = 0.30
    wrist_speed_response_ratio: float = 0.45
    prediction_step_ratio: float = 0.14
    prediction_total_ratio: float = 0.22
    bone_constraint_strength: float = 0.18
    bone_max_correction_ratio: float = 0.04


@dataclass
class SmoothedPose:
    raw_keypoints: np.ndarray
    keypoints: np.ndarray
    statuses: np.ndarray
    uncertain: bool
    interpolation_used: bool
    alphas: np.ndarray | None = None
    interpolation_counts: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.raw_keypoints = np.asarray(self.raw_keypoints, dtype=np.float32).reshape(17, 3)
        self.keypoints = np.asarray(self.keypoints, dtype=np.float32).reshape(17, 3)
        self.statuses = np.asarray(self.statuses, dtype="<U12").reshape(17)
        if self.alphas is None:
            self.alphas = np.full(17, np.nan, dtype=np.float32)
        else:
            self.alphas = np.asarray(self.alphas, dtype=np.float32).reshape(17)
        if self.interpolation_counts is None:
            self.interpolation_counts = np.zeros(17, dtype=np.int32)
        else:
            self.interpolation_counts = np.asarray(
                self.interpolation_counts, dtype=np.int32
            ).reshape(17)


class PoseSmoother:
    """Joint-adaptive EMA with causal, strictly bounded extrapolation.

    Torso points trade more response for stability.  Elbows, wrists, knees and
    ankles respond faster when confidence and observed velocity are high.  A
    rejected or absent observation may be predicted for one extremity frame
    or two other major-joint frames, then becomes missing.
    """

    def __init__(self, config: SmootherConfig | None = None) -> None:
        self.config = config or SmootherConfig()
        if not 0.0 < self.config.ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if self.config.max_interpolation_frames < 0:
            raise ValueError("max_interpolation_frames cannot be negative")
        self.reset()

    def reset(self) -> None:
        self._previous = np.full((17, 3), np.nan, dtype=np.float32)
        self._last_observed = np.full((17, 2), np.nan, dtype=np.float32)
        self._last_detected = np.full((17, 2), np.nan, dtype=np.float32)
        self._velocity = np.zeros((17, 2), dtype=np.float32)
        self._missing_counts = np.zeros(17, dtype=np.int32)
        self._previous_statuses = np.full(17, "missing", dtype="<U12")
        self._bone_lengths: dict[tuple[int, int], float] = {}
        self._last_body_scale = 100.0

    @staticmethod
    def _coerce_keypoints(pose: PoseDetection | np.ndarray | None) -> np.ndarray:
        if pose is None:
            result = np.full((17, 3), np.nan, dtype=np.float32)
            result[:, 2] = 0.0
            return result
        source = pose.keypoints if isinstance(pose, PoseDetection) else pose
        return np.asarray(source, dtype=np.float32).reshape(17, 3).copy()

    def _body_scale(self, raw: np.ndarray) -> float:
        valid_torso = [
            index for index in TORSO
            if raw[index, 2] >= self.config.keypoint_threshold
            and np.isfinite(raw[index, :2]).all()
        ]
        if len(valid_torso) >= 2:
            points = raw[valid_torso, :2]
            span = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
            if span > 1.0:
                self._last_body_scale = float(span)
                return self._last_body_scale
        # During an occlusion, keep the last reliable shoulder/hip scale.
        # A whole-pose span would silently loosen prediction gates.
        return self._last_body_scale

    def _alpha(self, index: int, confidence: float, speed_ratio: float) -> float:
        if index in TORSO_SET:
            low, high = self.config.torso_alpha_min, self.config.torso_alpha_max
        elif index in WRISTS:
            low, high = self.config.wrist_alpha_min, self.config.wrist_alpha_max
        elif index in ELBOWS:
            low, high = self.config.elbow_alpha_min, self.config.elbow_alpha_max
        elif index in RESPONSIVE_JOINTS:
            low, high = self.config.responsive_alpha_min, self.config.responsive_alpha_max
        else:
            low, high = self.config.other_alpha_min, self.config.other_alpha_max
        response_ratio = (
            self.config.wrist_speed_response_ratio
            if index in WRISTS else (
                self.config.elbow_speed_response_ratio
                if index in ELBOWS else self.config.speed_response_ratio
            )
        )
        speed_factor = float(
            np.clip(speed_ratio / max(response_ratio, 1e-6), 0.0, 1.0)
        )
        response = low + (high - low) * speed_factor
        confidence_factor = 0.82 + 0.18 * float(np.clip(confidence, 0.0, 1.0))
        legacy_scale = float(np.clip(self.config.ema_alpha / 0.45, 0.65, 1.35))
        return float(np.clip(response * confidence_factor * legacy_scale, 0.15, 0.95))

    def _joint_interpolation_limit(
        self, index: int, interpolation_limits: np.ndarray | None = None
    ) -> int:
        if interpolation_limits is not None:
            return max(
                0,
                min(
                    self.config.max_interpolation_frames,
                    int(interpolation_limits[index]),
                ),
            )
        group_limit = (
            self.config.extremity_interpolation_frames
            if index in WRISTS_AND_ANKLES
            else self.config.major_joint_interpolation_frames
        )
        return max(0, min(self.config.max_interpolation_frames, group_limit))

    def update(
        self,
        pose: PoseDetection | np.ndarray | None,
        interpolation_limits: np.ndarray | None = None,
        observation_statuses: np.ndarray | None = None,
    ) -> SmoothedPose:
        raw = self._coerce_keypoints(pose)
        output = np.full((17, 3), np.nan, dtype=np.float32)
        output[:, 2] = 0.0
        statuses = np.full(17, "missing", dtype="<U12")
        alphas = np.full(17, np.nan, dtype=np.float32)
        observed_speeds = np.zeros(17, dtype=np.float32)
        scale = self._body_scale(raw)
        observations = (
            np.asarray(observation_statuses, dtype="<U12").reshape(17)
            if observation_statuses is not None
            else np.full(17, "missing", dtype="<U12")
        )

        for index in range(17):
            point_valid = (
                np.isfinite(raw[index, :2]).all()
                and np.isfinite(raw[index, 2])
                and raw[index, 2] >= self.config.keypoint_threshold
            )
            previous_valid = np.isfinite(self._previous[index, :2]).all()
            observed_before = np.isfinite(self._last_observed[index]).all()
            speed_ratio = 0.0
            gated = False
            if point_valid and observed_before:
                raw_step = raw[index, :2] - self._last_observed[index]
                speed_ratio = float(np.linalg.norm(raw_step) / max(scale, 1.0))
                limit = (
                    self.config.extremity_max_speed_ratio
                    if index in WRISTS_AND_ANKLES
                    else self.config.general_max_speed_ratio
                )
                # A detected status means the upstream per-track validator has
                # already checked body-relative speed, bone length and chain
                # continuity.  Re-gating that accepted point here created a
                # second, hidden confirmation delay for fast forearm motion.
                gated = (
                    speed_ratio > limit and observations[index] != "detected"
                )
            observed_speeds[index] = speed_ratio

            if point_valid and not gated:
                confidence = float(np.clip(raw[index, 2], 0.0, 1.0))
                alpha = self._alpha(index, confidence, speed_ratio)
                if self._previous_statuses[index] in (
                    "predicted", "interpolated", "uncertain", "missing"
                ):
                    if index in WRISTS:
                        alpha = max(alpha, self.config.wrist_recovery_alpha)
                    elif index in ELBOWS:
                        alpha = max(alpha, self.config.elbow_recovery_alpha)
                alphas[index] = alpha
                if previous_valid:
                    smoothed_xy = alpha * raw[index, :2] + (1.0 - alpha) * self._previous[index, :2]
                else:
                    smoothed_xy = raw[index, :2]
                if observed_before:
                    raw_velocity = raw[index, :2] - self._last_observed[index]
                    max_velocity = scale * (
                        self.config.extremity_max_speed_ratio
                        if index in WRISTS_AND_ANKLES
                        else self.config.general_max_speed_ratio
                    )
                    norm = float(np.linalg.norm(raw_velocity))
                    if norm > max_velocity > 0:
                        raw_velocity *= max_velocity / norm
                    self._velocity[index] = 0.75 * raw_velocity + 0.25 * self._velocity[index]
                else:
                    self._velocity[index] = 0.0
                self._last_observed[index] = raw[index, :2]
                self._last_detected[index] = smoothed_xy
                output[index, :2] = smoothed_xy
                output[index, 2] = confidence
                statuses[index] = "detected"
                self._missing_counts[index] = 0
                continue

            self._missing_counts[index] += 1
            gap = int(self._missing_counts[index])
            interpolation_limit = self._joint_interpolation_limit(
                index, interpolation_limits
            )
            last_detected_valid = np.isfinite(self._last_detected[index]).all()
            if previous_valid and last_detected_valid and gap <= interpolation_limit:
                velocity = self._velocity[index] * (0.55 ** (gap - 1))
                step_limit = scale * self.config.prediction_step_ratio
                if index in WRISTS_AND_ANKLES:
                    step_limit *= 0.85
                norm = float(np.linalg.norm(velocity))
                if norm > step_limit > 0:
                    velocity *= step_limit / norm
                predicted = self._previous[index, :2] + velocity
                total_limit = scale * self.config.prediction_total_ratio
                if index in WRISTS_AND_ANKLES:
                    total_limit *= 0.82
                total_delta = predicted - self._last_detected[index]
                total_norm = float(np.linalg.norm(total_delta))
                if total_norm > total_limit > 0:
                    predicted = self._last_detected[index] + total_delta * (total_limit / total_norm)
                output[index, :2] = predicted
                output[index, 2] = max(0.01, float(self._previous[index, 2]) * 0.62)
                statuses[index] = (
                    "predicted"
                    if index in (7, 8, 9, 10)
                    and observations[index] in ("provisional", "rejected")
                    else "interpolated"
                )
            elif np.isfinite(raw[index, :2]).all() and raw[index, 2] > 0:
                statuses[index] = "uncertain"
            else:
                statuses[index] = "missing"

        self._apply_bone_constraints(output, statuses, scale, observed_speeds)
        self._previous = output.copy()
        self._previous_statuses = statuses.copy()
        return SmoothedPose(
            raw_keypoints=raw,
            keypoints=output,
            statuses=statuses,
            uncertain=bool(np.any(np.isin(statuses[5:17], ["uncertain", "missing"]))),
            interpolation_used=bool(
                np.any(np.isin(statuses, ["interpolated", "predicted"]))
            ),
            alphas=alphas,
            interpolation_counts=np.where(
                np.isin(statuses, ["interpolated", "predicted"]),
                self._missing_counts,
                0,
            ),
        )

    def _apply_bone_constraints(
        self,
        pose: np.ndarray,
        statuses: np.ndarray,
        scale: float,
        observed_speeds: np.ndarray,
    ) -> None:
        base_strength = float(np.clip(self.config.bone_constraint_strength, 0.0, 1.0))
        if base_strength <= 0:
            return
        usable = np.isin(statuses, ["detected", "interpolated", "predicted"])
        for proximal, distal in BONES:
            if not usable[proximal] or not usable[distal]:
                continue
            vector = pose[distal, :2] - pose[proximal, :2]
            length = float(np.linalg.norm(vector))
            if length <= 1e-6:
                continue
            bone = (proximal, distal)
            target = self._bone_lengths.get(bone)
            if target is None:
                target = length
            elif statuses[proximal] == "detected" and statuses[distal] == "detected":
                target = 0.82 * target + 0.18 * length
            self._bone_lengths[bone] = target
            constrained = pose[proximal, :2] + vector * (target / length)
            correction = constrained - pose[distal, :2]
            max_correction = max(0.0, scale * self.config.bone_max_correction_ratio)
            correction_norm = float(np.linalg.norm(correction))
            if correction_norm > max_correction > 0:
                correction *= max_correction / correction_norm
            strength = base_strength
            if statuses[distal] == "detected" and distal in RESPONSIVE_JOINTS:
                speed_factor = float(np.clip(observed_speeds[distal] / max(self.config.speed_response_ratio, 1e-6), 0.0, 1.0))
                strength *= 0.45 - 0.30 * speed_factor
            pose[distal, :2] += strength * correction
