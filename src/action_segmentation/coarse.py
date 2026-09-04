"""Causal coarse actions derived only from COCO-17 pose observations."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import exp
from statistics import median
from typing import Any

import numpy as np


COARSE_ACTIONS = (
    "idle",
    "reach",
    "retract",
    "lift",
    "lower",
    "move",
    "carry",
    "place",
    "hold",
    "release",
    "rotate",
    "push",
    "pull",
    "transition",
    "unknown",
    "lost",
)


@dataclass
class CoarseFrame:
    timestamp: float
    source_frame_index: int
    person_ref: str
    lock_epoch: int
    track_state: str
    lock_state: str
    candidate_person_count: int
    action: str
    anatomical_side: str
    observation_state: str
    detected_ratio: float
    predicted_ratio: float
    interpolated_ratio: float
    missing_ratio: float
    direction_clear: bool
    required_joints_reliable: bool
    keypoints: list[list[float]]
    keypoint_statuses: list[str]
    evidence_state: str = "raw"
    temporal_reason: str = "raw_pose_observation"
    bounded_uncertain_gap: bool = False
    hard_boundary: bool = False


class CausalCoarseActionClassifier:
    """Small pose heuristic; it never infers parts, grasp, or process semantics."""

    def __init__(self) -> None:
        self._previous: dict[tuple[str, int], dict[str, Any]] = {}

    @staticmethod
    def _body_scale(points: np.ndarray) -> float:
        torso = points[[5, 6, 11, 12], :2]
        valid = torso[np.isfinite(torso).all(axis=1)]
        if len(valid) >= 2:
            extent = np.ptp(valid, axis=0)
            return max(20.0, float(np.linalg.norm(extent)))
        return 100.0

    def classify(
        self,
        *,
        timestamp: float,
        person_ref: str,
        lock_epoch: int,
        track_state: str,
        lock_state: str,
        keypoints: np.ndarray | None,
        statuses: np.ndarray | list[str] | None,
    ) -> tuple[str, str, dict[str, Any]]:
        history_key = (person_ref, lock_epoch)
        if track_state != "tracked" or keypoints is None or statuses is None:
            # Lost/off-frame/missing observations are hard temporal boundaries.
            # A recovered frame must not derive velocity from pre-boundary pose.
            self._previous.pop(history_key, None)
            return (
                "lost",
                "bilateral",
                {
                    "observation_state": "lost",
                    "detected_ratio": 0.0,
                    "predicted_ratio": 0.0,
                    "interpolated_ratio": 0.0,
                    "missing_ratio": 1.0,
                    "direction_clear": False,
                    "required_joints_reliable": False,
                },
            )

        points = np.asarray(keypoints, dtype=np.float32).reshape(17, 3)
        states = np.asarray(statuses, dtype="<U16").reshape(17)
        counts = {
            name: float(np.count_nonzero(states == name)) / 17.0
            for name in ("detected", "predicted", "interpolated", "missing")
        }
        counts["missing"] += float(
            np.count_nonzero(np.isin(states, ["uncertain", "rejected"]))
        ) / 17.0
        arm_indices = [5, 6, 7, 8, 9, 10]
        reliable = np.isin(states[arm_indices], ["detected", "predicted", "interpolated"])
        finite = np.isfinite(points[arm_indices, :2]).all(axis=1)
        required_reliable = bool(np.count_nonzero(reliable & finite) >= 4)
        observation_state = (
            "detected"
            if counts["detected"] >= 0.65
            else "interpolated"
            if counts["interpolated"] + counts["predicted"] > 0
            else "missing"
        )
        evidence = {
            "observation_state": observation_state,
            "detected_ratio": counts["detected"],
            "predicted_ratio": counts["predicted"],
            "interpolated_ratio": counts["interpolated"],
            "missing_ratio": min(1.0, counts["missing"]),
            "direction_clear": False,
            "required_joints_reliable": required_reliable,
        }
        if not required_reliable:
            self._previous.pop(history_key, None)
            return "unknown", "bilateral", evidence

        current = {
            "timestamp": float(timestamp),
            "points": points.copy(),
        }
        wrist_now = points[[9, 10], :2]
        if not bool(np.isfinite(wrist_now).all(axis=1).any()):
            self._previous.pop(history_key, None)
            evidence["required_joints_reliable"] = False
            return "unknown", "bilateral", evidence
        previous = self._previous.get(history_key)
        if previous is None:
            self._previous[history_key] = current
            return "transition", "bilateral", evidence
        dt = max(1e-3, float(timestamp) - float(previous["timestamp"]))
        old_points = np.asarray(previous["points"], dtype=np.float32)
        scale = self._body_scale(points)
        wrist_old = old_points[[9, 10], :2]
        valid_wrist_motion = (
            np.isfinite(wrist_now).all(axis=1)
            & np.isfinite(wrist_old).all(axis=1)
        )
        if not bool(np.any(valid_wrist_motion)):
            # Prior missing wrists are a boundary; seed only the current,
            # genuinely visible wrist geometry for the following frame.
            self._previous[history_key] = current
            return "transition", "bilateral", evidence

        torso_now_points = points[[5, 6, 11, 12], :2]
        torso_old_points = old_points[[5, 6, 11, 12], :2]
        if (
            np.count_nonzero(np.isfinite(torso_now_points).all(axis=1)) < 2
            or np.count_nonzero(np.isfinite(torso_old_points).all(axis=1)) < 2
        ):
            self._previous[history_key] = current
            evidence["required_joints_reliable"] = False
            return "transition", "bilateral", evidence

        wrist_velocity = (wrist_now - wrist_old) / (scale * dt)
        wrist_velocity[~valid_wrist_motion] = np.nan
        torso_now = np.nanmean(torso_now_points, axis=0)
        torso_old = np.nanmean(torso_old_points, axis=0)
        torso_speed = float(np.linalg.norm(torso_now - torso_old) / (scale * dt))
        speeds = np.linalg.norm(wrist_velocity, axis=1)
        finite_speeds = np.isfinite(speeds)
        if not bool(np.any(finite_speeds)):
            self._previous[history_key] = current
            evidence["required_joints_reliable"] = False
            return "transition", "bilateral", evidence
        safe_speeds = np.where(finite_speeds, speeds, -np.inf)
        dominant = int(np.argmax(safe_speeds))
        side = "left" if dominant == 0 else "right"
        velocity = wrist_velocity[dominant]
        self._previous[history_key] = current

        shoulder_index = 5 if dominant == 0 else 6
        wrist_index = 9 if dominant == 0 else 10
        distance_now = float(
            np.linalg.norm(points[wrist_index, :2] - points[shoulder_index, :2])
            / scale
        )
        distance_old = float(
            np.linalg.norm(
                old_points[wrist_index, :2] - old_points[shoulder_index, :2]
            )
            / scale
        )
        extension_rate = (distance_now - distance_old) / dt

        if torso_speed >= 0.22:
            evidence["direction_clear"] = True
            return "move", "bilateral", evidence
        if float(speeds[dominant]) < 0.10:
            # Pose alone can see a stationary extended arm but cannot establish
            # that an object is being held.  Keep that distinction at idle.
            return "idle", "bilateral", evidence
        if abs(float(velocity[1])) > abs(float(velocity[0])) * 1.15:
            evidence["direction_clear"] = abs(float(velocity[1])) >= 0.14
            return ("lift" if velocity[1] < 0 else "lower"), side, evidence
        if extension_rate >= 0.11:
            evidence["direction_clear"] = True
            return "reach", side, evidence
        if extension_rate <= -0.11:
            evidence["direction_clear"] = True
            return "retract", side, evidence
        return "move", side, evidence


@dataclass(frozen=True)
class FrameActionStabilityConfig:
    """Time-based causal label confirmation for the stable-action input layer."""

    start_confirmation_seconds: float = 0.5
    stop_confirmation_seconds: float = 0.5
    temporal_context_seconds: float = 2.5
    bounded_uncertain_gap_seconds: float = 0.375

    def __post_init__(self) -> None:
        for name in (
            "start_confirmation_seconds",
            "stop_confirmation_seconds",
            "temporal_context_seconds",
            "bounded_uncertain_gap_seconds",
        ):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} cannot be negative")


@dataclass
class _TemporalLaneState:
    history: list[tuple[float, str]] = field(default_factory=list)
    active_action: str | None = None
    candidate_action: str | None = None
    candidate_evidence_seconds: float = 0.0
    candidate_output_indices: list[int] = field(default_factory=list)
    last_seen_timestamp: float | None = None


def _evidence_state_for_action(action: str) -> str:
    if action == "lost":
        return "lost"
    if action == "unknown":
        return "unknown"
    if action == "transition":
        return "transition"
    return "normal"


def _context_action(
    history: list[tuple[float, str]],
    *,
    timestamp: float,
    context_seconds: float,
) -> str:
    """Return a recency-weighted label without inventing an unobserved action."""

    if not history:
        return "transition"
    horizon = max(0.001, float(context_seconds))
    decay = max(0.125, horizon / 3.0)
    scores: dict[str, float] = {}
    latest: dict[str, float] = {}
    for observed_at, action in history:
        age = max(0.0, float(timestamp) - float(observed_at))
        weight = exp(-age / decay)
        scores[action] = scores.get(action, 0.0) + weight
        latest[action] = max(latest.get(action, float("-inf")), observed_at)
    return max(scores, key=lambda action: (scores[action], latest[action]))


def stabilize_coarse_frames(
    frames: list[CoarseFrame],
    config: FrameActionStabilityConfig | None = None,
) -> dict[str, Any]:
    """Confirm frame labels without altering the original raw observations.

    The returned frames are copies intended only as stable-event input.  A
    person/epoch change or a true lost state clears all temporal lanes.  Left,
    right, and bilateral observations keep independent state.  Brief
    missing/unreliable evidence is emitted as an explicit uncertain interval
    while context is retained only up to the configured bounded gap.
    """

    effective = config or FrameActionStabilityConfig()
    if not frames:
        return {
            "frames": [],
            "metrics": {
                "input_frame_count": 0,
                "output_frame_count": 0,
                "confirmed_switch_count": 0,
                "held_noise_frame_count": 0,
                "hard_boundary_frame_count": 0,
                "bounded_uncertain_gap_frame_count": 0,
                "identity_or_epoch_reset_count": 0,
                "anatomical_side_switch_count": 0,
            },
        }

    output: list[CoarseFrame] = []
    lanes: dict[str, _TemporalLaneState] = {}
    active_partition: tuple[str, int] | None = None
    uncertain_gap_start: float | None = None
    uncertain_gap_indices: list[int] = []
    previous_reliable_side: str | None = None
    confirmed_switches = 0
    held_noise_frames = 0
    identity_or_epoch_resets = 0
    anatomical_side_switches = 0

    def reset_all_lanes() -> None:
        nonlocal lanes, previous_reliable_side
        lanes = {}
        previous_reliable_side = None

    ordered = sorted(
        frames,
        key=lambda item: (float(item.timestamp), int(item.source_frame_index)),
    )
    positive_deltas = [
        float(right.timestamp) - float(left.timestamp)
        for left, right in zip(ordered, ordered[1:])
        if float(right.timestamp) - float(left.timestamp) > 1e-9
    ]
    sample_interval_seconds = (
        float(median(positive_deltas))
        if positive_deltas
        else max(0.001, effective.bounded_uncertain_gap_seconds / 3.0)
    )
    for frame in ordered:
        timestamp = float(frame.timestamp)
        partition = (frame.person_ref, frame.lock_epoch)
        if active_partition is not None and partition != active_partition:
            reset_all_lanes()
            uncertain_gap_start = None
            uncertain_gap_indices = []
            identity_or_epoch_resets += 1
        active_partition = partition

        lost_boundary = (
            frame.track_state != "tracked"
            or frame.lock_state
            in {
                "lost",
                "temporarily_lost",
                "awaiting_manual_relock",
                "off_frame",
            }
            or frame.observation_state == "lost"
            or frame.action == "lost"
        )
        if lost_boundary:
            output.append(
                replace(
                    frame,
                    action="lost",
                    evidence_state="lost",
                    temporal_reason="lost_or_tracking_hard_boundary",
                    bounded_uncertain_gap=False,
                    hard_boundary=True,
                )
            )
            reset_all_lanes()
            uncertain_gap_start = None
            uncertain_gap_indices = []
            continue

        quality_unreliable = (
            frame.observation_state == "missing"
            or not frame.required_joints_reliable
        )
        recovering_inside_gap = (
            uncertain_gap_start is not None
            and frame.action in {"unknown", "transition"}
        )
        if quality_unreliable or recovering_inside_gap:
            if uncertain_gap_start is None:
                uncertain_gap_start = timestamp
                uncertain_gap_indices = []
            output.append(
                replace(
                    frame,
                    action="unknown",
                    evidence_state="uncertain",
                    temporal_reason=(
                        "brief_missing_pose"
                        if frame.observation_state == "missing"
                        else "brief_unreliable_joints"
                        if quality_unreliable
                        else "bounded_gap_recovery_confirmation"
                    ),
                    bounded_uncertain_gap=True,
                    hard_boundary=False,
                )
            )
            uncertain_gap_indices.append(len(output) - 1)
            uncertain_span = (
                timestamp - uncertain_gap_start + sample_interval_seconds
            )
            if (
                uncertain_span
                > effective.bounded_uncertain_gap_seconds + 1e-9
            ):
                for output_index in uncertain_gap_indices:
                    output[output_index] = replace(
                        output[output_index],
                        action="unknown",
                        evidence_state="uncertain",
                        temporal_reason="long_missing_or_unreliable_hard_boundary",
                        bounded_uncertain_gap=False,
                        hard_boundary=True,
                    )
                reset_all_lanes()
            continue

        uncertain_gap_start = None
        uncertain_gap_indices = []
        side = (
            frame.anatomical_side
            if frame.anatomical_side in {"left", "right", "bilateral"}
            else "unknown"
        )
        if previous_reliable_side is not None and side != previous_reliable_side:
            anatomical_side_switches += 1
        previous_reliable_side = side

        lane = lanes.setdefault(side, _TemporalLaneState())
        if (
            lane.last_seen_timestamp is not None
            and timestamp - lane.last_seen_timestamp
            > effective.temporal_context_seconds + 1e-9
        ):
            lane = _TemporalLaneState()
            lanes[side] = lane
        lane.last_seen_timestamp = timestamp
        lane.history.append((timestamp, frame.action))
        cutoff = timestamp - effective.temporal_context_seconds
        lane.history = [
            item for item in lane.history if item[0] >= cutoff - 1e-9
        ]
        wanted = _context_action(
            lane.history,
            timestamp=timestamp,
            context_seconds=effective.temporal_context_seconds,
        )

        if lane.active_action is not None and wanted == lane.active_action:
            lane.candidate_action = None
            lane.candidate_evidence_seconds = 0.0
            lane.candidate_output_indices = []
            output.append(
                replace(
                    frame,
                    action=lane.active_action,
                    evidence_state=_evidence_state_for_action(
                        lane.active_action
                    ),
                    temporal_reason="confirmed_lane_context",
                    bounded_uncertain_gap=False,
                    hard_boundary=False,
                )
            )
            continue

        if lane.candidate_action != wanted:
            lane.candidate_action = wanted
            lane.candidate_evidence_seconds = 0.0
            lane.candidate_output_indices = []
        elif lane.candidate_output_indices:
            lane.candidate_evidence_seconds += sample_interval_seconds

        provisional = lane.active_action or (
            "unknown" if wanted == "unknown" else "transition"
        )
        output.append(
            replace(
                frame,
                action=provisional,
                evidence_state=_evidence_state_for_action(provisional),
                temporal_reason=(
                    "stop_hysteresis_held_prior_action"
                    if lane.active_action is not None
                    else "start_confirmation_pending"
                ),
                bounded_uncertain_gap=False,
                hard_boundary=False,
            )
        )
        lane.candidate_output_indices.append(len(output) - 1)
        required = (
            effective.stop_confirmation_seconds
            if lane.active_action is not None
            and wanted in {"idle", "unknown", "transition"}
            else effective.start_confirmation_seconds
        )
        if lane.candidate_evidence_seconds + 1e-9 < required:
            if lane.active_action is not None:
                held_noise_frames += 1
            continue

        if lane.active_action != wanted:
            confirmed_switches += 1
        lane.active_action = wanted
        for index in lane.candidate_output_indices:
            output[index] = replace(
                output[index],
                action=wanted,
                evidence_state=_evidence_state_for_action(wanted),
                temporal_reason="action_confirmed_by_lane_evidence_window",
            )
        lane.candidate_action = None
        lane.candidate_evidence_seconds = 0.0
        lane.candidate_output_indices = []

    hard_boundary_frames = sum(1 for item in output if item.hard_boundary)
    bounded_gap_frames = sum(
        1 for item in output if item.bounded_uncertain_gap
    )
    return {
        "frames": output,
        "metrics": {
            "input_frame_count": len(frames),
            "output_frame_count": len(output),
            "confirmed_switch_count": confirmed_switches,
            "held_noise_frame_count": held_noise_frames,
            "hard_boundary_frame_count": hard_boundary_frames,
            "bounded_uncertain_gap_frame_count": bounded_gap_frames,
            "lost_hard_boundary_frame_count": sum(
                1 for item in output if item.evidence_state == "lost"
            ),
            "long_gap_hard_boundary_frame_count": sum(
                1
                for item in output
                if item.hard_boundary and item.evidence_state == "uncertain"
            ),
            "identity_or_epoch_reset_count": identity_or_epoch_resets,
            "anatomical_side_switch_count": anatomical_side_switches,
            "lane_count": len(lanes),
            "configuration": {
                "start_confirmation_seconds": effective.start_confirmation_seconds,
                "stop_confirmation_seconds": effective.stop_confirmation_seconds,
                "temporal_context_seconds": effective.temporal_context_seconds,
                "bounded_uncertain_gap_seconds": (
                    effective.bounded_uncertain_gap_seconds
                ),
                "sample_interval_seconds": sample_interval_seconds,
            },
        },
    }


def _segment_status(action: str) -> str:
    if action == "lost":
        return "lost"
    if action in {"unknown", "transition"}:
        return "uncertain"
    return "proposed"


def build_pose_segments(
    frames: list[CoarseFrame],
    *,
    source_video_sha256: str,
    sample_interval_seconds: float,
    analysis_end_time: float | None = None,
) -> list[dict[str, Any]]:
    """Run-length encode without crossing identity, epoch, side, or lost boundaries."""

    if not frames:
        return []
    segments: list[dict[str, Any]] = []
    start = 0
    sequence = 1
    for index in range(1, len(frames) + 1):
        at_end = index == len(frames)
        if not at_end:
            left = frames[index - 1]
            right = frames[index]
            same_lane = (
                left.action == right.action
                and left.person_ref == right.person_ref
                and left.lock_epoch == right.lock_epoch
                and left.anatomical_side == right.anatomical_side
                and left.track_state == right.track_state
                and left.lock_state == right.lock_state
                and left.evidence_state == right.evidence_state
                and left.bounded_uncertain_gap == right.bounded_uncertain_gap
                and left.hard_boundary == right.hard_boundary
            )
            if same_lane:
                continue

        run = frames[start:index]
        first, last = run[0], run[-1]
        if index < len(frames):
            end_time = float(frames[index].timestamp)
        elif analysis_end_time is not None:
            end_time = float(analysis_end_time)
        else:
            end_time = float(last.timestamp) + sample_interval_seconds
        if end_time <= float(first.timestamp):
            end_time = float(last.timestamp) + sample_interval_seconds
        observation_counts = {
            name: sum(1 for item in run if item.observation_state == name)
            for name in ("detected", "predicted", "interpolated", "missing", "lost")
        }
        dominant_observation = max(observation_counts, key=observation_counts.get)
        segment_id = f"pose-segment-{sequence:05d}"
        segments.append(
            {
                "segment_id": segment_id,
                "action": first.action,
                "action_name": first.action,
                "person_ref": first.person_ref,
                "lock_epoch": first.lock_epoch,
                "side": first.anatomical_side,
                "anatomical_side": first.anatomical_side,
                "start_time": round(first.timestamp, 6),
                "end_time": round(end_time, 6),
                "duration_seconds": round(end_time - first.timestamp, 6),
                "start_frame": first.source_frame_index,
                "end_frame": last.source_frame_index,
                "source_frame_indices": [
                    item.source_frame_index for item in run
                ],
                "source_video_sha256": source_video_sha256,
                "status": _segment_status(first.action),
                "confirmation_status": "unconfirmed",
                "training_eligible": False,
                "training_approval": "pending",
                "track_state": first.track_state,
                "lock_state": first.lock_state,
                "observation_state": dominant_observation,
                "detected_ratio": sum(item.detected_ratio for item in run) / len(run),
                "predicted_ratio": sum(item.predicted_ratio for item in run) / len(run),
                "interpolated_ratio": sum(
                    item.interpolated_ratio for item in run
                )
                / len(run),
                "missing_ratio": sum(item.missing_ratio for item in run) / len(run),
                "required_joints_reliable": all(
                    item.required_joints_reliable for item in run
                ),
                "direction_clear": any(item.direction_clear for item in run),
                "multi_person_seen": any(
                    item.candidate_person_count > 1 for item in run
                ),
                "temporarily_lost": first.lock_state == "temporarily_lost",
                "raw_lost": first.action == "lost",
                "evidence_state": (
                    first.evidence_state
                    if first.evidence_state != "raw"
                    else _evidence_state_for_action(first.action)
                ),
                "temporal_reason": first.temporal_reason,
                "temporal_reasons": list(
                    dict.fromkeys(item.temporal_reason for item in run)
                ),
                "bounded_uncertain_gap": first.bounded_uncertain_gap,
                "hard_boundary": first.hard_boundary,
                "source_segment_ids": [segment_id],
            }
        )
        sequence += 1
        start = index
    return segments
