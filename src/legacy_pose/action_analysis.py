"""Causal, identity-free action features and neutral timeline segmentation.

The analyzer consumes pose records already produced for the locked anonymous
track.  It never owns a detector and therefore cannot add pose inference calls.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SEGMENT_TYPES = ("movement", "idle", "uncertain", "lost")
SEGMENT_SOURCES = ("detected", "predicted", "interpolated", "manual", "edited")
SEGMENT_STATUSES = ("proposed", "confirmed", "uncertain", "lost", "excluded")
DETECTED_STATES = {"detected"}
AUXILIARY_STATES = {"interpolated", "predicted"}
RELIABLE_STATES = DETECTED_STATES | AUXILIARY_STATES
HARD_LOST_STATES = {"lost", "temporarily_lost", "awaiting_manual_relock"}
HARD_UNCERTAIN_STATES = {"ambiguous", "ambiguous_association"}
HARD_LOST_OCCLUSIONS = {"off_frame"}
HARD_UNCERTAIN_OCCLUSIONS = {"severe"}
COARSE_ACTIONS = (
    "idle", "reach", "retract", "lift", "lower", "carry", "place", "hold",
    "release", "rotate", "push", "pull", "transition", "move", "unknown", "lost",
)
SHORT_DIRECTIONAL_ACTIONS = {"reach", "retract", "lift", "lower", "push", "pull"}
NORMAL_COARSE_ACTIONS = set(COARSE_ACTIONS) - {"transition", "unknown", "lost"}
FORBIDDEN_POSE_ACTION_TOKENS = {
    "grasp", "grabbing", "grab", "pickup", "pick_up", "take", "hand", "finger",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class ActionAnalysisConfig:
    enabled: bool = True
    movement_start_threshold: float = 0.42
    movement_stop_threshold: float = 0.20
    start_confirmation_seconds: float = 0.20
    stop_confirmation_seconds: float = 0.30
    minimum_segment_seconds: float = 0.30
    short_gap_merge_seconds: float = 0.25
    minimum_valid_keypoint_ratio: float = 0.55
    maximum_uncertain_ratio: float = 0.45
    stable_event_minimum_seconds: float = 1.00
    short_event_minimum_seconds: float = 0.50
    minimum_detected_evidence_ratio: float = 0.65
    short_event_minimum_detected_ratio: float = 0.75
    maximum_prediction_ratio: float = 0.35
    action_switch_confirmation_seconds: float = 0.30
    action_end_hysteresis_seconds: float = 0.30
    generic_move_minimum_seconds: float = 1.00
    idle_minimum_seconds: float = 1.00
    minimum_directional_displacement: float = 0.05
    representative_frame_strategy: str = "highest_quality_near_middle"
    distance_normalization_method: str = "torso_or_bbox_scale"

    def __post_init__(self) -> None:
        if self.movement_stop_threshold >= self.movement_start_threshold:
            raise ValueError("movement_stop_threshold must be below movement_start_threshold")
        for name in (
            "start_confirmation_seconds", "stop_confirmation_seconds",
            "minimum_segment_seconds", "short_gap_merge_seconds",
            "stable_event_minimum_seconds", "short_event_minimum_seconds",
            "action_switch_confirmation_seconds", "action_end_hysteresis_seconds",
            "generic_move_minimum_seconds", "idle_minimum_seconds",
            "minimum_directional_displacement",
        ):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} cannot be negative")
        for name in (
            "minimum_valid_keypoint_ratio", "maximum_uncertain_ratio",
            "minimum_detected_evidence_ratio", "short_event_minimum_detected_ratio",
            "maximum_prediction_ratio",
        ):
            if not 0 <= float(getattr(self, name)) <= 1:
                raise ValueError(f"{name} must be within [0, 1]")
        if self.short_event_minimum_seconds > self.stable_event_minimum_seconds:
            raise ValueError("short_event_minimum_seconds cannot exceed stable_event_minimum_seconds")
        if self.representative_frame_strategy != "highest_quality_near_middle":
            raise ValueError("Unsupported representative_frame_strategy")
        if self.distance_normalization_method != "torso_or_bbox_scale":
            raise ValueError("Unsupported distance_normalization_method")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ActionAnalysisConfig":
        """Read the deliberately flat YAML file without adding a YAML dependency."""

        values: dict[str, Any] = {}
        for raw_line in Path(path).read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            if ":" not in line:
                raise ValueError(f"Invalid action config line: {raw_line}")
            key, raw = (part.strip() for part in line.split(":", 1))
            if not hasattr(cls, key):
                raise ValueError(f"Unknown action config key: {key}")
            lowered = raw.lower()
            if lowered in {"true", "false"}:
                value: Any = lowered == "true"
            else:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw.strip("\"'")
            values[key] = value
        return cls(**values)


@dataclass
class ActionSegment:
    segment_id: str
    video_fingerprint: str
    locked_track_id: int | str | None
    action_name: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration_seconds: float
    segment_type: str
    source: str
    status: str
    confidence: float
    valid_frame_ratio: float
    uncertain_frame_ratio: float
    lost_frame_ratio: float
    interpolation_frame_ratio: float
    uncertainty_reasons: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    parent_segment_id: str | None = None
    revision: int = 0
    notes: str = ""
    representative_frame: str | None = None
    representative_time: float | None = None
    detected_frame_ratio: float = 1.0
    observation_state: str = "detected"
    person_ref: str | None = None
    lock_epoch: int | str | None = None
    side: str = "bilateral"
    display_eligible: bool = True
    source_segment_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.start_time = round(float(self.start_time), 9)
        self.end_time = round(float(self.end_time), 9)
        expected = round(self.end_time - self.start_time, 9)
        if expected <= 0:
            raise ValueError("Action segment end_time must be greater than start_time")
        if self.start_frame < 0 or self.end_frame < self.start_frame:
            raise ValueError("Action segment frame range is invalid")
        self.duration_seconds = expected
        if self.segment_type not in SEGMENT_TYPES:
            raise ValueError(f"Unsupported segment_type: {self.segment_type}")
        if self.source not in SEGMENT_SOURCES:
            raise ValueError(f"Unsupported source: {self.source}")
        if self.status not in SEGMENT_STATUSES:
            raise ValueError(f"Unsupported status: {self.status}")
        self.confidence = round(float(np.clip(self.confidence, 0.0, 1.0)), 6)
        for name in (
            "valid_frame_ratio", "uncertain_frame_ratio", "lost_frame_ratio",
            "interpolation_frame_ratio",
            "detected_frame_ratio",
        ):
            setattr(self, name, round(float(np.clip(getattr(self, name), 0.0, 1.0)), 6))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ActionSegment":
        known = cls.__dataclass_fields__.keys()
        return cls(**{key: value[key] for key in known if key in value})


@dataclass(frozen=True)
class ActionReviewEvent:
    event_id: str
    segment_id: str
    operation: str
    before: dict[str, Any] | None
    after: dict[str, Any] | None
    timestamp: str
    source: str = "manual"

    @classmethod
    def create(
        cls, segment_id: str, operation: str,
        before: dict[str, Any] | None, after: dict[str, Any] | None,
    ) -> "ActionReviewEvent":
        return cls(f"evt_{uuid.uuid4().hex}", segment_id, operation, before, after, utc_now())


def _finite_point(points: np.ndarray, index: int) -> bool:
    return points.shape == (17, 3) and bool(np.isfinite(points[index, :2]).all())


def _body_scale(record: dict[str, Any]) -> float:
    stated = record.get("body_scale")
    if stated is not None and np.isfinite(float(stated)) and float(stated) > 1:
        return float(stated)
    points = np.asarray(record.get("smoothed"), dtype=np.float32)
    distances: list[float] = []
    if points.shape == (17, 3):
        if _finite_point(points, 5) and _finite_point(points, 6):
            distances.append(float(np.linalg.norm(points[5, :2] - points[6, :2])))
        if all(_finite_point(points, index) for index in (5, 6, 11, 12)):
            shoulder = (points[5, :2] + points[6, :2]) / 2
            hip = (points[11, :2] + points[12, :2]) / 2
            distances.append(float(np.linalg.norm(shoulder - hip)))
    bbox = record.get("bbox")
    if bbox is not None:
        box = np.asarray(bbox, dtype=np.float32)
        if box.shape == (4,):
            distances.append(float(max(box[2] - box[0], box[3] - box[1]) * 0.45))
    return max(8.0, float(np.nanmedian(distances)) if distances else 100.0)


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float | None:
    first, second = a - b, c - b
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-6:
        return None
    cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    return round(math.degrees(math.acos(cosine)), 6)


def _state_tokens(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = str(value).replace(",", ";").split(";")
    return {str(item).strip().lower() for item in values if str(item).strip()}


def _identity_key(record: dict[str, Any]) -> tuple[Any, Any]:
    person = record.get("person_ref")
    if person in (None, ""):
        person = record.get("locked_track_id", record.get("track_id"))
    return person, record.get("lock_epoch")


def _record_quality(record: dict[str, Any]) -> tuple[float, float, float, bool, bool]:
    statuses = np.asarray(record.get("statuses", ["missing"] * 17)).astype(str)
    reliable = np.isin(statuses, list(RELIABLE_STATES))
    valid_ratio = float(np.mean(reliable)) if statuses.size else 0.0
    interpolated_ratio = float(np.mean(np.isin(statuses, ["interpolated", "predicted"]))) if statuses.size else 0.0
    missing_ratio = float(np.mean(np.isin(statuses, ["missing", "uncertain", "rejected", "provisional"]))) if statuses.size else 1.0
    tracking_states = set()
    for key in ("track_state", "lock_state", "lock_acquisition_state", "track_state_summary"):
        tracking_states.update(_state_tokens(record.get(key)))
    occlusions = set()
    for key in ("occlusion", "occlusion_summary", "raw_occlusion_states"):
        occlusions.update(_state_tokens(record.get(key)))
    human_boundary = _state_tokens(record.get("human_hard_boundary"))
    lost = bool(
        tracking_states & HARD_LOST_STATES
        or occlusions & HARD_LOST_OCCLUSIONS
        or human_boundary & (HARD_LOST_STATES | HARD_LOST_OCCLUSIONS)
        or record.get("temporarily_lost")
        or record.get("detection_present") is False
        or record.get("raw_lost")
        or record.get("raw_off_frame")
    )
    # The pose smoother's broad uncertainty flag can be true because a distal
    # point (commonly an ankle outside the camera view) is missing.  That must
    # not invalidate otherwise reliable upper-body action evidence.  Identity
    # association ambiguity is always uncertain; general pose uncertainty only
    # escalates when the usable-keypoint ratio is also materially low.
    uncertain = bool(
        record.get("ambiguous_association")
        or record.get("raw_ambiguous")
        or tracking_states & HARD_UNCERTAIN_STATES
        or occlusions & HARD_UNCERTAIN_OCCLUSIONS
        or record.get("association_state") == "ambiguous"
        or (record.get("pose_uncertain") and valid_ratio < 0.65)
    )
    return valid_ratio, interpolated_ratio, missing_ratio, lost, uncertain


def extract_motion_features(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract timestamp-aware normalized features using only existing pose records."""

    features: list[dict[str, Any]] = []
    previous_points: np.ndarray | None = None
    previous_statuses: np.ndarray | None = None
    previous_velocities: dict[int, float] = {}
    previous_time: float | None = None
    previous_identity: tuple[Any, Any] | None = None
    speed_indices = (5, 6, 7, 8, 9, 10, 11, 12)
    for record in records:
        timestamp = float(record["timestamp"])
        points = np.asarray(record.get("smoothed"), dtype=np.float32)
        statuses = np.asarray(record.get("statuses", ["missing"] * 17)).astype(str)
        scale = _body_scale(record)
        valid_ratio, interpolation_ratio, missing_ratio, lost, uncertain = _record_quality(record)
        detected_ratio = float(np.mean(statuses == "detected")) if statuses.size else 0.0
        identity = _identity_key(record)
        identity_changed = bool(
            previous_identity is not None and identity != previous_identity
            and any(value not in (None, "") for value in identity + previous_identity)
        )
        if identity_changed:
            uncertain = True
        hard_boundary = bool(lost or uncertain or identity_changed)
        dt = timestamp - previous_time if previous_time is not None else 0.0
        speeds: dict[int, float | None] = {}
        accelerations: dict[int, float | None] = {}
        for index in speed_indices:
            reliable = (
                not hard_boundary and dt > 1e-6 and previous_points is not None
                and previous_statuses is not None and _finite_point(points, index)
                and _finite_point(previous_points, index)
                and statuses[index] in DETECTED_STATES
                and previous_statuses[index] in DETECTED_STATES
            )
            speed = float(np.linalg.norm(points[index, :2] - previous_points[index, :2]) / scale / dt) if reliable else None
            speeds[index] = speed
            old_speed = previous_velocities.get(index)
            accelerations[index] = abs(speed - old_speed) / dt if speed is not None and old_speed is not None else None
            if speed is not None:
                previous_velocities[index] = speed

        def max_valid(indices: Iterable[int]) -> float:
            values = [float(speeds[index]) for index in indices if speeds.get(index) is not None]
            return max(values) if values else 0.0

        wrist_speed = max_valid((9, 10)); elbow_speed = max_valid((7, 8)); shoulder_speed = max_valid((5, 6))
        torso_center_speed = 0.0
        if not hard_boundary and dt > 1e-6 and previous_points is not None and previous_statuses is not None and all(
            _finite_point(points, index) and _finite_point(previous_points, index)
            and statuses[index] in DETECTED_STATES and previous_statuses[index] in DETECTED_STATES
            for index in (5, 6, 11, 12)
        ):
            center = np.mean(points[[5, 6, 11, 12], :2], axis=0)
            old_center = np.mean(previous_points[[5, 6, 11, 12], :2], axis=0)
            torso_center_speed = float(np.linalg.norm(center - old_center) / scale / dt)
        uncertain_keypoint_ratio = float(
            np.mean(np.isin(statuses, ["uncertain", "rejected", "provisional"]))
        ) if statuses.size else 1.0
        activity = (
            0.38 * wrist_speed + 0.27 * elbow_speed + 0.15 * shoulder_speed
            + 0.12 * torso_center_speed + 0.08 * max(shoulder_speed, torso_center_speed)
        )
        angles: dict[str, float | None] = {"left_elbow": None, "right_elbow": None, "left_shoulder": None, "right_shoulder": None}
        if points.shape == (17, 3):
            if all(_finite_point(points, i) for i in (5, 7, 9)): angles["left_elbow"] = _angle(points[5, :2], points[7, :2], points[9, :2])
            if all(_finite_point(points, i) for i in (6, 8, 10)): angles["right_elbow"] = _angle(points[6, :2], points[8, :2], points[10, :2])
            if all(_finite_point(points, i) for i in (7, 5, 11)): angles["left_shoulder"] = _angle(points[7, :2], points[5, :2], points[11, :2])
            if all(_finite_point(points, i) for i in (8, 6, 12)): angles["right_shoulder"] = _angle(points[8, :2], points[6, :2], points[12, :2])
        hand_distance = (
            float(np.linalg.norm(points[9, :2] - points[10, :2]) / scale)
            if points.shape == (17, 3) and _finite_point(points, 9) and _finite_point(points, 10) else None
        )
        wrist_relative_positions: dict[str, dict[str, list[float]] | None] = {
            "left": None, "right": None,
        }
        if points.shape == (17, 3) and all(_finite_point(points, i) for i in (5, 6, 11, 12)):
            torso_center = np.mean(points[[5, 6, 11, 12], :2], axis=0)
            for side, wrist, shoulder, hip in (("left", 9, 5, 11), ("right", 10, 6, 12)):
                if _finite_point(points, wrist):
                    wrist_relative_positions[side] = {
                        "torso_center": ((points[wrist, :2] - torso_center) / scale).round(6).tolist(),
                        "shoulder": ((points[wrist, :2] - points[shoulder, :2]) / scale).round(6).tolist(),
                        "hip": ((points[wrist, :2] - points[hip, :2]) / scale).round(6).tolist(),
                    }
        features.append({
            "frame_index": int(record["frame_index"]), "timestamp": timestamp,
            "locked_track_id": record.get("locked_track_id"), "body_scale": round(scale, 6),
            "left_wrist_speed": speeds.get(9), "right_wrist_speed": speeds.get(10),
            "left_elbow_speed": speeds.get(7), "right_elbow_speed": speeds.get(8),
            "left_shoulder_speed": speeds.get(5), "right_shoulder_speed": speeds.get(6),
            "left_wrist_acceleration": accelerations.get(9), "right_wrist_acceleration": accelerations.get(10),
            "elbow_angles": angles, "hand_distance": hand_distance,
            "wrist_relative_positions": wrist_relative_positions,
            "upper_body_center_speed": round(torso_center_speed, 6),
            "body_speed": round(max(shoulder_speed, torso_center_speed), 6),
            "activity_intensity": round(activity, 6), "valid_keypoint_ratio": round(valid_ratio, 6),
            "detected_keypoint_ratio": round(detected_ratio, 6),
            "uncertain_keypoint_ratio": round(uncertain_keypoint_ratio, 6),
            "interpolation_ratio": round(interpolation_ratio, 6), "missing_ratio": round(missing_ratio, 6),
            "lost": lost, "uncertain": uncertain, "hard_boundary": hard_boundary,
            "identity_boundary": identity_changed,
            "person_ref": record.get("person_ref"), "lock_epoch": record.get("lock_epoch"),
            "side": str(record.get("side", "bilateral")),
            "track_state": record.get("track_state", record.get("lock_state")),
            "occlusion": record.get("occlusion"),
            "near_frame_edge": bool(record.get("near_frame_edge") or record.get("edge_risk")),
            "multi_person": bool(record.get("multi_person") or int(record.get("candidate_person_count") or 0) > 1),
        })
        if hard_boundary:
            previous_points = None
            previous_statuses = None
            previous_velocities.clear()
        else:
            previous_points = points.copy() if points.shape == (17, 3) else None
            previous_statuses = statuses.copy() if statuses.shape == (17,) else None
        previous_time = timestamp
        previous_identity = identity
    return features


def _quality_label(feature: dict[str, Any], config: ActionAnalysisConfig) -> str | None:
    if feature["lost"]:
        return "lost"
    if (
        feature["uncertain"]
        or feature["valid_keypoint_ratio"] < config.minimum_valid_keypoint_ratio
        or feature.get("detected_keypoint_ratio", feature.get("valid_keypoint_ratio", 0.0)) < config.minimum_detected_evidence_ratio
        or feature.get("interpolation_ratio", 0.0) > config.maximum_prediction_ratio
        or feature.get("uncertain_keypoint_ratio", 0.0) > config.maximum_uncertain_ratio
    ):
        return "uncertain"
    return None


def _causal_labels(features: list[dict[str, Any]], config: ActionAnalysisConfig) -> list[str]:
    if not features:
        return []
    labels = ["idle"] * len(features)
    state = "idle"
    candidate_kind: str | None = None
    candidate_start = 0
    for index, feature in enumerate(features):
        quality = _quality_label(feature, config)
        if quality is not None:
            labels[index] = quality
            state = "idle"
            candidate_kind = None
            continue
        intensity = float(feature["activity_intensity"])
        wanted = (
            "movement" if state == "idle" and intensity >= config.movement_start_threshold
            else "idle" if state == "movement" and intensity <= config.movement_stop_threshold
            else state
        )
        if wanted == state:
            candidate_kind = None
            labels[index] = state
            continue
        if candidate_kind != wanted:
            candidate_kind, candidate_start = wanted, index
        elapsed = float(feature["timestamp"] - features[candidate_start]["timestamp"])
        required = config.start_confirmation_seconds if wanted == "movement" else config.stop_confirmation_seconds
        labels[index] = state
        if elapsed + 1e-9 >= required:
            state = wanted
            for buffered in range(candidate_start, index + 1):
                if _quality_label(features[buffered], config) is None:
                    labels[buffered] = state
            candidate_kind = None
    return labels


def _segment_from_run(
    features: list[dict[str, Any]], labels: list[str], start: int, stop: int,
    analysis_end_time: float, video_fingerprint: str, locked_track_id: Any,
    sequence: dict[str, int], config: ActionAnalysisConfig,
) -> ActionSegment:
    segment_type = labels[start]
    sequence[segment_type] = sequence.get(segment_type, 0) + 1
    first, last = features[start], features[stop]
    end_time = float(features[stop + 1]["timestamp"]) if stop + 1 < len(features) else float(analysis_end_time)
    if end_time <= float(first["timestamp"]):
        prior_dt = (
            float(last["timestamp"] - features[stop - 1]["timestamp"]) if stop > start else 0.001
        )
        end_time = float(last["timestamp"]) + max(prior_dt, 0.001)
    run = features[start:stop + 1]
    valid = float(np.mean([item["valid_keypoint_ratio"] for item in run]))
    uncertain = float(np.mean([item["uncertain"] for item in run]))
    lost = float(np.mean([item["lost"] for item in run]))
    interpolation = float(np.mean([item["interpolation_ratio"] for item in run]))
    detected = float(np.mean([
        item.get("detected_keypoint_ratio", item.get("valid_keypoint_ratio", 0.0))
        for item in run
    ]))
    reasons: list[str] = []
    if lost > 0: reasons.append("locked_track_lost")
    if uncertain > 0: reasons.append("ambiguous_or_uncertain_pose")
    if valid < config.minimum_valid_keypoint_ratio: reasons.append("low_valid_keypoint_ratio")
    if detected < config.minimum_detected_evidence_ratio: reasons.append("low_detected_keypoint_ratio")
    if interpolation > config.maximum_prediction_ratio: reasons.append("high_interpolation_ratio")
    status = (
        "lost" if segment_type == "lost"
        else "uncertain" if segment_type == "uncertain" or detected < config.minimum_detected_evidence_ratio
        or interpolation > config.maximum_prediction_ratio
        else "proposed"
    )
    confidence = valid * (1.0 - 0.6 * uncertain) * (1.0 - lost) * (1.0 - 0.35 * interpolation)
    source = "detected"
    observation_state = "detected"
    if segment_type == "lost":
        observation_state = "lost"
    elif interpolation > config.maximum_prediction_ratio or detected < config.minimum_detected_evidence_ratio:
        source = "predicted"
        observation_state = "predicted"
    return ActionSegment(
        segment_id=f"seg_{uuid.uuid4().hex}", video_fingerprint=video_fingerprint,
        locked_track_id=first.get("locked_track_id", locked_track_id),
        action_name=f"{segment_type}_{sequence[segment_type]:03d}",
        start_frame=int(first["frame_index"]), end_frame=int(last["frame_index"]),
        start_time=float(first["timestamp"]), end_time=end_time,
        duration_seconds=end_time - float(first["timestamp"]), segment_type=segment_type,
        source=source, status=status, confidence=confidence,
        valid_frame_ratio=valid, uncertain_frame_ratio=uncertain, lost_frame_ratio=lost,
        interpolation_frame_ratio=interpolation, uncertainty_reasons=reasons,
        detected_frame_ratio=detected, observation_state=observation_state,
        person_ref=first.get("person_ref"), lock_epoch=first.get("lock_epoch"),
        side=str(first.get("side", "bilateral")),
    )


def segment_actions(
    features: list[dict[str, Any]], config: ActionAnalysisConfig,
    *, video_fingerprint: str, locked_track_id: Any, analysis_end_time: float,
) -> tuple[list[ActionSegment], list[dict[str, Any]]]:
    """Return neutral automatic suggestions and separately recorded merge suggestions."""

    if not features:
        return [], []
    labels = _causal_labels(features, config)
    # Isolated movement below the minimum becomes uncertain, not a fabricated action.
    index = 0
    while index < len(labels):
        stop = index
        while stop + 1 < len(labels) and labels[stop + 1] == labels[index]: stop += 1
        run_end = float(features[stop + 1]["timestamp"]) if stop + 1 < len(features) else analysis_end_time
        if labels[index] == "movement" and run_end - float(features[index]["timestamp"]) < config.minimum_segment_seconds:
            labels[index:stop + 1] = ["uncertain"] * (stop - index + 1)
        index = stop + 1
    sequence: dict[str, int] = {}
    segments: list[ActionSegment] = []
    index = 0
    while index < len(labels):
        stop = index
        while stop + 1 < len(labels) and labels[stop + 1] == labels[index]: stop += 1
        segments.append(_segment_from_run(
            features, labels, index, stop, analysis_end_time,
            video_fingerprint, locked_track_id, sequence, config,
        ))
        index = stop + 1
    merge_suggestions: list[dict[str, Any]] = []
    for middle in range(1, len(segments) - 1):
        before, gap, after = segments[middle - 1:middle + 2]
        if (
            before.segment_type == after.segment_type == "movement"
            and gap.segment_type == "idle"
            and gap.duration_seconds <= config.short_gap_merge_seconds
        ):
            merge_suggestions.append({
                "suggestion_id": f"merge_{uuid.uuid4().hex}",
                "segment_ids": [before.segment_id, gap.segment_id, after.segment_id],
                "reason": "short_idle_gap", "gap_seconds": gap.duration_seconds,
                "applied": False,
            })
    return segments, merge_suggestions


def _event_float(event: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(event.get(key, default))
    except (TypeError, ValueError):
        return default


def _event_bool(event: dict[str, Any], key: str, default: bool = False) -> bool:
    value = event.get(key)
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _coarse_action(value: Any) -> str:
    action = str(value or "unknown").strip().lower().replace(" ", "_")
    aliases = {
        "movement": "move", "moving": "move", "reaching": "reach",
        "retracting": "retract", "lifting": "lift", "lowering": "lower",
        "carrying": "carry", "placing": "place", "holding": "hold",
        "releasing": "release", "turning": "rotate", "rotating": "rotate",
        "pushing": "push", "pulling": "pull", "unlabeled_action": "unknown",
    }
    action = aliases.get(action, action)
    if any(token in action for token in FORBIDDEN_POSE_ACTION_TOKENS):
        return "unknown"
    return action if action in COARSE_ACTIONS else "unknown"


def _event_duration(event: dict[str, Any]) -> float:
    stated = _event_float(event, "duration_seconds", -1.0)
    if stated >= 0:
        return stated
    return max(0.0, _event_float(event, "end_time") - _event_float(event, "start_time"))


def _event_identity(event: dict[str, Any]) -> tuple[str, str, str, str]:
    clip = str(event.get("clip_id", event.get("video_id", "")))
    person = str(event.get("person_ref", event.get("locked_track_id", "")))
    epoch = str(event.get("lock_epoch", ""))
    side = str(event.get("side", "unknown")).lower()
    return clip, person, epoch, side


def _event_boundary(event: dict[str, Any]) -> tuple[str | None, list[str]]:
    states: set[str] = set()
    for key in ("track_state", "lock_state", "track_state_summary", "raw_track_states"):
        states.update(_state_tokens(event.get(key)))
    occlusions: set[str] = set()
    for key in ("occlusion", "occlusion_summary", "raw_occlusion_states"):
        occlusions.update(_state_tokens(event.get(key)))
    human = _state_tokens(event.get("human_hard_boundary"))
    reasons: list[str] = []
    lost_reasons = sorted(
        (states & HARD_LOST_STATES)
        | (occlusions & HARD_LOST_OCCLUSIONS)
        | (human & (HARD_LOST_STATES | HARD_LOST_OCCLUSIONS))
    )
    if _event_bool(event, "raw_lost"):
        lost_reasons.append("raw_lost")
    if _event_bool(event, "raw_off_frame"):
        lost_reasons.append("raw_off_frame")
    if _event_bool(event, "temporarily_lost"):
        lost_reasons.append("temporarily_lost")
    if lost_reasons:
        return "lost", sorted(set(lost_reasons))
    uncertain_reasons = sorted(
        (states & HARD_UNCERTAIN_STATES) | (occlusions & HARD_UNCERTAIN_OCCLUSIONS)
    )
    if _event_bool(event, "ambiguous_seen") or _event_bool(event, "raw_ambiguous"):
        uncertain_reasons.append("ambiguous")
    observation = str(event.get("observation_state", event.get("dominant_observation_state", ""))).lower()
    if observation == "missing":
        uncertain_reasons.append("missing_pose_evidence")
    if _event_bool(event, "human_semantic_defer"):
        uncertain_reasons.append("human_semantic_defer")
    if uncertain_reasons:
        return "uncertain", sorted(set(uncertain_reasons))
    return None, reasons


def _event_ratios(event: dict[str, Any]) -> tuple[float, float, float, float]:
    observation = str(event.get("observation_state", event.get("dominant_observation_state", ""))).lower()
    explicit = any(key in event and event.get(key) not in (None, "") for key in (
        "detected_ratio", "predicted_ratio", "interpolated_ratio", "missing_ratio",
    ))
    if explicit:
        return (
            _event_float(event, "detected_ratio"), _event_float(event, "predicted_ratio"),
            _event_float(event, "interpolated_ratio"), _event_float(event, "missing_ratio"),
        )
    return (
        1.0 if observation == "detected" else 0.0,
        1.0 if observation == "predicted" else 0.0,
        1.0 if observation == "interpolated" else 0.0,
        1.0 if observation in {"missing", "uncertain"} else 0.0,
    )


def _stable_event_copy(
    event: dict[str, Any], *, action: str, status: str, observation_state: str,
    reason: str, boundary_reasons: list[str] | None = None,
) -> dict[str, Any]:
    result = dict(event)
    result["action"] = action
    result["action_name"] = action
    result["start_time"] = round(_event_float(event, "start_time"), 9)
    result["end_time"] = round(_event_float(event, "end_time"), 9)
    result["duration_seconds"] = round(result["end_time"] - result["start_time"], 9)
    result["status"] = status
    result["confirmation_status"] = "unconfirmed"
    result["training_eligible"] = False
    result["training_approval"] = "pending"
    result["observation_state"] = observation_state
    result["display_eligible"] = True
    result["stabilization_reason"] = reason
    result["boundary_reasons"] = list(boundary_reasons or [])
    result["source_event_ids"] = str(event.get("source_event_ids") or event.get("event_id") or event.get("segment_id") or "")
    result["source_segment_ids"] = str(event.get("source_segment_ids") or event.get("source_pose_segment_ids") or "")
    return result


def _between_has_barrier(
    barriers: list[dict[str, Any]], identity: tuple[str, str, str, str],
    start: float, end: float,
) -> bool:
    return any(
        _event_identity(item) == identity
        and _event_float(item, "end_time") > start + 1e-9
        and _event_float(item, "start_time") < end - 1e-9
        for item in barriers
    )


def stabilize_action_events(
    events: list[dict[str, Any]], config: ActionAnalysisConfig,
) -> dict[str, Any]:
    """Build the outward stable-action layer without deleting pose evidence.

    Input rows remain byte-for-value equivalent in ``pose_evidence``.  Suppressed
    fragments are returned separately with reasons, so duration stabilization
    never erases lost, missing, predicted, or transition evidence.
    """

    pose_evidence = [dict(item) for item in events]
    candidates: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    barriers: list[dict[str, Any]] = []
    hard_boundary_count = 0
    short_preserved_count = 0

    ordered = sorted(
        (dict(item) for item in events),
        key=lambda item: (_event_identity(item), _event_float(item, "start_time"), _event_float(item, "end_time")),
    )
    for event in ordered:
        action = _coarse_action(event.get("action", event.get("action_name")))
        duration = _event_duration(event)
        detected, predicted, interpolated, missing = _event_ratios(event)
        prediction_ratio = predicted + interpolated
        boundary, boundary_reasons = _event_boundary(event)
        if boundary:
            hard_boundary_count += 1
            barriers.append(event)
        if boundary == "lost":
            candidates.append(_stable_event_copy(
                event, action="lost", status="lost", observation_state="lost",
                reason="hard_lost_or_off_frame_boundary", boundary_reasons=boundary_reasons,
            ))
            continue
        if _event_bool(event, "human_semantic_defer"):
            deferred = _stable_event_copy(
                event, action="unknown", status="uncertain", observation_state="uncertain",
                reason="human_semantic_fact_requires_pose_action_review", boundary_reasons=boundary_reasons,
            )
            deferred["display_eligible"] = False
            deferred["suppressed"] = True
            deferred["candidate_action_options"] = str(event.get("candidate_action_options") or "reach;lift;retract;unknown")
            suppressed.append(deferred)
            continue
        if boundary == "uncertain":
            if duration >= config.stable_event_minimum_seconds:
                candidates.append(_stable_event_copy(
                    event, action="unknown", status="uncertain", observation_state="uncertain",
                    reason="hard_uncertain_or_missing_boundary", boundary_reasons=boundary_reasons,
                ))
            else:
                uncertain = _stable_event_copy(
                    event, action="unknown", status="uncertain", observation_state="uncertain",
                    reason="short_hard_boundary_retained_as_non_display_evidence", boundary_reasons=boundary_reasons,
                )
                uncertain["display_eligible"] = False
                uncertain["suppressed"] = True
                suppressed.append(uncertain)
            continue

        identity = _event_identity(event)
        identity_reliable = bool(identity[1] and identity[2] and identity[3] in {"left", "right", "bilateral"})
        required_reliable = _event_bool(event, "required_joints_reliable", detected >= config.minimum_detected_evidence_ratio)
        direction_clear = _event_bool(event, "direction_clear", False)
        risky_context = (
            _event_bool(event, "raw_multi") or _event_bool(event, "multi_person_seen")
            or _event_bool(event, "raw_edge") or _event_bool(event, "edge_risk")
        )
        evidence_reliable = (
            detected >= config.minimum_detected_evidence_ratio
            and prediction_ratio <= config.maximum_prediction_ratio
            and missing <= config.maximum_uncertain_ratio
        )
        if not evidence_reliable:
            if duration >= config.stable_event_minimum_seconds:
                candidates.append(_stable_event_copy(
                    event, action="unknown", status="uncertain", observation_state="uncertain",
                    reason="prediction_or_missing_evidence_dominates",
                ))
            else:
                weak = _stable_event_copy(
                    event, action="transition", status="uncertain", observation_state="uncertain",
                    reason="short_prediction_or_missing_evidence_suppressed",
                )
                weak["display_eligible"] = False
                weak["suppressed"] = True
                suppressed.append(weak)
            continue

        if duration < config.short_event_minimum_seconds - 1e-9:
            tiny = _stable_event_copy(
                event, action="transition", status="uncertain", observation_state="detected",
                reason="below_short_event_minimum_retained_as_pose_evidence",
            )
            tiny["display_eligible"] = False
            tiny["suppressed"] = True
            suppressed.append(tiny)
            continue

        if duration < config.stable_event_minimum_seconds - 1e-9:
            high_risk_threshold = min(1.0, config.short_event_minimum_detected_ratio + 0.10)
            short_allowed = (
                action in SHORT_DIRECTIONAL_ACTIONS and direction_clear and required_reliable
                and identity_reliable and detected >= config.short_event_minimum_detected_ratio
                and (not risky_context or detected >= high_risk_threshold)
            )
            if not short_allowed:
                short = _stable_event_copy(
                    event, action="transition", status="uncertain", observation_state="detected",
                    reason="subsecond_event_failed_direction_visibility_or_lock_gate",
                )
                short["display_eligible"] = False
                short["suppressed"] = True
                suppressed.append(short)
                continue
            short_preserved_count += 1

        minimum = (
            config.generic_move_minimum_seconds if action == "move"
            else config.idle_minimum_seconds if action == "idle"
            else config.stable_event_minimum_seconds
        )
        if action in {"move", "idle"} and duration < minimum - 1e-9:
            generic = _stable_event_copy(
                event, action="transition", status="uncertain", observation_state="detected",
                reason=f"short_{action}_suppressed_by_strict_generic_gate",
            )
            generic["display_eligible"] = False
            generic["suppressed"] = True
            suppressed.append(generic)
            continue
        if action in {"unknown", "transition"} and duration < config.stable_event_minimum_seconds - 1e-9:
            unknown = _stable_event_copy(
                event, action=action, status="uncertain", observation_state="uncertain",
                reason="short_unknown_or_transition_retained_as_non_display_evidence",
            )
            unknown["display_eligible"] = False
            unknown["suppressed"] = True
            suppressed.append(unknown)
            continue
        candidates.append(_stable_event_copy(
            event, action=action, status="proposed" if action in NORMAL_COARSE_ACTIONS else "uncertain",
            observation_state="detected", reason="duration_and_visibility_gate_passed",
        ))

    lanes: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for candidate in candidates:
        lanes.setdefault(_event_identity(candidate), []).append(candidate)
    stable: list[dict[str, Any]] = []
    merge_count = 0
    for identity, lane in sorted(lanes.items()):
        lane.sort(key=lambda item: (_event_float(item, "start_time"), _event_float(item, "end_time")))
        merged_lane: list[dict[str, Any]] = []
        for candidate in lane:
            if merged_lane:
                prior = merged_lane[-1]
                gap = _event_float(candidate, "start_time") - _event_float(prior, "end_time")
                same_action = candidate["action"] == prior["action"]
                normal_merge = candidate["action"] not in {"lost", "unknown", "transition"}
                barrier = _between_has_barrier(
                    barriers, identity, _event_float(prior, "end_time"), _event_float(candidate, "start_time"),
                )
                if same_action and normal_merge and -1e-6 <= gap <= config.short_gap_merge_seconds + 1e-9 and not barrier:
                    prior["end_time"] = max(_event_float(prior, "end_time"), _event_float(candidate, "end_time"))
                    prior["duration_seconds"] = round(prior["end_time"] - _event_float(prior, "start_time"), 9)
                    prior["source_event_ids"] = ";".join(filter(None, (str(prior.get("source_event_ids", "")), str(candidate.get("source_event_ids", "")))))
                    prior["source_segment_ids"] = ";".join(filter(None, (str(prior.get("source_segment_ids", "")), str(candidate.get("source_segment_ids", "")))))
                    prior["stabilization_reason"] = "same_person_epoch_side_action_fragments_merged"
                    merge_count += 1
                    continue
            merged_lane.append(candidate)
        stable.extend(merged_lane)
    stable.sort(key=lambda item: (_event_identity(item), _event_float(item, "start_time"), _event_float(item, "end_time")))
    return {
        "pose_evidence": pose_evidence,
        "stable_events": stable,
        "suppressed_events": suppressed,
        "metrics": {
            "input_pose_event_count": len(pose_evidence),
            "stable_event_count": len(stable),
            "suppressed_count": len(suppressed),
            "merge_count": merge_count,
            "hard_boundary_count": hard_boundary_count,
            "short_directional_action_preserved_count": short_preserved_count,
        },
    }


def _stabilize_neutral_segments(
    pose_phases: list[ActionSegment], config: ActionAnalysisConfig,
) -> tuple[list[ActionSegment], list[ActionSegment]]:
    stable: list[ActionSegment] = []
    suppressed: list[ActionSegment] = []
    for phase in pose_phases:
        keep = (
            phase.segment_type == "lost"
            or phase.duration_seconds >= config.stable_event_minimum_seconds - 1e-9
        )
        if phase.segment_type in {"movement", "idle"} and phase.status == "uncertain":
            keep = False
        if keep:
            stable.append(phase)
        else:
            phase.display_eligible = False
            suppressed.append(phase)
    return stable, suppressed


def choose_representative_indices(
    segments: list[ActionSegment], features: list[dict[str, Any]],
) -> dict[str, int]:
    result: dict[str, int] = {}
    for segment in segments:
        candidates = [
            item for item in features
            if segment.start_time <= float(item["timestamp"]) < segment.end_time
            and not item["lost"]
        ]
        if not candidates:
            continue
        midpoint = (segment.start_time + segment.end_time) / 2.0
        best = max(
            candidates,
            key=lambda item: (
                float(item["valid_keypoint_ratio"]) - 0.5 * float(item["interpolation_ratio"]),
                -abs(float(item["timestamp"]) - midpoint),
            ),
        )
        result[segment.segment_id] = int(best["frame_index"])
        segment.representative_time = float(best["timestamp"])
    return result


def analyze_actions(
    records: list[dict[str, Any]], config: ActionAnalysisConfig, *,
    video_fingerprint: str, locked_track_id: Any, analysis_end_time: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    feature_started = time.perf_counter()
    features = extract_motion_features(records)
    feature_ms = (time.perf_counter() - feature_started) * 1000.0
    segmentation_started = time.perf_counter()
    pose_phases, merges = segment_actions(
        features, config, video_fingerprint=video_fingerprint,
        locked_track_id=locked_track_id, analysis_end_time=analysis_end_time,
    )
    segments, suppressed_phases = _stabilize_neutral_segments(pose_phases, config)
    segmentation_ms = (time.perf_counter() - segmentation_started) * 1000.0
    representative_started = time.perf_counter()
    representative_indices = choose_representative_indices(segments, features)
    representative_ms = (time.perf_counter() - representative_started) * 1000.0
    return {
        "features": features, "pose_phases": pose_phases, "segments": segments,
        "stable_segments": segments,
        "suppressed_pose_phases": suppressed_phases,
        "merge_suggestions": merges,
        "representative_indices": representative_indices,
        "timings": {
            "feature_extraction_ms": round(feature_ms, 6),
            "segmentation_ms": round(segmentation_ms, 6),
            "representative_frame_ms": round(representative_ms, 6),
            "action_analysis_total_ms": round((time.perf_counter() - started) * 1000.0, 6),
        },
    }
