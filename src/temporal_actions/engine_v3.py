"""Explainable Temporal Action Engine V3 shadow implementation.

The engine derives traceable numerical features from real Body Pose and
qualified real Hand observations.  It never mutates or replaces the accepted
Phase B ``action_events`` timeline.  Object-dependent semantics remain
unavailable unless real Object and Interaction lineage is supplied.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from src.contracts import LayerState

from .contracts import (
    OBJECT_EVIDENCE_REQUIRED_ACTIONS,
    TemporalActionCandidate,
    TemporalFeatureFrame,
)
from .provider import TemporalActionOutput


_SIDES = ("left", "right", "bilateral")
_BODY_INDICES = {
    "left": {"shoulder": 5, "elbow": 7, "wrist": 9},
    "right": {"shoulder": 6, "elbow": 8, "wrist": 10},
}
_HARD_TRACK_STATES = {
    "lost",
    "off_frame",
    "awaiting_manual_relock",
}
_UNCERTAIN_ACTIONS = {"transition", "unknown", "lost"}
_POSE_ONLY_FALLBACK = {
    "carry": "move",
    "place": "lower",
    "hold": "unknown",
    "release": "unknown",
}


def _canonical_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _finite_point(
    keypoints: list[Any],
    statuses: list[Any],
    index: int,
) -> tuple[float, float] | None:
    if index >= len(keypoints):
        return None
    point = keypoints[index]
    if not isinstance(point, list) or len(point) < 2:
        return None
    if index < len(statuses) and statuses[index] in {
        "missing",
        "rejected",
    }:
        return None
    x_value, y_value = point[0], point[1]
    if not isinstance(x_value, (int, float)) or not isinstance(
        y_value,
        (int, float),
    ):
        return None
    x = float(x_value)
    y = float(y_value)
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y


def _distance(
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _mean_point(
    points: Iterable[tuple[float, float] | None],
) -> tuple[float, float] | None:
    available = [point for point in points if point is not None]
    if not available:
        return None
    return (
        sum(point[0] for point in available) / len(available),
        sum(point[1] for point in available) / len(available),
    )


def _direction(dx: float, dy: float) -> str:
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return "stationary"
    if abs(dx) >= abs(dy):
        return "rightward" if dx > 0.0 else "leftward"
    return "downward" if dy > 0.0 else "upward"


@dataclass(frozen=True)
class _Profile:
    rule_version: str
    profile_id: str
    profile_sha256: str
    context_seconds: float
    minimum_stable_seconds: float


class TemporalActionEngineV3:
    """Build lane-isolated shadow features and conservative candidates."""

    model_version = "temporal_action_engine_v3_interpretable_shadow"

    def __init__(
        self,
        config_path: str | Path = "configs/temporal_action_v3.json",
    ) -> None:
        self.config_path = Path(config_path)
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        payload = config["parameter_payload"]
        expected_hash = config["parameter_profile"][
            "parameter_profile_sha256"
        ]
        actual_hash = _canonical_sha256(payload)
        if actual_hash != expected_hash:
            raise ValueError(
                "Temporal V3 parameter profile SHA256 mismatch"
            )
        if config["engine"].get("shadow_mode") is not True:
            raise ValueError("Temporal V3 must remain shadow_mode=true")
        if config["engine"].get("publishes_primary_action_events") is not False:
            raise ValueError(
                "Temporal V3 cannot publish primary action_events"
            )
        temporal = payload["temporal"]
        self.profile = _Profile(
            rule_version=config["engine"]["rule_version"],
            profile_id=config["parameter_profile"][
                "parameter_profile_id"
            ],
            profile_sha256=expected_hash,
            context_seconds=float(temporal["temporal_context_seconds"]),
            minimum_stable_seconds=1.2,
        )
        self.context_seconds = self.profile.context_seconds
        self.rule_version = self.profile.rule_version
        self.parameter_profile_id = self.profile.profile_id
        self.parameter_profile_sha256 = self.profile.profile_sha256

    @staticmethod
    def _segment_index(
        segments: list[dict[str, Any]],
    ) -> tuple[dict[int, list[str]], set[str]]:
        by_frame: dict[int, list[str]] = defaultdict(list)
        known: set[str] = set()
        for segment in segments:
            segment_id = str(segment.get("segment_id", "")).strip()
            if not segment_id:
                continue
            known.add(segment_id)
            for frame_index in segment.get("source_frame_indices", []):
                if isinstance(frame_index, int):
                    by_frame[frame_index].append(segment_id)
        return by_frame, known

    @staticmethod
    def _hand_index(
        hand_frames: list[dict[str, Any]],
    ) -> tuple[
        dict[tuple[int, str, int, str], list[dict[str, Any]]],
        set[str],
    ]:
        by_binding: dict[
            tuple[int, str, int, str],
            list[dict[str, Any]],
        ] = defaultdict(list)
        known: set[str] = set()
        for hand in hand_frames:
            hand_id = str(hand.get("hand_pose_id", "")).strip()
            if not hand_id:
                continue
            known.add(hand_id)
            key = (
                int(hand.get("frame_index", -1)),
                str(hand.get("person_ref", "")),
                int(hand.get("lock_epoch", -1)),
                str(hand.get("anatomical_side", "")),
            )
            by_binding[key].append(hand)
        return by_binding, known

    @staticmethod
    def _hard_boundary(
        frame: dict[str, Any],
    ) -> tuple[bool, list[str], bool]:
        reasons: list[str] = []
        track_state = str(frame.get("track_state", ""))
        raw_action = str(frame.get("action", "unknown"))
        lost = (
            track_state in {"lost", "off_frame"}
            or raw_action == "lost"
            or frame.get("observation_state") == "lost"
        )
        if lost:
            reasons.append(f"track_state_{track_state or 'lost'}")
        if frame.get("awaiting_manual_relock"):
            reasons.append("awaiting_manual_relock")
        if frame.get("switch_exposed"):
            reasons.append("person_changed")
        if frame.get("hard_boundary"):
            reasons.append(
                str(
                    frame.get("temporal_reason")
                    or "upstream_hard_boundary"
                )
            )
        if track_state in _HARD_TRACK_STATES:
            reasons.append(track_state)
        return bool(reasons), _ordered_unique(reasons), lost

    @staticmethod
    def _body_geometry(
        frame: dict[str, Any],
        side: str,
    ) -> tuple[
        tuple[float, float] | None,
        float | None,
        float,
        int,
    ]:
        keypoints = frame.get("keypoints", [])
        statuses = frame.get("keypoint_statuses", [])
        sides = ("left", "right") if side == "bilateral" else (side,)
        wrists: list[tuple[float, float] | None] = []
        extensions: list[float] = []
        available_joint_count = 0
        required_joint_count = len(sides) * 3
        bbox = frame.get("bbox")
        scale = 1.0
        if (
            isinstance(bbox, list)
            and len(bbox) == 4
            and all(isinstance(item, (int, float)) for item in bbox)
        ):
            scale = max(
                1.0,
                math.hypot(
                    float(bbox[2]) - float(bbox[0]),
                    float(bbox[3]) - float(bbox[1]),
                ),
            )
        for anatomical_side in sides:
            indices = _BODY_INDICES[anatomical_side]
            shoulder = _finite_point(
                keypoints,
                statuses,
                indices["shoulder"],
            )
            elbow = _finite_point(
                keypoints,
                statuses,
                indices["elbow"],
            )
            wrist = _finite_point(
                keypoints,
                statuses,
                indices["wrist"],
            )
            available_joint_count += sum(
                item is not None for item in (shoulder, elbow, wrist)
            )
            wrists.append(wrist)
            if wrist is not None and shoulder is not None:
                extensions.append(_distance(wrist, shoulder) / scale)
        wrist_center = _mean_point(wrists)
        extension = (
            sum(extensions) / len(extensions) if extensions else None
        )
        reliability = (
            available_joint_count / required_joint_count
            if required_joint_count
            else 0.0
        )
        return wrist_center, extension, reliability, available_joint_count

    @staticmethod
    def _qualified_hand(
        hand: dict[str, Any],
        *,
        person_ref: str,
        lock_epoch: int,
        side: str,
    ) -> bool:
        checks = hand.get("association_checks")
        warnings = checks.get("warnings", []) if isinstance(checks, dict) else []
        return bool(
            hand.get("person_ref") == person_ref
            and hand.get("lock_epoch") == lock_epoch
            and hand.get("anatomical_side") == side
            and hand.get("observation_state") == "detected"
            and hand.get("quality_state") == "qualified"
            and hand.get("action_feature_eligible") is True
            and hand.get("landmark_count") == 21
            and isinstance(hand.get("landmarks"), list)
            and len(hand["landmarks"]) == 21
            and not warnings
            and not (
                isinstance(checks, dict)
                and checks.get("duplicate_across_sides")
            )
        )

    @staticmethod
    def _hand_points(
        hand: dict[str, Any],
    ) -> list[tuple[float, float]] | None:
        result: list[tuple[float, float]] = []
        indices: set[int] = set()
        for landmark in hand.get("landmarks", []):
            if not isinstance(landmark, dict):
                return None
            index = landmark.get("index")
            x_value = landmark.get("x")
            y_value = landmark.get("y")
            if (
                not isinstance(index, int)
                or index in indices
                or not isinstance(x_value, (int, float))
                or not isinstance(y_value, (int, float))
            ):
                return None
            x = float(x_value)
            y = float(y_value)
            if not math.isfinite(x) or not math.isfinite(y):
                return None
            indices.add(index)
            result.append((x, y))
        if indices != set(range(21)):
            return None
        return result

    @staticmethod
    def _unqualified_hand_state(
        hands: list[dict[str, Any]],
        *,
        lost: bool,
    ) -> str:
        if lost:
            return "lost"
        states = {str(hand.get("quality_state", "")) for hand in hands}
        if "association_uncertain" in states:
            return "association_uncertain"
        if "insufficient_geometry" in states:
            return "insufficient_geometry"
        return "not_observed"

    def build_feature_frames(
        self,
        analysis: dict[str, Any],
    ) -> tuple[
        list[TemporalFeatureFrame],
        dict[str, Any],
        set[str],
        set[str],
    ]:
        source_video = analysis["source_video"]
        source_sha256 = str(source_video["sha256"])
        recording_group_id = str(source_video["recording_group_id"])
        segment_by_frame, known_segment_ids = self._segment_index(
            analysis.get("pose_segments", [])
        )
        hand_by_binding, known_hand_ids = self._hand_index(
            analysis.get("hand_pose_frames", [])
        )
        buffers: dict[
            tuple[str, int, str],
            deque[tuple[float, tuple[float, float], float]],
        ] = defaultdict(deque)
        previous_hand_center: dict[
            tuple[str, int, str],
            tuple[float, tuple[float, float]],
        ] = {}
        feature_frames: list[TemporalFeatureFrame] = []
        hand_features_used = 0
        hand_observations_rejected = 0
        hard_boundary_features = 0
        for frame in analysis.get("pose_frames", []):
            frame_index = int(frame["source_frame_index"])
            timestamp = float(frame["timestamp"])
            person_ref = str(frame["person_ref"])
            lock_epoch = int(frame["lock_epoch"])
            raw_side = str(frame.get("anatomical_side", "bilateral"))
            raw_action = str(frame.get("action", "unknown"))
            hard_boundary, boundary_reasons, lost = self._hard_boundary(frame)
            for side in _SIDES:
                lane_key = (person_ref, lock_epoch, side)
                if hard_boundary:
                    buffers[lane_key].clear()
                wrist, extension, reliability, joint_count = (
                    self._body_geometry(frame, side)
                )
                body_state = "available"
                body_features_used = False
                body_features: dict[str, Any] | None = None
                previous = (
                    buffers[lane_key][-1] if buffers[lane_key] else None
                )
                if lost:
                    body_state = "lost"
                elif wrist is None:
                    body_state = "missing"
                elif hard_boundary or reliability < (2.0 / 3.0):
                    body_state = "uncertain"
                else:
                    dt = timestamp - previous[0] if previous else 0.0
                    dx = wrist[0] - previous[1][0] if previous else 0.0
                    dy = wrist[1] - previous[1][1] if previous else 0.0
                    displacement = math.hypot(dx, dy)
                    speed = displacement / dt if dt > 0.0 else 0.0
                    buffer = buffers[lane_key]
                    if dt > 0.0:
                        buffer.append((timestamp, wrist, speed))
                    elif not buffer:
                        buffer.append((timestamp, wrist, 0.0))
                    while (
                        buffer
                        and timestamp - buffer[0][0]
                        > self.profile.context_seconds
                    ):
                        buffer.popleft()
                    speeds = [item[2] for item in buffer]
                    context_span = (
                        buffer[-1][0] - buffer[0][0]
                        if len(buffer) > 1
                        else 0.0
                    )
                    body_features = {
                        "wrist_velocity": {
                            "dx_pixels_per_second": (
                                dx / dt if dt > 0.0 else 0.0
                            ),
                            "dy_pixels_per_second": (
                                dy / dt if dt > 0.0 else 0.0
                            ),
                        },
                        "wrist_speed": speed,
                        "wrist_displacement": displacement,
                        "wrist_extension": extension,
                        "motion_direction": _direction(dx, dy),
                        "motion_amplitude": displacement,
                        "joint_reliability": reliability,
                        "available_joint_count": joint_count,
                        "context_sample_count": len(buffer),
                        "context_span_seconds": context_span,
                        "context_mean_speed": (
                            sum(speeds) / len(speeds) if speeds else 0.0
                        ),
                        "context_peak_speed": max(speeds) if speeds else 0.0,
                    }
                    body_features_used = True

                relevant_hand_sides = (
                    ("left", "right") if side == "bilateral" else (side,)
                )
                relevant_hands: list[dict[str, Any]] = []
                for hand_side in relevant_hand_sides:
                    relevant_hands.extend(
                        hand_by_binding.get(
                            (
                                frame_index,
                                person_ref,
                                lock_epoch,
                                hand_side,
                            ),
                            [],
                        )
                    )
                qualified: list[
                    tuple[dict[str, Any], list[tuple[float, float]]]
                ] = []
                for hand in relevant_hands:
                    hand_side = str(hand.get("anatomical_side", ""))
                    points = self._hand_points(hand)
                    if (
                        self._qualified_hand(
                            hand,
                            person_ref=person_ref,
                            lock_epoch=lock_epoch,
                            side=hand_side,
                        )
                        and points is not None
                    ):
                        qualified.append((hand, points))
                    elif hand.get("observation_state") != "missing":
                        hand_observations_rejected += 1

                hand_state = self._unqualified_hand_state(
                    relevant_hands,
                    lost=lost,
                )
                hand_features: dict[str, Any] | None = None
                source_hand_ids: list[str] = []
                hand_features_used_now = False
                if qualified and not hard_boundary:
                    per_side: list[dict[str, Any]] = []
                    for hand, points in qualified:
                        hand_side = str(hand["anatomical_side"])
                        center = _mean_point(points)
                        assert center is not None
                        history_key = (
                            person_ref,
                            lock_epoch,
                            hand_side,
                        )
                        prior = previous_hand_center.get(history_key)
                        speed = 0.0
                        if prior is not None and timestamp > prior[0]:
                            speed = _distance(center, prior[1]) / (
                                timestamp - prior[0]
                            )
                        previous_hand_center[history_key] = (
                            timestamp,
                            center,
                        )
                        crop = hand.get("crop_bbox")
                        crop_scale = 1.0
                        if (
                            isinstance(crop, list)
                            and len(crop) == 4
                            and all(
                                isinstance(value, (int, float))
                                for value in crop
                            )
                        ):
                            crop_scale = max(
                                1.0,
                                math.hypot(
                                    float(crop[2]) - float(crop[0]),
                                    float(crop[3]) - float(crop[1]),
                                ),
                            )
                        per_side.append(
                            {
                                "anatomical_side": hand_side,
                                "centroid_x": center[0],
                                "centroid_y": center[1],
                                "speed_pixels_per_second": speed,
                                "hand_span_roi_ratio": (
                                    _distance(points[5], points[17])
                                    / crop_scale
                                ),
                            }
                        )
                        source_hand_ids.append(str(hand["hand_pose_id"]))
                    hand_state = "qualified"
                    hand_features_used_now = True
                    hand_features = {
                        "hand_motion_feature": {
                            "qualified_side_count": len(per_side),
                            "per_side": per_side,
                        },
                        "hand_shape_feature": {
                            "landmark_count_per_hand": 21,
                            "derived_from_real_landmarks": True,
                        },
                    }
                    hand_features_used += 1

                lane_matches_raw = (
                    side == "bilateral"
                    or raw_side == side
                    or raw_side == "bilateral"
                )
                coarse_action = raw_action if lane_matches_raw else "transition"
                if coarse_action in OBJECT_EVIDENCE_REQUIRED_ACTIONS:
                    coarse_action = _POSE_ONLY_FALLBACK[coarse_action]
                if lost:
                    coarse_action = "lost"
                    evidence_state = "lost"
                elif hard_boundary:
                    coarse_action = "unknown"
                    evidence_state = "uncertain"
                elif not lane_matches_raw or coarse_action == "transition":
                    coarse_action = "transition"
                    evidence_state = "transition"
                elif coarse_action == "unknown":
                    evidence_state = "unknown"
                elif body_state != "available":
                    coarse_action = "unknown"
                    evidence_state = "uncertain"
                else:
                    evidence_state = "normal"
                status = (
                    "proposed"
                    if (
                        evidence_state == "normal"
                        and coarse_action not in _UNCERTAIN_ACTIONS
                        and body_features_used
                    )
                    else "uncertain"
                )
                if hard_boundary:
                    hand_features = None
                    source_hand_ids = []
                    hand_features_used_now = False
                    hard_boundary_features += 1
                feature = TemporalFeatureFrame(
                    temporal_feature_id=(
                        f"tf-{source_sha256[:12]}-{frame_index:08d}-{side}"
                    ),
                    coarse_action=coarse_action,
                    evidence_state=evidence_state,
                    person_ref=person_ref,
                    lock_epoch=lock_epoch,
                    anatomical_side=side,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    source_video_sha256=source_sha256,
                    recording_group_id=recording_group_id,
                    rule_version=self.profile.rule_version,
                    parameter_profile_id=self.profile.profile_id,
                    parameter_profile_sha256=self.profile.profile_sha256,
                    body_feature_state=body_state,
                    hand_feature_state=hand_state,
                    object_feature_state="unavailable",
                    body_features_used=body_features_used,
                    hand_features_used=hand_features_used_now,
                    object_features_used=False,
                    body_motion_features=body_features,
                    hand_motion_features=hand_features,
                    object_motion_features=None,
                    source_frame_indices=[frame_index],
                    source_segment_ids=_ordered_unique(
                        segment_by_frame.get(frame_index, [])
                    ),
                    source_hand_pose_ids=_ordered_unique(source_hand_ids),
                    source_object_track_ids=[],
                    source_interaction_ids=[],
                    hard_boundary=hard_boundary,
                    boundary_reasons=boundary_reasons,
                    status=status,
                )
                feature.validate()
                feature_frames.append(feature)
                if hard_boundary:
                    buffers[lane_key].clear()
        diagnostics = {
            "lane_count": 3,
            "feature_frame_count": len(feature_frames),
            "body_pose_frame_count": len(analysis.get("pose_frames", [])),
            "hand_feature_use_count": hand_features_used,
            "hand_observation_rejection_count": hand_observations_rejected,
            "object_feature_use_count": 0,
            "hard_boundary_feature_count": hard_boundary_features,
        }
        return (
            feature_frames,
            diagnostics,
            known_segment_ids,
            known_hand_ids,
        )

    def _project_candidates(
        self,
        action_events: list[dict[str, Any]],
        feature_frames: list[TemporalFeatureFrame],
    ) -> tuple[
        list[TemporalActionCandidate],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        by_binding_frame = {
            (
                feature.person_ref,
                feature.lock_epoch,
                feature.anatomical_side,
                feature.frame_index,
            ): feature
            for feature in feature_frames
        }
        candidates: list[TemporalActionCandidate] = []
        change_points: list[dict[str, Any]] = []
        suppressed: list[dict[str, Any]] = []
        for event in action_events:
            person_ref = str(event.get("person_ref", ""))
            lock_epoch = int(event.get("lock_epoch", -1))
            side = str(
                event.get("anatomical_side")
                or event.get("side")
                or "bilateral"
            )
            if side not in _SIDES:
                side = "bilateral"
            source_indices = [
                int(value)
                for value in event.get("source_frame_indices", [])
                if isinstance(value, int)
            ]
            sources = [
                by_binding_frame[
                    (person_ref, lock_epoch, side, frame_index)
                ]
                for frame_index in source_indices
                if (
                    person_ref,
                    lock_epoch,
                    side,
                    frame_index,
                )
                in by_binding_frame
            ]
            sources.sort(key=lambda item: (item.timestamp, item.frame_index))
            action = str(event.get("action", "unknown"))
            if action in OBJECT_EVIDENCE_REQUIRED_ACTIONS:
                action = _POSE_ONLY_FALLBACK[action]
            hard_boundary = bool(event.get("hard_boundary"))
            if len(sources) < 2:
                suppressed.append(
                    {
                        "source_action_event_id": event.get(
                            "action_event_id"
                        ),
                        "reason": "insufficient_temporal_feature_span",
                        "source_feature_count": len(sources),
                    }
                )
                continue
            duration = sources[-1].timestamp - sources[0].timestamp
            is_normal = (
                event.get("event_kind") == "stable_action"
                and event.get("display_eligible") is True
                and event.get("evidence_state") == "normal"
                and action not in _UNCERTAIN_ACTIONS
                and not hard_boundary
            )
            if is_normal:
                if duration < self.profile.minimum_stable_seconds:
                    suppressed.append(
                        {
                            "source_action_event_id": event.get(
                                "action_event_id"
                            ),
                            "reason": "below_v3_minimum_stable_seconds",
                            "duration_seconds": duration,
                        }
                    )
                    continue
                if not all(
                    source.body_feature_state == "available"
                    and source.body_features_used
                    and not source.hard_boundary
                    for source in sources
                ):
                    suppressed.append(
                        {
                            "source_action_event_id": event.get(
                                "action_event_id"
                            ),
                            "reason": "body_quality_gate_not_continuous",
                            "duration_seconds": duration,
                        }
                    )
                    continue
                evidence_state = "normal"
                status = "proposed"
                body_state = "available"
                boundary_reasons: list[str] = []
            elif hard_boundary and action in {"lost", "unknown"}:
                evidence_state = "lost" if action == "lost" else "uncertain"
                status = "uncertain"
                body_state = "lost" if action == "lost" else "uncertain"
                boundary_reasons = _ordered_unique(
                    str(item)
                    for item in event.get("boundary_reasons", [])
                ) or ["upstream_hard_boundary"]
            else:
                suppressed.append(
                    {
                        "source_action_event_id": event.get(
                            "action_event_id"
                        ),
                        "reason": "not_stable_normal_or_explicit_boundary",
                        "duration_seconds": duration,
                    }
                )
                continue
            source_feature_ids = [
                source.temporal_feature_id for source in sources
            ]
            source_segment_ids = _ordered_unique(
                segment_id
                for source in sources
                for segment_id in source.source_segment_ids
            )
            if not source_segment_ids:
                suppressed.append(
                    {
                        "source_action_event_id": event.get(
                            "action_event_id"
                        ),
                        "reason": "missing_pose_segment_lineage",
                    }
                )
                continue
            source_hand_ids = _ordered_unique(
                hand_id
                for source in sources
                if source.hand_features_used
                for hand_id in source.source_hand_pose_ids
            )
            uses_hand = bool(source_hand_ids) and is_normal
            start_change = {
                "change_type": (
                    "start_confirmation" if is_normal else "hard_boundary"
                ),
                "timestamp": sources[0].timestamp,
                "reason": (
                    "v3_lane_start_confirmed"
                    if is_normal
                    else "upstream_hard_boundary_entered"
                ),
                "source_temporal_feature_ids": [source_feature_ids[0]],
            }
            end_change = {
                "change_type": (
                    "stop_confirmation" if is_normal else "hard_boundary"
                ),
                "timestamp": sources[-1].timestamp,
                "reason": (
                    "v3_lane_stop_hysteresis_confirmed"
                    if is_normal
                    else "upstream_hard_boundary_exited"
                ),
                "source_temporal_feature_ids": [source_feature_ids[-1]],
            }
            change_evidence = [start_change, end_change]
            candidate = TemporalActionCandidate(
                temporal_action_candidate_id=(
                    "tac-"
                    + str(event.get("action_event_id", len(candidates) + 1))
                ),
                action=action,
                evidence_state=evidence_state,
                person_ref=person_ref,
                lock_epoch=lock_epoch,
                anatomical_side=side,
                start_frame_index=sources[0].frame_index,
                end_frame_index=sources[-1].frame_index,
                start_time=sources[0].timestamp,
                end_time=sources[-1].timestamp,
                duration_seconds=duration,
                source_video_sha256=sources[0].source_video_sha256,
                recording_group_id=sources[0].recording_group_id,
                engine_kind="interpretable_rules",
                shadow_mode=True,
                rule_version=self.profile.rule_version,
                parameter_profile_id=self.profile.profile_id,
                parameter_profile_sha256=self.profile.profile_sha256,
                temporal_context_seconds=self.profile.context_seconds,
                body_feature_state=body_state,
                hand_feature_state=(
                    "qualified" if uses_hand else "not_observed"
                ),
                object_feature_state="unavailable",
                body_features_used=is_normal,
                hand_features_used=uses_hand,
                object_features_used=False,
                source_temporal_feature_ids=source_feature_ids,
                source_segment_ids=source_segment_ids,
                source_frame_indices=[
                    source.frame_index for source in sources
                ],
                source_hand_pose_ids=source_hand_ids,
                source_object_track_ids=[],
                source_interaction_ids=[],
                change_point_evidence=change_evidence,
                hard_boundary=hard_boundary,
                boundary_reasons=boundary_reasons,
                status=status,
            )
            candidate.validate()
            candidates.append(candidate)
            change_points.extend(
                [
                    {
                        **item,
                        "source_action_event_id": event.get(
                            "action_event_id"
                        ),
                        "status": "proposed" if is_normal else "uncertain",
                        "training_eligible": False,
                    }
                    for item in change_evidence
                ]
            )
        return candidates, change_points, suppressed

    def analyze_analysis(
        self,
        analysis: dict[str, Any],
    ) -> TemporalActionOutput:
        (
            features,
            feature_diagnostics,
            known_segment_ids,
            known_hand_ids,
        ) = self.build_feature_frames(analysis)
        candidates, change_points, suppressed = self._project_candidates(
            analysis.get("action_events", []),
            features,
        )
        normal_candidates = [
            item
            for item in candidates
            if item.evidence_state == "normal"
        ]
        output = TemporalActionOutput(
            state=LayerState(
                layer="temporal_action_engine_v3_shadow",
                status="available",
                reason=(
                    "interpretable_rules_shadow_only_primary_action_events_"
                    "unchanged"
                ),
                model_version=self.model_version,
                evidence_count=len(features) + len(candidates),
            ),
            action_candidates=candidates,
            feature_frames=features,
            change_point_candidates=change_points,
            diagnostics={
                **feature_diagnostics,
                "provider_state": "available",
                "shadow_mode": True,
                "publishes_primary_action_events": False,
                "fallback_primary_action_events": True,
                "rule_version": self.profile.rule_version,
                "parameter_profile_id": self.profile.profile_id,
                "parameter_profile_sha256": self.profile.profile_sha256,
                "temporal_context_seconds": self.profile.context_seconds,
                "stable_minimum_seconds": (
                    self.profile.minimum_stable_seconds
                ),
                "candidate_count": len(candidates),
                "stable_normal_candidate_count": len(normal_candidates),
                "sub_1s_normal_candidate_count": sum(
                    item.duration_seconds < 1.0
                    for item in normal_candidates
                ),
                "suppressed_candidate_count": len(suppressed),
                "suppressed_candidates": suppressed,
                "accuracy_status": "not_evaluable",
                "semantic_accuracy": None,
                "macro_f1": None,
                "object_feature_state": "unavailable",
                "object_feature_use_count": 0,
                "training_eligible": False,
            },
            known_pose_segment_ids=frozenset(known_segment_ids),
            known_hand_pose_ids=frozenset(known_hand_ids),
            known_object_track_ids=frozenset(),
            known_interaction_ids=frozenset(),
        )
        output.validate()
        return output

    def analyze_analysis_safe(
        self,
        analysis: dict[str, Any],
    ) -> TemporalActionOutput:
        """Fail closed while leaving accepted Phase B events available."""

        try:
            return self.analyze_analysis(analysis)
        except Exception as exc:
            output = TemporalActionOutput(
                state=LayerState(
                    layer="temporal_action_engine_v3_shadow",
                    status="unavailable",
                    reason="shadow_provider_error_primary_action_events_retained",
                    model_version=self.model_version,
                    evidence_count=0,
                ),
                diagnostics={
                    "provider_state": "error",
                    "error_type": type(exc).__name__,
                    "shadow_mode": True,
                    "publishes_primary_action_events": False,
                    "fallback_primary_action_events": True,
                    "accuracy_status": "not_evaluable",
                    "training_eligible": False,
                },
            )
            output.validate()
            return output
