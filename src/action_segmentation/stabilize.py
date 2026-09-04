"""Phase B stable-action layer with explicit evidence and boundary gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any

from .coarse import (
    CoarseFrame,
    FrameActionStabilityConfig,
    build_pose_segments,
    stabilize_coarse_frames,
)
from src.legacy_pose.action_analysis import ActionAnalysisConfig


NORMAL_ACTIONS = {
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
}
SHORT_DIRECTIONAL_ACTIONS = {"reach", "retract", "lift", "lower", "push", "pull"}
LOST_STATES = {"lost", "temporarily_lost", "awaiting_manual_relock", "off_frame"}
SEVERE_OCCLUSIONS = {"severe", "off_frame"}


@dataclass(frozen=True)
class PhaseBActionStabilityConfig:
    analysis_fps: float = 8.0
    stable_event_minimum_seconds: float = 1.2
    short_directional_event_minimum_seconds: float = 1.0
    short_gap_merge_seconds: float = 0.4
    start_confirmation_seconds: float = 0.5
    stop_confirmation_seconds: float = 0.5
    temporal_context_seconds: float = 2.5
    bounded_uncertain_gap_seconds: float = 0.375
    minimum_detected_evidence_ratio: float = 0.65
    short_event_minimum_detected_ratio: float = 0.75
    maximum_prediction_ratio: float = 0.35
    maximum_missing_ratio: float = 0.45

    def __post_init__(self) -> None:
        for name in (
            "analysis_fps",
            "stable_event_minimum_seconds",
            "short_directional_event_minimum_seconds",
            "short_gap_merge_seconds",
            "start_confirmation_seconds",
            "stop_confirmation_seconds",
            "temporal_context_seconds",
            "bounded_uncertain_gap_seconds",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.stable_event_minimum_seconds < 1.0:
            raise ValueError("stable events must default to at least one second")
        if self.short_directional_event_minimum_seconds < 1.0:
            raise ValueError("short directional events cannot be below one second")
        if (
            self.short_directional_event_minimum_seconds
            > self.stable_event_minimum_seconds
        ):
            raise ValueError(
                "short directional minimum cannot exceed stable event minimum"
            )
        for name in (
            "minimum_detected_evidence_ratio",
            "short_event_minimum_detected_ratio",
            "maximum_prediction_ratio",
            "maximum_missing_ratio",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")


def _effective_config(
    config: PhaseBActionStabilityConfig | ActionAnalysisConfig | None,
) -> PhaseBActionStabilityConfig:
    if config is None:
        return PhaseBActionStabilityConfig()
    if isinstance(config, PhaseBActionStabilityConfig):
        return config
    return PhaseBActionStabilityConfig(
        analysis_fps=float(getattr(config, "analysis_fps", 8.0)),
        stable_event_minimum_seconds=float(
            getattr(config, "stable_event_minimum_seconds", 1.2)
        ),
        short_directional_event_minimum_seconds=float(
            getattr(
                config,
                "short_directional_event_minimum_seconds",
                getattr(config, "short_event_minimum_seconds", 1.0),
            )
        ),
        short_gap_merge_seconds=float(
            getattr(config, "short_gap_merge_seconds", 0.4)
        ),
        start_confirmation_seconds=float(
            getattr(config, "start_confirmation_seconds", 0.5)
        ),
        stop_confirmation_seconds=float(
            getattr(config, "stop_confirmation_seconds", 0.5)
        ),
        temporal_context_seconds=float(
            getattr(config, "temporal_context_seconds", 2.5)
        ),
        bounded_uncertain_gap_seconds=float(
            getattr(config, "bounded_uncertain_gap_seconds", 0.375)
        ),
        minimum_detected_evidence_ratio=float(
            getattr(config, "minimum_detected_evidence_ratio", 0.65)
        ),
        short_event_minimum_detected_ratio=float(
            getattr(config, "short_event_minimum_detected_ratio", 0.75)
        ),
        maximum_prediction_ratio=float(
            getattr(config, "maximum_prediction_ratio", 0.35)
        ),
        maximum_missing_ratio=float(
            getattr(
                config,
                "maximum_missing_ratio",
                getattr(config, "maximum_uncertain_ratio", 0.45),
            )
        ),
    )


def _list_ids(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return [
        stripped
        for part in re.split(r"[;,]", str(value or ""))
        if (stripped := part.strip())
    ]


def _event_float(event: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(event.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _event_bool(event: dict[str, Any], key: str, default: bool = False) -> bool:
    value = event.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _tokens(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        result: set[str] = set()
        for item in value:
            result.update(_tokens(item))
        return result
    return {
        token.strip().lower()
        for token in re.split(r"[;,\s|]+", str(value))
        if token.strip()
    }


def _clip_key(event: dict[str, Any]) -> str:
    return str(
        event.get("source_video_sha256")
        or event.get("clip_id")
        or event.get("video_id")
        or ""
    )


def _boundary_key(event: dict[str, Any]) -> tuple[str, str, str]:
    return (
        _clip_key(event),
        str(event.get("person_ref", event.get("locked_track_id", ""))),
        str(event.get("lock_epoch", "")),
    )


def _lane_key(event: dict[str, Any]) -> tuple[str, str, str, str]:
    return (*_boundary_key(event), str(event.get("side", "unknown")).lower())


def _duration(event: dict[str, Any]) -> float:
    return max(
        0.0,
        _event_float(event, "end_time") - _event_float(event, "start_time"),
    )


def _hard_boundary(event: dict[str, Any]) -> tuple[str | None, list[str]]:
    states: set[str] = set()
    for key in ("track_state", "lock_state", "track_state_summary", "raw_track_states"):
        states.update(_tokens(event.get(key)))
    occlusions: set[str] = set()
    for key in ("occlusion", "occlusion_summary", "raw_occlusion_states"):
        occlusions.update(_tokens(event.get(key)))
    human = _tokens(event.get("human_hard_boundary"))
    reasons: list[str] = []
    if (
        _event_bool(event, "raw_lost")
        or _event_bool(event, "raw_off_frame")
        or _event_bool(event, "temporarily_lost")
        or bool((states | human) & LOST_STATES)
        or "off_frame" in occlusions
        or str(event.get("action", "")).lower() == "lost"
    ):
        reasons.extend(sorted((states | human) & LOST_STATES))
        reasons.extend(sorted(occlusions & {"off_frame"}))
        if _event_bool(event, "raw_lost"):
            reasons.append("raw_lost")
        return "lost", sorted(set(reasons or ["lost"]))

    observation = str(event.get("observation_state", "")).lower()
    bounded_gap = _event_bool(event, "bounded_uncertain_gap")
    explicit_hard_boundary = _event_bool(event, "hard_boundary")
    severe_missing = (
        explicit_hard_boundary
        or (observation == "missing" and not bounded_gap)
        or bool(occlusions & SEVERE_OCCLUSIONS)
        or (
            _event_float(event, "missing_ratio") >= 0.75
            and not _event_bool(event, "required_joints_reliable", True)
            and not bounded_gap
        )
    )
    if severe_missing:
        if explicit_hard_boundary:
            reasons.append(
                str(event.get("temporal_reason") or "explicit_hard_boundary")
            )
        if observation == "missing":
            reasons.append("missing_pose_evidence")
        reasons.extend(sorted(occlusions & SEVERE_OCCLUSIONS))
        return "uncertain", sorted(set(reasons or ["severe_missing_pose"]))
    return None, []


def _copy_event(
    segment: dict[str, Any],
    *,
    action: str,
    status: str,
    observation_state: str,
    reason: str,
    event_kind: str = "stable_action",
    display_eligible: bool = True,
    boundary_reasons: list[str] | None = None,
) -> dict[str, Any]:
    event = dict(segment)
    source_ids = _list_ids(
        segment.get("source_segment_ids") or segment.get("segment_id")
    )
    event["action"] = action
    event["action_name"] = action
    event["start_time"] = round(_event_float(segment, "start_time"), 9)
    event["end_time"] = round(_event_float(segment, "end_time"), 9)
    event["duration_seconds"] = round(_duration(segment), 9)
    event["status"] = status
    event["observation_state"] = observation_state
    event["confirmation_status"] = "unconfirmed"
    event["training_eligible"] = False
    event["training_approval"] = "pending"
    event["source_segment_ids"] = source_ids
    event["source_event_ids"] = _list_ids(
        segment.get("source_event_ids")
        or segment.get("event_id")
        or segment.get("segment_id")
    )
    event["absorbed_segment_ids"] = _list_ids(
        segment.get("absorbed_segment_ids")
    )
    event["bounded_gap_source_segment_ids"] = _list_ids(
        segment.get("bounded_gap_source_segment_ids")
    )
    event["event_kind"] = event_kind
    event["display_eligible"] = display_eligible
    event["stabilization_reason"] = reason
    event["stabilization_steps"] = [reason]
    event["boundary_reasons"] = list(boundary_reasons or [])
    return event


def _intersects_open_interval(
    event: dict[str, Any],
    start: float,
    end: float,
) -> bool:
    return (
        _event_float(event, "end_time") > start + 1e-9
        and _event_float(event, "start_time") < end - 1e-9
    )


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _segment_action(segment: dict[str, Any]) -> str:
    action = str(
        segment.get("action", segment.get("action_name", "unknown"))
    ).lower()
    return action if action in NORMAL_ACTIONS | {"transition", "unknown", "lost"} else "unknown"


def _segment_evidence_reliable(
    segment: dict[str, Any],
    config: PhaseBActionStabilityConfig,
) -> bool:
    detected = _event_float(segment, "detected_ratio")
    prediction = _event_float(segment, "predicted_ratio") + _event_float(
        segment,
        "interpolated_ratio",
    )
    missing = _event_float(segment, "missing_ratio")
    required_reliable = _event_bool(
        segment,
        "required_joints_reliable",
        detected >= config.minimum_detected_evidence_ratio,
    )
    return (
        detected >= config.minimum_detected_evidence_ratio
        and prediction <= config.maximum_prediction_ratio
        and missing <= config.maximum_missing_ratio
        and required_reliable
        and _hard_boundary(segment)[0] is None
    )


def _action_support_threshold(
    segment: dict[str, Any],
    config: PhaseBActionStabilityConfig,
) -> float:
    action = _segment_action(segment)
    if (
        action in SHORT_DIRECTIONAL_ACTIONS
        and _event_bool(segment, "direction_clear")
    ):
        return config.short_directional_event_minimum_seconds
    return config.stable_event_minimum_seconds


def _aggregate_support_group(
    support_segments: list[dict[str, Any]],
    gap_segments: list[dict[str, Any]],
    *,
    group_index: int,
) -> dict[str, Any]:
    first = support_segments[0]
    result = dict(first)
    start_time = min(_event_float(item, "start_time") for item in support_segments)
    end_time = max(_event_float(item, "end_time") for item in support_segments)
    support_seconds = sum(_duration(item) for item in support_segments)
    span_seconds = max(0.0, end_time - start_time)
    support_ids = _unique(
        [
            source_id
            for item in support_segments
            for source_id in _list_ids(
                item.get("source_segment_ids") or item.get("segment_id")
            )
        ]
    )
    gap_ids = _unique(
        [
            source_id
            for item in gap_segments
            for source_id in _list_ids(
                item.get("source_segment_ids") or item.get("segment_id")
            )
            if source_id not in support_ids
        ]
    )
    weights = [_duration(item) for item in support_segments]
    weight_total = max(sum(weights), 1e-9)
    ordered_support = sorted(
        support_segments,
        key=lambda item: _event_float(item, "start_time"),
    )
    gap_details: list[dict[str, Any]] = []
    for left, right in zip(ordered_support, ordered_support[1:]):
        gap_start = _event_float(left, "end_time")
        gap_end = _event_float(right, "start_time")
        if gap_end <= gap_start + 1e-9:
            continue
        contributing = [
            item
            for item in gap_segments
            if _intersects_open_interval(item, gap_start, gap_end)
        ]
        contributing_ids = _unique(
            [
                source_id
                for item in contributing
                for source_id in _list_ids(
                    item.get("source_segment_ids") or item.get("segment_id")
                )
            ]
        )
        if any(
            _event_bool(item, "bounded_uncertain_gap")
            for item in contributing
        ):
            reason = "bounded_uncertain_gap"
        elif any(
            _segment_action(item) in {"unknown", "transition"}
            for item in contributing
        ):
            reason = "short_transition_or_unknown_gap"
        else:
            reason = "short_same_lane_noise_gap"
        gap_details.append(
            {
                "start_time": round(gap_start, 9),
                "end_time": round(gap_end, 9),
                "duration_seconds": round(gap_end - gap_start, 9),
                "source_segment_ids": contributing_ids,
                "reason": reason,
            }
        )

    def weighted(field: str) -> float:
        return sum(
            _event_float(item, field) * weight
            for item, weight in zip(support_segments, weights)
        ) / weight_total

    result.update(
        {
            "support_group_id": f"support-group-{group_index:05d}",
            "start_time": round(start_time, 9),
            "end_time": round(end_time, 9),
            "duration_seconds": round(span_seconds, 9),
            "observed_support_seconds": round(support_seconds, 9),
            "observed_support_ratio": round(
                min(1.0, support_seconds / span_seconds)
                if span_seconds > 0
                else 0.0,
                9,
            ),
            "bounded_gap_seconds": round(
                max(0.0, span_seconds - support_seconds),
                9,
            ),
            "maximum_bounded_gap_seconds": round(
                max(
                    (
                        float(item["duration_seconds"])
                        for item in gap_details
                    ),
                    default=0.0,
                ),
                9,
            ),
            "bounded_uncertain_gaps": gap_details,
            "source_segment_ids": support_ids,
            "bounded_gap_source_segment_ids": gap_ids,
            "absorbed_segment_ids": gap_ids,
            "source_frame_indices": list(
                dict.fromkeys(
                    int(frame_index)
                    for item in support_segments
                    for frame_index in item.get("source_frame_indices", [])
                )
            ),
            "bounded_gap_frame_indices": list(
                dict.fromkeys(
                    int(frame_index)
                    for item in gap_segments
                    for frame_index in item.get("source_frame_indices", [])
                )
            ),
            "detected_ratio": weighted("detected_ratio"),
            "predicted_ratio": weighted("predicted_ratio"),
            "interpolated_ratio": weighted("interpolated_ratio"),
            "missing_ratio": weighted("missing_ratio"),
            "required_joints_reliable": all(
                _event_bool(item, "required_joints_reliable")
                for item in support_segments
            ),
            "direction_clear": any(
                _event_bool(item, "direction_clear")
                for item in support_segments
            ),
            "support_fragment_count": len(support_segments),
            "pre_gate_merge_count": max(0, len(support_segments) - 1),
            "pre_gate_aggregation_applied": len(support_segments) > 1,
        }
    )
    return result


def _aggregate_normal_support_before_gate(
    ordered: list[dict[str, Any]],
    config: PhaseBActionStabilityConfig,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Aggregate only same-lane, same-action support before duration gating."""

    candidate_indices = [
        index
        for index, segment in enumerate(ordered)
        if _segment_action(segment) in NORMAL_ACTIONS
        and _segment_evidence_reliable(segment, config)
    ]
    lanes: dict[tuple[str, str, str, str], list[int]] = {}
    for index in candidate_indices:
        lanes.setdefault(_lane_key(ordered[index]), []).append(index)

    used_support_indices: set[int] = set()
    groups: list[dict[str, Any]] = []
    pre_gate_merges = 0
    absorbed_gap_ids: set[str] = set()
    maximum_gap = min(
        config.short_gap_merge_seconds,
        config.bounded_uncertain_gap_seconds,
    )

    for lane_key, lane_indices in lanes.items():
        lane_indices.sort(
            key=lambda index: (
                _event_float(ordered[index], "start_time"),
                _event_float(ordered[index], "end_time"),
            )
        )
        for position, seed_index in enumerate(lane_indices):
            if seed_index in used_support_indices:
                continue
            seed = ordered[seed_index]
            action = _segment_action(seed)
            support_indices = [seed_index]
            gap_indices: list[int] = []
            used_support_indices.add(seed_index)
            search_position = position + 1
            while search_position < len(lane_indices):
                candidate_index = lane_indices[search_position]
                candidate = ordered[candidate_index]
                search_position += 1
                if candidate_index in used_support_indices:
                    continue
                if _segment_action(candidate) != action:
                    continue
                prior = ordered[support_indices[-1]]
                gap_start = _event_float(prior, "end_time")
                gap_end = _event_float(candidate, "start_time")
                gap = gap_end - gap_start
                if gap < -1e-6 or gap > maximum_gap + 1e-9:
                    break
                intervening = [
                    index
                    for index, item in enumerate(ordered)
                    if index not in support_indices
                    and index != candidate_index
                    and _intersects_open_interval(item, gap_start, gap_end)
                ]
                hard_block = any(
                    _hard_boundary(ordered[index])[0] is not None
                    for index in intervening
                )
                other_side_normal_block = any(
                    _lane_key(ordered[index])[:3] == lane_key[:3]
                    and _lane_key(ordered[index])[3] != lane_key[3]
                    and _segment_action(ordered[index]) in NORMAL_ACTIONS
                    and _segment_evidence_reliable(ordered[index], config)
                    for index in intervening
                )
                stable_conflict = any(
                    _lane_key(ordered[index]) == lane_key
                    and _segment_action(ordered[index]) != action
                    and _segment_action(ordered[index]) in NORMAL_ACTIONS
                    and _segment_evidence_reliable(ordered[index], config)
                    and _duration(ordered[index])
                    >= _action_support_threshold(ordered[index], config) - 1e-9
                    for index in intervening
                )
                if hard_block or other_side_normal_block or stable_conflict:
                    break
                support_indices.append(candidate_index)
                used_support_indices.add(candidate_index)
                gap_indices.extend(intervening)
                pre_gate_merges += 1

            support_segments = [ordered[index] for index in support_indices]
            unique_gap_indices = list(dict.fromkeys(gap_indices))
            gap_segments = [ordered[index] for index in unique_gap_indices]
            group = _aggregate_support_group(
                support_segments,
                gap_segments,
                group_index=len(groups) + 1,
            )
            absorbed_gap_ids.update(
                _list_ids(group.get("bounded_gap_source_segment_ids"))
            )
            groups.append(group)

    passthrough = [
        segment
        for index, segment in enumerate(ordered)
        if index not in set(candidate_indices)
    ]
    result = passthrough + groups
    result.sort(
        key=lambda item: (
            _event_float(item, "start_time"),
            _event_float(item, "end_time"),
            _lane_key(item),
        )
    )
    return result, {
        "pre_gate_merged_fragment_count": pre_gate_merges,
        "pre_gate_support_group_count": len(groups),
        "pre_gate_absorbed_gap_segment_count": len(absorbed_gap_ids),
    }


def _merge_candidates(
    candidates: list[dict[str, Any]],
    pose_segments: list[dict[str, Any]],
    barriers: list[dict[str, Any]],
    config: PhaseBActionStabilityConfig,
) -> tuple[list[dict[str, Any]], int, int]:
    lanes: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    passthrough: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate["event_kind"] != "stable_action":
            passthrough.append(candidate)
            continue
        lanes.setdefault(_lane_key(candidate), []).append(candidate)

    merged: list[dict[str, Any]] = []
    merge_count = 0
    absorbed_count = 0
    for lane_key, lane in lanes.items():
        lane.sort(
            key=lambda item: (
                _event_float(item, "start_time"),
                _event_float(item, "end_time"),
            )
        )
        lane_output: list[dict[str, Any]] = []
        for candidate in lane:
            if not lane_output:
                lane_output.append(candidate)
                continue
            prior = lane_output[-1]
            gap_start = _event_float(prior, "end_time")
            gap_end = _event_float(candidate, "start_time")
            gap = gap_end - gap_start
            same_action = prior["action"] == candidate["action"]
            if (
                not same_action
                or prior["action"] not in NORMAL_ACTIONS
                or gap < -1e-6
                or gap > config.short_gap_merge_seconds + 1e-9
            ):
                lane_output.append(candidate)
                continue

            boundary_key = lane_key[:3]
            blocked = any(
                _boundary_key(barrier) == boundary_key
                and _intersects_open_interval(barrier, gap_start, gap_end)
                for barrier in barriers
            )
            prior_ids = set(_list_ids(prior.get("source_segment_ids")))
            candidate_ids = set(_list_ids(candidate.get("source_segment_ids")))
            intervening = [
                segment
                for segment in pose_segments
                if _boundary_key(segment) == boundary_key
                and not (
                    set(
                        _list_ids(
                            segment.get("source_segment_ids")
                            or segment.get("segment_id")
                        )
                    )
                    & (prior_ids | candidate_ids)
                )
                and _intersects_open_interval(segment, gap_start, gap_end)
            ]
            # The coarse classifier currently emits one dominant side per
            # frame.  An explicit different-side observation therefore blocks
            # continuity instead of being treated as concurrent hidden motion.
            different_side = any(
                str(item.get("side", "unknown")).lower() != lane_key[3]
                and not _event_bool(item, "bounded_uncertain_gap")
                for item in intervening
            )
            absorbable = all(
                _duration(item) <= config.short_gap_merge_seconds + 1e-9
                and _hard_boundary(item)[0] is None
                for item in intervening
            )
            if blocked or different_side or not absorbable:
                lane_output.append(candidate)
                continue

            absorbed_ids = [
                source_id
                for item in sorted(
                    intervening,
                    key=lambda value: _event_float(value, "start_time"),
                )
                for source_id in _list_ids(
                    item.get("source_segment_ids") or item.get("segment_id")
                )
            ]
            prior["end_time"] = max(
                _event_float(prior, "end_time"),
                _event_float(candidate, "end_time"),
            )
            prior["duration_seconds"] = round(
                prior["end_time"] - _event_float(prior, "start_time"),
                9,
            )
            prior["source_segment_ids"] = _unique(
                _list_ids(prior.get("source_segment_ids"))
                + _list_ids(candidate.get("source_segment_ids"))
            )
            prior["bounded_gap_source_segment_ids"] = _unique(
                _list_ids(prior.get("bounded_gap_source_segment_ids"))
                + absorbed_ids
                + _list_ids(
                    candidate.get("bounded_gap_source_segment_ids")
                )
            )
            prior["source_event_ids"] = _unique(
                _list_ids(prior.get("source_event_ids"))
                + _list_ids(candidate.get("source_event_ids"))
            )
            prior["absorbed_segment_ids"] = _unique(
                _list_ids(prior.get("absorbed_segment_ids"))
                + absorbed_ids
                + _list_ids(candidate.get("absorbed_segment_ids"))
            )
            observed_support = _event_float(
                prior,
                "observed_support_seconds",
                _duration(prior),
            ) + _event_float(
                candidate,
                "observed_support_seconds",
                _duration(candidate),
            )
            prior["observed_support_seconds"] = round(observed_support, 9)
            prior["bounded_gap_seconds"] = round(
                max(0.0, prior["duration_seconds"] - observed_support),
                9,
            )
            prior["observed_support_ratio"] = round(
                min(
                    1.0,
                    observed_support / max(prior["duration_seconds"], 1e-9),
                ),
                9,
            )
            prior["support_fragment_count"] = int(
                prior.get("support_fragment_count", 1)
            ) + int(candidate.get("support_fragment_count", 1))
            prior["pre_gate_merge_count"] = int(
                prior.get("pre_gate_merge_count", 0)
            ) + int(candidate.get("pre_gate_merge_count", 0))
            prior["bounded_uncertain_gaps"] = list(
                prior.get("bounded_uncertain_gaps", [])
            ) + list(candidate.get("bounded_uncertain_gaps", []))
            if gap > 1e-9:
                prior["bounded_uncertain_gaps"].append(
                    {
                        "start_time": round(gap_start, 9),
                        "end_time": round(gap_end, 9),
                        "duration_seconds": round(gap, 9),
                        "source_segment_ids": absorbed_ids,
                        "reason": "post_gate_same_action_gap",
                    }
                )
            prior["maximum_bounded_gap_seconds"] = round(
                max(
                    (
                        float(item.get("duration_seconds", 0.0))
                        for item in prior["bounded_uncertain_gaps"]
                    ),
                    default=0.0,
                ),
                9,
            )
            prior["stabilization_reason"] = (
                "same_person_epoch_side_action_fragments_safely_merged"
            )
            prior["stabilization_steps"] = _unique(
                list(prior.get("stabilization_steps", []))
                + ["same_action_gap_hysteresis_merge"]
                + list(candidate.get("stabilization_steps", []))
            )
            merge_count += 1
            absorbed_count += len(absorbed_ids)
        merged.extend(lane_output)

    result = merged + passthrough
    result.sort(
        key=lambda item: (
            _event_float(item, "start_time"),
            _event_float(item, "end_time"),
            _lane_key(item),
        )
    )
    return result, merge_count, absorbed_count


def build_stable_action_events(
    pose_segments: list[dict[str, Any]],
    config: PhaseBActionStabilityConfig | ActionAnalysisConfig | None = None,
) -> dict[str, Any]:
    """Gate each raw pose segment, then merge only evidence-safe fragments."""

    effective = _effective_config(config)
    pose_evidence = [dict(item) for item in pose_segments]
    raw_ordered = sorted(
        (dict(item) for item in pose_segments),
        key=lambda item: (
            _event_float(item, "start_time"),
            _event_float(item, "end_time"),
            _lane_key(item),
        ),
    )
    ordered, aggregation_metrics = _aggregate_normal_support_before_gate(
        raw_ordered,
        effective,
    )
    candidates: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    barriers: list[dict[str, Any]] = []
    short_directional_count = 0

    for segment in ordered:
        action = _segment_action(segment)
        duration = _duration(segment)
        observed_support = _event_float(
            segment,
            "observed_support_seconds",
            duration,
        )
        boundary, boundary_reasons = _hard_boundary(segment)
        if boundary is not None:
            barriers.append(segment)
        if boundary == "lost":
            candidates.append(
                _copy_event(
                    segment,
                    action="lost",
                    status="lost",
                    observation_state="lost",
                    reason="hard_lost_or_off_frame_boundary",
                    event_kind="hard_boundary",
                    display_eligible=duration >= 1.0,
                    boundary_reasons=boundary_reasons,
                )
            )
            continue
        if boundary == "uncertain":
            uncertain = _copy_event(
                segment,
                action="unknown",
                status="uncertain",
                observation_state="uncertain",
                reason="severe_occlusion_or_missing_hard_boundary",
                event_kind="hard_boundary",
                display_eligible=duration >= effective.stable_event_minimum_seconds,
                boundary_reasons=boundary_reasons,
            )
            if duration >= effective.stable_event_minimum_seconds:
                candidates.append(uncertain)
            else:
                uncertain["suppressed"] = True
                suppressed.append(uncertain)
            continue

        detected = _event_float(segment, "detected_ratio")
        direction_clear = _event_bool(segment, "direction_clear")
        evidence_reliable = _segment_evidence_reliable(segment, effective)
        if not evidence_reliable:
            weak = _copy_event(
                segment,
                action="unknown"
                if duration >= effective.stable_event_minimum_seconds
                else "transition",
                status="uncertain",
                observation_state="uncertain",
                reason="prediction_missing_or_required_joint_gate",
                display_eligible=duration >= effective.stable_event_minimum_seconds,
            )
            if duration >= effective.stable_event_minimum_seconds:
                candidates.append(weak)
            else:
                weak["suppressed"] = True
                suppressed.append(weak)
            continue

        threshold = _action_support_threshold(segment, effective)
        if action in {"unknown", "transition"}:
            threshold = effective.stable_event_minimum_seconds
        if observed_support + 1e-9 < threshold:
            short = _copy_event(
                segment,
                action="transition",
                status="uncertain",
                observation_state=str(
                    segment.get("observation_state", "detected")
                ),
                reason="below_phase_b_observed_support_gate",
                display_eligible=False,
            )
            short["suppressed"] = True
            short["required_duration_seconds"] = threshold
            short["observed_support_seconds"] = round(observed_support, 9)
            suppressed.append(short)
            continue

        if (
            action in SHORT_DIRECTIONAL_ACTIONS
            and observed_support < effective.stable_event_minimum_seconds
        ):
            if detected < effective.short_event_minimum_detected_ratio:
                short = _copy_event(
                    segment,
                    action="transition",
                    status="uncertain",
                    observation_state="uncertain",
                    reason="short_directional_visibility_gate_failed",
                    display_eligible=False,
                )
                short["suppressed"] = True
                suppressed.append(short)
                continue
            short_directional_count += 1
            reason = "one_second_directional_exception_passed"
        else:
            reason = "phase_b_observed_support_and_visibility_gate_passed"
        candidates.append(
            _copy_event(
                segment,
                action=action,
                status="proposed" if action in NORMAL_ACTIONS else "uncertain",
                observation_state=str(
                    segment.get("observation_state", "detected")
                ),
                reason=reason,
            )
        )

    stable, post_gate_merge_count, post_gate_absorbed_count = _merge_candidates(
        candidates,
        ordered,
        barriers,
        effective,
    )
    for index, event in enumerate(stable, start=1):
        event["action_event_id"] = f"action-event-{index:05d}"

    normal_stable = [
        item
        for item in stable
        if item.get("event_kind") == "stable_action"
        and item.get("action") in NORMAL_ACTIONS
    ]
    sub_1s = [
        item for item in normal_stable if _duration(item) < 1.0 - 1e-9
    ]
    published_pre_gate_merges = sum(
        int(item.get("pre_gate_merge_count", 0))
        for item in normal_stable
    )
    suppressed_pre_gate_merges = max(
        0,
        aggregation_metrics["pre_gate_merged_fragment_count"]
        - published_pre_gate_merges,
    )
    published_gap_ids = {
        source_id
        for item in normal_stable
        for source_id in _list_ids(
            item.get("bounded_gap_source_segment_ids")
        )
    }
    window_start = min(
        (_event_float(item, "start_time") for item in ordered),
        default=0.0,
    )
    window_end = max(
        (_event_float(item, "end_time") for item in ordered),
        default=window_start,
    )
    window_seconds = max(0.0, window_end - window_start)
    normal_overlap_with_barrier = 0
    for event in normal_stable:
        if any(
            _boundary_key(barrier) == _boundary_key(event)
            and _event_float(barrier, "end_time")
            > _event_float(event, "start_time") + 1e-9
            and _event_float(barrier, "start_time")
            < _event_float(event, "end_time") - 1e-9
            for barrier in barriers
        ):
            normal_overlap_with_barrier += 1

    return {
        "pose_evidence": pose_evidence,
        "stable_events": stable,
        "suppressed_events": suppressed,
        "metrics": {
            "input_pose_event_count": len(pose_evidence),
            "input_pose_segment_count": len(pose_evidence),
            "stable_event_count": len(stable),
            "stable_normal_action_count": len(normal_stable),
            "suppressed_count": len(suppressed),
            "suppressed_fragment_count": len(suppressed),
            "merge_count": published_pre_gate_merges + post_gate_merge_count,
            "merged_fragment_count": (
                published_pre_gate_merges + post_gate_merge_count
            ),
            "pre_gate_merged_fragment_count": aggregation_metrics[
                "pre_gate_merged_fragment_count"
            ],
            "published_pre_gate_merged_fragment_count": (
                published_pre_gate_merges
            ),
            "suppressed_pre_gate_merged_fragment_count": (
                suppressed_pre_gate_merges
            ),
            "post_gate_merged_fragment_count": post_gate_merge_count,
            "pre_gate_support_group_count": aggregation_metrics[
                "pre_gate_support_group_count"
            ],
            "absorbed_segment_count": (
                len(published_gap_ids) + post_gate_absorbed_count
            ),
            "pre_gate_absorbed_gap_segment_count": aggregation_metrics[
                "pre_gate_absorbed_gap_segment_count"
            ],
            "stable_normal_observed_support_seconds": round(
                sum(
                    _event_float(item, "observed_support_seconds", _duration(item))
                    for item in normal_stable
                ),
                9,
            ),
            "hard_boundary_count": len(barriers),
            "short_directional_action_preserved_count": short_directional_count,
            "sub_1s_stable_event_count": len(sub_1s),
            "events_per_minute": (
                round(len(normal_stable) * 60.0 / window_seconds, 6)
                if window_seconds > 0
                else 0.0
            ),
            "unknown_transition_duration_seconds": round(
                sum(
                    _duration(item)
                    for item in ordered
                    if str(
                        item.get("action", item.get("action_name", "unknown"))
                    ).lower()
                    in {"unknown", "transition"}
                ),
                9,
            ),
            "displayed_unknown_transition_duration_seconds": round(
                sum(
                    _duration(item)
                    for item in stable
                    if item.get("action") in {"unknown", "transition"}
                ),
                9,
            ),
            "lost_normal_action_overlap_count": normal_overlap_with_barrier,
            "cross_identity_or_epoch_merge_count": 0,
            "configuration": asdict(effective),
        },
    }


def _timeline_state(segment: dict[str, Any]) -> tuple[str, str, str]:
    action = str(
        segment.get("action", segment.get("action_name", "unknown"))
    ).lower()
    if _event_bool(segment, "hard_boundary"):
        if action == "lost" or str(segment.get("evidence_state", "")).lower() == "lost":
            return "lost", "lost", "hard_boundary"
        return "uncertain", "unknown", "hard_boundary"
    if _event_bool(segment, "bounded_uncertain_gap"):
        return "uncertain", "unknown", "bounded_uncertain_gap"
    explicit = str(segment.get("evidence_state", "")).lower()
    if explicit in {"normal", "transition", "unknown", "uncertain", "lost"}:
        evidence_state = explicit
    elif action == "lost":
        evidence_state = "lost"
    elif action == "unknown":
        evidence_state = "unknown"
    elif action == "transition":
        evidence_state = "transition"
    else:
        evidence_state = "normal"
    if evidence_state == "lost":
        return "lost", "lost", "hard_boundary"
    if evidence_state == "uncertain":
        return "uncertain", "unknown", "observed"
    return evidence_state, action, "observed"


def _raw_switch_metrics(
    raw_pose_segments: list[dict[str, Any]],
    *,
    window_seconds: float,
) -> dict[str, Any]:
    ordered = sorted(
        raw_pose_segments,
        key=lambda item: (
            _event_float(item, "start_time"),
            _event_float(item, "end_time"),
        ),
    )
    switches = 0
    denominator = 0
    for left, right in zip(ordered, ordered[1:]):
        if _boundary_key(left) != _boundary_key(right):
            continue
        if _hard_boundary(left)[0] == "lost" or _hard_boundary(right)[0] == "lost":
            continue
        denominator += 1
        left_action = str(
            left.get("action", left.get("action_name", "unknown"))
        ).lower()
        right_action = str(
            right.get("action", right.get("action_name", "unknown"))
        ).lower()
        if left_action != right_action:
            switches += 1
    return {
        "raw_action_switch_count": switches,
        "raw_action_switch_denominator": denominator,
        "raw_action_switch_rate": (
            round(switches / denominator, 9) if denominator else 0.0
        ),
        "raw_action_switches_per_minute": (
            round(switches * 60.0 / window_seconds, 6)
            if window_seconds > 0
            else 0.0
        ),
    }


def validate_evidence_timeline(
    intervals: list[dict[str, Any]],
    *,
    analysis_start_time: float,
    analysis_end_time: float,
    tolerance_seconds: float,
) -> None:
    """Validate temporal continuity and conservative automatic-result invariants."""

    start = float(analysis_start_time)
    end = float(analysis_end_time)
    if end <= start:
        raise ValueError("Evidence timeline requires a positive analysis window")
    if not intervals:
        raise ValueError("Evidence timeline cannot be empty for a decoded window")
    cursor = start
    normal_actions = NORMAL_ACTIONS
    for index, interval in enumerate(intervals):
        interval_start = _event_float(interval, "start_time")
        interval_end = _event_float(interval, "end_time")
        if interval_end <= interval_start:
            raise ValueError(f"Evidence interval {index} has non-positive duration")
        if abs(interval_start - cursor) > tolerance_seconds + 1e-9:
            relation = "gap" if interval_start > cursor else "overlap"
            raise ValueError(
                f"Evidence timeline {relation} before interval {index}: "
                f"{interval_start - cursor:.9f}s"
            )
        if not isinstance(interval.get("source_segment_ids"), list) or not interval[
            "source_segment_ids"
        ]:
            raise ValueError("Every evidence interval needs source_segment_ids[]")
        if bool(interval.get("training_eligible", False)):
            raise ValueError("Automatic evidence cannot be training eligible")
        state = str(interval.get("evidence_state", ""))
        action = str(interval.get("action", ""))
        status = str(interval.get("status", ""))
        continuity = str(interval.get("continuity_kind", ""))
        if state == "normal" and (
            action not in normal_actions or status != "proposed"
        ):
            raise ValueError("Normal timeline evidence must be a proposed normal action")
        if state in {"transition", "unknown", "uncertain"} and status != "uncertain":
            raise ValueError("Non-normal timeline evidence must remain uncertain")
        if state == "lost" and (
            action != "lost"
            or status != "lost"
            or continuity != "hard_boundary"
        ):
            raise ValueError("Lost timeline evidence must remain a hard boundary")
        cursor = interval_end
    if abs(cursor - end) > tolerance_seconds + 1e-9:
        raise ValueError(
            f"Evidence timeline does not end at analysis window: {cursor} != {end}"
        )


def build_evidence_timeline(
    stable_input_segments: list[dict[str, Any]],
    raw_pose_segments: list[dict[str, Any]],
    *,
    analysis_start_time: float,
    analysis_end_time: float,
    sample_interval_seconds: float,
) -> dict[str, Any]:
    """Publish a continuous evidence track without promoting weak evidence."""

    window_start = float(analysis_start_time)
    window_end = float(analysis_end_time)
    window_seconds = max(0.0, window_end - window_start)
    if window_seconds <= 0:
        raise ValueError("Evidence timeline requires a positive analysis window")
    raw_by_id = {
        str(segment.get("segment_id")): segment
        for segment in raw_pose_segments
        if segment.get("segment_id")
    }
    ordered = sorted(
        (dict(item) for item in stable_input_segments),
        key=lambda item: (
            _event_float(item, "start_time"),
            _event_float(item, "end_time"),
        ),
    )
    if not ordered:
        raise ValueError("Decoded evidence is required to build a timeline")

    intervals: list[dict[str, Any]] = []
    cursor = window_start

    def append_interval(
        segment: dict[str, Any],
        *,
        start_time: float,
        end_time: float,
        sampling_gap: bool = False,
    ) -> None:
        if end_time <= start_time + 1e-12:
            return
        source_ids = _list_ids(
            segment.get("source_segment_ids") or segment.get("segment_id")
        )
        raw_sources = [
            raw_by_id[source_id]
            for source_id in source_ids
            if source_id in raw_by_id
        ]
        if sampling_gap:
            evidence_state, action, continuity = (
                "uncertain",
                "unknown",
                "bounded_uncertain_gap",
            )
            reasons = ["sample_timestamp_gap"]
            observation_state = "uncertain"
        else:
            evidence_state, action, continuity = _timeline_state(segment)
            configured_reasons = segment.get("temporal_reasons")
            reasons = (
                [str(reason) for reason in configured_reasons]
                if isinstance(configured_reasons, list) and configured_reasons
                else [
                    str(
                        segment.get("temporal_reason")
                        or segment.get("stabilization_reason")
                        or "observed_pose_evidence"
                    )
                ]
            )
            observation_state = str(
                segment.get("observation_state", "uncertain")
            ).lower()
            if evidence_state == "uncertain":
                observation_state = (
                    observation_state
                    if observation_state in {"missing", "uncertain"}
                    else "uncertain"
                )
            if evidence_state == "lost":
                observation_state = "lost"
        status = (
            "proposed"
            if evidence_state == "normal"
            else "lost"
            if evidence_state == "lost"
            else "uncertain"
        )
        intervals.append(
            {
                "evidence_interval_id": f"evidence-{len(intervals) + 1:05d}",
                "evidence_state": evidence_state,
                "action": action,
                "raw_action_names": _unique(
                    [
                        str(
                            source.get(
                                "action",
                                source.get("action_name", "unknown"),
                            )
                        ).lower()
                        for source in raw_sources
                    ]
                ),
                "person_ref": str(segment.get("person_ref", "")),
                "lock_epoch": int(segment.get("lock_epoch", 0)),
                "anatomical_side": str(
                    segment.get(
                        "anatomical_side",
                        segment.get("side", "unknown"),
                    )
                ).lower(),
                "start_time": round(start_time, 9),
                "end_time": round(end_time, 9),
                "duration_seconds": round(end_time - start_time, 9),
                "track_state": str(segment.get("track_state", "unknown")),
                "lock_state": str(segment.get("lock_state", "unknown")),
                "observation_state": observation_state,
                "continuity_kind": continuity,
                "continuity_reasons": _unique(reasons),
                "source_frame_indices": [
                    int(value)
                    for value in segment.get("source_frame_indices", [])
                ],
                "source_segment_ids": source_ids,
                "source_video_sha256": str(
                    segment.get("source_video_sha256", "")
                ),
                "recording_group_id": segment.get("recording_group_id"),
                "source_model_version": segment.get("source_model_version"),
                "status": status,
                "stabilization_reasons": _unique(reasons),
                "display_eligible": True,
                "training_eligible": False,
            }
        )

    for segment in ordered:
        segment_start = max(
            window_start,
            _event_float(segment, "start_time"),
        )
        segment_end = min(window_end, _event_float(segment, "end_time"))
        if segment_end <= window_start or segment_start >= window_end:
            continue
        if segment_start < cursor - 1e-6:
            raise ValueError(
                "Stable input segments overlap inside the analysis window"
            )
        if segment_start > cursor + 1e-9:
            append_interval(
                segment,
                start_time=cursor,
                end_time=segment_start,
                sampling_gap=True,
            )
        segment_start = max(segment_start, cursor)
        append_interval(
            segment,
            start_time=segment_start,
            end_time=segment_end,
        )
        cursor = max(cursor, segment_end)

    if cursor < window_end - 1e-9:
        append_interval(
            ordered[-1],
            start_time=cursor,
            end_time=window_end,
            sampling_gap=True,
        )

    validate_evidence_timeline(
        intervals,
        analysis_start_time=window_start,
        analysis_end_time=window_end,
        tolerance_seconds=max(1e-6, float(sample_interval_seconds) / 100.0),
    )
    duration_by_state = {
        state: round(
            sum(
                _duration(interval)
                for interval in intervals
                if interval["evidence_state"] == state
            ),
            9,
        )
        for state in ("normal", "transition", "unknown", "uncertain", "lost")
    }
    covered_seconds = round(
        sum(_duration(interval) for interval in intervals),
        9,
    )
    normal_seconds = duration_by_state["normal"]
    metrics = {
        "window_start_time": window_start,
        "window_end_time": window_end,
        "window_seconds": round(window_seconds, 9),
        "covered_seconds": covered_seconds,
        "uncovered_seconds": round(max(0.0, window_seconds - covered_seconds), 9),
        "overlap_seconds": 0.0,
        "coverage_ratio": round(
            min(1.0, covered_seconds / window_seconds),
            9,
        ),
        "tolerance_seconds": max(
            1e-6,
            float(sample_interval_seconds) / 100.0,
        ),
        "interval_count": len(intervals),
        "duration_by_state_seconds": duration_by_state,
        "normal_action_seconds": normal_seconds,
        "normal_action_coverage_ratio": round(
            normal_seconds / window_seconds,
            9,
        ),
        "bounded_uncertain_gap_count": sum(
            1
            for interval in intervals
            if interval["continuity_kind"] == "bounded_uncertain_gap"
        ),
        "bounded_uncertain_gap_seconds": round(
            sum(
                _duration(interval)
                for interval in intervals
                if interval["continuity_kind"] == "bounded_uncertain_gap"
            ),
            9,
        ),
        "hard_boundary_count": sum(
            1
            for interval in intervals
            if interval["continuity_kind"] == "hard_boundary"
        ),
        "pose_fragment_count": len(raw_pose_segments),
        **_raw_switch_metrics(
            raw_pose_segments,
            window_seconds=window_seconds,
        ),
    }
    return {"intervals": intervals, "metrics": metrics}


def build_stable_action_events_from_frames(
    frames: list[CoarseFrame],
    raw_pose_segments: list[dict[str, Any]],
    *,
    source_video_sha256: str,
    sample_interval_seconds: float,
    analysis_end_time: float,
    analysis_start_time: float | None = None,
    frame_config: FrameActionStabilityConfig | None = None,
    event_config: PhaseBActionStabilityConfig | ActionAnalysisConfig | None = None,
) -> dict[str, Any]:
    """Build stable events from confirmed frame copies with raw-segment lineage."""

    frame_result = stabilize_coarse_frames(frames, frame_config)
    stable_input_segments = build_pose_segments(
        frame_result["frames"],
        source_video_sha256=source_video_sha256,
        sample_interval_seconds=sample_interval_seconds,
        analysis_end_time=analysis_end_time,
    )
    raw_by_id = {
        str(segment["segment_id"]): segment
        for segment in raw_pose_segments
        if segment.get("segment_id")
    }
    raw_ids_by_frame: dict[int, list[str]] = {}
    for segment_id, segment in raw_by_id.items():
        for frame_index in segment.get("source_frame_indices", []):
            raw_ids_by_frame.setdefault(int(frame_index), []).append(segment_id)

    provenance_fields = (
        "recording_group_id",
        "source_model_version",
        "reviewer",
        "reviewed_at",
    )
    for segment in stable_input_segments:
        source_ids = _unique(
            [
                source_id
                for frame_index in segment.get("source_frame_indices", [])
                for source_id in raw_ids_by_frame.get(int(frame_index), [])
            ]
        )
        if not source_ids:
            raise ValueError(
                "Stable frame segment has no traceable raw pose segment"
            )
        segment["source_segment_ids"] = source_ids
        sources = [raw_by_id[source_id] for source_id in source_ids]
        for field in provenance_fields:
            values = {
                source.get(field)
                for source in sources
                if field in source
            }
            if len(values) == 1:
                segment[field] = values.pop()

    result = build_stable_action_events(stable_input_segments, event_config)
    timeline = build_evidence_timeline(
        stable_input_segments,
        raw_pose_segments,
        analysis_start_time=(
            float(analysis_start_time)
            if analysis_start_time is not None
            else min(
                (
                    _event_float(segment, "start_time")
                    for segment in raw_pose_segments
                ),
                default=0.0,
            )
        ),
        analysis_end_time=analysis_end_time,
        sample_interval_seconds=sample_interval_seconds,
    )
    result["evidence_timeline"] = timeline["intervals"]
    result["evidence_timeline_metrics"] = timeline["metrics"]
    result["stable_input_segments"] = result["pose_evidence"]
    result["pose_evidence"] = [dict(item) for item in raw_pose_segments]
    metrics = result["metrics"]
    metrics["input_stabilized_segment_count"] = metrics[
        "input_pose_segment_count"
    ]
    metrics["input_pose_event_count"] = len(raw_pose_segments)
    metrics["input_pose_segment_count"] = len(raw_pose_segments)
    metrics["frame_stabilization"] = frame_result["metrics"]
    metrics["evidence_timeline"] = timeline["metrics"]
    return result
