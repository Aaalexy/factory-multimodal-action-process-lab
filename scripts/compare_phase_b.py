"""Compare Phase A replay and Phase B results without inventing accuracy.

The expected layout is::

    <validation-root>/<clip-id>/before/analysis.json
    <validation-root>/<clip-id>/after/analysis.json

Only complete pairs are compared.  Missing pairs are reported explicitly, and
the comparison remains ``not_evaluable`` until independent human ground truth
exists.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


COUNT_FIELDS = (
    "processed_frame_count",
    "pose_inference_calls",
    "hand_inference_calls",
    "hand_detected_frame_count",
    "hand_uncertain_frame_count",
    "hand_missing_frame_count",
    "hand_unavailable_frame_count",
    "hand_detected_observation_count",
    "hand_uncertain_observation_count",
    "hand_missing_observation_count",
    "left_right_association_error_count",
    "association_warning_count",
    "pose_segment_count",
    "stable_action_event_count",
    "stable_normal_action_count",
    "sub_1s_stable_event_count",
    "suppressed_fragment_count",
    "merged_fragment_count",
    "lost_normal_action_false_positive_count",
    "cross_identity_or_epoch_merge_count",
)
SUM_FLOAT_FIELDS = (
    "analysis_duration_seconds",
    "unknown_transition_duration_seconds",
    "displayed_unknown_transition_duration_seconds",
    "processing_seconds",
    "end_to_end_seconds",
)
RATE_FIELDS = (
    "events_per_minute",
    "mean_pose_inference_ms",
    "mean_hand_inference_ms",
    "processing_frames_per_second",
    "end_to_end_frames_per_second",
)
METRIC_FIELDS = COUNT_FIELDS + SUM_FLOAT_FIELDS + RATE_FIELDS
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
ASSOCIATION_ERROR_WARNINGS = {
    "model_hand_is_closer_to_opposite_body_wrist",
    "model_wrist_too_far_from_own_body_wrist",
    "own_body_wrist_unavailable_for_consistency_check",
    "duplicate_hand_candidate_across_sides",
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Analysis must be a JSON object: {path}")
    return payload


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return default


def _integer(value: Any, default: int = 0) -> int:
    return int(round(_number(value, float(default))))


def _duration(item: dict[str, Any]) -> float:
    explicit = item.get("duration_seconds")
    if isinstance(explicit, (int, float)) and math.isfinite(float(explicit)):
        return max(0.0, float(explicit))
    return max(
        0.0,
        _number(item.get("end_time")) - _number(item.get("start_time")),
    )


def _analysis_duration(payload: dict[str, Any]) -> float:
    window = payload.get("source_video", {}).get("analysis_window", {})
    start = _number(window.get("start_time"))
    end = _number(window.get("end_time"))
    if end > start:
        return end - start
    events = payload.get("action_events", [])
    if events:
        first = min(_number(item.get("start_time")) for item in events)
        last = max(_number(item.get("end_time")) for item in events)
        return max(0.0, last - first)
    return 0.0


def _hand_fallback(
    payload: dict[str, Any],
) -> tuple[dict[str, int], dict[str, int], int]:
    observation_counts = {"detected": 0, "uncertain": 0, "missing": 0}
    states_by_frame: dict[int, list[str]] = {}
    association_error_events: set[tuple[int, str]] = set()
    for record in payload.get("hand_pose_frames", []):
        raw_state = str(record.get("observation_state", "missing")).lower()
        state = (
            "detected"
            if raw_state == "detected"
            else "uncertain"
            if raw_state in {"uncertain", "predicted", "interpolated"}
            else "missing"
        )
        observation_counts[state] += 1
        states_by_frame.setdefault(
            _integer(record.get("frame_index"), -1), []
        ).append(state)
        warnings = record.get("association_checks", {}).get("warnings", [])
        frame_index = _integer(record.get("frame_index"), -1)
        for warning in warnings:
            if str(warning) in ASSOCIATION_ERROR_WARNINGS:
                association_error_events.add((frame_index, str(warning)))
    frame_counts = {"detected": 0, "uncertain": 0, "missing": 0}
    for states in states_by_frame.values():
        if "detected" in states:
            frame_counts["detected"] += 1
        elif "uncertain" in states:
            frame_counts["uncertain"] += 1
        else:
            frame_counts["missing"] += 1
    return frame_counts, observation_counts, len(association_error_events)


def extract_metrics(payload: dict[str, Any]) -> dict[str, int | float | None]:
    """Extract comparable runtime evidence with conservative fallbacks."""

    runtime = payload.get("runtime", {})
    stabilization = payload.get("stabilization_metrics", {})
    frames = payload.get("pose_frames", [])
    segments = payload.get("pose_segments", [])
    events = payload.get("action_events", [])
    normal_events = [
        item
        for item in events
        if str(item.get("action", item.get("action_name", ""))).lower()
        in NORMAL_ACTIONS
    ]
    hand_frames, hand_observations, association_errors = _hand_fallback(
        payload
    )
    duration_seconds = _analysis_duration(payload)

    def runtime_or(
        key: str,
        fallback: int | float | None,
    ) -> int | float | None:
        value = runtime.get(key)
        return fallback if value is None else value

    processed_frames = _integer(
        runtime_or("processed_frame_count", len(frames))
    )
    stable_count = _integer(
        runtime_or("stable_action_event_count", len(events))
    )
    metrics: dict[str, int | float | None] = {
        "processed_frame_count": processed_frames,
        "pose_inference_calls": _integer(
            runtime_or("pose_inference_calls", processed_frames)
        ),
        "hand_inference_calls": _integer(
            runtime_or("hand_inference_calls", 0)
        ),
        "hand_detected_frame_count": _integer(
            runtime_or("hand_detected_frame_count", hand_frames["detected"])
        ),
        "hand_uncertain_frame_count": _integer(
            runtime_or(
                "hand_uncertain_frame_count", hand_frames["uncertain"]
            )
        ),
        "hand_missing_frame_count": _integer(
            runtime_or("hand_missing_frame_count", hand_frames["missing"])
        ),
        "hand_unavailable_frame_count": _integer(
            runtime_or("hand_unavailable_frame_count", 0)
        ),
        "hand_detected_observation_count": _integer(
            runtime_or(
                "hand_detected_observation_count",
                hand_observations["detected"],
            )
        ),
        "hand_uncertain_observation_count": _integer(
            runtime_or(
                "hand_uncertain_observation_count",
                hand_observations["uncertain"],
            )
        ),
        "hand_missing_observation_count": _integer(
            runtime_or(
                "hand_missing_observation_count",
                hand_observations["missing"],
            )
        ),
        "left_right_association_error_count": _integer(
            runtime_or(
                "left_right_association_error_count", association_errors
            )
        ),
        "association_warning_count": _integer(
            runtime_or("association_warning_count", 0)
        ),
        "pose_segment_count": _integer(
            runtime_or("pose_segment_count", len(segments))
        ),
        "stable_action_event_count": stable_count,
        "stable_normal_action_count": _integer(
            runtime_or("stable_normal_action_count", len(normal_events))
        ),
        "sub_1s_stable_event_count": _integer(
            runtime_or(
                "sub_1s_stable_event_count",
                sum(_duration(item) < 1.0 - 1e-9 for item in normal_events),
            )
        ),
        "events_per_minute": float(
            runtime_or(
                "events_per_minute",
                stable_count * 60.0 / duration_seconds
                if duration_seconds > 0
                else 0.0,
            )
        ),
        "suppressed_fragment_count": _integer(
            runtime_or(
                "suppressed_fragment_count",
                stabilization.get(
                    "suppressed_fragment_count",
                    stabilization.get("suppressed_count", 0),
                ),
            )
        ),
        "merged_fragment_count": _integer(
            runtime_or(
                "merged_fragment_count",
                stabilization.get(
                    "merged_fragment_count",
                    stabilization.get("merge_count", 0),
                ),
            )
        ),
        "unknown_transition_duration_seconds": float(
            runtime_or(
                "unknown_transition_duration_seconds",
                stabilization.get(
                    "unknown_transition_duration_seconds",
                    sum(
                        _duration(item)
                        for item in events
                        if str(
                            item.get("action", item.get("action_name", ""))
                        ).lower()
                        in {"unknown", "transition"}
                    ),
                ),
            )
        ),
        "displayed_unknown_transition_duration_seconds": float(
            runtime_or(
                "displayed_unknown_transition_duration_seconds",
                stabilization.get(
                    "displayed_unknown_transition_duration_seconds",
                    sum(
                        _duration(item)
                        for item in events
                        if str(
                            item.get("action", item.get("action_name", ""))
                        ).lower()
                        in {"unknown", "transition"}
                    ),
                ),
            )
        ),
        "lost_normal_action_false_positive_count": _integer(
            runtime_or(
                "lost_normal_action_false_positive_count",
                stabilization.get("lost_normal_action_overlap_count", 0),
            )
        ),
        "cross_identity_or_epoch_merge_count": _integer(
            runtime_or(
                "cross_identity_or_epoch_merge_count",
                stabilization.get(
                    "cross_identity_or_epoch_merge_count", 0
                ),
            )
        ),
        "mean_pose_inference_ms": (
            None
            if runtime.get("mean_pose_inference_ms") is None
            else float(runtime["mean_pose_inference_ms"])
        ),
        "mean_hand_inference_ms": (
            None
            if runtime.get("mean_hand_inference_ms") is None
            else float(runtime["mean_hand_inference_ms"])
        ),
        "analysis_duration_seconds": round(duration_seconds, 6),
        "processing_seconds": float(runtime.get("processing_seconds", 0.0)),
        "end_to_end_seconds": float(
            runtime.get(
                "end_to_end_seconds",
                runtime.get("processing_seconds", 0.0),
            )
        ),
        "end_to_end_frames_per_second": float(
            runtime_or(
                "end_to_end_frames_per_second",
                processed_frames / _number(runtime.get("end_to_end_seconds"))
                if _number(runtime.get("end_to_end_seconds")) > 0
                else 0.0,
            )
        ),
        "processing_frames_per_second": float(
            runtime_or(
                "processing_frames_per_second",
                processed_frames / _number(runtime.get("processing_seconds"))
                if _number(runtime.get("processing_seconds")) > 0
                else 0.0,
            )
        ),
    }
    return metrics


def _profile(payload: dict[str, Any], phase: str) -> dict[str, str]:
    runtime = payload.get("runtime", {})
    name = str(
        payload.get("action_profile")
        or runtime.get("action_profile")
        or f"{phase}_profile_unreported"
    )
    if phase == "before":
        description = (
            "Phase A parameter replay with the all-NaN crash guard; this is "
            "a controlled timeline baseline, not a model-accuracy score."
        )
    else:
        description = (
            "Phase B stable-event profile with confirmation, temporal "
            "context, duration gates, and hard identity/lost boundaries."
        )
    return {"name": name, "description": description}


def _evaluation(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": "not_evaluable",
        "reason": (
            "No independently human-confirmed action, hand, or boundary "
            "ground truth was supplied; values are runtime stability evidence."
        ),
        "before_source_status": before.get("evaluation", {}).get("status"),
        "after_source_status": after.get("evaluation", {}).get("status"),
        "accuracy_metrics_computed": False,
    }


def _delta(
    before: dict[str, int | float | None],
    after: dict[str, int | float | None],
) -> dict[str, int | float | None]:
    result: dict[str, int | float | None] = {}
    for key in METRIC_FIELDS:
        left, right = before.get(key), after.get(key)
        if left is None or right is None:
            result[key] = None
        elif key in COUNT_FIELDS:
            result[key] = int(right) - int(left)
        else:
            result[key] = round(float(right) - float(left), 6)
    return result


def compare_pair(
    clip_id: str,
    before_path: Path,
    after_path: Path,
) -> dict[str, Any]:
    before_payload = _load_json(before_path)
    after_payload = _load_json(after_path)
    before_metrics = extract_metrics(before_payload)
    after_metrics = extract_metrics(after_payload)
    before_sha = before_payload.get("source_video", {}).get("sha256")
    after_sha = after_payload.get("source_video", {}).get("sha256")
    return {
        "clip_id": clip_id,
        "source_video_sha256": after_sha or before_sha,
        "source_hash_match": bool(before_sha and before_sha == after_sha),
        "paths": {
            "before": str(before_path.resolve()),
            "after": str(after_path.resolve()),
        },
        "profiles": {
            "before": _profile(before_payload, "before"),
            "after": _profile(after_payload, "after"),
        },
        "evaluation": _evaluation(before_payload, after_payload),
        "before": before_metrics,
        "after": after_metrics,
        "delta_after_minus_before": _delta(before_metrics, after_metrics),
    }


def _weighted_mean(
    entries: Iterable[dict[str, int | float | None]],
    value_key: str,
    weight_key: str,
) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for metrics in entries:
        value = metrics.get(value_key)
        weight = _number(metrics.get(weight_key))
        if value is not None and weight > 0:
            numerator += float(value) * weight
            denominator += weight
    return round(numerator / denominator, 6) if denominator else None


def aggregate_metrics(
    metrics_entries: list[dict[str, int | float | None]],
) -> dict[str, int | float | None]:
    totals: dict[str, int | float | None] = {
        key: sum(_integer(item.get(key)) for item in metrics_entries)
        for key in COUNT_FIELDS
    }
    totals.update(
        {
            key: round(
                sum(_number(item.get(key)) for item in metrics_entries), 6
            )
            for key in SUM_FLOAT_FIELDS
        }
    )
    total_runtime = _number(totals["end_to_end_seconds"])
    total_frames = _number(totals["processed_frame_count"])
    total_analysis_duration = _number(totals["analysis_duration_seconds"])
    totals["events_per_minute"] = (
        round(
            sum(
                _number(item.get("events_per_minute"))
                * _number(item.get("analysis_duration_seconds"))
                for item in metrics_entries
            )
            / total_analysis_duration,
            6,
        )
        if total_analysis_duration > 0
        else 0.0
    )
    totals["mean_pose_inference_ms"] = _weighted_mean(
        metrics_entries,
        "mean_pose_inference_ms",
        "pose_inference_calls",
    )
    totals["mean_hand_inference_ms"] = _weighted_mean(
        metrics_entries,
        "mean_hand_inference_ms",
        "hand_inference_calls",
    )
    totals["end_to_end_frames_per_second"] = (
        round(total_frames / total_runtime, 6) if total_runtime > 0 else 0.0
    )
    total_processing = sum(
        _number(item.get("processing_seconds")) for item in metrics_entries
    )
    totals["processing_frames_per_second"] = (
        round(total_frames / total_processing, 6)
        if total_processing > 0
        else 0.0
    )
    return totals


def discover_pairs(
    validation_root: Path,
) -> tuple[list[tuple[str, Path, Path]], list[dict[str, Any]]]:
    pairs: list[tuple[str, Path, Path]] = []
    incomplete: list[dict[str, Any]] = []
    candidates = {
        before.parent.parent
        for before in validation_root.rglob("before/analysis.json")
    }
    candidates.update(
        after.parent.parent
        for after in validation_root.rglob("after/analysis.json")
    )
    for clip_dir in sorted(candidates, key=lambda item: item.as_posix()):
        before = clip_dir / "before" / "analysis.json"
        after = clip_dir / "after" / "analysis.json"
        clip_id = clip_dir.relative_to(validation_root).as_posix()
        if before.is_file() and after.is_file():
            pairs.append((clip_id, before, after))
        else:
            incomplete.append(
                {
                    "clip_id": clip_id,
                    "before_present": before.is_file(),
                    "after_present": after.is_file(),
                }
            )
    return pairs, incomplete


def compare_validation_root(validation_root: Path) -> dict[str, Any]:
    validation_root = validation_root.resolve()
    pairs, incomplete = discover_pairs(validation_root)
    if not pairs:
        raise FileNotFoundError(
            f"No complete before/after analysis pair under {validation_root}"
        )
    clips = [
        compare_pair(clip_id, before, after)
        for clip_id, before, after in pairs
    ]
    before_summary = aggregate_metrics([item["before"] for item in clips])
    after_summary = aggregate_metrics([item["after"] for item in clips])
    return {
        "schema_version": "phase_b_hand_action_comparison_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "validation_root": str(validation_root),
        "evaluation": {
            "status": "not_evaluable",
            "reason": (
                "No independently human-confirmed ground truth; comparison "
                "covers runtime evidence and timeline stability only."
            ),
            "accuracy_metrics_computed": False,
        },
        "profile_notes": {
            "before": (
                "Phase A parameter replay with crash guard, used only as a "
                "controlled pre-upgrade timeline baseline."
            ),
            "after": (
                "Phase B hand-enabled stable-action run using higher temporal "
                "sampling and explicit confirmation/boundary gates."
            ),
        },
        "clips": clips,
        "summary": {
            "clip_count": len(clips),
            "incomplete_clip_count": len(incomplete),
            "evaluation": "not_evaluable",
            "before": before_summary,
            "after": after_summary,
            "delta_after_minus_before": _delta(
                before_summary, after_summary
            ),
        },
        "incomplete_clips": incomplete,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation-root",
        default="outputs/phase_b_validation",
    )
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = compare_validation_root(Path(args.validation_root))
    _write_json(Path(args.output), payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
