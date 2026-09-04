"""Compare a frozen Phase B baseline with a Phase B.1 candidate honestly."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_phase_b import NORMAL_ACTIONS, extract_metrics


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Analysis must be an object: {path}")
    return value


def _duration(item: dict[str, Any]) -> float:
    return max(
        0.0,
        float(item.get("end_time", 0.0))
        - float(item.get("start_time", 0.0)),
    )


def _union_duration(
    items: list[dict[str, Any]],
    *,
    window_start: float,
    window_end: float,
) -> float:
    intervals = sorted(
        (
            max(window_start, float(item.get("start_time", window_start))),
            min(window_end, float(item.get("end_time", window_start))),
        )
        for item in items
    )
    total = 0.0
    cursor_start: float | None = None
    cursor_end: float | None = None
    for start, end in intervals:
        if end <= start:
            continue
        if cursor_start is None:
            cursor_start, cursor_end = start, end
        elif start <= float(cursor_end) + 1e-9:
            cursor_end = max(float(cursor_end), end)
        else:
            total += float(cursor_end) - cursor_start
            cursor_start, cursor_end = start, end
    if cursor_start is not None:
        total += float(cursor_end) - cursor_start
    return total


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _raw_pose_core(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select immutable observation fields, excluding new derived annotations."""

    fields = (
        "segment_id",
        "action",
        "action_name",
        "person_ref",
        "lock_epoch",
        "side",
        "anatomical_side",
        "start_time",
        "end_time",
        "duration_seconds",
        "start_frame",
        "end_frame",
        "source_frame_indices",
        "source_video_sha256",
        "track_state",
        "lock_state",
        "observation_state",
        "detected_ratio",
        "predicted_ratio",
        "interpolated_ratio",
        "missing_ratio",
        "required_joints_reliable",
        "direction_clear",
        "raw_lost",
    )
    return [
        {key: segment.get(key) for key in fields}
        for segment in segments
    ]


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    return (
        resolved.relative_to(PROJECT_ROOT).as_posix()
        if resolved.is_relative_to(PROJECT_ROOT)
        else str(resolved)
    )


def _raw_switches(segments: list[dict[str, Any]]) -> tuple[int, int]:
    ordered = sorted(
        segments,
        key=lambda item: (
            float(item.get("start_time", 0.0)),
            float(item.get("end_time", 0.0)),
        ),
    )
    switches = 0
    denominator = 0
    for left, right in zip(ordered, ordered[1:]):
        left_key = (
            left.get("source_video_sha256"),
            left.get("person_ref"),
            left.get("lock_epoch"),
        )
        right_key = (
            right.get("source_video_sha256"),
            right.get("person_ref"),
            right.get("lock_epoch"),
        )
        if left_key != right_key:
            continue
        left_action = str(left.get("action", "unknown")).lower()
        right_action = str(right.get("action", "unknown")).lower()
        if "lost" in {left_action, right_action}:
            continue
        denominator += 1
        switches += left_action != right_action
    return switches, denominator


def _normal_event_integrity(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_by_id = {
        str(segment.get("segment_id")): segment
        for segment in payload.get("pose_segments", [])
        if segment.get("segment_id")
    }
    configuration = payload.get("stabilization_metrics", {}).get(
        "configuration",
        {},
    )
    stable_minimum = float(
        configuration.get("stable_event_minimum_seconds", 1.2)
    )
    directional_minimum = float(
        configuration.get("short_directional_event_minimum_seconds", 1.0)
    )
    bounded_limit = float(
        configuration.get("bounded_uncertain_gap_seconds", 0.375)
    )
    directional = {"reach", "retract", "lift", "lower", "push", "pull"}
    violations = []
    for event in payload.get("action_events", []):
        action = str(event.get("action", "")).lower()
        if event.get("event_kind") != "stable_action" or action not in NORMAL_ACTIONS:
            continue
        support_ids = [str(value) for value in event.get("source_segment_ids", [])]
        gap_ids = [
            str(value)
            for value in event.get("bounded_gap_source_segment_ids", [])
        ]
        support = float(
            event.get("observed_support_seconds", _duration(event))
        )
        threshold = (
            directional_minimum
            if action in directional and bool(event.get("direction_clear"))
            else stable_minimum
        )
        event_key = (
            event.get("person_ref"),
            event.get("lock_epoch"),
            event.get("anatomical_side", event.get("side")),
        )
        source_keys = {
            (
                raw_by_id[source_id].get("person_ref"),
                raw_by_id[source_id].get("lock_epoch"),
                raw_by_id[source_id].get(
                    "anatomical_side",
                    raw_by_id[source_id].get("side"),
                ),
            )
            for source_id in support_ids
            if source_id in raw_by_id
        }
        reasons = []
        if not support_ids or any(
            source_id not in raw_by_id for source_id in support_ids
        ):
            reasons.append("missing_raw_source_segment")
        if set(support_ids) & set(gap_ids):
            reasons.append("gap_source_promoted_to_action_support")
        if support + 1e-9 < threshold:
            reasons.append("observed_support_below_threshold")
        if float(event.get("maximum_bounded_gap_seconds", 0.0)) > bounded_limit + 1e-9:
            reasons.append("bounded_gap_exceeds_configuration")
        if source_keys and source_keys != {event_key}:
            reasons.append("source_crosses_person_epoch_or_side")
        if reasons:
            violations.append(
                {
                    "action_event_id": event.get("action_event_id"),
                    "reasons": reasons,
                }
            )
    return violations


def extract_phase_b1_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(extract_metrics(payload))
    window = payload.get("source_video", {}).get("analysis_window", {})
    start = float(window.get("start_time", 0.0))
    end = float(window.get("end_time", start))
    window_seconds = max(0.0, end - start)
    events = payload.get("action_events", [])
    normal_events = [
        item
        for item in events
        if item.get("event_kind") == "stable_action"
        and str(item.get("action", "")).lower() in NORMAL_ACTIONS
    ]
    display_events = [
        item for item in events if item.get("display_eligible") is not False
    ]
    timeline_metrics = payload.get("evidence_timeline_metrics", {})
    frame_metrics = payload.get("stabilization_metrics", {}).get(
        "frame_stabilization",
        {},
    )
    raw_switch_count, raw_switch_denominator = _raw_switches(
        payload.get("pose_segments", [])
    )
    integrity_violations = _normal_event_integrity(payload)
    normal_seconds = _union_duration(
        normal_events,
        window_start=start,
        window_end=end,
    )
    display_seconds = _union_duration(
        display_events,
        window_start=start,
        window_end=end,
    )
    metrics.update(
        {
            "analysis_window_start": start,
            "analysis_window_end": end,
            "analysis_window_seconds": window_seconds,
            "source_video_sha256": payload.get("source_video", {}).get("sha256"),
            "pose_segments_sha256": _canonical_sha(
                payload.get("pose_segments", [])
            ),
            "pose_segment_core_sha256": _canonical_sha(
                _raw_pose_core(payload.get("pose_segments", []))
            ),
            "stable_normal_action_seconds": round(normal_seconds, 9),
            "stable_normal_action_coverage_ratio": (
                round(normal_seconds / window_seconds, 9)
                if window_seconds
                else 0.0
            ),
            "display_eligible_seconds": round(display_seconds, 9),
            "display_eligible_ratio": (
                round(display_seconds / window_seconds, 9)
                if window_seconds
                else 0.0
            ),
            "evidence_timeline_present": bool(
                payload.get("evidence_timeline")
            ),
            "evidence_timeline_coverage_ratio": float(
                timeline_metrics.get("coverage_ratio", 0.0)
            ),
            "evidence_timeline_uncovered_seconds": float(
                timeline_metrics.get("uncovered_seconds", window_seconds)
            ),
            "evidence_normal_action_coverage_ratio": float(
                timeline_metrics.get("normal_action_coverage_ratio", 0.0)
            ),
            "raw_action_switch_count": raw_switch_count,
            "raw_action_switch_denominator": raw_switch_denominator,
            "raw_action_switch_rate": (
                round(raw_switch_count / raw_switch_denominator, 9)
                if raw_switch_denominator
                else 0.0
            ),
            "frame_hard_boundary_count": int(
                frame_metrics.get("hard_boundary_frame_count", 0)
            ),
            "bounded_uncertain_gap_frame_count": int(
                frame_metrics.get("bounded_uncertain_gap_frame_count", 0)
            ),
            "actual_non_tracked_frame_count": sum(
                frame.get("track_state") != "tracked"
                for frame in payload.get("pose_frames", [])
            ),
            "accuracy_status": str(
                payload.get("evaluation", {}).get("status", "not_evaluable")
            ),
            "validation_flags_all_false": all(
                value is False
                for value in payload.get("validation_flags", {}).values()
            ),
            "normal_event_integrity_violation_count": len(
                integrity_violations
            ),
            "normal_event_integrity_violations": integrity_violations,
        }
    )
    return metrics


def _numeric_delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in candidate.items():
        prior = baseline.get(key)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isinstance(prior, (int, float))
            and not isinstance(prior, bool)
        ):
            result[key] = round(float(value) - float(prior), 9)
    return result


def compare_roots(baseline_root: Path, candidate_root: Path) -> dict[str, Any]:
    clips = []
    clip_ids = sorted(
        path.name for path in baseline_root.iterdir() if path.is_dir()
    )
    for clip_id in clip_ids:
        baseline_path = baseline_root / clip_id / "after" / "analysis.json"
        candidate_path = candidate_root / clip_id / "candidate" / "analysis.json"
        if not baseline_path.is_file() or not candidate_path.is_file():
            continue
        baseline_payload = _load(baseline_path)
        candidate_payload = _load(candidate_path)
        baseline = extract_phase_b1_metrics(baseline_payload)
        candidate = extract_phase_b1_metrics(candidate_payload)
        same_source = (
            baseline["source_video_sha256"] == candidate["source_video_sha256"]
        )
        same_window = (
            baseline["analysis_window_start"]
            == candidate["analysis_window_start"]
            and baseline["analysis_window_end"]
            == candidate["analysis_window_end"]
        )
        gates = {
            "same_source_video": same_source,
            "same_analysis_window": same_window,
            "raw_pose_segments_unchanged": (
                baseline["pose_segment_core_sha256"]
                == candidate["pose_segment_core_sha256"]
            ),
            "continuous_evidence_timeline": (
                candidate["evidence_timeline_present"]
                and candidate["evidence_timeline_coverage_ratio"]
                >= 1.0 - 1e-9
                and candidate["evidence_timeline_uncovered_seconds"] <= 1e-9
            ),
            "no_sub_1s_normal_stable_events": (
                candidate["sub_1s_stable_event_count"] == 0
            ),
            "no_lost_normal_action_false_positive": (
                candidate["lost_normal_action_false_positive_count"] == 0
            ),
            "no_cross_identity_or_epoch_merge": (
                candidate["cross_identity_or_epoch_merge_count"] == 0
            ),
            "normal_stable_coverage_not_reduced": (
                candidate["stable_normal_action_seconds"]
                >= baseline["stable_normal_action_seconds"] - 1e-9
            ),
            "hard_boundary_frames_not_increased": (
                candidate["frame_hard_boundary_count"]
                <= baseline["frame_hard_boundary_count"]
            ),
            "accuracy_remains_not_evaluable": (
                candidate["accuracy_status"] == "not_evaluable"
            ),
            "validation_flags_remain_false": candidate[
                "validation_flags_all_false"
            ],
            "normal_event_support_and_lineage_are_consistent": (
                candidate["normal_event_integrity_violation_count"] == 0
            ),
        }
        clips.append(
            {
                "clip_id": clip_id,
                "baseline_analysis": _display_path(baseline_path),
                "candidate_analysis": _display_path(candidate_path),
                "baseline": baseline,
                "candidate": candidate,
                "delta_candidate_minus_baseline": _numeric_delta(
                    candidate,
                    baseline,
                ),
                "gates": gates,
                "passed": all(gates.values()),
            }
        )
    if not clips:
        raise ValueError("No complete baseline/candidate replay pairs found")

    def aggregate(side: str) -> dict[str, Any]:
        selected = [clip[side] for clip in clips]
        window_seconds = sum(item["analysis_window_seconds"] for item in selected)
        normal_seconds = sum(
            item["stable_normal_action_seconds"] for item in selected
        )
        display_seconds = sum(item["display_eligible_seconds"] for item in selected)
        covered_seconds = sum(
            item["analysis_window_seconds"]
            - item["evidence_timeline_uncovered_seconds"]
            for item in selected
        )
        sum_keys = (
            "processed_frame_count",
            "pose_segment_count",
            "stable_action_event_count",
            "stable_normal_action_count",
            "sub_1s_stable_event_count",
            "suppressed_fragment_count",
            "merged_fragment_count",
            "lost_normal_action_false_positive_count",
            "cross_identity_or_epoch_merge_count",
            "frame_hard_boundary_count",
            "bounded_uncertain_gap_frame_count",
            "actual_non_tracked_frame_count",
            "raw_action_switch_count",
            "raw_action_switch_denominator",
            "normal_event_integrity_violation_count",
            "end_to_end_seconds",
        )
        result = {
            key: round(sum(float(item.get(key, 0.0)) for item in selected), 9)
            for key in sum_keys
        }
        result.update(
            {
                "clip_count": len(selected),
                "analysis_window_seconds": round(window_seconds, 9),
                "stable_normal_action_seconds": round(normal_seconds, 9),
                "stable_normal_action_coverage_ratio": round(
                    normal_seconds / window_seconds,
                    9,
                ),
                "display_eligible_seconds": round(display_seconds, 9),
                "display_eligible_ratio": round(
                    display_seconds / window_seconds,
                    9,
                ),
                "evidence_timeline_coverage_ratio": round(
                    covered_seconds / window_seconds,
                    9,
                ),
                "accuracy_status": "not_evaluable",
            }
        )
        return result

    baseline_aggregate = aggregate("baseline")
    candidate_aggregate = aggregate("candidate")
    return {
        "schema_version": "factory_phase_b1_comparison_v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "comparison_kind": "frozen_phase_b_baseline_to_phase_b1_candidate",
        "clips": clips,
        "summary": {
            "baseline": baseline_aggregate,
            "candidate": candidate_aggregate,
            "delta_candidate_minus_baseline": _numeric_delta(
                candidate_aggregate,
                baseline_aggregate,
            ),
        },
        "all_gates_passed": all(clip["passed"] for clip in clips),
        "evaluation": {
            "status": "not_evaluable",
            "accuracy_metrics_computed": False,
            "reason": "No independently human-confirmed action ground truth",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_roots(
        args.baseline_root.resolve(),
        args.candidate_root.resolve(),
    )
    output = args.output.resolve()
    if PROJECT_ROOT not in output.parents:
        raise ValueError("Comparison output must stay inside the project workspace")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "clip_count": len(result["clips"]),
                "all_gates_passed": result["all_gates_passed"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
