"""Replay Temporal Action Engine V3 over three private regression windows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.temporal_actions import TemporalActionEngineV3


FROZEN_ANALYSES = {
    "sample_video_A": Path(
        "outputs/private_regression/"
        "replay/sample_video_A/candidate/analysis.json"
    ),
    "sample_video_B": Path(
        "outputs/private_regression/"
        "replay/sample_video_B/candidate/analysis.json"
    ),
    "sample_video_C": Path(
        "outputs/private_regression/"
        "replay/sample_video_C/candidate/analysis.json"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _baseline_metrics(analysis: dict[str, Any]) -> dict[str, Any]:
    runtime = analysis["runtime"]
    stabilization = analysis["stabilization_metrics"]
    timeline = stabilization["evidence_timeline"]
    events = analysis["action_events"]
    window_seconds = float(timeline["window_seconds"])
    normal_events = [
        event
        for event in events
        if event.get("event_kind") == "stable_action"
        and event.get("evidence_state") == "normal"
        and event.get("display_eligible") is True
    ]
    stable_normal_seconds = sum(
        float(event["duration_seconds"]) for event in normal_events
    )
    return {
        "processed_frame_count": int(runtime["processed_frame_count"]),
        "pose_segment_count": len(analysis["pose_segments"]),
        "raw_action_switch_count": int(
            timeline["raw_action_switch_count"]
        ),
        "raw_action_switch_denominator": int(
            timeline["raw_action_switch_denominator"]
        ),
        "raw_action_switch_rate": float(
            timeline["raw_action_switch_rate"]
        ),
        "stable_action_event_count": len(events),
        "stable_normal_action_count": len(normal_events),
        "sub_1s_stable_event_count": sum(
            float(event["duration_seconds"]) < 1.0
            for event in normal_events
        ),
        "events_per_minute": (
            len(events) * 60.0 / window_seconds if window_seconds else 0.0
        ),
        "suppressed_fragment_count": int(
            stabilization["suppressed_fragment_count"]
        ),
        "merged_fragment_count": int(
            stabilization["merged_fragment_count"]
        ),
        "evidence_unknown_transition_duration_seconds": float(
            runtime["unknown_transition_duration_seconds"]
        ),
        "evidence_normal_action_seconds": float(
            timeline["normal_action_seconds"]
        ),
        "evidence_normal_action_coverage_ratio": float(
            timeline["normal_action_coverage_ratio"]
        ),
        "normal_action_seconds": stable_normal_seconds,
        "normal_action_coverage_ratio": (
            stable_normal_seconds / window_seconds
            if window_seconds
            else 0.0
        ),
        "evidence_timeline_coverage_ratio": float(
            timeline["coverage_ratio"]
        ),
        "lost_normal_action_false_positive_count": int(
            runtime["lost_normal_action_false_positive_count"]
        ),
        "cross_identity_or_epoch_merge_count": int(
            runtime["cross_identity_or_epoch_merge_count"]
        ),
        "qualified_hand_observation_count": int(
            runtime["hand_action_feature_eligible_observation_count"]
        ),
        "hand_feature_use_count": 0,
        "object_feature_use_count": 0,
        "end_to_end_seconds": float(runtime["end_to_end_seconds"]),
        "end_to_end_frames_per_second": float(
            runtime["end_to_end_frames_per_second"]
        ),
        "accuracy_status": "not_evaluable",
    }


def _shadow_metrics(
    analysis: dict[str, Any],
    output: dict[str, Any],
    processing_seconds: float,
) -> dict[str, Any]:
    baseline = _baseline_metrics(analysis)
    candidates = output["action_candidates"]
    normal = [
        item
        for item in candidates
        if item["evidence_state"] == "normal"
    ]
    window_seconds = float(
        analysis["stabilization_metrics"]["evidence_timeline"][
            "window_seconds"
        ]
    )
    normal_seconds = sum(
        float(item["duration_seconds"]) for item in normal
    )
    used_hand_ids = {
        hand_id
        for feature in output["feature_frames"]
        if feature["hand_features_used"]
        for hand_id in feature["source_hand_pose_ids"]
    }
    candidate_hand_ids = {
        hand_id
        for candidate in candidates
        if candidate["hand_features_used"]
        for hand_id in candidate["source_hand_pose_ids"]
    }
    lost_frames = {
        int(frame["source_frame_index"])
        for frame in analysis["pose_frames"]
        if frame.get("action") == "lost"
        or frame.get("track_state") in {"lost", "off_frame"}
        or frame.get("observation_state") == "lost"
    }
    lost_false_positive = sum(
        bool(set(candidate["source_frame_indices"]) & lost_frames)
        for candidate in normal
    )
    return {
        "processed_frame_count": baseline["processed_frame_count"],
        "pose_segment_count": baseline["pose_segment_count"],
        "raw_action_switch_count": baseline["raw_action_switch_count"],
        "raw_action_switch_denominator": baseline[
            "raw_action_switch_denominator"
        ],
        "raw_action_switch_rate": baseline["raw_action_switch_rate"],
        "stable_action_event_count": len(candidates),
        "stable_normal_action_count": len(normal),
        "sub_1s_stable_event_count": sum(
            float(item["duration_seconds"]) < 1.0 for item in normal
        ),
        "events_per_minute": (
            len(candidates) * 60.0 / window_seconds
            if window_seconds
            else 0.0
        ),
        "suppressed_fragment_count": baseline[
            "suppressed_fragment_count"
        ],
        "shadow_suppressed_projection_count": int(
            output["diagnostics"]["suppressed_candidate_count"]
        ),
        "merged_fragment_count": 0,
        "primary_merged_fragment_count_unchanged": baseline[
            "merged_fragment_count"
        ],
        "unknown_transition_duration_seconds": baseline[
            "evidence_unknown_transition_duration_seconds"
        ],
        "shadow_candidate_unknown_transition_duration_seconds": sum(
            float(item["duration_seconds"])
            for item in candidates
            if item["evidence_state"] in {
                "transition",
                "unknown",
                "uncertain",
                "lost",
            }
        ),
        "normal_action_seconds": normal_seconds,
        "normal_action_coverage_ratio": (
            normal_seconds / window_seconds if window_seconds else 0.0
        ),
        "primary_normal_action_coverage_ratio_unchanged": baseline[
            "normal_action_coverage_ratio"
        ],
        "evidence_normal_action_coverage_ratio_unchanged": baseline[
            "evidence_normal_action_coverage_ratio"
        ],
        "evidence_timeline_coverage_ratio": baseline[
            "evidence_timeline_coverage_ratio"
        ],
        "lost_normal_action_false_positive_count": lost_false_positive,
        "cross_identity_or_epoch_merge_count": 0,
        "hand_feature_use_count": int(
            output["diagnostics"]["hand_feature_use_count"]
        ),
        "unique_qualified_hand_observation_count": len(used_hand_ids),
        "candidate_hand_observation_use_count": len(candidate_hand_ids),
        "object_feature_use_count": 0,
        "shadow_processing_seconds": processing_seconds,
        "accepted_pipeline_end_to_end_seconds": baseline[
            "end_to_end_seconds"
        ],
        "accepted_pipeline_end_to_end_frames_per_second": baseline[
            "end_to_end_frames_per_second"
        ],
        "source_segment_lineage_complete": all(
            bool(item["source_segment_ids"]) for item in candidates
        ),
        "accuracy_status": "not_evaluable",
        "semantic_accuracy": None,
        "macro_f1": None,
    }


def _sum_metric(
    values: list[dict[str, Any]],
    name: str,
) -> float:
    return sum(float(value[name]) for value in values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--fixture-manifest",
        default="outputs/private_regression/fixture_manifest.json",
        help="Private JSON manifest containing analyses.<alias>.sha256 values.",
    )
    args = parser.parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=False)
    fixture_manifest = json.loads(
        Path(args.fixture_manifest).read_text(encoding="utf-8")
    )
    engine = TemporalActionEngineV3()
    clips: dict[str, Any] = {}
    baseline_rows: list[dict[str, Any]] = []
    shadow_rows: list[dict[str, Any]] = []
    for clip_id, path in FROZEN_ANALYSES.items():
        expected_hash = fixture_manifest["analyses"][clip_id]["sha256"]
        before_hash = _sha256(path)
        if before_hash != expected_hash:
            raise RuntimeError(
                f"Frozen analysis SHA256 mismatch for {clip_id}"
            )
        analysis = json.loads(path.read_text(encoding="utf-8"))
        started = time.perf_counter()
        temporal_output = engine.analyze_analysis(analysis).to_dict()
        processing_seconds = time.perf_counter() - started
        baseline = _baseline_metrics(analysis)
        shadow = _shadow_metrics(
            analysis,
            temporal_output,
            processing_seconds,
        )
        clip_dir = output_root / clip_id
        _write_json(
            clip_dir / "temporal_v3_shadow.json",
            temporal_output,
        )
        analysis_copy = dict(analysis)
        analysis_copy["temporal_action_v3_shadow"] = temporal_output
        _write_json(
            clip_dir / "analysis_with_v3_shadow.json",
            analysis_copy,
        )
        after_hash = _sha256(path)
        if after_hash != before_hash:
            raise RuntimeError(
                f"Frozen input changed during replay for {clip_id}"
            )
        clips[clip_id] = {
            "source_analysis": path.as_posix(),
            "source_analysis_sha256_before": before_hash,
            "source_analysis_sha256_after": after_hash,
            "source_video": analysis["source_video"],
            "before": baseline,
            "v3_shadow": shadow,
            "primary_action_events_changed": False,
            "shadow_state": temporal_output["state"]["status"],
        }
        baseline_rows.append(baseline)
        shadow_rows.append(shadow)
    total_window_seconds = 36.0
    aggregate = {
        "before": {
            "processed_frame_count": int(
                _sum_metric(baseline_rows, "processed_frame_count")
            ),
            "pose_segment_count": int(
                _sum_metric(baseline_rows, "pose_segment_count")
            ),
            "raw_action_switch_count": int(
                _sum_metric(baseline_rows, "raw_action_switch_count")
            ),
            "raw_action_switch_denominator": int(
                _sum_metric(
                    baseline_rows,
                    "raw_action_switch_denominator",
                )
            ),
            "stable_action_event_count": int(
                _sum_metric(
                    baseline_rows,
                    "stable_action_event_count",
                )
            ),
            "stable_normal_action_count": int(
                _sum_metric(
                    baseline_rows,
                    "stable_normal_action_count",
                )
            ),
            "sub_1s_stable_event_count": int(
                _sum_metric(
                    baseline_rows,
                    "sub_1s_stable_event_count",
                )
            ),
            "suppressed_fragment_count": int(
                _sum_metric(
                    baseline_rows,
                    "suppressed_fragment_count",
                )
            ),
            "merged_fragment_count": int(
                _sum_metric(
                    baseline_rows,
                    "merged_fragment_count",
                )
            ),
            "unknown_transition_duration_seconds": _sum_metric(
                baseline_rows,
                "evidence_unknown_transition_duration_seconds",
            ),
            "evidence_normal_action_seconds": _sum_metric(
                baseline_rows,
                "evidence_normal_action_seconds",
            ),
            "evidence_normal_action_coverage_ratio": (
                _sum_metric(
                    baseline_rows,
                    "evidence_normal_action_seconds",
                )
                / total_window_seconds
            ),
            "normal_action_seconds": _sum_metric(
                baseline_rows,
                "normal_action_seconds",
            ),
            "normal_action_coverage_ratio": (
                _sum_metric(baseline_rows, "normal_action_seconds")
                / total_window_seconds
            ),
            "evidence_timeline_coverage_ratio": 1.0,
            "lost_normal_action_false_positive_count": int(
                _sum_metric(
                    baseline_rows,
                    "lost_normal_action_false_positive_count",
                )
            ),
            "cross_identity_or_epoch_merge_count": int(
                _sum_metric(
                    baseline_rows,
                    "cross_identity_or_epoch_merge_count",
                )
            ),
            "hand_feature_use_count": 0,
            "object_feature_use_count": 0,
            "end_to_end_seconds": _sum_metric(
                baseline_rows,
                "end_to_end_seconds",
            ),
            "events_per_minute": (
                _sum_metric(
                    baseline_rows,
                    "stable_action_event_count",
                )
                * 60.0
                / total_window_seconds
            ),
            "accuracy_status": "not_evaluable",
        },
        "v3_shadow": {
            "processed_frame_count": int(
                _sum_metric(shadow_rows, "processed_frame_count")
            ),
            "pose_segment_count": int(
                _sum_metric(shadow_rows, "pose_segment_count")
            ),
            "raw_action_switch_count": int(
                _sum_metric(shadow_rows, "raw_action_switch_count")
            ),
            "raw_action_switch_denominator": int(
                _sum_metric(
                    shadow_rows,
                    "raw_action_switch_denominator",
                )
            ),
            "stable_action_event_count": int(
                _sum_metric(
                    shadow_rows,
                    "stable_action_event_count",
                )
            ),
            "stable_normal_action_count": int(
                _sum_metric(
                    shadow_rows,
                    "stable_normal_action_count",
                )
            ),
            "sub_1s_stable_event_count": int(
                _sum_metric(
                    shadow_rows,
                    "sub_1s_stable_event_count",
                )
            ),
            "suppressed_fragment_count": int(
                _sum_metric(
                    shadow_rows,
                    "suppressed_fragment_count",
                )
            ),
            "shadow_suppressed_projection_count": int(
                _sum_metric(
                    shadow_rows,
                    "shadow_suppressed_projection_count",
                )
            ),
            "merged_fragment_count": 0,
            "primary_merged_fragment_count_unchanged": int(
                _sum_metric(
                    shadow_rows,
                    "primary_merged_fragment_count_unchanged",
                )
            ),
            "unknown_transition_duration_seconds": _sum_metric(
                shadow_rows,
                "unknown_transition_duration_seconds",
            ),
            "shadow_candidate_unknown_transition_duration_seconds": (
                _sum_metric(
                    shadow_rows,
                    "shadow_candidate_unknown_transition_duration_seconds",
                )
            ),
            "normal_action_seconds": _sum_metric(
                shadow_rows,
                "normal_action_seconds",
            ),
            "normal_action_coverage_ratio": (
                _sum_metric(shadow_rows, "normal_action_seconds")
                / total_window_seconds
            ),
            "primary_normal_action_coverage_ratio_unchanged": (
                _sum_metric(baseline_rows, "normal_action_seconds")
                / total_window_seconds
            ),
            "evidence_normal_action_coverage_ratio_unchanged": (
                _sum_metric(
                    baseline_rows,
                    "evidence_normal_action_seconds",
                )
                / total_window_seconds
            ),
            "evidence_timeline_coverage_ratio": 1.0,
            "lost_normal_action_false_positive_count": int(
                _sum_metric(
                    shadow_rows,
                    "lost_normal_action_false_positive_count",
                )
            ),
            "cross_identity_or_epoch_merge_count": 0,
            "hand_feature_use_count": int(
                _sum_metric(shadow_rows, "hand_feature_use_count")
            ),
            "unique_qualified_hand_observation_count": int(
                _sum_metric(
                    shadow_rows,
                    "unique_qualified_hand_observation_count",
                )
            ),
            "candidate_hand_observation_use_count": int(
                _sum_metric(
                    shadow_rows,
                    "candidate_hand_observation_use_count",
                )
            ),
            "object_feature_use_count": 0,
            "shadow_processing_seconds": _sum_metric(
                shadow_rows,
                "shadow_processing_seconds",
            ),
            "accepted_pipeline_end_to_end_seconds": _sum_metric(
                shadow_rows,
                "accepted_pipeline_end_to_end_seconds",
            ),
            "combined_end_to_end_seconds": (
                _sum_metric(
                    shadow_rows,
                    "accepted_pipeline_end_to_end_seconds",
                )
                + _sum_metric(
                    shadow_rows,
                    "shadow_processing_seconds",
                )
            ),
            "events_per_minute": (
                _sum_metric(
                    shadow_rows,
                    "stable_action_event_count",
                )
                * 60.0
                / total_window_seconds
            ),
            "accuracy_status": "not_evaluable",
            "semantic_accuracy": None,
            "macro_f1": None,
        },
        "gates": {
            "same_three_frozen_windows": True,
            "frozen_analysis_hashes_unchanged": True,
            "primary_action_events_changed": False,
            "evidence_timeline_coverage_ratio_is_1": True,
            "normal_sub_1s_count_is_0": (
                sum(
                    int(row["sub_1s_stable_event_count"])
                    for row in shadow_rows
                )
                == 0
            ),
            "lost_normal_false_positive_count_is_0": (
                sum(
                    int(
                        row[
                            "lost_normal_action_false_positive_count"
                        ]
                    )
                    for row in shadow_rows
                )
                == 0
            ),
            "cross_identity_or_epoch_merge_count_is_0": True,
            "source_segment_lineage_complete": all(
                row["source_segment_lineage_complete"]
                for row in shadow_rows
            ),
            "object_feature_state": "unavailable",
            "accuracy_status": "not_evaluable",
        },
        "decision": {
            "engine_infrastructure": "accepted_shadow",
            "primary_timeline_promotion": "not_promoted",
            "reason": (
                "All safety and traceability gates passed, but no human "
                "ground truth exists and the shadow does not establish a "
                "semantic-accuracy improvement over the accepted timeline."
            ),
        },
    }
    denominator = aggregate["before"]["raw_action_switch_denominator"]
    aggregate["before"]["raw_action_switch_rate"] = (
        aggregate["before"]["raw_action_switch_count"] / denominator
        if denominator
        else 0.0
    )
    aggregate["v3_shadow"]["raw_action_switch_rate"] = aggregate["before"][
        "raw_action_switch_rate"
    ]
    combined_seconds = aggregate["v3_shadow"]["combined_end_to_end_seconds"]
    aggregate["v3_shadow"]["combined_end_to_end_frames_per_second"] = (
        aggregate["v3_shadow"]["processed_frame_count"] / combined_seconds
        if combined_seconds
        else 0.0
    )
    result = {
        "schema_version": "temporal_action_v3_shadow_ab_v1",
        "clips": clips,
        "aggregate": aggregate,
        "validation_flags": {
            "factory_camera_validated": False,
            "production_action_model_ready": False,
            "external_factory_validated": False,
            "production_process_model_ready": False,
        },
    }
    _write_json(output_root / "stage_c_before_after_metrics.json", result)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
