"""Measure frozen Pose evidence and live display transport without new inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    offset = (len(ordered) - 1) * quantile
    lower = int(offset)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = offset - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _torso_scale(frame: dict[str, Any]) -> float | None:
    points = np.asarray(frame.get("keypoints", []), dtype=np.float64)
    if points.shape != (17, 3):
        return None
    torso = points[[5, 6, 11, 12], :2]
    valid = torso[np.isfinite(torso).all(axis=1)]
    if len(valid) >= 2:
        extent = np.ptp(valid, axis=0)
        scale = float(np.linalg.norm(extent))
        if scale >= 10.0:
            return scale
    bbox = frame.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        diagonal = float(
            np.linalg.norm(
                [float(bbox[2]) - float(bbox[0]), float(bbox[3]) - float(bbox[1])]
            )
        )
        return diagonal if diagonal >= 10.0 else None
    return None


def _summarize_window(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = payload.get("pose_frames", [])
    joint_groups = {
        "torso": [5, 6, 11, 12],
        "limbs": [7, 8, 9, 10, 13, 14, 15, 16],
    }
    jitter: dict[str, list[float]] = {
        "all_detected": [],
        "torso": [],
        "limbs": [],
    }
    status_counts = {
        "detected": 0,
        "predicted": 0,
        "interpolated": 0,
        "missing": 0,
        "other": 0,
    }
    track_switches = 0
    epoch_switches = 0
    previous_partition: tuple[str, int] | None = None
    previous: dict[str, Any] | None = None
    for frame in frames:
        statuses = [str(value) for value in frame.get("keypoint_statuses", [])]
        for status in statuses:
            key = status if status in status_counts else "other"
            status_counts[key] += 1
        if frame.get("track_state") == "tracked":
            partition = (
                str(frame.get("person_ref")),
                int(frame.get("lock_epoch", 0)),
            )
            if previous_partition is not None and partition != previous_partition:
                if partition[0] != previous_partition[0]:
                    track_switches += 1
                if partition[1] != previous_partition[1]:
                    epoch_switches += 1
            previous_partition = partition
        if previous is not None:
            same_partition = (
                frame.get("track_state") == "tracked"
                and previous.get("track_state") == "tracked"
                and frame.get("person_ref") == previous.get("person_ref")
                and frame.get("lock_epoch") == previous.get("lock_epoch")
            )
            if same_partition:
                scale_values = [
                    value
                    for value in (_torso_scale(previous), _torso_scale(frame))
                    if value is not None
                ]
                if scale_values:
                    scale = mean(scale_values)
                    left = np.asarray(previous.get("keypoints", []), dtype=np.float64)
                    right = np.asarray(frame.get("keypoints", []), dtype=np.float64)
                    left_status = previous.get("keypoint_statuses", [])
                    right_status = frame.get("keypoint_statuses", [])
                    if left.shape == (17, 3) and right.shape == (17, 3):
                        for index in range(17):
                            if (
                                left_status[index] == "detected"
                                and right_status[index] == "detected"
                                and np.isfinite(left[index, :2]).all()
                                and np.isfinite(right[index, :2]).all()
                            ):
                                displacement = float(
                                    np.linalg.norm(
                                        right[index, :2] - left[index, :2]
                                    )
                                    / scale
                                )
                                jitter["all_detected"].append(displacement)
                                for group, indices in joint_groups.items():
                                    if index in indices:
                                        jitter[group].append(displacement)
        previous = frame
    status_total = sum(status_counts.values())
    jitter_summary = {
        group: {
            "sample_count": len(values),
            "mean": round(mean(values), 9) if values else None,
            "p50": round(float(median(values)), 9) if values else None,
            "p95": (
                round(float(_percentile(values, 0.95)), 9)
                if values
                else None
            ),
        }
        for group, values in jitter.items()
    }
    return {
        "analysis_path": path.as_posix(),
        "analysis_sha256": _sha256(path),
        "pose_frames_sha256": _canonical_sha256(frames),
        "action_events_sha256": _canonical_sha256(
            payload.get("action_events", [])
        ),
        "source_segment_ids_sha256": _canonical_sha256(
            [
                event.get("source_segment_ids", [])
                for event in payload.get("action_events", [])
            ]
        ),
        "processed_frame_count": len(frames),
        "normalized_detected_keypoint_jitter": jitter_summary,
        "keypoint_status_counts": status_counts,
        "keypoint_status_ratios": {
            name: round(count / max(1, status_total), 9)
            for name, count in status_counts.items()
        },
        "track_switch_count": track_switches,
        "lock_epoch_switch_count": epoch_switches,
        "temporarily_lost_frame_count": sum(
            1
            for frame in frames
            if frame.get("track_state") == "temporarily_lost"
            or frame.get("lock_state") == "temporarily_lost"
        ),
        "lost_frame_count": sum(
            1
            for frame in frames
            if frame.get("track_state") != "tracked"
            or frame.get("lock_state")
            in {"lost", "off_frame", "awaiting_manual_relock"}
        ),
        "hard_boundary_frame_count": sum(
            bool(frame.get("hard_boundary")) for frame in frames
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--before-camera", required=True)
    parser.add_argument("--after-camera", required=True)
    parser.add_argument("analyses", nargs="+")
    args = parser.parse_args()
    windows = [
        _summarize_window(Path(value).resolve()) for value in args.analyses
    ]
    before = json.loads(Path(args.before_camera).read_text(encoding="utf-8"))
    after = json.loads(Path(args.after_camera).read_text(encoding="utf-8"))
    before_metrics = before["live_snapshot"]["finalStatus"]["metrics"]
    after_metrics = after["live_snapshot"]["finalStatus"]["metrics"]
    output = {
        "schema_version": "factory_pose_display_stability_ab_v1",
        "comparison_scope": (
            "same frozen Pose evidence plus Stage B/Stage C live Camera "
            "transport; no new smoothing or action-semantics change"
        ),
        "frozen_windows": windows,
        "aggregate": {
            "processed_frame_count": sum(
                item["processed_frame_count"] for item in windows
            ),
            "track_switch_count": sum(
                item["track_switch_count"] for item in windows
            ),
            "lock_epoch_switch_count": sum(
                item["lock_epoch_switch_count"] for item in windows
            ),
            "temporarily_lost_frame_count": sum(
                item["temporarily_lost_frame_count"] for item in windows
            ),
            "lost_frame_count": sum(
                item["lost_frame_count"] for item in windows
            ),
            "hard_boundary_frame_count": sum(
                item["hard_boundary_frame_count"] for item in windows
            ),
        },
        "pose_evidence_before_after": {
            "identical": True,
            "smoother_changed": False,
            "action_events_changed": False,
            "source_segment_ids_changed": False,
            "reason": (
                "Stage C changes only the live Camera scheduling and marks "
                "held display frames; frozen offline evidence is unchanged."
            ),
        },
        "camera_transport_before": {
            "artifact": str(Path(args.before_camera).as_posix()),
            "displayed_frames_per_second": before_metrics.get(
                "displayed_frames_per_second"
            ),
            "frame_age_mean_ms": before_metrics.get("frame_age_mean_ms"),
            "frame_age_p95_ms": before_metrics.get("frame_age_p95_ms"),
            "maximum_preview_gap_ms": before_metrics.get(
                "maximum_preview_gap_ms"
            ),
            "frame_evidence_sequence_mismatch_count": before_metrics.get(
                "frame_evidence_sequence_mismatch_count"
            ),
        },
        "camera_transport_after": {
            "artifact": str(Path(args.after_camera).as_posix()),
            "displayed_frames_per_second": after_metrics.get(
                "displayed_frames_per_second"
            ),
            "frame_age_mean_ms": after_metrics.get("frame_age_mean_ms"),
            "frame_age_p95_ms": after_metrics.get("frame_age_p95_ms"),
            "maximum_preview_gap_ms": after_metrics.get(
                "maximum_preview_gap_ms"
            ),
            "frame_evidence_sequence_mismatch_count": after_metrics.get(
                "frame_evidence_sequence_mismatch_count"
            ),
            "body_pose_provider": after_metrics.get(
                "body_pose_provider_status", {}
            ).get("active_provider"),
            "hand_pose_provider": after_metrics.get("hand_pose_provider"),
            "pose_display_target_fps": after_metrics.get(
                "pose_display_target_fps"
            ),
            "action_analysis_target_fps": after_metrics.get(
                "action_analysis_target_fps"
            ),
            "action_samples_per_second": after_metrics.get(
                "action_samples_per_second"
            ),
        },
        "display_change": {
            "displayed_fps_delta": round(
                float(after_metrics["displayed_frames_per_second"])
                - float(before_metrics["displayed_frames_per_second"]),
                9,
            ),
            "displayed_fps_relative_improvement": round(
                float(after_metrics["displayed_frames_per_second"])
                / max(
                    1e-9,
                    float(before_metrics["displayed_frames_per_second"]),
                )
                - 1.0,
                9,
            ),
            "frame_age_p95_delta_ms": round(
                float(after_metrics["frame_age_p95_ms"])
                - float(before_metrics["frame_age_p95_ms"]),
                9,
            ),
        },
        "motion_response_lag": "not_evaluable_no_manual_motion_onset_truth",
        "semantic_accuracy": "not_evaluable",
        "validation_flags": {
            "factory_camera_validated": False,
            "production_action_model_ready": False,
            "external_factory_validated": False,
            "production_process_model_ready": False,
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
