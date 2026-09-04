"""Classify real Hand failures and render Body-guided ROI evidence.

This diagnostic consumes accepted real-video outputs. It does not recompute or
invent Body/Hand geometry and never promotes an observation state.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable

import cv2
import numpy as np


EXPECTED_ROOT = Path(__file__).resolve().parents[1]
SIDES = ("left", "right")
BODY_EDGES = (
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)
HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _percentile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(
        float(value)
        for value in values
        if math.isfinite(float(value))
    )
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return round(ordered[low], 6)
    weight = position - low
    return round(
        ordered[low] * (1.0 - weight) + ordered[high] * weight,
        6,
    )


def _source_video(root: Path, analysis_path: Path, payload: dict[str, Any]) -> Path:
    local = analysis_path.parent / "source_video.mp4"
    if local.is_file():
        return local.resolve()
    raw = Path(str(payload["source_video"]["path"]))
    candidate = raw if raw.is_absolute() else root / raw
    if not candidate.is_file():
        raise FileNotFoundError(f"Real source video is missing: {candidate}")
    return candidate.resolve()


def _failure_category(record: dict[str, Any]) -> str:
    state = str(record.get("observation_state", "missing")).lower()
    reason = str(record.get("reason", ""))
    warnings = list(
        (record.get("association_checks") or {}).get("warnings", [])
    )
    landmarks = list(record.get("landmarks") or [])
    if state == "lost" or reason == "person_or_lock_hard_boundary":
        return "lost_or_person_boundary"
    if reason == "body_tracking_not_reliable_for_hand_roi":
        return "body_tracking_not_reliable"
    if record.get("crop_bbox") is None:
        return "body_guided_roi_unavailable"
    if str(record.get("backend_state", "")).lower() == "error":
        return "backend_error"
    if len(landmarks) == 21:
        if warnings or state == "uncertain":
            return "real_21_points_association_downgraded"
        return "real_21_points_qualified_or_detected"
    if reason == "no_hand_detected_in_body_guided_roi":
        return "roi_available_model_no_output"
    return "other_no_geometry"


def _metrics(records: list[dict[str, Any]], pose_frames: list[dict[str, Any]]) -> dict[str, Any]:
    failure = Counter(_failure_category(record) for record in records)
    observations = Counter(
        str(record.get("observation_state", "missing")).lower()
        for record in records
    )
    qualities = Counter(
        str(record.get("quality_state", "unknown")).lower()
        for record in records
    )
    warnings: Counter[str] = Counter()
    timings: list[float] = []
    crop_sizes: list[float] = []
    geometry_frames: set[int] = set()
    drawable_frames: set[int] = set()
    qualified_frames: set[int] = set()
    duplicate_frames: set[int] = set()
    per_side: dict[str, Any] = {}
    for record in records:
        warning_values = list(
            (record.get("association_checks") or {}).get("warnings", [])
        )
        warnings.update(str(value) for value in warning_values)
        elapsed = record.get("inference_time_ms")
        if isinstance(elapsed, (int, float)) and math.isfinite(float(elapsed)):
            timings.append(float(elapsed))
        bbox = record.get("crop_bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            crop_sizes.append(float(bbox[2]) - float(bbox[0]))
        landmarks = list(record.get("landmarks") or [])
        frame_index = int(record.get("frame_index", -1))
        if len(landmarks) == 21:
            geometry_frames.add(frame_index)
            if str(record.get("observation_state", "")).lower() in {
                "detected",
                "uncertain",
            }:
                drawable_frames.add(frame_index)
        if str(record.get("quality_state", "")).lower() == "qualified":
            qualified_frames.add(frame_index)
        if bool(
            (record.get("association_checks") or {}).get(
                "duplicate_across_sides", False
            )
        ):
            duplicate_frames.add(frame_index)

    for side in SIDES:
        selected = [
            record
            for record in records
            if str(record.get("anatomical_side", "")).lower() == side
        ]
        side_timings = [
            float(record["inference_time_ms"])
            for record in selected
            if isinstance(record.get("inference_time_ms"), (int, float))
        ]
        per_side[side] = {
            "record_count": len(selected),
            "roi_available_frame_count": sum(
                record.get("crop_bbox") is not None for record in selected
            ),
            "hand_inference_call_count": len(side_timings),
            "real_21_point_frame_count": sum(
                len(record.get("landmarks") or []) == 21
                for record in selected
            ),
            "detected": sum(
                str(record.get("observation_state", "")).lower() == "detected"
                for record in selected
            ),
            "uncertain": sum(
                str(record.get("observation_state", "")).lower() == "uncertain"
                for record in selected
            ),
            "missing": sum(
                str(record.get("observation_state", "")).lower() == "missing"
                for record in selected
            ),
            "lost": sum(
                str(record.get("observation_state", "")).lower() == "lost"
                for record in selected
            ),
            "qualified": sum(
                str(record.get("quality_state", "")).lower() == "qualified"
                for record in selected
            ),
            "association_uncertain": sum(
                str(record.get("quality_state", "")).lower()
                == "association_uncertain"
                for record in selected
            ),
            "association_warning_count": sum(
                len(
                    (record.get("association_checks") or {}).get(
                        "warnings", []
                    )
                )
                for record in selected
            ),
            "duplicate_side_frame_count": len(
                {
                    int(record.get("frame_index", -1))
                    for record in selected
                    if bool(
                        (record.get("association_checks") or {}).get(
                            "duplicate_across_sides", False
                        )
                    )
                }
            ),
            "visible_render_frame_count": len(
                {
                    int(record.get("frame_index", -1))
                    for record in selected
                    if len(record.get("landmarks") or []) == 21
                    and str(record.get("observation_state", "")).lower()
                    in {"detected", "uncertain"}
                }
            ),
            "inference_mean_ms": (
                round(statistics.fmean(side_timings), 6)
                if side_timings else None
            ),
            "inference_p50_ms": _percentile(side_timings, 0.5),
            "inference_p95_ms": _percentile(side_timings, 0.95),
        }

    return {
        "pose_frame_count": len(pose_frames),
        "hand_record_count": len(records),
        "failure_reason_counts": dict(sorted(failure.items())),
        "observation_state_counts": dict(sorted(observations.items())),
        "quality_state_counts": dict(sorted(qualities.items())),
        "association_warning_counts": dict(sorted(warnings.items())),
        "roi_available_observation_count": sum(
            record.get("crop_bbox") is not None for record in records
        ),
        "roi_unavailable_observation_count": sum(
            record.get("crop_bbox") is None for record in records
        ),
        "hand_inference_call_count": len(timings),
        "real_21_point_observation_count": sum(
            len(record.get("landmarks") or []) == 21 for record in records
        ),
        "real_21_point_frame_count": len(geometry_frames),
        "visible_render_frame_count": len(drawable_frames),
        "qualified_frame_count": len(qualified_frames),
        "duplicate_side_frame_count": len(duplicate_frames),
        "association_warning_occurrence_count": sum(warnings.values()),
        "crop_size_pixels": {
            "minimum": round(min(crop_sizes), 3) if crop_sizes else None,
            "p50": _percentile(crop_sizes, 0.5),
            "p95": _percentile(crop_sizes, 0.95),
            "maximum": round(max(crop_sizes), 3) if crop_sizes else None,
        },
        "inference_mean_ms": (
            round(statistics.fmean(timings), 6) if timings else None
        ),
        "inference_p50_ms": _percentile(timings, 0.5),
        "inference_p95_ms": _percentile(timings, 0.95),
        "per_side": per_side,
        "ui_draw_rule": (
            "detected_or_uncertain_and_exactly_21_real_points"
        ),
        "ui_hidden_real_21_point_observation_count": sum(
            len(record.get("landmarks") or []) == 21
            and str(record.get("observation_state", "")).lower()
            not in {"detected", "uncertain"}
            for record in records
        ),
        "precision_recall_f1": "not_evaluable",
    }


def _point(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, dict):
        return None
    x, y = value.get("x"), value.get("y")
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return None
    return int(round(float(x))), int(round(float(y)))


def _render(
    frame: np.ndarray,
    pose: dict[str, Any],
    records: list[dict[str, Any]],
    label: str,
) -> np.ndarray:
    rendered = frame.copy()
    keypoints = list(pose.get("keypoints") or [])
    statuses = list(pose.get("keypoint_statuses") or [])

    def body_point(index: int) -> tuple[int, int] | None:
        if index >= len(keypoints):
            return None
        raw = keypoints[index]
        if not isinstance(raw, list) or len(raw) < 2:
            return None
        state = statuses[index] if index < len(statuses) else "missing"
        if str(state).lower() in {"missing", "lost", "rejected"}:
            return None
        if raw[0] is None or raw[1] is None:
            return None
        return int(round(float(raw[0]))), int(round(float(raw[1])))

    for first, second in BODY_EDGES:
        left, right = body_point(first), body_point(second)
        if left is not None and right is not None:
            cv2.line(rendered, left, right, (255, 220, 40), 2, cv2.LINE_AA)

    side_colors = {"left": (255, 80, 220), "right": (40, 190, 255)}
    for record in records:
        side = str(record.get("anatomical_side", ""))
        color = side_colors.get(side, (220, 220, 220))
        bbox = record.get("crop_bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            cv2.rectangle(
                rendered,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                color,
                2,
                cv2.LINE_AA,
            )
        landmarks = list(record.get("landmarks") or [])
        for first, second in HAND_EDGES:
            if first >= len(landmarks) or second >= len(landmarks):
                continue
            left, right = _point(landmarks[first]), _point(landmarks[second])
            if left is not None and right is not None:
                cv2.line(rendered, left, right, color, 2, cv2.LINE_AA)
        for landmark in landmarks:
            location = _point(landmark)
            if location is not None:
                cv2.circle(rendered, location, 2, color, -1, cv2.LINE_AA)

    rendered = cv2.resize(rendered, (640, 360), interpolation=cv2.INTER_AREA)
    cv2.rectangle(rendered, (0, 0), (640, 44), (10, 18, 28), -1)
    cv2.putText(
        rendered,
        label[:92],
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (238, 244, 248),
        1,
        cv2.LINE_AA,
    )
    return rendered


def _contact_sheet(
    root: Path,
    analyses: list[tuple[Path, dict[str, Any], Path]],
    output: Path,
) -> dict[str, Any]:
    panels: list[np.ndarray] = []
    selections: list[dict[str, Any]] = []
    for analysis_path, payload, video_path in analyses:
        pose_by_index = {
            int(frame["source_frame_index"]): frame
            for frame in payload.get("pose_frames", [])
        }
        hand_by_index: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in payload.get("hand_pose_frames", []):
            hand_by_index[int(record["frame_index"])].append(record)
        desired = (
            "real_21_points_association_downgraded",
            "roi_available_model_no_output",
            "body_guided_roi_unavailable",
        )
        selected_indices: list[tuple[int, str]] = []
        for category in desired:
            for frame_index, records in sorted(hand_by_index.items()):
                if frame_index in {item[0] for item in selected_indices}:
                    continue
                if category in {_failure_category(item) for item in records}:
                    selected_indices.append((frame_index, category))
                    break
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open real source video: {video_path}")
        try:
            for frame_index, category in selected_indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue
                records = hand_by_index[frame_index]
                label = (
                    f"{analysis_path.parent.parent.name} f={frame_index} "
                    f"{category}"
                )
                panels.append(
                    _render(frame, pose_by_index[frame_index], records, label)
                )
                selections.append(
                    {
                        "analysis": str(analysis_path.relative_to(root)),
                        "source_frame_index": frame_index,
                        "timestamp": pose_by_index[frame_index]["timestamp"],
                        "category": category,
                        "hand_pose_ids": [
                            record.get("hand_pose_id") for record in records
                        ],
                    }
                )
        finally:
            capture.release()
    if not panels:
        raise RuntimeError("No real contact-sheet panels were decoded")
    columns = 3
    rows: list[np.ndarray] = []
    blank = np.zeros_like(panels[0])
    for index in range(0, len(panels), columns):
        row = panels[index:index + columns]
        while len(row) < columns:
            row.append(blank.copy())
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)
    if not cv2.imwrite(str(output), sheet):
        raise RuntimeError(f"Could not write contact sheet: {output}")
    return {
        "path": str(output.relative_to(root)),
        "sha256": _sha256(output),
        "panel_count": len(panels),
        "selections": selections,
        "truthfulness": {
            "mock_landmarks": False,
            "old_hand_geometry_reused": False,
            "missing_geometry_drawn": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", nargs=3, type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    if root != EXPECTED_ROOT:
        raise RuntimeError(f"Workspace gate mismatch: {root}")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    analyses: list[tuple[Path, dict[str, Any], Path]] = []
    aggregate_records: list[dict[str, Any]] = []
    aggregate_pose: list[dict[str, Any]] = []
    clips: list[dict[str, Any]] = []
    for raw_path in args.analysis:
        analysis_path = raw_path.resolve()
        payload = _json(analysis_path)
        source = _source_video(root, analysis_path, payload)
        expected_hash = str(payload["source_video"]["sha256"])
        actual_hash = _sha256(source)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Source SHA256 mismatch for {source}: {actual_hash}"
            )
        records = list(payload.get("hand_pose_frames", []))
        pose_frames = list(payload.get("pose_frames", []))
        metrics = _metrics(records, pose_frames)
        clips.append(
            {
                "clip_id": analysis_path.parent.parent.name,
                "analysis": str(analysis_path.relative_to(root)),
                "analysis_sha256": _sha256(analysis_path),
                "source_video": str(source.relative_to(root)),
                "source_video_sha256": actual_hash,
                "analysis_window": payload["source_video"]["analysis_window"],
                "metrics": metrics,
                "pose_frame_core_sha256": hashlib.sha256(
                    json.dumps(
                        pose_frames,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
                "action_events_sha256": hashlib.sha256(
                    json.dumps(
                        payload.get("action_events", []),
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            }
        )
        analyses.append((analysis_path, payload, source))
        aggregate_records.extend(records)
        aggregate_pose.extend(pose_frames)

    failure_payload = {
        "schema_version": "factory_hand_failure_reason_counts_v1",
        "run_id": output_root.name,
        "clips": clips,
        "aggregate": _metrics(aggregate_records, aggregate_pose),
        "classification_notes": {
            "ui_draws_uncertain_real_21_points": True,
            "missing_or_lost_draws_geometry": False,
            "body_cuda_is_not_hand_cuda": True,
            "precision_recall_f1": "not_evaluable",
        },
    }
    failure_path = output_root / "HAND_FAILURE_REASON_COUNTS.json"
    failure_path.write_text(
        json.dumps(failure_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    contact = _contact_sheet(
        root,
        analyses,
        output_root / "HAND_ROI_CONTACT_SHEET.jpg",
    )
    historical = _json(
        root
        / "outputs/private_regression/pose_gpu_latency/"
        "REAL_USB_CAMERA_GPU_VALIDATION_FINAL.json"
    )
    camera_config = _json(root / "configs/camera.json")
    baseline = {
        "schema_version": "factory_baseline_hand_latency_diagnostic_v1",
        "run_id": output_root.name,
        "project_path": str(root),
        "body_model": {
            "path": "models/yolov8n-pose.onnx",
            "sha256": _sha256(root / "models/yolov8n-pose.onnx"),
            "configured_policy": camera_config["body_provider_policy"],
            "historical_active_provider": (
                historical["live_snapshot"]["finalStatus"]["metrics"]
                ["body_pose_provider_status"]["active_provider"]
            ),
            "historical_fallback_active": (
                historical["live_snapshot"]["finalStatus"]["metrics"]
                ["body_pose_provider_status"]["fallback_active"]
            ),
        },
        "hand_model": {
            "path": "models/hand_pose/hand_landmarker.task",
            "sha256": _sha256(
                root / "models/hand_pose/hand_landmarker.task"
            ),
            "provider": "CPU",
            "backend_mode": "image",
            "landmarks_per_real_hand": 21,
        },
        "camera_config": camera_config,
        "code_audit": {
            "full_resolution_roi_input": True,
            "left_and_right_inference_serial_in_same_body_frame": True,
            "hand_inference_after_body_inference": True,
            "ui_draws_detected_real_21_points": True,
            "ui_draws_uncertain_real_21_points_as_dashed_transparent": True,
            "ui_hides_missing_or_lost_geometry": True,
            "current_camera_hand_sampling": (
                "every processed Body display frame when enabled"
            ),
            "not_sampled_state_present": False,
        },
        "same_real_window_aggregate": failure_payload["aggregate"],
        "historical_real_usb": {
            "requested_resolution": [
                camera_config["requested_width"],
                camera_config["requested_height"],
            ],
            "requested_fps": camera_config["requested_fps"],
            "displayed_fps": historical["live_snapshot"]["finalStatus"]
            ["metrics"]["displayed_frames_per_second"],
            "frame_age_p95_ms": historical["live_snapshot"]["finalStatus"]
            ["metrics"]["frame_age_p95_ms"],
            "maximum_preview_gap_ms": historical["live_snapshot"]
            ["finalStatus"]["metrics"]["maximum_preview_gap_ms"],
            "mean_body_inference_ms": historical["live_snapshot"]
            ["finalStatus"]["metrics"]["mean_pose_inference_ms"],
            "p95_body_inference_ms": historical["live_snapshot"]
            ["finalStatus"]["metrics"]["p95_pose_inference_ms"],
            "hand_inference_mean_ms": historical["live_snapshot"]
            ["finalStatus"]["metrics"]["mean_hand_inference_ms"],
            "hand_provider": historical["live_snapshot"]["finalStatus"]
            ["metrics"]["hand_pose_provider"],
            "sequence_mismatch_count": historical["live_snapshot"]
            ["finalStatus"]["metrics"]
            ["frame_evidence_sequence_mismatch_count"],
        },
        "contact_sheet": contact,
        "camera_hand_on_off": "pending_current_run",
        "cold_start_vs_steady_state": "pending_current_run",
        "accuracy": "not_evaluable",
        "validation_flags": {
            "factory_camera_validated": False,
            "production_action_model_ready": False,
            "external_factory_validated": False,
            "production_process_model_ready": False,
        },
    }
    baseline_path = output_root / "BASELINE_HAND_LATENCY_DIAGNOSTIC.json"
    baseline_path.write_text(
        json.dumps(baseline, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "baseline": str(baseline_path),
                "failures": str(failure_path),
                "contact_sheet": contact,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
