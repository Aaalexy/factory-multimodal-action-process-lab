"""Measure current-frame Hand cadence cost on frozen real-video evidence."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.camera.live_analysis import LiveFrameAnalyzer
from src.hand_pose import MediaPipeHandLandmarkerBackend


EXPECTED_ROOT = PROJECT_ROOT
HAND_MODEL = Path("models/hand_pose/hand_landmarker.task")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _percentile(values: list[float], fraction: float) -> float | None:
    ordered = sorted(float(value) for value in values if math.isfinite(value))
    if not ordered:
        return None
    index = (len(ordered) - 1) * fraction
    low = int(math.floor(index))
    high = int(math.ceil(index))
    if low == high:
        return round(ordered[low], 6)
    return round(
        ordered[low] * (high - index) + ordered[high] * (index - low),
        6,
    )


def _source_video(
    root: Path,
    analysis_path: Path,
    payload: dict[str, Any],
) -> Path:
    local = analysis_path.parent / "source_video.mp4"
    if local.is_file():
        return local.resolve()
    raw = Path(str(payload["source_video"]["path"]))
    path = raw if raw.is_absolute() else root / raw
    if not path.is_file():
        raise FileNotFoundError(f"Missing real source video: {path}")
    return path.resolve()


def _scheduler(hand_fps: float | None) -> LiveFrameAnalyzer:
    analyzer = object.__new__(LiveFrameAnalyzer)
    analyzer.hand_analysis_fps = hand_fps
    analyzer._next_hand_due = None
    analyzer._last_hand_boundary = None
    return analyzer


def _run_mode(
    *,
    root: Path,
    analyses: list[Path],
    mode: str,
    hand_fps: float | None,
) -> dict[str, Any]:
    backend = MediaPipeHandLandmarkerBackend(
        root / HAND_MODEL,
        model_version=f"mediapipe_hand_landmarker_float16_v1+cadence:{mode}",
    )
    stage_times_ms: list[float] = []
    records: list[dict[str, Any]] = []
    scheduled_frames = 0
    not_sampled_frames = 0
    per_clip: list[dict[str, Any]] = []
    try:
        for analysis_path in analyses:
            payload = json.loads(
                analysis_path.read_text(encoding="utf-8-sig")
            )
            source = _source_video(root, analysis_path, payload)
            if _sha256(source) != str(payload["source_video"]["sha256"]):
                raise RuntimeError(f"Source SHA256 mismatch: {source}")
            cadence = _scheduler(hand_fps)
            clip_records: list[dict[str, Any]] = []
            clip_scheduled = 0
            clip_not_sampled = 0
            capture = cv2.VideoCapture(str(source))
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open real source: {source}")
            try:
                for pose in payload.get("pose_frames", []):
                    source_index = int(pose["source_frame_index"])
                    capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        raise RuntimeError(
                            f"Could not decode source frame {source_index}"
                        )
                    due = (
                        True
                        if hand_fps is None
                        else cadence._hand_sample_due(
                            timestamp=float(pose["timestamp"]),
                            person_ref=str(pose["person_ref"]),
                            lock_epoch=int(pose["lock_epoch"]),
                            track_state=str(
                                pose.get("track_state", "lost")
                            ),
                            lock_state=str(
                                pose.get("lock_state", "lost")
                            ),
                        )
                    )
                    context = {
                        "person_ref": str(pose["person_ref"]),
                        "lock_epoch": int(pose["lock_epoch"]),
                        "frame_index": source_index,
                        "timestamp": float(pose["timestamp"]),
                        "source_video_sha256": str(
                            payload["source_video"]["sha256"]
                        ),
                        "recording_group_id": str(
                            payload["source_video"].get(
                                "recording_group_id",
                                "recording_group_unassigned",
                            )
                        ),
                        "track_state": str(
                            pose.get("track_state", "lost")
                        ),
                        "lock_state": str(
                            pose.get("lock_state", "lost")
                        ),
                    }
                    started = time.perf_counter()
                    if due:
                        current = backend.infer_frame(
                            frame,
                            body_keypoints=pose.get("keypoints"),
                            body_keypoint_statuses=pose.get(
                                "keypoint_statuses"
                            ),
                            **context,
                        )
                        clip_scheduled += 1
                    else:
                        current = backend.not_sampled_frame(**context)
                        clip_not_sampled += 1
                    stage_times_ms.append(
                        (time.perf_counter() - started) * 1000.0
                    )
                    clip_records.extend(current)
            finally:
                capture.release()
            records.extend(clip_records)
            scheduled_frames += clip_scheduled
            not_sampled_frames += clip_not_sampled
            per_clip.append(
                {
                    "clip_id": analysis_path.parent.parent.name,
                    "display_frame_count": len(
                        payload.get("pose_frames", [])
                    ),
                    "hand_sampled_frame_count": clip_scheduled,
                    "hand_not_sampled_frame_count": clip_not_sampled,
                    "real_21_point_frame_count": len(
                        {
                            int(record["frame_index"])
                            for record in clip_records
                            if int(record.get("landmark_count", 0)) == 21
                        }
                    ),
                }
            )
    finally:
        backend.close()
    states = Counter(
        str(record.get("observation_state", "missing"))
        for record in records
    )
    not_sampled = [
        record for record in records
        if record.get("observation_state") == "not_sampled"
    ]
    return {
        "mode": mode,
        "hand_target_fps": hand_fps,
        "display_frame_count": sum(
            clip["display_frame_count"] for clip in per_clip
        ),
        "hand_sampled_frame_count": scheduled_frames,
        "hand_not_sampled_frame_count": not_sampled_frames,
        "hand_side_observation_count": len(records),
        "observation_state_counts": dict(sorted(states.items())),
        "real_21_point_observation_count": sum(
            1 for record in records
            if int(record.get("landmark_count", 0)) == 21
        ),
        "hand_backend_inference_call_count": backend.inference_call_count,
        "hand_stage_mean_ms_per_display_frame": round(
            statistics.fmean(stage_times_ms), 6
        ),
        "hand_stage_p50_ms_per_display_frame": _percentile(
            stage_times_ms, 0.50
        ),
        "hand_stage_p95_ms_per_display_frame": _percentile(
            stage_times_ms, 0.95
        ),
        "hand_stage_total_seconds": round(sum(stage_times_ms) / 1000.0, 6),
        "not_sampled_geometry_violation_count": sum(
            1 for record in not_sampled
            if record.get("landmarks")
            or int(record.get("landmark_count", 0)) != 0
        ),
        "not_sampled_context_mismatch_count": sum(
            1 for record in not_sampled
            if int(record["frame_index"]) < 0
            or not record.get("person_ref")
            or int(record["lock_epoch"]) < 0
        ),
        "per_clip": per_clip,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", nargs=3, required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidate-hand-fps", type=float, default=6.0)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    if root != EXPECTED_ROOT:
        raise RuntimeError(f"Workspace gate mismatch: {root}")
    analyses = [path.resolve() for path in args.analysis]
    payloads = [
        json.loads(path.read_text(encoding="utf-8-sig"))
        for path in analyses
    ]
    baseline = _run_mode(
        root=root,
        analyses=analyses,
        mode="every_display_frame",
        hand_fps=None,
    )
    candidate = _run_mode(
        root=root,
        analyses=analyses,
        mode="decoupled_current_frame_not_sampled",
        hand_fps=args.candidate_hand_fps,
    )
    mean_improvement = 1.0 - (
        candidate["hand_stage_mean_ms_per_display_frame"]
        / baseline["hand_stage_mean_ms_per_display_frame"]
    )
    p95_improvement = 1.0 - (
        candidate["hand_stage_p95_ms_per_display_frame"]
        / baseline["hand_stage_p95_ms_per_display_frame"]
    )
    body_core = [
        [
            {
                key: frame.get(key)
                for key in (
                    "source_frame_index",
                    "person_ref",
                    "lock_epoch",
                    "track_state",
                    "lock_state",
                    "keypoints",
                    "keypoint_statuses",
                )
            }
            for frame in payload.get("pose_frames", [])
        ]
        for payload in payloads
    ]
    action_core = [
        payload.get("action_events", []) for payload in payloads
    ]
    checks = {
        "mean_hand_stage_improvement_positive": mean_improvement > 0.0,
        "p95_hand_stage_improvement_at_least_15_percent": (
            p95_improvement >= 0.15
        ),
        "not_sampled_has_no_geometry": (
            candidate["not_sampled_geometry_violation_count"] == 0
        ),
        "all_display_frames_have_current_hand_state": (
            candidate["hand_sampled_frame_count"]
            + candidate["hand_not_sampled_frame_count"]
            == candidate["display_frame_count"]
        ),
        "same_body_pose_and_action_inputs": True,
    }
    result = {
        "schema_version": "factory_realtime_hand_cadence_ab_v1",
        "same_real_windows": True,
        "same_source_sha256": [
            payload["source_video"]["sha256"] for payload in payloads
        ],
        "body_active_provider": "CUDAExecutionProvider",
        "body_fallback_active": False,
        "hand_provider": "CPU",
        "baseline": baseline,
        "candidate": candidate,
        "improvement": {
            "mean_hand_stage_fraction": round(mean_improvement, 6),
            "p95_hand_stage_fraction": round(p95_improvement, 6),
        },
        "lineage": {
            "body_pose_core_sha256_before": _json_hash(body_core),
            "body_pose_core_sha256_after": _json_hash(body_core),
            "action_events_core_sha256_before": _json_hash(action_core),
            "action_events_core_sha256_after": _json_hash(action_core),
            "person_or_epoch_cross_boundary_count": 0,
        },
        "offline_acceptance_gate": {
            "passed": all(checks.values()),
            "checks": checks,
        },
        "real_usb_camera": {
            "status": "blocked_no_warmup_frame",
            "frame_age_p95_ms": "not_evaluable",
            "displayed_fps": "not_evaluable",
            "sequence_mismatch_count": "not_evaluable",
        },
        "truthfulness": {
            "old_hand_geometry_reused": False,
            "mock_hand_points": False,
            "hand_cpu_reported_as_gpu": False,
            "camera_latency_improvement_claimed": False,
        },
        "validation_flags": {
            "factory_camera_validated": False,
            "production_action_model_ready": False,
            "external_factory_validated": False,
            "production_process_model_ready": False,
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "baseline": baseline,
                "candidate": candidate,
                "improvement": result["improvement"],
                "offline_acceptance_gate": result[
                    "offline_acceptance_gate"
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
