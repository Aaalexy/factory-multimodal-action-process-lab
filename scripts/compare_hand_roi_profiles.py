"""Compare bounded Hand ROI profiles on the accepted frozen Pose windows.

The script reuses the accepted Body keypoints but decodes every corresponding
real source frame and runs the real MediaPipe IMAGE backend. It never invents
Hand geometry and never changes association quality gates.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hand_pose import MediaPipeHandLandmarkerBackend


EXPECTED_ROOT = PROJECT_ROOT
HAND_MODEL = Path("models/hand_pose/hand_landmarker.task")
COMPACT_PROFILES = (
    {
        "profile_id": "accepted_baseline",
        "minimum_crop_pixels": 96,
        "forearm_scale": 2.4,
        "upper_arm_scale": 1.05,
        "wrist_extension_ratio": 0.18,
    },
    {
        "profile_id": "balanced_hand_scale",
        "minimum_crop_pixels": 88,
        "forearm_scale": 2.0,
        "upper_arm_scale": 0.75,
        "wrist_extension_ratio": 0.24,
    },
    {
        "profile_id": "compact_hand_scale",
        "minimum_crop_pixels": 80,
        "forearm_scale": 1.8,
        "upper_arm_scale": 0.65,
        "wrist_extension_ratio": 0.28,
    },
)
EXTENSION_PROFILES = (
    COMPACT_PROFILES[0],
    {
        "profile_id": "extended_wrist_center",
        "minimum_crop_pixels": 96,
        "forearm_scale": 2.4,
        "upper_arm_scale": 1.05,
        "wrist_extension_ratio": 0.38,
    },
    {
        "profile_id": "extended_context",
        "minimum_crop_pixels": 96,
        "forearm_scale": 2.6,
        "upper_arm_scale": 1.10,
        "wrist_extension_ratio": 0.38,
    },
)
WRIST_CENTER_PROFILES = (
    COMPACT_PROFILES[0],
    {
        "profile_id": "body_wrist_center",
        "minimum_crop_pixels": 96,
        "forearm_scale": 2.4,
        "upper_arm_scale": 1.05,
        "wrist_extension_ratio": 0.0,
    },
    {
        "profile_id": "body_wrist_context",
        "minimum_crop_pixels": 96,
        "forearm_scale": 2.6,
        "upper_arm_scale": 1.10,
        "wrist_extension_ratio": 0.0,
    },
)
PROFILE_SETS = {
    "compact": COMPACT_PROFILES,
    "extension": EXTENSION_PROFILES,
    "wrist_center": WRIST_CENTER_PROFILES,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    candidate = raw if raw.is_absolute() else root / raw
    if not candidate.is_file():
        raise FileNotFoundError(f"Real source video is missing: {candidate}")
    return candidate.resolve()


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    states = Counter(
        str(record.get("observation_state", "missing")).lower()
        for record in records
    )
    real = [
        record for record in records
        if int(record.get("landmark_count", 0)) == 21
        and len(record.get("landmarks") or []) == 21
    ]
    qualified = [
        record for record in real
        if str(record.get("quality_state")) == "qualified"
    ]
    warnings = [
        warning
        for record in real
        for warning in (
            record.get("association_checks", {}).get("warnings", []) or []
        )
    ]
    duplicate_frames = {
        int(record["frame_index"])
        for record in real
        if "duplicate_hand_candidate_across_sides" in (
            record.get("association_checks", {}).get("warnings", []) or []
        )
    }
    cross_side = [
        warning for warning in warnings
        if warning in {
            "model_hand_is_closer_to_opposite_body_wrist",
            "duplicate_hand_candidate_across_sides",
        }
    ]
    inference = [
        float(record["inference_time_ms"])
        for record in records
        if record.get("inference_time_ms") is not None
    ]
    crop_sizes = [
        float(record["crop_transform"]["x_scale"])
        for record in records
        if isinstance(record.get("crop_transform"), dict)
    ]
    unique_real_frames = {
        int(record["frame_index"]) for record in real
    }
    return {
        "hand_side_observation_count": len(records),
        "roi_available_observation_count": len(crop_sizes),
        "inference_call_count": len(inference),
        "real_21_point_observation_count": len(real),
        "real_21_point_frame_count": len(unique_real_frames),
        "qualified_observation_count": len(qualified),
        "detected": states["detected"],
        "uncertain": states["uncertain"],
        "missing": states["missing"],
        "lost": states["lost"],
        "association_warning_occurrence_count": len(warnings),
        "association_warning_rate_per_real_observation": (
            round(len(warnings) / len(real), 6) if real else 0.0
        ),
        "cross_side_warning_occurrence_count": len(cross_side),
        "duplicate_side_frame_count": len(duplicate_frames),
        "inference_mean_ms": (
            round(statistics.fmean(inference), 6) if inference else None
        ),
        "inference_p95_ms": _percentile(inference, 0.95),
        "crop_size_pixels": {
            "minimum": min(crop_sizes) if crop_sizes else None,
            "p50": _percentile(crop_sizes, 0.5),
            "p95": _percentile(crop_sizes, 0.95),
            "maximum": max(crop_sizes) if crop_sizes else None,
        },
    }


def _run_profile(
    *,
    root: Path,
    analysis_paths: list[Path],
    profile: dict[str, Any],
) -> dict[str, Any]:
    backend = MediaPipeHandLandmarkerBackend(
        root / HAND_MODEL,
        model_version=(
            "mediapipe_hand_landmarker_float16_v1"
            f"+roi:{profile['profile_id']}"
        ),
        minimum_crop_pixels=int(profile["minimum_crop_pixels"]),
        forearm_scale=float(profile["forearm_scale"]),
        upper_arm_scale=float(profile["upper_arm_scale"]),
        wrist_extension_ratio=float(profile["wrist_extension_ratio"]),
    )
    all_records: list[dict[str, Any]] = []
    clips: list[dict[str, Any]] = []
    try:
        for analysis_path in analysis_paths:
            payload = json.loads(
                analysis_path.read_text(encoding="utf-8-sig")
            )
            source = _source_video(root, analysis_path, payload)
            if _sha256(source) != str(payload["source_video"]["sha256"]):
                raise RuntimeError(f"Source SHA256 mismatch: {source}")
            capture = cv2.VideoCapture(str(source))
            if not capture.isOpened():
                raise RuntimeError(f"Cannot decode real source: {source}")
            records: list[dict[str, Any]] = []
            try:
                for pose in payload.get("pose_frames", []):
                    source_index = int(pose["source_frame_index"])
                    capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        raise RuntimeError(
                            f"Could not decode source frame {source_index}"
                        )
                    records.extend(
                        backend.infer_frame(
                            frame,
                            body_keypoints=pose.get("keypoints"),
                            body_keypoint_statuses=pose.get(
                                "keypoint_statuses"
                            ),
                            person_ref=str(pose["person_ref"]),
                            lock_epoch=int(pose["lock_epoch"]),
                            frame_index=source_index,
                            timestamp=float(pose["timestamp"]),
                            source_video_sha256=str(
                                payload["source_video"]["sha256"]
                            ),
                            recording_group_id=str(
                                payload["source_video"].get(
                                    "recording_group_id",
                                    "recording_group_unassigned",
                                )
                            ),
                            track_state=str(pose.get("track_state", "lost")),
                            lock_state=str(pose.get("lock_state", "lost")),
                        )
                    )
            finally:
                capture.release()
            clips.append(
                {
                    "clip_id": analysis_path.parent.parent.name,
                    "analysis": str(analysis_path.relative_to(root)),
                    "analysis_sha256": _sha256(analysis_path),
                    "source_video_sha256": _sha256(source),
                    "metrics": _summarize(records),
                }
            )
            all_records.extend(records)
    finally:
        backend.close()
    return {
        "profile": profile,
        "aggregate": _summarize(all_records),
        "clips": clips,
    }


def _gate(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    base = baseline["aggregate"]
    cand = candidate["aggregate"]
    baseline_by_clip = {
        clip["clip_id"]: clip["metrics"] for clip in baseline["clips"]
    }
    per_clip_non_regression = all(
        clip["metrics"]["real_21_point_frame_count"]
        >= baseline_by_clip[clip["clip_id"]]["real_21_point_frame_count"]
        for clip in candidate["clips"]
    )
    checks = {
        "aggregate_real_21_frame_count_increased": (
            cand["real_21_point_frame_count"]
            > base["real_21_point_frame_count"]
        ),
        "per_clip_real_21_frame_count_non_regression": per_clip_non_regression,
        "qualified_observation_count_non_regression": (
            cand["qualified_observation_count"]
            >= base["qualified_observation_count"]
        ),
        "cross_side_warning_count_non_increase": (
            cand["cross_side_warning_occurrence_count"]
            <= base["cross_side_warning_occurrence_count"]
        ),
        "duplicate_side_frame_count_non_increase": (
            cand["duplicate_side_frame_count"]
            <= base["duplicate_side_frame_count"]
        ),
        "association_warning_rate_non_increase": (
            cand["association_warning_rate_per_real_observation"]
            <= base["association_warning_rate_per_real_observation"]
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", nargs=3, required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--profile-set",
        choices=tuple(PROFILE_SETS),
        default="compact",
    )
    args = parser.parse_args()
    root = Path.cwd().resolve()
    if root != EXPECTED_ROOT:
        raise RuntimeError(f"Workspace gate mismatch: {root}")
    analyses = [path.resolve() for path in args.analysis]
    results = [
        _run_profile(root=root, analysis_paths=analyses, profile=profile)
        for profile in PROFILE_SETS[args.profile_set]
    ]
    baseline = results[0]
    candidate_gates = [
        {
            "profile_id": result["profile"]["profile_id"],
            "gate": _gate(baseline, result),
        }
        for result in results[1:]
    ]
    passing = [
        item["profile_id"] for item in candidate_gates
        if item["gate"]["passed"]
    ]
    payload = {
        "schema_version": "factory_hand_roi_profile_ab_v1",
        "profile_set": args.profile_set,
        "same_real_windows": True,
        "same_accepted_body_pose_evidence": True,
        "model_path": str(HAND_MODEL),
        "model_sha256": _sha256(root / HAND_MODEL),
        "results": results,
        "candidate_gates": candidate_gates,
        "selected_profile": passing[0] if passing else None,
        "precision_recall_f1": "not_evaluable",
        "truthfulness": {
            "real_source_frames": True,
            "real_mediapipe_image_inference": True,
            "mock_hand_points": False,
            "old_hand_geometry_reused": False,
            "association_gates_weakened": False,
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
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "selected_profile": payload["selected_profile"],
                "candidate_gates": candidate_gates,
                "aggregates": {
                    result["profile"]["profile_id"]: result["aggregate"]
                    for result in results
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
