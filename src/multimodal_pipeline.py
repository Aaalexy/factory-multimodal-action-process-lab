"""Runnable, evidence-gated offline multimodal kickoff pipeline."""

from __future__ import annotations

import ast
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import cv2
import numpy as np

from src.action_segmentation import (
    CausalCoarseActionClassifier,
    FrameActionStabilityConfig,
    PhaseBActionStabilityConfig,
    build_pose_segments,
    build_stable_action_events_from_frames,
)
from src.action_segmentation.coarse import CoarseFrame
from src.contracts import LayerState, MultimodalResult, ValidationFlags
from src.evaluation import not_evaluable_manifest
from src.hand_pose import DisabledHandBackend, MediaPipeHandLandmarkerBackend
from src.interaction_fusion import InteractionFusionEngine
from src.legacy_pose.action_analysis import (
    ActionAnalysisConfig,
    stabilize_action_events as stabilize_phase_a_events,
)
from src.legacy_pose.manual_selection import ManualSelectionSeed
from src.legacy_pose.person_tracker import TrackerConfig
from src.object_perception import NotConfiguredObjectPerception
from src.pose_core import PoseRuntime, overlay_pose
from src.process_reasoning import ProcessReasoner
from src.provenance import sha256_file
from src.temporal_actions import NotConfiguredTemporalActionModel
from src.tracking import AnonymousPersonLock
from src.video_io import iter_video_frames, probe_video


@dataclass(frozen=True)
class BaselineConfig:
    project_root: str
    source_video: str
    model_path: str = "models/yolov8n-pose.onnx"
    hand_model_path: str = "models/hand_pose/hand_landmarker.task"
    output_dir: str = "outputs/baseline_run"
    sample_fps: float = 8.0
    start_time: float = 0.0
    duration_seconds: float = 12.0
    providers: tuple[str, ...] | None = None
    body_provider_policy: str = "prefer_cuda"
    recording_group_id: str = "recording_group_unassigned"
    hand_enabled: bool = True
    action_profile: str = "phase_b"
    manual_selection_seed: dict[str, Any] | None = None

    def resolved_project_root(self) -> Path:
        return Path(self.project_root).expanduser().resolve()


def _project_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _load_phase_b_action_config(
    root: Path,
    *,
    analysis_fps: float,
) -> PhaseBActionStabilityConfig:
    """Load Phase B thresholds from the active project configuration."""

    project_config_path = root / "configs" / "project.json"
    project_config = json.loads(project_config_path.read_text(encoding="utf-8"))
    action_config = project_config.get("action_stability")
    if not isinstance(action_config, dict):
        raise ValueError(
            "configs/project.json must contain an action_stability object"
        )
    supported_fields = {
        "stable_event_minimum_seconds",
        "short_directional_event_minimum_seconds",
        "short_gap_merge_seconds",
        "start_confirmation_seconds",
        "stop_confirmation_seconds",
        "temporal_context_seconds",
        "bounded_uncertain_gap_seconds",
        "minimum_detected_evidence_ratio",
        "short_event_minimum_detected_ratio",
        "maximum_prediction_ratio",
        "maximum_missing_ratio",
    }
    configured_values = {
        key: action_config[key]
        for key in supported_fields
        if key in action_config
    }
    return PhaseBActionStabilityConfig(
        analysis_fps=analysis_fps,
        **configured_values,
    )


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _safe_float_list(values: np.ndarray, digits: int = 3) -> list[Any]:
    array = np.asarray(values)
    result: list[Any] = []
    for row in array:
        result.append(
            [
                None if not np.isfinite(value) else round(float(value), digits)
                for value in row
            ]
        )
    return result


def _candidate_payload(detections: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_ref": f"anonymous-candidate-{index:02d}",
            "bbox": [round(float(value), 2) for value in detection.bbox],
            "confidence": round(float(detection.confidence), 5),
        }
        for index, detection in enumerate(detections, start=1)
    ]


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

BODY_ACTION_POINTS = [
    "left_shoulder[5]",
    "right_shoulder[6]",
    "left_elbow[7]",
    "right_elbow[8]",
    "left_wrist[9]",
    "right_wrist[10]",
    "left_hip[11]",
    "right_hip[12]",
]

HAND_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)


def _id_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    text = str(value or "").strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, (list, tuple)):
            return [str(item) for item in parsed if str(item)]
    return [
        item.strip()
        for item in text.replace(",", ";").split(";")
        if item.strip()
    ]


def _event_duration(event: dict[str, Any]) -> float:
    return max(
        0.0,
        float(event.get("end_time", 0.0)) - float(event.get("start_time", 0.0)),
    )


def _phase_a_stabilization(
    pose_segments: list[dict[str, Any]],
    *,
    analysis_fps: float,
) -> dict[str, Any]:
    """Replay the kickoff duration profile for an honest before comparison."""

    result = stabilize_phase_a_events(
        pose_segments,
        ActionAnalysisConfig(
            stable_event_minimum_seconds=1.0,
            short_event_minimum_seconds=0.75,
            short_gap_merge_seconds=0.25,
        ),
    )
    stable = [dict(item) for item in result["stable_events"]]
    source_by_id = {
        str(item["segment_id"]): item
        for item in pose_segments
        if item.get("segment_id")
    }
    cross_boundary = 0
    for index, event in enumerate(stable, start=1):
        source_ids = _id_list(
            event.get("source_segment_ids") or event.get("segment_id")
        )
        event["source_segment_ids"] = source_ids
        event["source_event_ids"] = _id_list(
            event.get("source_event_ids") or event.get("segment_id")
        )
        event["action_event_id"] = f"action-event-{index:05d}"
        event["training_eligible"] = False
        event["training_approval"] = "pending"
        event["event_kind"] = (
            "hard_boundary"
            if event.get("action") == "lost"
            else "stable_action"
        )
        identities = {
            (
                str(source_by_id[source_id].get("person_ref", "")),
                int(source_by_id[source_id].get("lock_epoch", 0)),
            )
            for source_id in source_ids
            if source_id in source_by_id
        }
        if len(identities) > 1:
            cross_boundary += 1

    normal = [
        item
        for item in stable
        if str(item.get("action", "")).lower() in NORMAL_ACTIONS
    ]
    boundaries = [
        item
        for item in pose_segments
        if item.get("raw_lost")
        or str(item.get("track_state", "")).lower()
        in {"lost", "temporarily_lost", "off_frame"}
        or str(item.get("action", "")).lower() == "lost"
    ]
    lost_overlap = 0
    for event in normal:
        if any(
            str(boundary.get("person_ref")) == str(event.get("person_ref"))
            and int(boundary.get("lock_epoch", 0))
            == int(event.get("lock_epoch", 0))
            and float(boundary.get("end_time", 0.0))
            > float(event.get("start_time", 0.0)) + 1e-9
            and float(boundary.get("start_time", 0.0))
            < float(event.get("end_time", 0.0)) - 1e-9
            for boundary in boundaries
        ):
            lost_overlap += 1
    window_start = min(
        (float(item.get("start_time", 0.0)) for item in pose_segments),
        default=0.0,
    )
    window_end = max(
        (float(item.get("end_time", window_start)) for item in pose_segments),
        default=window_start,
    )
    window_seconds = max(0.0, window_end - window_start)
    metrics = dict(result["metrics"])
    metrics.update(
        {
            "input_pose_segment_count": len(pose_segments),
            "stable_normal_action_count": len(normal),
            "suppressed_fragment_count": int(metrics.get("suppressed_count", 0)),
            "merged_fragment_count": int(metrics.get("merge_count", 0)),
            "sub_1s_stable_event_count": sum(
                _event_duration(item) < 1.0 - 1e-9 for item in normal
            ),
            "events_per_minute": (
                round(len(normal) * 60.0 / window_seconds, 6)
                if window_seconds > 0
                else 0.0
            ),
            "unknown_transition_duration_seconds": round(
                sum(
                    _event_duration(item)
                    for item in pose_segments
                    if item.get("action") in {"unknown", "transition"}
                ),
                9,
            ),
            "displayed_unknown_transition_duration_seconds": round(
                sum(
                    _event_duration(item)
                    for item in stable
                    if item.get("action") in {"unknown", "transition"}
                ),
                9,
            ),
            "lost_normal_action_overlap_count": lost_overlap,
            "cross_identity_or_epoch_merge_count": cross_boundary,
            "configuration": {
                "analysis_fps": analysis_fps,
                "stable_event_minimum_seconds": 1.0,
                "short_directional_event_minimum_seconds": 0.75,
                "short_gap_merge_seconds": 0.25,
                "start_confirmation_seconds": None,
                "stop_confirmation_seconds": None,
                "temporal_context_seconds": None,
            },
        }
    )
    return {
        "profile": "phase_a_parameter_replay_with_crash_guard",
        "pose_evidence": [dict(item) for item in pose_segments],
        "stable_events": stable,
        "suppressed_events": result["suppressed_events"],
        "metrics": metrics,
    }


def _hand_summary(
    hand_records: list[dict[str, Any]],
    processed_frame_count: int,
    *,
    backend_enabled: bool,
) -> dict[str, Any]:
    by_frame: dict[int, list[str]] = {}
    eligible_frames: set[int] = set()
    observation_counts = {
        "detected": 0,
        "uncertain": 0,
        "missing": 0,
    }
    backend_state_counts = {
        "available": 0,
        "unavailable": 0,
        "error": 0,
        "unknown": 0,
    }
    quality_state_counts = {
        "qualified": 0,
        "association_uncertain": 0,
        "insufficient_geometry": 0,
        "not_observed": 0,
        "lost": 0,
        "unknown": 0,
    }
    validation_state_counts = {
        "not_reviewed": 0,
        "review_required": 0,
        "not_evaluable": 0,
        "unknown": 0,
    }
    eligible_observation_count = 0
    warnings: list[dict[str, Any]] = []
    association_error_names = {
        "model_hand_is_closer_to_opposite_body_wrist",
        "model_wrist_too_far_from_own_body_wrist",
        "own_body_wrist_unavailable_for_consistency_check",
        "duplicate_hand_candidate_across_sides",
    }
    association_error_events: set[tuple[int, str]] = set()
    all_warning_records: set[tuple[int, str, str]] = set()
    for record in hand_records:
        frame_index = int(record.get("frame_index", -1))
        state = str(record.get("observation_state", "missing")).lower()
        normalized = (
            "detected"
            if state == "detected"
            else "uncertain"
            if state in {"uncertain", "predicted", "interpolated"}
            else "missing"
        )
        observation_counts[normalized] += 1
        by_frame.setdefault(frame_index, []).append(normalized)

        backend_state = str(record.get("backend_state", "unknown")).lower()
        backend_state_counts[
            backend_state if backend_state in backend_state_counts else "unknown"
        ] += 1
        quality_state = str(record.get("quality_state", "unknown")).lower()
        quality_state_counts[
            quality_state if quality_state in quality_state_counts else "unknown"
        ] += 1
        validation_state = str(
            record.get("validation_state", "unknown")
        ).lower()
        validation_state_counts[
            validation_state
            if validation_state in validation_state_counts
            else "unknown"
        ] += 1
        if record.get("action_feature_eligible") is True:
            eligible_observation_count += 1
            eligible_frames.add(frame_index)

        association = record.get("association_checks", {})
        record_warnings = list(association.get("warnings", []))
        if record_warnings:
            side = str(record.get("anatomical_side", "unknown"))
            for warning in record_warnings:
                all_warning_records.add((frame_index, side, str(warning)))
                if warning in association_error_names:
                    association_error_events.add((frame_index, str(warning)))
            warnings.append(
                {
                    "hand_pose_id": record.get("hand_pose_id"),
                    "anatomical_side": record.get("anatomical_side"),
                    "warnings": record_warnings,
                }
            )

    frame_counts = {"detected": 0, "uncertain": 0, "missing": 0}
    for states in by_frame.values():
        if "detected" in states:
            frame_counts["detected"] += 1
        elif "uncertain" in states:
            frame_counts["uncertain"] += 1
        else:
            frame_counts["missing"] += 1
    unavailable = (
        max(0, processed_frame_count - len(by_frame))
        if not backend_enabled
        else 0
    )
    return {
        "hand_detected_frame_count": frame_counts["detected"],
        "hand_uncertain_frame_count": frame_counts["uncertain"],
        "hand_missing_frame_count": frame_counts["missing"],
        "hand_unavailable_frame_count": unavailable,
        "hand_detected_observation_count": observation_counts["detected"],
        "hand_uncertain_observation_count": observation_counts["uncertain"],
        "hand_missing_observation_count": observation_counts["missing"],
        "hand_backend_state_counts": backend_state_counts,
        "hand_quality_state_counts": quality_state_counts,
        "hand_validation_state_counts": validation_state_counts,
        "hand_action_feature_eligible_observation_count": (
            eligible_observation_count
        ),
        "hand_action_feature_eligible_frame_count": len(eligible_frames),
        "left_right_association_error_count": len(association_error_events),
        "association_warning_count": len(all_warning_records),
        "association_counting_unit": (
            "unique (frame_index, warning_type) for left/right errors; "
            "unique (frame_index, anatomical_side, warning_type) for warnings"
        ),
        "association_warnings": warnings,
    }


def _stable_event_at(
    action_events: list[dict[str, Any]],
    timestamp: float,
) -> dict[str, Any] | None:
    return next(
        (
            item
            for item in action_events
            if float(item.get("start_time", 0.0)) <= timestamp
            < float(item.get("end_time", 0.0))
        ),
        None,
    )


def _overlay_hand_evidence(
    frame: np.ndarray,
    hand_records: list[dict[str, Any]],
    *,
    stable_event: dict[str, Any] | None,
) -> np.ndarray:
    rendered = frame.copy()
    colors = {"left": (212, 182, 53), "right": (255, 140, 156)}
    labels: list[str] = []
    for record in hand_records:
        side = str(record.get("anatomical_side", "unknown"))
        state = str(record.get("observation_state", "missing"))
        labels.append(f"{side} hand: {state}")
        landmarks = record.get("landmarks", [])
        if state not in {"detected", "uncertain"} or not landmarks:
            continue
        points = {
            int(item["index"]): (int(round(item["x"])), int(round(item["y"])))
            for item in landmarks
            if item.get("x") is not None and item.get("y") is not None
        }
        color = colors.get(side, (210, 210, 210))
        line_width = 2 if state == "detected" else 1
        for first, second in HAND_EDGES:
            if first in points and second in points:
                cv2.line(
                    rendered,
                    points[first],
                    points[second],
                    color,
                    line_width,
                    cv2.LINE_AA,
                )
        for point in points.values():
            cv2.circle(rendered, point, 3, color, -1, cv2.LINE_AA)

    stable_label = (
        f"{stable_event.get('action')} "
        f"{_event_duration(stable_event):.2f}s"
        if stable_event is not None
        else "unavailable at this timestamp"
    )
    labels.insert(0, f"stable action: {stable_label}")
    height = rendered.shape[0]
    line_spacing = 40
    y_start = max(40, height - line_spacing * len(labels) - 18)
    for index, label in enumerate(labels):
        cv2.putText(
            rendered,
            label,
            (18, y_start + index * line_spacing),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.05,
            (238, 240, 244),
            3,
            cv2.LINE_AA,
        )
    return rendered


def _write_contact_sheet(frames: list[np.ndarray], path: Path) -> None:
    if not frames:
        return
    thumbnails: list[np.ndarray] = []
    for frame in frames[:12]:
        height, width = frame.shape[:2]
        target_width = 520
        target_height = max(1, int(round(height * target_width / width)))
        thumbnails.append(
            cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        )
    tile_height = max(item.shape[0] for item in thumbnails)
    padded: list[np.ndarray] = []
    for item in thumbnails:
        if item.shape[0] < tile_height:
            item = cv2.copyMakeBorder(
                item,
                0,
                tile_height - item.shape[0],
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(12, 18, 28),
            )
        padded.append(item)
    columns = 3
    blank = np.zeros_like(padded[0])
    rows: list[np.ndarray] = []
    for offset in range(0, len(padded), columns):
        row = padded[offset : offset + columns]
        row.extend([blank] * (columns - len(row)))
        rows.append(cv2.hconcat(row))
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.vconcat(rows))


def _copy_source_video(source: Path, output_dir: Path) -> Path:
    suffix = source.suffix.lower() or ".mp4"
    target = output_dir / f"source_video{suffix}"
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return target


def _decode_real_warmup_frame(path: Path, timestamp: float) -> np.ndarray:
    """Decode one real source frame, then release the decoder before CUDA warmup."""

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"OpenCV cannot decode video: {path}")
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp) * 1000.0)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Unable to decode real Body Pose warmup frame: {path}")
    return frame


def run_baseline(
    config: BaselineConfig,
    *,
    progress_callback: Callable[[float, str, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    def publish(value: float, message: str, stage: str) -> None:
        if progress_callback is not None:
            progress_callback(
                max(0.0, min(1.0, float(value))),
                str(message),
                str(stage),
            )

    def raise_if_cancelled(message: str) -> None:
        if cancel_check is not None and cancel_check():
            raise InterruptedError(message)

    pipeline_started = perf_counter()
    root = config.resolved_project_root()
    source = Path(config.source_video).expanduser().resolve()
    model = _project_path(root, config.model_path)
    hand_model = _project_path(root, config.hand_model_path)
    output_dir = _project_path(root, config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not source.is_file():
        raise FileNotFoundError(f"Source video is missing: {source}")
    if not model.is_file():
        raise FileNotFoundError(f"Pose model is missing: {model}")
    if config.sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    if config.action_profile not in {"phase_a", "phase_b"}:
        raise ValueError("action_profile must be 'phase_a' or 'phase_b'")

    raise_if_cancelled("Analysis cancelled before source preparation")
    publish(0.02, "Preparing the uploaded MP4.", "queued")
    local_video = _copy_source_video(source, output_dir)
    video = probe_video(local_video)
    model_hash = sha256_file(model)
    hand_model_hash = sha256_file(hand_model) if hand_model.is_file() else None
    end_time = min(
        video.duration_seconds,
        max(config.start_time, config.start_time + config.duration_seconds),
    )
    if end_time <= config.start_time:
        raise ValueError("Requested analysis window is empty")

    manual_selection: ManualSelectionSeed | None = None
    if config.manual_selection_seed is not None:
        manual_selection = ManualSelectionSeed.from_mapping(
            dict(config.manual_selection_seed)
        )
        if Path(manual_selection.video_path).resolve() != source:
            raise ValueError("manual selection seed belongs to another video")
        frame_tolerance = max(1.0 / max(video.fps, 1e-9), 0.05)
        if (
            abs(manual_selection.selection_timestamp - config.start_time)
            > frame_tolerance
        ):
            raise ValueError(
                "manual selection seed does not match the analysis start frame"
            )

    raise_if_cancelled("Analysis cancelled before Body Pose initialization")
    publish(
        0.04,
        "Initializing and warming the real Body Pose provider.",
        "gpu_warming_up",
    )
    pose_runtime = PoseRuntime(
        model,
        providers=list(config.providers) if config.providers else None,
        provider_policy=config.body_provider_policy,
    )
    pose_warmup_times_ms = pose_runtime.warmup(
        _decode_real_warmup_frame(local_video, config.start_time),
    )
    raise_if_cancelled("Analysis cancelled after Body Pose warmup")
    publish(
        0.08,
        "Body Pose warmup completed; initializing Hand Pose.",
        "analyzing",
    )
    hand_backend_error: str | None = None
    if config.hand_enabled and hand_model.is_file():
        try:
            hand_backend = MediaPipeHandLandmarkerBackend(
                hand_model,
                model_version=(
                    "mediapipe_hand_landmarker_float16_v1"
                    f"+sha256:{hand_model_hash}"
                ),
            )
        except Exception as exc:
            hand_backend = DisabledHandBackend()
            hand_backend_error = (
                f"hand_backend_initialization_failed:{type(exc).__name__}"
            )
            hand_backend.reason = hand_backend_error
    else:
        hand_backend = DisabledHandBackend()
        if config.hand_enabled:
            hand_backend_error = "hand_model_file_missing"
            hand_backend.reason = hand_backend_error

    tracker_config = (
        TrackerConfig(
            selection_mode="manual",
            manual_selection_seed=manual_selection,
        )
        if manual_selection is not None
        else TrackerConfig()
    )
    anonymous_lock = AnonymousPersonLock(tracker_config)
    classifier = CausalCoarseActionClassifier()
    sampled_frames: list[CoarseFrame] = []
    pose_frame_payloads: list[dict[str, Any]] = []
    hand_pose_frames: list[dict[str, Any]] = []
    contact_entries: list[dict[str, Any]] = []
    maximum_candidates = 0
    expected_samples = max(
        1,
        int(np.ceil((end_time - config.start_time) * config.sample_fps)),
    )
    contact_stride = max(1, int(np.ceil(expected_samples / 12.0)))
    processing_started = perf_counter()
    try:
        for packet in iter_video_frames(
            local_video,
            start_time=config.start_time,
            end_time=end_time,
            output_fps=config.sample_fps,
        ):
            raise_if_cancelled(
                "Analysis cancelled inside the video frame loop"
            )
            detections = pose_runtime.detect(packet.image)
            maximum_candidates = max(maximum_candidates, len(detections))
            lock = anonymous_lock.update(detections, packet.image.shape)
            raw = lock.raw_result
            pose = raw.smoothed_pose if lock.usable_pose else None
            keypoints = pose.keypoints if pose is not None else None
            statuses = pose.statuses if pose is not None else None
            action, anatomical_side, evidence = classifier.classify(
                timestamp=packet.timestamp,
                person_ref=lock.person_ref,
                lock_epoch=lock.lock_epoch,
                track_state=lock.track_state,
                lock_state=lock.lock_state,
                keypoints=keypoints,
                statuses=statuses,
            )
            frame_evidence = CoarseFrame(
                timestamp=float(packet.timestamp),
                source_frame_index=int(packet.source_frame_index),
                person_ref=lock.person_ref,
                lock_epoch=lock.lock_epoch,
                track_state=lock.track_state,
                lock_state=lock.lock_state,
                candidate_person_count=len(detections),
                action=action,
                anatomical_side=anatomical_side,
                observation_state=str(evidence["observation_state"]),
                detected_ratio=float(evidence["detected_ratio"]),
                predicted_ratio=float(evidence["predicted_ratio"]),
                interpolated_ratio=float(evidence["interpolated_ratio"]),
                missing_ratio=float(evidence["missing_ratio"]),
                direction_clear=bool(evidence["direction_clear"]),
                required_joints_reliable=bool(
                    evidence["required_joints_reliable"]
                ),
                keypoints=(
                    _safe_float_list(keypoints)
                    if keypoints is not None
                    else []
                ),
                keypoint_statuses=(
                    [str(item) for item in statuses]
                    if statuses is not None
                    else []
                ),
            )
            sampled_frames.append(frame_evidence)
            frame_hand_records = hand_backend.infer_frame(
                packet.image,
                body_keypoints=keypoints,
                body_keypoint_statuses=statuses,
                person_ref=lock.person_ref,
                lock_epoch=lock.lock_epoch,
                frame_index=int(packet.source_frame_index),
                timestamp=float(packet.timestamp),
                source_video_sha256=video.sha256,
                recording_group_id=config.recording_group_id,
                track_state=lock.track_state,
                lock_state=lock.lock_state,
            )
            hand_pose_frames.extend(frame_hand_records)
            raise_if_cancelled(
                "Analysis cancelled after current-frame Hand Pose inference"
            )
            bbox = (
                [round(float(value), 2) for value in raw.detection.bbox]
                if lock.usable_pose and raw.detection is not None
                else None
            )
            pose_frame_payloads.append(
                {
                    **asdict(frame_evidence),
                    "bbox": bbox,
                    "person_confidence": (
                        round(float(raw.detection.confidence), 5)
                        if lock.usable_pose and raw.detection is not None
                        else None
                    ),
                    "anonymous_candidates": _candidate_payload(detections),
                    "switch_exposed": lock.switch_exposed,
                    "awaiting_manual_relock": lock.awaiting_manual_relock,
                    "hand_pose_ids": [
                        item["hand_pose_id"] for item in frame_hand_records
                    ],
                    "evidence_kind": "real_yolov8_pose",
                }
            )
            processed_index = len(sampled_frames) - 1
            if (
                len(contact_entries) < 12
                and processed_index % contact_stride == 0
            ):
                contact_entries.append(
                    {
                        "timestamp": float(packet.timestamp),
                        "hand_records": [dict(item) for item in frame_hand_records],
                        "frame": overlay_pose(
                            packet.image,
                            raw,
                            person_ref=lock.person_ref,
                            lock_epoch=lock.lock_epoch,
                            coarse_action=f"raw {action}",
                            geometry_allowed=lock.usable_pose,
                        ),
                    }
                )
            processed_count = len(sampled_frames)
            publish(
                0.10 + 0.78 * min(1.0, processed_count / expected_samples),
                (
                    f"Analyzing real Body and Hand evidence "
                    f"({processed_count}/{expected_samples} sampled frames)."
                ),
                "analyzing",
            )
    finally:
        hand_backend.close()
    processing_seconds = perf_counter() - processing_started

    if not sampled_frames:
        raise RuntimeError("No video frames were decoded in the requested window")
    raise_if_cancelled("Analysis cancelled before temporal stabilization")
    publish(
        0.90,
        "Building evidence and stable-action timelines.",
        "writing",
    )

    sample_interval = 1.0 / config.sample_fps
    pose_segments = build_pose_segments(
        sampled_frames,
        source_video_sha256=video.sha256,
        sample_interval_seconds=sample_interval,
        analysis_end_time=end_time,
    )
    for segment in pose_segments:
        segment["recording_group_id"] = config.recording_group_id
        segment["source_model_version"] = f"sha256:{model_hash}"
        segment["body_points_used"] = list(BODY_ACTION_POINTS)
        segment["reviewer"] = None
        segment["reviewed_at"] = None
    if config.action_profile == "phase_a":
        stabilization = _phase_a_stabilization(
            pose_segments,
            analysis_fps=config.sample_fps,
        )
    else:
        phase_b_config = _load_phase_b_action_config(
            root,
            analysis_fps=config.sample_fps,
        )
        stabilization = build_stable_action_events_from_frames(
            sampled_frames,
            pose_segments,
            source_video_sha256=video.sha256,
            sample_interval_seconds=sample_interval,
            analysis_end_time=end_time,
            frame_config=FrameActionStabilityConfig(
                start_confirmation_seconds=(
                    phase_b_config.start_confirmation_seconds
                ),
                stop_confirmation_seconds=phase_b_config.stop_confirmation_seconds,
                temporal_context_seconds=phase_b_config.temporal_context_seconds,
                bounded_uncertain_gap_seconds=(
                    phase_b_config.bounded_uncertain_gap_seconds
                ),
            ),
            event_config=phase_b_config,
            analysis_start_time=config.start_time,
        )
    action_events = stabilization["stable_events"]
    for event in action_events:
        event["body_points_used"] = list(BODY_ACTION_POINTS)
    for event in stabilization["suppressed_events"]:
        event["body_points_used"] = list(BODY_ACTION_POINTS)
    hand_metrics = _hand_summary(
        hand_pose_frames,
        len(sampled_frames),
        backend_enabled=hand_backend.enabled,
    )

    object_output = NotConfiguredObjectPerception().analyze(pose_frame_payloads)
    interaction_output = InteractionFusionEngine().derive(
        pose_frames=pose_frame_payloads,
        object_tracks=object_output.object_tracks,
        object_layer_status=object_output.state.status,
    )
    temporal_output = NotConfiguredTemporalActionModel().analyze(
        action_events,
        interaction_output.interaction_events,
    )
    process_output = ProcessReasoner().infer(
        pose_action_events=action_events,
        interaction_events=interaction_output.interaction_events,
        temporal_action_candidates=temporal_output.action_candidates,
        required_layers_available=False,
    )

    relative_video = local_video.relative_to(root).as_posix()
    layer_states = [
        LayerState(
            layer="pose_tracking",
            status="available",
            reason="real YOLOv8n-Pose ONNX detections and anonymous lock",
            model_version=f"sha256:{model_hash}",
            evidence_count=len(pose_frame_payloads),
        ),
        LayerState(
            layer="hand_pose",
            status=hand_backend.availability_status,
            reason=hand_backend.reason,
            model_version=hand_backend.model_version,
            evidence_count=hand_metrics["hand_detected_observation_count"],
        ),
        LayerState(
            layer="pose_action_segmentation",
            status="available",
            reason=(
                "causal pose heuristic plus start/stop confirmation, "
                "context, duration, visibility, and hard-boundary gates"
                if config.action_profile == "phase_b"
                else "Phase A parameter replay for controlled comparison"
            ),
            model_version=(
                "pose_heuristic_phase_b_v2"
                if config.action_profile == "phase_b"
                else "pose_heuristic_phase_a_replay"
            ),
            evidence_count=len(action_events),
        ),
        object_output.state,
        interaction_output.state,
        temporal_output.state,
        process_output.state,
    ]
    result = MultimodalResult(
        schema_version="factory_multimodal_analysis_v1",
        project="Factory Multimodal Action Process Lab",
        source_video={
            **video.to_dict(),
            "path": relative_video,
            "intake_source_path": str(source),
            "analysis_window": {
                "start_time": config.start_time,
                "end_time": end_time,
                "sample_fps": config.sample_fps,
            },
            "recording_group_id": config.recording_group_id,
        },
        validation_flags=ValidationFlags(),
        pose_segments=pose_segments,
        action_events=action_events,
        evidence_timeline=stabilization.get("evidence_timeline", []),
        evidence_timeline_metrics=stabilization.get(
            "evidence_timeline_metrics",
            {},
        ),
        hand_pose_frames=hand_pose_frames,
        object_tracks=object_output.object_tracks,
        interaction_events=interaction_output.interaction_events,
        process_steps=process_output.process_steps,
        layer_states=layer_states,
        tracking_summary={
            "person_ref": anonymous_lock.person_ref,
            "lock_epoch": anonymous_lock.lock_epoch,
            "active_track_id": anonymous_lock.active_track_id,
            "maximum_candidate_person_count": maximum_candidates,
            "switch_events": anonymous_lock.switch_events,
            "switch_event_count": len(
                [
                    item
                    for item in anonymous_lock.switch_events
                    if "switch" in str(item.get("event", ""))
                ]
            ),
            "silent_switch_count": 0,
            "awaiting_manual_relock": anonymous_lock.awaiting_manual_relock,
            "selection_mode": (
                "manual" if manual_selection is not None else "automatic"
            ),
            "selected_candidate_id": (
                manual_selection.candidate_id
                if manual_selection is not None
                else None
            ),
            "manual_seed_match": (
                dict(anonymous_lock.tracker.last_manual_match)
                if manual_selection is not None
                else None
            ),
            "tracker_lock_events": list(anonymous_lock.tracker.lock_events),
        },
        evaluation=not_evaluable_manifest(
            "No independently human-confirmed ground truth is available for this run"
        ),
    )
    payload = result.to_dict()
    payload["pose_frames"] = pose_frame_payloads
    hand_backend_modes = sorted(
        {
            str(record.get("backend_mode"))
            for record in hand_pose_frames
            if record.get("backend_mode")
        }
    )
    hand_quality_gate_versions = sorted(
        {
            str(record.get("quality_gate_version"))
            for record in hand_pose_frames
            if record.get("quality_gate_version")
        }
    )
    payload["hand_model"] = {
        "backend": type(hand_backend).__name__,
        "backend_state": getattr(
            hand_backend,
            "availability_status",
            "available" if hand_backend.enabled else "unavailable",
        ),
        "backend_mode": (
            hand_backend_modes[0]
            if len(hand_backend_modes) == 1
            else "mixed"
            if hand_backend_modes
            else "disabled"
            if not hand_backend.enabled
            else "unknown"
        ),
        "version": hand_backend.model_version,
        "sha256": hand_model_hash,
        "runtime_version": getattr(hand_backend, "runtime_version", None),
        "provider": "CPU" if hand_backend.enabled else "unavailable",
        "hand_gpu_status": (
            "unsupported_current_backend_on_windows"
            if hand_backend.enabled
            else "not_configured"
        ),
        "quality_gate_version": (
            hand_quality_gate_versions[0]
            if len(hand_quality_gate_versions) == 1
            else "mixed"
            if hand_quality_gate_versions
            else None
        ),
        "enabled": hand_backend.enabled,
        "initialization_error": hand_backend_error,
        "confidence_note": (
            "MediaPipe Python result does not expose raw detection, "
            "presence, or tracking confidence values"
            if hand_backend.enabled
            else "unavailable"
        ),
    }
    payload["action_profile"] = (
        stabilization.get("profile") or config.action_profile
    )
    payload["stabilization_metrics"] = stabilization["metrics"]
    payload["suppressed_action_evidence"] = stabilization["suppressed_events"]
    payload["evidence_timeline"] = stabilization.get("evidence_timeline", [])
    payload["evidence_timeline_metrics"] = stabilization.get(
        "evidence_timeline_metrics",
        {},
    )
    mean_hand_ms = (
        round(float(np.mean(hand_backend.inference_times_ms)), 3)
        if hand_backend.inference_times_ms
        else None
    )
    hand_p50_ms = (
        round(float(np.percentile(hand_backend.inference_times_ms, 50)), 3)
        if hand_backend.inference_times_ms
        else None
    )
    hand_p95_ms = (
        round(float(np.percentile(hand_backend.inference_times_ms, 95)), 3)
        if hand_backend.inference_times_ms
        else None
    )
    end_to_end_seconds = perf_counter() - pipeline_started
    payload["runtime"] = {
        "processed_frame_count": len(sampled_frames),
        "analysis_fps": config.sample_fps,
        "action_profile": config.action_profile,
        "pose_providers": pose_runtime.providers,
        "pose_provider_status": pose_runtime.provider_status,
        "pose_warmup_call_count": pose_runtime.detector.warmup_call_count,
        "pose_warmup_times_ms": [
            round(value, 3) for value in pose_warmup_times_ms
        ],
        "pose_inference_calls": pose_runtime.detector.inference_call_count,
        "mean_pose_inference_ms": (
            round(
                float(np.mean(pose_runtime.detector.inference_times_ms)),
                3,
            )
            if pose_runtime.detector.inference_times_ms
            else None
        ),
        "hand_backend": type(hand_backend).__name__,
        "hand_provider": "CPU" if hand_backend.enabled else "unavailable",
        "hand_gpu_status": (
            "unsupported_current_backend_on_windows"
            if hand_backend.enabled
            else "not_configured"
        ),
        "hand_model_version": hand_backend.model_version,
        "hand_inference_calls": hand_backend.inference_call_count,
        "mean_hand_inference_ms": mean_hand_ms,
        "hand_inference_p50_ms": hand_p50_ms,
        "hand_inference_p95_ms": hand_p95_ms,
        "hand_inference_error_count": getattr(
            hand_backend,
            "inference_error_count",
            0,
        ),
        **hand_metrics,
        "pose_segment_count": len(pose_segments),
        "action_event_record_count": len(action_events),
        "stable_action_event_count": len(action_events),
        "stable_normal_action_count": stabilization["metrics"].get(
            "stable_normal_action_count",
            0,
        ),
        "sub_1s_stable_event_count": stabilization["metrics"].get(
            "sub_1s_stable_event_count",
            0,
        ),
        "events_per_minute": stabilization["metrics"].get(
            "events_per_minute",
            0.0,
        ),
        "suppressed_fragment_count": stabilization["metrics"].get(
            "suppressed_fragment_count",
            stabilization["metrics"].get("suppressed_count", 0),
        ),
        "merged_fragment_count": stabilization["metrics"].get(
            "merged_fragment_count",
            stabilization["metrics"].get("merge_count", 0),
        ),
        "unknown_transition_duration_seconds": stabilization["metrics"].get(
            "unknown_transition_duration_seconds",
            0.0,
        ),
        "displayed_unknown_transition_duration_seconds": stabilization[
            "metrics"
        ].get(
            "displayed_unknown_transition_duration_seconds",
            0.0,
        ),
        "lost_normal_action_false_positive_count": stabilization["metrics"].get(
            "lost_normal_action_overlap_count",
            0,
        ),
        "cross_identity_or_epoch_merge_count": stabilization["metrics"].get(
            "cross_identity_or_epoch_merge_count",
            0,
        ),
        "evidence_timeline_coverage_ratio": stabilization.get(
            "evidence_timeline_metrics",
            {},
        ).get("coverage_ratio", 0.0),
        "evidence_timeline_uncovered_seconds": stabilization.get(
            "evidence_timeline_metrics",
            {},
        ).get("uncovered_seconds", 0.0),
        "normal_action_coverage_ratio": stabilization.get(
            "evidence_timeline_metrics",
            {},
        ).get("normal_action_coverage_ratio", 0.0),
        "processing_seconds": round(processing_seconds, 6),
        "processing_frames_per_second": round(
            len(sampled_frames) / max(processing_seconds, 1e-9),
            6,
        ),
        "end_to_end_seconds": round(end_to_end_seconds, 6),
        "end_to_end_frames_per_second": round(
            len(sampled_frames) / max(end_to_end_seconds, 1e-9),
            6,
        ),
        "mock_keypoints_used": False,
        "mock_hand_landmarks_used": False,
        "preset_actions_used": False,
        "deepseek_called": False,
        "model_training_performed": False,
        "usb_camera_used": False,
        "rtsp_used": False,
    }
    raise_if_cancelled("Analysis cancelled before writing result files")
    _atomic_json(output_dir / "analysis.json", payload)
    _atomic_json(
        output_dir / "dataset_manifest.json",
        {
            "schema_version": "dataset_manifest_v1",
            "source_video_sha256": video.sha256,
            "source_video_path": relative_video,
            "recording_group_id": config.recording_group_id,
            "person_ref": anonymous_lock.person_ref,
            "lock_epoch": anonymous_lock.lock_epoch,
            "source_model_version": f"sha256:{model_hash}",
            "hand_source_model_version": hand_backend.model_version,
            "status": "proposed",
            "training_eligible": False,
            "reviewer": None,
            "reviewed_at": None,
            "training_approval": "pending",
            "split": "unassigned",
            "split_reason": "fewer_than_three_independent_recording_groups",
        },
    )
    _atomic_json(
        output_dir / "model_evaluation_manifest.json",
        {
            "schema_version": "model_evaluation_manifest_v1",
            "source_model_version": f"sha256:{model_hash}",
            **payload["evaluation"],
        },
    )
    _atomic_json(
        output_dir / "process_review_queue.json",
        {
            "schema_version": "process_review_queue_v1",
            "items": [],
            "status": "unavailable",
            "reason": "No evidence-qualified process candidates",
        },
    )
    rendered_frames = [
        _overlay_hand_evidence(
            item["frame"],
            item["hand_records"],
            stable_event=_stable_event_at(action_events, item["timestamp"]),
        )
        for item in contact_entries
    ]
    _write_contact_sheet(
        rendered_frames,
        output_dir / "phase_b_contact_sheet.jpg",
    )
    _write_contact_sheet(
        rendered_frames,
        output_dir / "pose_contact_sheet.jpg",
    )
    publish(1.0, "Analysis result files are complete.", "completed")
    return {
        "analysis_path": str(output_dir / "analysis.json"),
        "local_video_path": str(local_video),
        "pose_contact_sheet": str(output_dir / "pose_contact_sheet.jpg"),
        "phase_b_contact_sheet": str(
            output_dir / "phase_b_contact_sheet.jpg"
        ),
        "sampled_frame_count": len(sampled_frames),
        "pose_segment_count": len(pose_segments),
        "action_event_record_count": len(action_events),
        "stable_action_event_count": len(action_events),
        **{
            key: value
            for key, value in hand_metrics.items()
            if key != "association_warnings"
        },
        "pose_providers": pose_runtime.providers,
        "source_video_sha256": video.sha256,
        "model_sha256": model_hash,
        "hand_model_sha256": hand_model_hash,
        "action_profile": config.action_profile,
    }
