"""Controlled IMAGE versus VIDEO Hand Landmarker comparison.

The command consumes exactly three accepted ``analysis.json`` files.  It
reuses their frozen Body Pose observations and exact source-frame indices,
decodes only the project-local ``source_video.mp4`` files, and runs the
project's default IMAGE and VIDEO Hand backends on identical inputs.

This script does not run Body Pose, smooth a Hand ROI, infer factory
semantics, or calculate accuracy without independent human ground truth.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
EXPECTED_WORKSPACE = PROJECT_ROOT
HAND_MODEL_MANIFEST = PROJECT_ROOT / "HAND_MODEL_MANIFEST.json"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "hand_pose" / (
    "hand_landmarker.task"
)
ANATOMICAL_SIDES = ("left", "right")
VALID_LOCK_STATES = {"tracked", "tracking", "locked"}
VALIDATION_FLAG_NAMES = {
    "factory_camera_validated",
    "production_action_model_ready",
    "external_factory_validated",
    "production_process_model_ready",
}
CORE_EXCLUDED_FIELDS = {
    "inference_time_ms",
    "processing_time_ms",
    "elapsed_time_ms",
}
MAX_REPORTED_ISSUES = 200


class Issues:
    """Bound reported issue detail while retaining an exact total."""

    def __init__(self) -> None:
        self.count = 0
        self.items: list[dict[str, Any]] = []

    def add(self, category: str, **detail: Any) -> None:
        self.count += 1
        if len(self.items) < MAX_REPORTED_ISSUES:
            self.items.append({"category": category, **detail})

    def payload(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "reported_count": len(self.items),
            "truncated": self.count > len(self.items),
            "items": self.items,
        }


def _require_exact_workspace() -> None:
    if str(Path.cwd().resolve()) != str(EXPECTED_WORKSPACE):
        raise RuntimeError("Exact workspace gate failed")
    if str(PROJECT_ROOT.resolve()) != str(EXPECTED_WORKSPACE):
        raise RuntimeError("Script project root does not match the workspace")


def _inside_project(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError(f"{label} must remain inside the project workspace")
    return resolved


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _value_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic_new(path: Path, payload: dict[str, Any]) -> Path:
    resolved = _inside_project(path, label="Output")
    if resolved.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {resolved}")
    temporary = resolved.with_suffix(resolved.suffix + ".part")
    if temporary.exists():
        raise FileExistsError(
            f"Interrupted output needs inspection before retry: {temporary}"
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(resolved)
    return resolved


def _record_key(record: dict[str, Any]) -> tuple[int, str, int, str]:
    return (
        int(record.get("frame_index", -1)),
        str(record.get("person_ref", "")),
        int(record.get("lock_epoch", -1)),
        str(record.get("anatomical_side", "")),
    )


def _key_text(key: tuple[int, str, int, str]) -> str:
    frame_index, person_ref, lock_epoch, side = key
    return (
        f"frame={frame_index}|person={person_ref}|"
        f"epoch={lock_epoch}|side={side}"
    )


def _pose_key(
    pose_frame: dict[str, Any],
    anatomical_side: str,
) -> tuple[int, str, int, str]:
    return (
        int(pose_frame["source_frame_index"]),
        str(pose_frame["person_ref"]),
        int(pose_frame["lock_epoch"]),
        anatomical_side,
    )


def _record_map(
    records: Iterable[dict[str, Any]],
) -> tuple[
    dict[tuple[int, str, int, str], dict[str, Any]],
    list[str],
]:
    mapped: dict[tuple[int, str, int, str], dict[str, Any]] = {}
    duplicate_keys: list[str] = []
    for record in records:
        key = _record_key(record)
        if key in mapped:
            duplicate_keys.append(_key_text(key))
            continue
        mapped[key] = record
    return mapped, duplicate_keys


def _validated_pose_frames(
    payload: dict[str, Any],
    *,
    analysis_path: Path,
) -> list[dict[str, Any]]:
    frames = payload.get("pose_frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"Accepted analysis has no pose_frames: {analysis_path}")
    required = {
        "source_frame_index",
        "timestamp",
        "person_ref",
        "lock_epoch",
        "track_state",
        "lock_state",
        "keypoints",
        "keypoint_statuses",
    }
    previous_frame = -1
    previous_timestamp = -math.inf
    seen: set[int] = set()
    validated: list[dict[str, Any]] = []
    for ordinal, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(
                f"pose_frames[{ordinal}] is not an object: {analysis_path}"
            )
        missing = sorted(required - set(frame))
        if missing:
            raise ValueError(
                f"pose_frames[{ordinal}] misses {missing}: {analysis_path}"
            )
        frame_index = int(frame["source_frame_index"])
        timestamp = float(frame["timestamp"])
        if frame_index in seen:
            raise ValueError(
                f"Duplicate source_frame_index {frame_index}: {analysis_path}"
            )
        if frame_index <= previous_frame:
            raise ValueError(
                f"Nonmonotonic source_frame_index at {ordinal}: {analysis_path}"
            )
        if timestamp <= previous_timestamp:
            raise ValueError(
                f"Nonmonotonic accepted timestamp at {ordinal}: {analysis_path}"
            )
        seen.add(frame_index)
        previous_frame = frame_index
        previous_timestamp = timestamp
        validated.append(frame)
    return validated


def _clip_id(analysis_path: Path) -> str:
    candidate = analysis_path.parent.parent.name
    if not candidate or candidate.lower() in {"candidate", "replay"}:
        candidate = analysis_path.parent.name
    if not candidate or any(char in candidate for char in "\\/:"):
        raise ValueError(f"Cannot derive a safe clip id from {analysis_path}")
    return candidate


def _resolve_source_video(
    analysis_path: Path,
    payload: dict[str, Any],
) -> tuple[Path, str]:
    local_video = _inside_project(
        analysis_path.parent / "source_video.mp4",
        label="Source video",
    )
    if not local_video.is_file():
        raise FileNotFoundError(
            f"Project-local source_video.mp4 is missing: {local_video}"
        )
    metadata = payload.get("source_video")
    if not isinstance(metadata, dict):
        raise ValueError(f"source_video metadata is missing: {analysis_path}")
    declared_path = metadata.get("path")
    if declared_path:
        path_value = Path(str(declared_path))
        if not path_value.is_absolute():
            path_value = PROJECT_ROOT / path_value
        declared = _inside_project(path_value, label="Declared source video")
        if declared != local_video:
            raise ValueError(
                "Accepted source_video.path does not identify the adjacent "
                f"source_video.mp4: {analysis_path}"
            )
    expected_hash = str(metadata.get("sha256", "")).lower()
    actual_hash = _file_sha256(local_video)
    if not expected_hash or actual_hash != expected_hash:
        raise ValueError(
            f"Source video SHA256 mismatch for {analysis_path}: "
            f"expected={expected_hash or 'missing'} actual={actual_hash}"
        )
    return local_video, actual_hash


def _manifest_model() -> tuple[Path, str | None]:
    if not HAND_MODEL_MANIFEST.is_file():
        return DEFAULT_MODEL_PATH.resolve(), None
    manifest = _load_json(HAND_MODEL_MANIFEST)
    model = manifest.get("model")
    if not isinstance(model, dict):
        return DEFAULT_MODEL_PATH.resolve(), None
    local_path = Path(
        str(model.get("local_path") or DEFAULT_MODEL_PATH.relative_to(PROJECT_ROOT))
    )
    if not local_path.is_absolute():
        local_path = PROJECT_ROOT / local_path
    return (
        _inside_project(local_path, label="Hand model"),
        str(model.get("sha256", "")).lower() or None,
    )


def _analysis_model_path(payload: dict[str, Any]) -> Path | None:
    hand_model = payload.get("hand_model")
    if not isinstance(hand_model, dict):
        return None
    value = hand_model.get("local_path") or hand_model.get("model_path")
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return _inside_project(path, label="Analysis Hand model")


def _expected_model_hashes(payloads: Iterable[dict[str, Any]]) -> set[str]:
    hashes: set[str] = set()
    for payload in payloads:
        hand_model = payload.get("hand_model")
        if isinstance(hand_model, dict) and hand_model.get("sha256"):
            hashes.add(str(hand_model["sha256"]).lower())
        for record in payload.get("hand_pose_frames", []):
            version = str(record.get("source_model_version", ""))
            marker = "sha256:"
            if marker in version:
                hashes.add(version.rsplit(marker, 1)[1].lower())
    return {value for value in hashes if len(value) == 64}


def _resolve_model(
    override: Path | None,
    payloads: list[dict[str, Any]],
) -> tuple[Path, str]:
    manifest_path, manifest_hash = _manifest_model()
    analysis_paths = {
        path
        for path in (_analysis_model_path(payload) for payload in payloads)
        if path is not None
    }
    if len(analysis_paths) > 1:
        raise ValueError("Accepted analyses declare different Hand model paths")
    if override is not None:
        selected = _inside_project(override, label="Hand model override")
    elif analysis_paths:
        selected = next(iter(analysis_paths))
    else:
        selected = manifest_path
    if not selected.is_file():
        raise FileNotFoundError(f"Hand model is missing: {selected}")
    actual_hash = _file_sha256(selected)
    expected = _expected_model_hashes(payloads)
    if manifest_hash:
        expected.add(manifest_hash)
    if expected and expected != {actual_hash}:
        raise ValueError(
            "Hand model hashes do not resolve to one frozen model: "
            f"expected={sorted(expected)} actual={actual_hash}"
        )
    return selected, actual_hash


def _model_version(payload: dict[str, Any], model_hash: str) -> str:
    versions = {
        str(record.get("source_model_version"))
        for record in payload.get("hand_pose_frames", [])
        if record.get("source_model_version")
    }
    if len(versions) > 1:
        raise ValueError("Frozen Hand records contain multiple model versions")
    if versions:
        return next(iter(versions))
    hand_model = payload.get("hand_model")
    if isinstance(hand_model, dict):
        version = hand_model.get("version") or hand_model.get("model_version")
        if version:
            return str(version)
    return f"mediapipe_hand_landmarker+sha256:{model_hash}"


def _recording_group_id(payload: dict[str, Any]) -> str:
    source = payload.get("source_video")
    if isinstance(source, dict) and source.get("recording_group_id"):
        return str(source["recording_group_id"])
    records = payload.get("hand_pose_frames", [])
    groups = {
        str(record.get("recording_group_id"))
        for record in records
        if record.get("recording_group_id")
    }
    if len(groups) != 1:
        raise ValueError("Cannot resolve one recording_group_id")
    return next(iter(groups))


def _backend_metrics(
    records: list[dict[str, Any]],
    *,
    backend: Any | None,
    frozen_runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    observations: Counter[str] = Counter()
    qualities: Counter[str] = Counter()
    validations: Counter[str] = Counter()
    warnings: Counter[str] = Counter()
    warning_records = 0
    duplicate_records = 0
    duplicate_frames: set[int] = set()
    eligible_records = 0
    eligible_frames: set[int] = set()
    geometry_frames: set[int] = set()
    for record in records:
        observation = str(record.get("observation_state", "missing")).lower()
        if observation in {"predicted", "interpolated"}:
            observation = "uncertain"
        if observation in {"lost", "off_frame"}:
            observation = "missing"
        observations[observation] += 1
        qualities[str(record.get("quality_state", "unknown")).lower()] += 1
        validations[
            str(record.get("validation_state", "unknown")).lower()
        ] += 1
        record_warnings = list(
            (record.get("association_checks") or {}).get("warnings", [])
        )
        if record_warnings:
            warning_records += 1
            warnings.update(str(item) for item in record_warnings)
        duplicate = bool(
            (record.get("association_checks") or {}).get(
                "duplicate_across_sides",
                False,
            )
        )
        if duplicate:
            duplicate_records += 1
            duplicate_frames.add(int(record.get("frame_index", -1)))
        if record.get("action_feature_eligible") is True:
            eligible_records += 1
            eligible_frames.add(int(record.get("frame_index", -1)))
        if record.get("landmarks"):
            geometry_frames.add(int(record.get("frame_index", -1)))

    if backend is not None:
        timings = [
            float(value)
            for value in getattr(backend, "inference_times_ms", [])
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        ]
        inference_calls = int(getattr(backend, "inference_call_count", 0))
        mean_ms = sum(timings) / len(timings) if timings else None
        error_count = int(getattr(backend, "inference_error_count", 0))
    else:
        runtime = frozen_runtime or {}
        inference_calls = int(runtime.get("hand_inference_calls", 0))
        raw_mean = runtime.get("mean_hand_inference_ms")
        mean_ms = float(raw_mean) if raw_mean is not None else None
        error_count = int(runtime.get("hand_inference_error_count", 0))

    return {
        "record_count": len(records),
        "observation_counts": dict(sorted(observations.items())),
        "quality_state_counts": dict(sorted(qualities.items())),
        "validation_state_counts": dict(sorted(validations.items())),
        "action_feature_eligible_observation_count": eligible_records,
        "action_feature_eligible_frame_count": len(eligible_frames),
        "geometry_frame_count": len(geometry_frames),
        "association_warning_record_count": warning_records,
        "association_warning_occurrence_count": sum(warnings.values()),
        "association_warning_counts": dict(sorted(warnings.items())),
        "duplicate_record_count": duplicate_records,
        "duplicate_frame_count": len(duplicate_frames),
        "inference_call_count": inference_calls,
        "mean_inference_ms": round(mean_ms, 6) if mean_ms is not None else None,
        "inference_error_count": error_count,
    }


def _true_runs(values: list[bool]) -> list[int]:
    runs: list[int] = []
    active = 0
    for value in values:
        if value:
            active += 1
        elif active:
            runs.append(active)
            active = 0
    if active:
        runs.append(active)
    return runs


def _run_payload(values: list[bool]) -> dict[str, Any]:
    runs = _true_runs(values)
    return {
        "positive_frame_count": sum(values),
        "run_count": len(runs),
        "isolated_positive_frame_count": sum(run == 1 for run in runs),
        "longest_run_frames": max(runs, default=0),
        "mean_run_frames": (
            round(sum(runs) / len(runs), 6) if runs else 0.0
        ),
    }


def _continuity_metrics(
    records: list[dict[str, Any]],
    pose_frames: list[dict[str, Any]],
) -> dict[str, Any]:
    record_map, duplicate_keys = _record_map(records)
    sides: dict[str, Any] = {}
    for side in ANATOMICAL_SIDES:
        geometry_values: list[bool] = []
        detected_values: list[bool] = []
        qualified_values: list[bool] = []
        eligible_values: list[bool] = []
        state_values: list[str] = []
        for frame in pose_frames:
            record = record_map.get(_pose_key(frame, side), {})
            geometry_values.append(bool(record.get("landmarks")))
            detected_values.append(
                str(record.get("observation_state", "")).lower()
                == "detected"
            )
            qualified_values.append(
                str(record.get("quality_state", "")).lower() == "qualified"
            )
            eligible_values.append(
                record.get("action_feature_eligible") is True
            )
            state_values.append(
                str(record.get("observation_state", "missing")).lower()
            )
        sides[side] = {
            "geometry": _run_payload(geometry_values),
            "detected": _run_payload(detected_values),
            "qualified": _run_payload(qualified_values),
            "eligible": _run_payload(eligible_values),
            "observation_state_switch_count": sum(
                previous != current
                for previous, current in zip(
                    state_values,
                    state_values[1:],
                )
            ),
        }
    return {
        "frame_count": len(pose_frames),
        "duplicate_record_key_count": len(duplicate_keys),
        "sides": sides,
    }


def _complete_detected_geometry(record: dict[str, Any]) -> bool:
    landmarks = record.get("landmarks")
    if not isinstance(landmarks, list) or len(landmarks) != 21:
        return False
    indices: list[int] = []
    for landmark in landmarks:
        if not isinstance(landmark, dict):
            return False
        index = landmark.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            return False
        indices.append(index)
    return set(indices) == set(range(21))


def _safety_audit(
    records: list[dict[str, Any]],
    pose_frames: list[dict[str, Any]],
    *,
    expected_mode: str,
    source_width: int,
    source_height: int,
) -> dict[str, Any]:
    boundary = Issues()
    geometry = Issues()
    training = Issues()
    timestamp = Issues()
    session = Issues()
    duplicate_eligible_count = 0
    missing_or_lost_geometry_count = 0
    nonmonotonic_accepted_count = 0
    cross_person_epoch_session_count = 0

    pose_by_index = {
        int(frame["source_frame_index"]): frame for frame in pose_frames
    }
    expected_keys = {
        _pose_key(frame, side)
        for frame in pose_frames
        for side in ANATOMICAL_SIDES
    }
    mapped, duplicate_keys = _record_map(records)
    actual_keys = set(mapped)
    for key in sorted(expected_keys - actual_keys):
        session.add("missing_expected_side_record", record_key=_key_text(key))
    for key in sorted(actual_keys - expected_keys):
        cross_person_epoch_session_count += 1
        session.add("unexpected_context_record", record_key=_key_text(key))
    for value in duplicate_keys:
        session.add("duplicate_context_record", record_key=value)

    seen_ids: set[str] = set()
    previous_by_side: dict[str, tuple[float, tuple[str, int]]] = {}
    previous_by_video_session: dict[tuple[str, int], int] = {}
    video_session_geometry_contexts: dict[
        tuple[str, int],
        set[tuple[str, int]],
    ] = {}
    for record in records:
        key = _record_key(record)
        frame_index, person_ref, lock_epoch, side = key
        record_id = str(record.get("hand_pose_id", _key_text(key)))
        if record_id in seen_ids:
            session.add("duplicate_hand_pose_id", hand_pose_id=record_id)
        seen_ids.add(record_id)
        pose = pose_by_index.get(frame_index)
        if pose is None:
            cross_person_epoch_session_count += 1
            boundary.add(
                "record_has_no_frozen_pose_frame",
                hand_pose_id=record_id,
                frame_index=frame_index,
            )
            continue

        expected_context = (
            str(pose["person_ref"]),
            int(pose["lock_epoch"]),
        )
        if (person_ref, lock_epoch) != expected_context:
            cross_person_epoch_session_count += 1
            boundary.add(
                "person_or_epoch_context_mismatch",
                hand_pose_id=record_id,
                actual=[person_ref, lock_epoch],
                expected=list(expected_context),
            )
        if side not in ANATOMICAL_SIDES:
            cross_person_epoch_session_count += 1
            boundary.add(
                "invalid_anatomical_side",
                hand_pose_id=record_id,
                anatomical_side=side,
            )
        mode = str(record.get("backend_mode", "")).lower()
        if mode != expected_mode:
            session.add(
                "backend_mode_mismatch",
                hand_pose_id=record_id,
                actual=mode,
                expected=expected_mode,
            )

        record_timestamp = float(record.get("timestamp", math.nan))
        pose_timestamp = float(pose["timestamp"])
        if (
            not math.isfinite(record_timestamp)
            or abs(record_timestamp - pose_timestamp) > 1e-6
        ):
            timestamp.add(
                "record_timestamp_mismatch",
                hand_pose_id=record_id,
                actual=record.get("timestamp"),
                expected=pose_timestamp,
            )
        previous = previous_by_side.get(side)
        accepted = bool(record.get("landmarks")) or (
            str(record.get("observation_state", "")).lower() == "detected"
        )
        context = (person_ref, lock_epoch)
        if previous is not None and previous[1] == context:
            if record_timestamp <= previous[0]:
                timestamp.add(
                    "nonmonotonic_side_timestamp",
                    hand_pose_id=record_id,
                    previous=previous[0],
                    current=record_timestamp,
                )
                if accepted:
                    nonmonotonic_accepted_count += 1
        previous_by_side[side] = (record_timestamp, context)

        if expected_mode == "video":
            backend_timestamp = record.get("backend_timestamp_ms")
            expected_backend_timestamp = int(round(pose_timestamp * 1000.0))
            if (
                isinstance(backend_timestamp, bool)
                or not isinstance(backend_timestamp, int)
                or backend_timestamp != expected_backend_timestamp
            ):
                timestamp.add(
                    "backend_timestamp_mismatch",
                    hand_pose_id=record_id,
                    actual=backend_timestamp,
                    expected=expected_backend_timestamp,
                )
            generation = record.get("tracker_session_generation")
            if isinstance(generation, bool) or not isinstance(generation, int):
                session.add(
                    "invalid_tracker_session_generation",
                    hand_pose_id=record_id,
                    value=generation,
                )
            elif record.get("landmarks"):
                session_key = (side, generation)
                video_session_geometry_contexts.setdefault(
                    session_key,
                    set(),
                ).add(context)
                previous_backend_timestamp = previous_by_video_session.get(
                    session_key
                )
                if (
                    previous_backend_timestamp is not None
                    and isinstance(backend_timestamp, int)
                    and backend_timestamp <= previous_backend_timestamp
                ):
                    nonmonotonic_accepted_count += 1
                    timestamp.add(
                        "nonmonotonic_accepted_video_session_timestamp",
                        hand_pose_id=record_id,
                        session_generation=generation,
                        previous=previous_backend_timestamp,
                        current=backend_timestamp,
                    )
                if isinstance(backend_timestamp, int):
                    previous_by_video_session[session_key] = backend_timestamp

        landmarks = record.get("landmarks")
        landmark_count = int(record.get("landmark_count", -1))
        observation = str(
            record.get("observation_state", "missing")
        ).lower()
        if observation in {"missing", "lost", "off_frame"} and (
            landmarks != [] or landmark_count != 0
        ):
            missing_or_lost_geometry_count += 1
            geometry.add(
                "missing_or_lost_has_geometry",
                hand_pose_id=record_id,
                observation_state=observation,
                landmark_count=landmark_count,
            )
        if observation == "detected" and not _complete_detected_geometry(record):
            geometry.add(
                "detected_without_21_unique_landmarks",
                hand_pose_id=record_id,
                landmark_count=landmark_count,
            )
        if isinstance(landmarks, list):
            if landmark_count != len(landmarks):
                geometry.add(
                    "landmark_count_mismatch",
                    hand_pose_id=record_id,
                    declared=landmark_count,
                    actual=len(landmarks),
                )
            for landmark in landmarks:
                if not isinstance(landmark, dict):
                    geometry.add(
                        "landmark_not_object",
                        hand_pose_id=record_id,
                    )
                    continue
                try:
                    x_value = float(landmark.get("x"))
                    y_value = float(landmark.get("y"))
                except (TypeError, ValueError):
                    x_value = math.nan
                    y_value = math.nan
                if not math.isfinite(x_value) or not math.isfinite(y_value):
                    geometry.add(
                        "nonfinite_landmark",
                        hand_pose_id=record_id,
                        landmark_index=landmark.get("index"),
                    )
                elif not (
                    0.0 <= x_value < float(source_width)
                    and 0.0 <= y_value < float(source_height)
                ):
                    geometry.add(
                        "landmark_outside_source_frame",
                        hand_pose_id=record_id,
                        landmark_index=landmark.get("index"),
                        point=[x_value, y_value],
                    )

        tracked = str(pose.get("track_state", "")).lower() == "tracked"
        lock_valid = (
            str(pose.get("lock_state", "")).lower() in VALID_LOCK_STATES
        )
        if (not tracked or not lock_valid) and landmarks:
            boundary.add(
                "geometry_on_tracking_or_lock_boundary",
                hand_pose_id=record_id,
                track_state=pose.get("track_state"),
                lock_state=pose.get("lock_state"),
            )

        eligible = record.get("action_feature_eligible") is True
        duplicate = bool(
            (record.get("association_checks") or {}).get(
                "duplicate_across_sides",
                False,
            )
        )
        if duplicate and eligible:
            duplicate_eligible_count += 1
            geometry.add(
                "duplicate_record_marked_eligible",
                hand_pose_id=record_id,
            )
        if record.get("training_eligible") is not False:
            training.add(
                "automatic_record_training_eligible",
                hand_pose_id=record_id,
                value=record.get("training_eligible"),
            )
        if str(record.get("status", "")).lower() == "confirmed":
            training.add(
                "automatic_record_confirmed",
                hand_pose_id=record_id,
            )
        if record.get("reviewer") is not None or record.get("reviewed_at") is not None:
            training.add(
                "automatic_record_has_human_review_identity",
                hand_pose_id=record_id,
            )

        checks = record.get("association_checks")
        if isinstance(checks, dict):
            declared_session = checks.get("session_context")
            if isinstance(declared_session, dict):
                declared = (
                    str(declared_session.get("person_ref", "")),
                    int(declared_session.get("lock_epoch", -1)),
                    str(declared_session.get("anatomical_side", "")),
                )
                expected = (person_ref, lock_epoch, side)
                if declared != expected:
                    cross_person_epoch_session_count += 1
                    session.add(
                        "declared_session_context_mismatch",
                        hand_pose_id=record_id,
                        actual=list(declared),
                        expected=list(expected),
                    )

    for session_key, contexts in video_session_geometry_contexts.items():
        if len(contexts) <= 1:
            continue
        cross_person_epoch_session_count += len(contexts) - 1
        session.add(
            "tracker_session_geometry_crosses_person_or_epoch",
            anatomical_side=session_key[0],
            tracker_session_generation=session_key[1],
            contexts=[list(item) for item in sorted(contexts)],
        )

    payloads = {
        "boundary": boundary.payload(),
        "geometry": geometry.payload(),
        "training": training.payload(),
        "timestamp": timestamp.payload(),
        "session": session.payload(),
    }
    gates = {
        "missing_or_lost_geometry_zero": (
            missing_or_lost_geometry_count == 0
        ),
        "nonmonotonic_accepted_zero": nonmonotonic_accepted_count == 0,
        "cross_person_epoch_session_zero": (
            cross_person_epoch_session_count == 0
        ),
        "duplicate_eligible_zero": duplicate_eligible_count == 0,
        "boundary_violations_zero": boundary.count == 0,
        "geometry_violations_zero": geometry.count == 0,
        "training_violations_zero": training.count == 0,
        "timestamp_violations_zero": timestamp.count == 0,
        "session_violations_zero": session.count == 0,
    }
    return {
        "status": "passed" if all(gates.values()) else "failed",
        "gates": gates,
        "required_zero_counts": {
            "missing_or_lost_geometry": missing_or_lost_geometry_count,
            "nonmonotonic_accepted": nonmonotonic_accepted_count,
            "cross_person_epoch_session": cross_person_epoch_session_count,
            "duplicate_eligible": duplicate_eligible_count,
        },
        "violations": payloads,
    }


def _frozen_core_comparison(
    frozen: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> dict[str, Any]:
    frozen_map, frozen_duplicates = _record_map(frozen)
    candidate_map, candidate_duplicates = _record_map(candidate)
    keys = sorted(
        {
            field
            for record in frozen
            for field in record
            if field not in CORE_EXCLUDED_FIELDS
        }
    )

    def project(
        records: dict[
            tuple[int, str, int, str],
            dict[str, Any],
        ],
    ) -> list[dict[str, Any]]:
        return [
            {
                "_comparison_key": _key_text(key),
                **{
                    field: (
                        record[field]
                        if field in record
                        else {"__field_missing__": True}
                    )
                    for field in keys
                },
            }
            for key, record in sorted(records.items())
        ]

    frozen_projected = project(frozen_map)
    candidate_projected = project(candidate_map)
    frozen_hash = _value_sha256(frozen_projected)
    candidate_hash = _value_sha256(candidate_projected)
    mismatches = Issues()
    for key in sorted(frozen_map.keys() - candidate_map.keys()):
        mismatches.add("missing_candidate_record", record_key=_key_text(key))
    for key in sorted(candidate_map.keys() - frozen_map.keys()):
        mismatches.add("extra_candidate_record", record_key=_key_text(key))
    for key in sorted(frozen_map.keys() & candidate_map.keys()):
        left = {
            field: frozen_map[key].get(
                field,
                {"__field_missing__": True},
            )
            for field in keys
        }
        right = {
            field: candidate_map[key].get(
                field,
                {"__field_missing__": True},
            )
            for field in keys
        }
        changed = sorted(
            field for field in keys if left[field] != right[field]
        )
        if changed:
            mismatches.add(
                "changed_core_fields",
                record_key=_key_text(key),
                fields=changed,
                frozen_sha256=_value_sha256(left),
                candidate_sha256=_value_sha256(right),
            )
    for key in frozen_duplicates:
        mismatches.add("duplicate_frozen_record_key", record_key=key)
    for key in candidate_duplicates:
        mismatches.add("duplicate_candidate_record_key", record_key=key)
    return {
        "alignment_key": [
            "source_frame_index/frame_index",
            "person_ref",
            "lock_epoch",
            "anatomical_side",
        ],
        "excluded_timing_fields": sorted(CORE_EXCLUDED_FIELDS),
        "compared_core_fields": keys,
        "frozen_record_count": len(frozen),
        "candidate_record_count": len(candidate),
        "frozen_core_sha256": frozen_hash,
        "candidate_core_sha256": candidate_hash,
        "match": (
            frozen_hash == candidate_hash
            and mismatches.count == 0
        ),
        "mismatches": mismatches.payload(),
    }


def _evaluation_payload() -> dict[str, Any]:
    return {
        "status": "not_evaluable",
        "accuracy": "not_evaluable",
        "precision": "not_evaluable",
        "recall": "not_evaluable",
        "reason": (
            "No independently human-confirmed Hand landmark, association, "
            "or continuity ground truth is available for these windows."
        ),
    }


def _candidate_payload(
    *,
    clip_id: str,
    mode: str,
    analysis_path: Path,
    source_video: Path,
    source_video_sha256: str,
    model_path: Path,
    model_sha256: str,
    model_version: str,
    pose_frames: list[dict[str, Any]],
    records: list[dict[str, Any]],
    metrics: dict[str, Any],
    continuity: dict[str, Any],
    safety: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "schema_version": "hand_video_mode_candidate_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "clip_id": clip_id,
        "mode": mode,
        "source": {
            "accepted_analysis": str(analysis_path),
            "source_video": str(source_video),
            "source_video_sha256": source_video_sha256,
            "pose_frame_count": len(pose_frames),
            "pose_frames_sha256": _value_sha256(pose_frames),
            "body_pose_recomputed": False,
        },
        "model": {
            "path": str(model_path),
            "sha256": model_sha256,
            "version": model_version,
            "device": "CPU",
        },
        "experiment_controls": {
            "same_source_frame_indices": True,
            "same_frozen_body_pose": True,
            "roi_smoothing_enabled": False,
            "decoder_path": (
                "cv2_seek_to_first_accepted_source_frame_then_sequential"
            ),
            "decoder_first_source_frame_index": min(
                int(frame["source_frame_index"]) for frame in pose_frames
            ),
            "model_training_performed": False,
        },
        "metrics": metrics,
        "continuity": continuity,
        "safety": safety,
        "processing_elapsed_seconds": round(elapsed_seconds, 6),
        "evaluation": _evaluation_payload(),
        "hand_pose_frames": records,
    }


def _decode_and_run_modes(
    *,
    payload: dict[str, Any],
    pose_frames: list[dict[str, Any]],
    source_video: Path,
    source_video_sha256: str,
    model_path: Path,
    model_version: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    Any,
    Any,
    dict[str, float],
]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for replay") from exc
    try:
        from src.hand_pose import (
            MediaPipeHandLandmarkerBackend,
            MediaPipeHandLandmarkerVideoBackend,
        )
    except ImportError as exc:
        raise RuntimeError(
            "IMAGE and VIDEO Hand backends must be available from src.hand_pose"
        ) from exc

    image_backend = MediaPipeHandLandmarkerBackend(
        model_path,
        model_version=model_version,
    )
    video_backend = MediaPipeHandLandmarkerVideoBackend(
        model_path,
        model_version=model_version,
    )
    image_records: list[dict[str, Any]] = []
    video_records: list[dict[str, Any]] = []
    mode_elapsed = {"image": 0.0, "video": 0.0}
    recording_group_id = _recording_group_id(payload)
    frames_by_index = {
        int(frame["source_frame_index"]): frame for frame in pose_frames
    }
    target_indices = sorted(frames_by_index)
    target_position = 0
    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        for backend in (image_backend, video_backend):
            close = getattr(backend, "close", None)
            if callable(close):
                close()
        raise RuntimeError(f"Cannot open source video: {source_video}")
    try:
        first_target_index = target_indices[0]
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, first_target_index):
            raise RuntimeError(
                "OpenCV could not seek to the accepted analysis-window "
                f"start frame {first_target_index}: {source_video}"
            )
        frame_index = first_target_index
        while target_position < len(target_indices):
            ok, image = capture.read()
            if not ok or image is None:
                raise RuntimeError(
                    "Video ended before accepted source frame "
                    f"{target_indices[target_position]} in {source_video}"
                )
            target = target_indices[target_position]
            if frame_index == target:
                pose = frames_by_index[target]
                kwargs = {
                    "body_keypoints": pose["keypoints"],
                    "body_keypoint_statuses": pose["keypoint_statuses"],
                    "person_ref": str(pose["person_ref"]),
                    "lock_epoch": int(pose["lock_epoch"]),
                    "frame_index": target,
                    "timestamp": float(pose["timestamp"]),
                    "source_video_sha256": source_video_sha256,
                    "recording_group_id": recording_group_id,
                    "track_state": str(pose["track_state"]),
                    "lock_state": str(pose["lock_state"]),
                }
                started = perf_counter()
                image_records.extend(
                    image_backend.infer_frame(image, **kwargs)
                )
                mode_elapsed["image"] += perf_counter() - started
                started = perf_counter()
                video_records.extend(
                    video_backend.infer_frame(image, **kwargs)
                )
                mode_elapsed["video"] += perf_counter() - started
                target_position += 1
            frame_index += 1
    finally:
        capture.release()
        for backend in (image_backend, video_backend):
            close = getattr(backend, "close", None)
            if callable(close):
                close()
    return (
        image_records,
        video_records,
        image_backend,
        video_backend,
        mode_elapsed,
    )


def _clip_comparison(
    *,
    analysis_path: Path,
    payload: dict[str, Any],
    pose_frames: list[dict[str, Any]],
    frozen_records: list[dict[str, Any]],
    image_records: list[dict[str, Any]],
    video_records: list[dict[str, Any]],
    image_metrics: dict[str, Any],
    video_metrics: dict[str, Any],
    image_continuity: dict[str, Any],
    video_continuity: dict[str, Any],
    image_safety: dict[str, Any],
    video_safety: dict[str, Any],
) -> dict[str, Any]:
    frozen_runtime = payload.get("runtime")
    if not isinstance(frozen_runtime, dict):
        frozen_runtime = {}
    frozen_metrics = _backend_metrics(
        frozen_records,
        backend=None,
        frozen_runtime=frozen_runtime,
    )
    core = _frozen_core_comparison(frozen_records, image_records)
    gates = {
        "frozen_image_core_matches_candidate_image": core["match"],
        "same_image_and_video_record_count": (
            len(image_records) == len(video_records)
            == len(pose_frames) * len(ANATOMICAL_SIDES)
        ),
        "candidate_image_safety_passed": image_safety["status"] == "passed",
        "candidate_video_safety_passed": video_safety["status"] == "passed",
        "four_validation_flags_false": (
            isinstance(payload.get("validation_flags"), dict)
            and VALIDATION_FLAG_NAMES.issubset(payload["validation_flags"])
            and all(
                payload["validation_flags"][name] is False
                for name in VALIDATION_FLAG_NAMES
            )
        ),
        "accuracy_precision_recall_not_evaluable": True,
    }
    return {
        "clip_id": _clip_id(analysis_path),
        "status": "passed" if all(gates.values()) else "failed",
        "accepted_analysis": str(analysis_path),
        "pose_frame_count": len(pose_frames),
        "pose_frames_sha256": _value_sha256(pose_frames),
        "frozen_image_vs_candidate_image": core,
        "frozen_image_metrics": frozen_metrics,
        "candidate_image_metrics": image_metrics,
        "candidate_video_metrics": video_metrics,
        "candidate_image_continuity": image_continuity,
        "candidate_video_continuity": video_continuity,
        "candidate_image_safety": image_safety,
        "candidate_video_safety": video_safety,
        "mode_delta_video_minus_image": {
            "detected_observations": (
                video_metrics["observation_counts"].get("detected", 0)
                - image_metrics["observation_counts"].get("detected", 0)
            ),
            "uncertain_observations": (
                video_metrics["observation_counts"].get("uncertain", 0)
                - image_metrics["observation_counts"].get("uncertain", 0)
            ),
            "missing_observations": (
                video_metrics["observation_counts"].get("missing", 0)
                - image_metrics["observation_counts"].get("missing", 0)
            ),
            "eligible_observations": (
                video_metrics[
                    "action_feature_eligible_observation_count"
                ]
                - image_metrics[
                    "action_feature_eligible_observation_count"
                ]
            ),
            "association_warning_occurrences": (
                video_metrics["association_warning_occurrence_count"]
                - image_metrics["association_warning_occurrence_count"]
            ),
            "duplicate_frames": (
                video_metrics["duplicate_frame_count"]
                - image_metrics["duplicate_frame_count"]
            ),
        },
        "gates": gates,
        "evaluation": _evaluation_payload(),
    }


def _aggregate_mode(
    clips: list[dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    observations: Counter[str] = Counter()
    qualities: Counter[str] = Counter()
    warnings: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    weighted_ms = 0.0
    timed_calls = 0
    for clip in clips:
        metrics = clip[field]
        observations.update(metrics["observation_counts"])
        qualities.update(metrics["quality_state_counts"])
        warnings.update(metrics["association_warning_counts"])
        for name in (
            "record_count",
            "action_feature_eligible_observation_count",
            "action_feature_eligible_frame_count",
            "geometry_frame_count",
            "association_warning_record_count",
            "association_warning_occurrence_count",
            "duplicate_record_count",
            "duplicate_frame_count",
            "inference_call_count",
            "inference_error_count",
        ):
            totals[name] += int(metrics[name])
        if metrics["mean_inference_ms"] is not None:
            calls = int(metrics["inference_call_count"])
            weighted_ms += float(metrics["mean_inference_ms"]) * calls
            timed_calls += calls
    return {
        **dict(totals),
        "observation_counts": dict(sorted(observations.items())),
        "quality_state_counts": dict(sorted(qualities.items())),
        "association_warning_counts": dict(sorted(warnings.items())),
        "mean_inference_ms": (
            round(weighted_ms / timed_calls, 6) if timed_calls else None
        ),
    }


def _aggregate_safety(
    clips: list[dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    required: Counter[str] = Counter()
    violation_counts: Counter[str] = Counter()
    for clip in clips:
        safety = clip[field]
        required.update(safety["required_zero_counts"])
        for category, detail in safety["violations"].items():
            violation_counts[category] += int(detail["count"])
    gates = {
        "missing_or_lost_geometry_zero": (
            required["missing_or_lost_geometry"] == 0
        ),
        "nonmonotonic_accepted_zero": (
            required["nonmonotonic_accepted"] == 0
        ),
        "cross_person_epoch_session_zero": (
            required["cross_person_epoch_session"] == 0
        ),
        "duplicate_eligible_zero": required["duplicate_eligible"] == 0,
        "all_violation_categories_zero": not any(
            violation_counts.values()
        ),
    }
    return {
        "status": "passed" if all(gates.values()) else "failed",
        "gates": gates,
        "required_zero_counts": dict(required),
        "violation_counts": dict(violation_counts),
    }


def run_comparison(
    analysis_paths: list[Path],
    *,
    output_root: Path,
    model_override: Path | None,
    summary_output: Path | None,
) -> tuple[dict[str, Any], list[tuple[Path, dict[str, Any]]]]:
    if len(analysis_paths) != 3:
        raise ValueError("Exactly three accepted analysis.json paths are required")
    resolved_analyses = [
        _inside_project(path, label="Accepted analysis")
        for path in analysis_paths
    ]
    if len(set(resolved_analyses)) != 3:
        raise ValueError("The three accepted analysis paths must be unique")
    for path in resolved_analyses:
        if path.name != "analysis.json" or not path.is_file():
            raise FileNotFoundError(f"Accepted analysis.json is missing: {path}")
    payloads = [_load_json(path) for path in resolved_analyses]
    clip_ids = [_clip_id(path) for path in resolved_analyses]
    if len(set(clip_ids)) != 3:
        raise ValueError(f"Clip ids must be unique: {clip_ids}")

    output_root = _inside_project(output_root, label="Output root")
    model_path, model_hash = _resolve_model(model_override, payloads)
    summary_path = (
        _inside_project(summary_output, label="Summary output")
        if summary_output is not None
        else output_root / "HAND_VIDEO_MODE_COMPARISON.json"
    )
    planned_outputs = [summary_path]
    for clip_id in clip_ids:
        planned_outputs.extend(
            [
                output_root / clip_id / "candidate_image_hand.json",
                output_root / clip_id / "candidate_video_hand.json",
            ]
        )
    for path in planned_outputs:
        if path.exists() or path.with_suffix(path.suffix + ".part").exists():
            raise FileExistsError(
                f"Refusing to overwrite or bypass interrupted output: {path}"
            )

    clip_reports: list[dict[str, Any]] = []
    artifacts: list[tuple[Path, dict[str, Any]]] = []
    for analysis_path, payload in zip(
        resolved_analyses,
        payloads,
        strict=True,
    ):
        clip_id = _clip_id(analysis_path)
        pose_frames = _validated_pose_frames(
            payload,
            analysis_path=analysis_path,
        )
        frozen_records = list(payload.get("hand_pose_frames", []))
        if len(frozen_records) != len(pose_frames) * len(ANATOMICAL_SIDES):
            raise ValueError(
                f"Frozen Hand record count does not match pose frames: {clip_id}"
            )
        source_video, source_hash = _resolve_source_video(
            analysis_path,
            payload,
        )
        source_metadata = payload.get("source_video", {})
        source_width = int(source_metadata.get("width", 0))
        source_height = int(source_metadata.get("height", 0))
        if source_width <= 0 or source_height <= 0:
            raise ValueError(f"Invalid source dimensions: {analysis_path}")
        model_version = _model_version(payload, model_hash)
        (
            image_records,
            video_records,
            image_backend,
            video_backend,
            mode_elapsed,
        ) = _decode_and_run_modes(
            payload=payload,
            pose_frames=pose_frames,
            source_video=source_video,
            source_video_sha256=source_hash,
            model_path=model_path,
            model_version=model_version,
        )
        image_metrics = _backend_metrics(
            image_records,
            backend=image_backend,
        )
        video_metrics = _backend_metrics(
            video_records,
            backend=video_backend,
        )
        image_continuity = _continuity_metrics(
            image_records,
            pose_frames,
        )
        video_continuity = _continuity_metrics(
            video_records,
            pose_frames,
        )
        image_safety = _safety_audit(
            image_records,
            pose_frames,
            expected_mode="image",
            source_width=source_width,
            source_height=source_height,
        )
        video_safety = _safety_audit(
            video_records,
            pose_frames,
            expected_mode="video",
            source_width=source_width,
            source_height=source_height,
        )
        image_payload = _candidate_payload(
            clip_id=clip_id,
            mode="image",
            analysis_path=analysis_path,
            source_video=source_video,
            source_video_sha256=source_hash,
            model_path=model_path,
            model_sha256=model_hash,
            model_version=model_version,
            pose_frames=pose_frames,
            records=image_records,
            metrics=image_metrics,
            continuity=image_continuity,
            safety=image_safety,
            elapsed_seconds=mode_elapsed["image"],
        )
        video_payload = _candidate_payload(
            clip_id=clip_id,
            mode="video",
            analysis_path=analysis_path,
            source_video=source_video,
            source_video_sha256=source_hash,
            model_path=model_path,
            model_sha256=model_hash,
            model_version=model_version,
            pose_frames=pose_frames,
            records=video_records,
            metrics=video_metrics,
            continuity=video_continuity,
            safety=video_safety,
            elapsed_seconds=mode_elapsed["video"],
        )
        clip_report = _clip_comparison(
            analysis_path=analysis_path,
            payload=payload,
            pose_frames=pose_frames,
            frozen_records=frozen_records,
            image_records=image_records,
            video_records=video_records,
            image_metrics=image_metrics,
            video_metrics=video_metrics,
            image_continuity=image_continuity,
            video_continuity=video_continuity,
            image_safety=image_safety,
            video_safety=video_safety,
        )
        clip_reports.append(clip_report)
        artifacts.extend(
            [
                (
                    output_root / clip_id / "candidate_image_hand.json",
                    image_payload,
                ),
                (
                    output_root / clip_id / "candidate_video_hand.json",
                    video_payload,
                ),
            ]
        )

    image_safety = _aggregate_safety(
        clip_reports,
        "candidate_image_safety",
    )
    video_safety = _aggregate_safety(
        clip_reports,
        "candidate_video_safety",
    )
    aggregate_gates = {
        "exactly_three_clips_compared": len(clip_reports) == 3,
        "all_frozen_image_cores_match_candidate_image": all(
            clip["frozen_image_vs_candidate_image"]["match"]
            for clip in clip_reports
        ),
        "all_clip_gates_passed": all(
            clip["status"] == "passed" for clip in clip_reports
        ),
        "candidate_image_safety_passed": image_safety["status"] == "passed",
        "candidate_video_safety_passed": video_safety["status"] == "passed",
        "accuracy_precision_recall_not_evaluable": True,
    }
    summary = {
        "schema_version": "hand_image_video_mode_comparison_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(aggregate_gates.values()) else "failed",
        "project_path": str(PROJECT_ROOT),
        "experiment": {
            "name": "MediaPipe Hand Landmarker IMAGE versus VIDEO",
            "accepted_analysis_paths": [
                str(path) for path in resolved_analyses
            ],
            "model_path": str(model_path),
            "model_sha256": model_hash,
            "body_pose_recomputed": False,
            "same_pose_frames_and_source_frame_indices": True,
            "roi_smoothing_enabled": False,
        },
        "summary": {
            "clip_count": len(clip_reports),
            "pose_frame_count": sum(
                int(clip["pose_frame_count"]) for clip in clip_reports
            ),
            "frozen_image": _aggregate_mode(
                clip_reports,
                "frozen_image_metrics",
            ),
            "candidate_image": _aggregate_mode(
                clip_reports,
                "candidate_image_metrics",
            ),
            "candidate_video": _aggregate_mode(
                clip_reports,
                "candidate_video_metrics",
            ),
            "candidate_image_safety": image_safety,
            "candidate_video_safety": video_safety,
            "aggregate_gates": aggregate_gates,
        },
        "clips": clip_reports,
        "evaluation": _evaluation_payload(),
    }
    artifacts.append((summary_path, summary))
    return summary, artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a controlled Hand IMAGE/VIDEO A/B on three accepted "
            "analysis.json files without recomputing Body Pose."
        )
    )
    parser.add_argument(
        "--analysis",
        type=Path,
        nargs=3,
        required=True,
        metavar=("ANALYSIS_1", "ANALYSIS_2", "ANALYSIS_3"),
        help="Exactly three accepted project-local analysis.json files",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="New project-local output root; existing outputs are not overwritten",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help=(
            "Optional project-local path override for the frozen Hand model; "
            "bytes must match accepted metadata"
        ),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help=(
            "Optional project-local summary JSON path; defaults to "
            "<output-root>/HAND_VIDEO_MODE_COMPARISON.json"
        ),
    )
    return parser


def main() -> None:
    _require_exact_workspace()
    args = build_parser().parse_args()
    summary, artifacts = run_comparison(
        args.analysis,
        output_root=args.output_root,
        model_override=args.model,
        summary_output=args.summary_output,
    )
    written = [
        str(_write_atomic_new(path, payload)) for path, payload in artifacts
    ]
    print(
        json.dumps(
            {
                "status": summary["status"],
                "written": written,
                **summary["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
