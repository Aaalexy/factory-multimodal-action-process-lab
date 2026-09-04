"""Compare frozen Phase B.1 Hand evidence with a Phase B.2 quality candidate.

This comparison is deliberately semantic and fail-closed:

* the frozen Hand record fields and action-event fields must not change;
* Phase B.2 may only add derived Hand quality/gating fields;
* automatic evidence remains non-training data;
* no accuracy claim is produced without independent human ground truth.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_WORKSPACE = PROJECT_ROOT
DEFAULT_LAYOUT = "{clip_id}/candidate/analysis.json"
EXPECTED_OBSERVATION_COUNTS = {
    "detected": 7,
    "uncertain": 48,
    "missing": 521,
}
EXPECTED_ASSOCIATION_WARNING_COUNT = 39
VALIDATION_FLAG_NAMES = {
    "factory_camera_validated",
    "production_action_model_ready",
    "external_factory_validated",
    "production_process_model_ready",
}
ELIGIBLE_FIELD = "action_feature_eligible"
DERIVED_HAND_FIELDS = {
    "backend_state",
    "quality_state",
    "validation_state",
    ELIGIBLE_FIELD,
    "hand_feature_eligible",
    "temporal_feature_eligible",
    "feature_eligible",
    "eligible",
    "quality_gate_version",
    "quality_reasons",
    "validation_reasons",
    "eligibility_reasons",
    "derived_quality",
}
TIMING_HAND_FIELDS = {
    "inference_time_ms",
    "processing_time_ms",
    "elapsed_time_ms",
}
ELIGIBLE_VALIDATION_STATES = {"not_reviewed"}
BACKEND_STATES = ("available", "unavailable", "error", "unknown")
QUALITY_STATES = (
    "qualified",
    "association_uncertain",
    "insufficient_geometry",
    "not_observed",
    "lost",
    "unknown",
)
VALIDATION_STATES = (
    "not_reviewed",
    "review_required",
    "not_evaluable",
    "unknown",
)
MAX_REPORTED_ISSUES = 250


class _Issues:
    def __init__(self) -> None:
        self.total = 0
        self.items: list[dict[str, Any]] = []

    def add(self, category: str, **detail: Any) -> None:
        self.total += 1
        if len(self.items) < MAX_REPORTED_ISSUES:
            self.items.append({"category": category, **detail})

    def payload(self) -> dict[str, Any]:
        return {
            "count": self.total,
            "reported_count": len(self.items),
            "truncated": self.total > len(self.items),
            "items": self.items,
        }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Analysis must be a JSON object: {path}")
    return payload


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _normalise_layout(layout: str) -> str:
    normalised = str(layout).replace("\\", "/")
    if normalised.count("{clip_id}") != 1:
        raise ValueError("Layout must contain exactly one {clip_id} placeholder")
    probe = PurePosixPath(normalised.replace("{clip_id}", "clip"))
    if probe.is_absolute() or ".." in probe.parts:
        raise ValueError("Layout must stay relative to its replay root")
    return normalised


def _discover(root: Path, layout: str) -> dict[str, Path]:
    root = root.resolve()
    layout = _normalise_layout(layout)
    prefix, suffix = layout.split("{clip_id}", 1)
    pattern = layout.replace("{clip_id}", "*")
    found: dict[str, Path] = {}
    for path in sorted(root.glob(pattern)):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if not relative.startswith(prefix) or (
            suffix and not relative.endswith(suffix)
        ):
            continue
        end = len(relative) - len(suffix) if suffix else len(relative)
        clip_id = relative[len(prefix) : end]
        if not clip_id or "/" in clip_id or "\\" in clip_id:
            continue
        if clip_id in found:
            raise ValueError(f"Duplicate analysis path for clip {clip_id}")
        found[clip_id] = path.resolve()
    return found


def _observation_bucket(value: Any) -> str:
    state = str(value or "missing").lower()
    if state == "detected":
        return "detected"
    if state in {"uncertain", "predicted", "interpolated"}:
        return "uncertain"
    if state in {"missing", "lost", "off_frame"}:
        return "missing"
    return state


def _warnings(record: dict[str, Any]) -> list[str]:
    checks = record.get("association_checks")
    if not isinstance(checks, dict):
        return []
    raw = checks.get("warnings", [])
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw]


def _is_duplicate(record: dict[str, Any]) -> bool:
    checks = record.get("association_checks")
    return bool(
        isinstance(checks, dict)
        and checks.get("duplicate_across_sides", False)
    )


def _has_21_unique_landmark_indices(record: dict[str, Any]) -> bool:
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
    return len(set(indices)) == 21 and set(indices) == set(range(21))


def _record_identifier(
    record: dict[str, Any],
    *,
    field: str,
    index: int,
) -> str:
    value = record.get(field)
    return str(value) if value not in {None, ""} else f"index:{index}"


def _core_keys(records: Iterable[dict[str, Any]]) -> list[str]:
    keys = {
        key
        for record in records
        for key in record
        if key not in DERIVED_HAND_FIELDS and key not in TIMING_HAND_FIELDS
    }
    return sorted(keys)


def _project_records(
    records: list[dict[str, Any]],
    *,
    keys: list[str],
    identifier_field: str,
) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        projected.append(
            {
                "_comparison_id": _record_identifier(
                    record,
                    field=identifier_field,
                    index=index,
                ),
                **{
                    key: (
                        record[key]
                        if key in record
                        else {"__field_missing__": True}
                    )
                    for key in keys
                },
            }
        )
    projected.sort(key=lambda item: item["_comparison_id"])
    return projected


def _core_mismatch_details(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    identifier_field: str,
) -> list[dict[str, Any]]:
    left = {item["_comparison_id"]: item for item in baseline}
    right = {item["_comparison_id"]: item for item in candidate}
    details: list[dict[str, Any]] = []
    for identifier in sorted(left.keys() - right.keys()):
        details.append(
            {"record_id": identifier, "difference": "missing_in_candidate"}
        )
    for identifier in sorted(right.keys() - left.keys()):
        details.append(
            {"record_id": identifier, "difference": "extra_in_candidate"}
        )
    for identifier in sorted(left.keys() & right.keys()):
        if left[identifier] == right[identifier]:
            continue
        fields = sorted(
            key
            for key in set(left[identifier]) | set(right[identifier])
            if key != "_comparison_id"
            and left[identifier].get(key) != right[identifier].get(key)
        )
        details.append(
            {
                "record_id": identifier,
                "difference": "changed_core_fields",
                "fields": fields,
                "baseline_record_sha256": _sha256(left[identifier]),
                "candidate_record_sha256": _sha256(right[identifier]),
            }
        )
    return details


def _hand_core_comparison(
    baseline_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
) -> dict[str, Any]:
    keys = _core_keys(baseline_records)
    baseline = _project_records(
        baseline_records,
        keys=keys,
        identifier_field="hand_pose_id",
    )
    candidate = _project_records(
        candidate_records,
        keys=keys,
        identifier_field="hand_pose_id",
    )
    baseline_sha = _sha256(baseline)
    candidate_sha = _sha256(candidate)
    return {
        "core_field_count": len(keys),
        "excluded_derived_fields": sorted(DERIVED_HAND_FIELDS),
        "excluded_timing_fields": sorted(TIMING_HAND_FIELDS),
        "baseline_sha256": baseline_sha,
        "candidate_sha256": candidate_sha,
        "match": baseline_sha == candidate_sha,
        "mismatches": _core_mismatch_details(
            baseline,
            candidate,
            identifier_field="hand_pose_id",
        ),
    }


def _action_core_comparison(
    baseline_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
) -> dict[str, Any]:
    keys = sorted({key for record in baseline_records for key in record})
    baseline = _project_records(
        baseline_records,
        keys=keys,
        identifier_field="action_event_id",
    )
    candidate = _project_records(
        candidate_records,
        keys=keys,
        identifier_field="action_event_id",
    )
    baseline_sha = _sha256(baseline)
    candidate_sha = _sha256(candidate)
    return {
        "core_field_count": len(keys),
        "baseline_sha256": baseline_sha,
        "candidate_sha256": candidate_sha,
        "match": baseline_sha == candidate_sha,
        "mismatches": _core_mismatch_details(
            baseline,
            candidate,
            identifier_field="action_event_id",
        ),
    }


def _hand_counts(records: list[dict[str, Any]]) -> dict[str, Any]:
    observations = Counter(
        _observation_bucket(record.get("observation_state"))
        for record in records
    )
    warning_records = {
        (
            record.get("frame_index"),
            str(record.get("anatomical_side", "unknown")),
            warning,
        )
        for record in records
        for warning in _warnings(record)
    }
    return {
        "record_count": len(records),
        "observation_counts": dict(sorted(observations.items())),
        "association_warning_count": len(warning_records),
    }


def _quality_contract(
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    issues = _Issues()
    backend_counts: Counter[str] = Counter(
        {state: 0 for state in BACKEND_STATES}
    )
    quality_counts: Counter[str] = Counter(
        {state: 0 for state in QUALITY_STATES}
    )
    validation_counts: Counter[str] = Counter(
        {state: 0 for state in VALIDATION_STATES}
    )
    eligible_observations = 0
    eligible_frames: set[Any] = set()

    for index, record in enumerate(records):
        record_id = _record_identifier(
            record,
            field="hand_pose_id",
            index=index,
        )
        for field in (
            "backend_state",
            "quality_state",
            "validation_state",
            ELIGIBLE_FIELD,
        ):
            if field not in record:
                issues.add(
                    "missing_phase_b2_field",
                    hand_pose_id=record_id,
                    field=field,
                )

        backend = str(record.get("backend_state", "__missing__")).lower()
        quality = str(record.get("quality_state", "__missing__")).lower()
        validation = str(
            record.get("validation_state", "__missing__")
        ).lower()
        backend_counts[backend] += 1
        quality_counts[quality] += 1
        validation_counts[validation] += 1

        eligible_raw = record.get(ELIGIBLE_FIELD)
        if not isinstance(eligible_raw, bool):
            issues.add(
                "eligible_field_not_boolean",
                hand_pose_id=record_id,
                value_type=type(eligible_raw).__name__,
            )
            eligible = False
        else:
            eligible = eligible_raw

        warning_list = _warnings(record)
        duplicate = _is_duplicate(record)
        unique_landmarks = _has_21_unique_landmark_indices(record)
        observation = str(record.get("observation_state", "missing")).lower()
        expected_eligible = (
            backend == "available"
            and observation == "detected"
            and quality == "qualified"
            and validation in ELIGIBLE_VALIDATION_STATES
            and unique_landmarks
            and not warning_list
            and not duplicate
        )
        if eligible != expected_eligible:
            issues.add(
                "eligible_does_not_match_quality_gate",
                hand_pose_id=record_id,
                actual=eligible,
                expected=expected_eligible,
                backend_state=backend,
                observation_state=observation,
                quality_state=quality,
                validation_state=validation,
                unique_landmark_indices=unique_landmarks,
                warning_count=len(warning_list),
                duplicate_across_sides=duplicate,
            )
        if warning_list and eligible:
            issues.add(
                "warning_record_marked_eligible",
                hand_pose_id=record_id,
                warnings=warning_list,
            )
        if duplicate and eligible:
            issues.add(
                "duplicate_record_marked_eligible",
                hand_pose_id=record_id,
            )
        if observation in {"missing", "lost", "off_frame"}:
            landmarks = record.get("landmarks")
            if landmarks != [] or int(record.get("landmark_count", -1)) != 0:
                issues.add(
                    "missing_or_lost_record_has_geometry",
                    hand_pose_id=record_id,
                    observation_state=observation,
                    landmark_count=record.get("landmark_count"),
                )
            if eligible:
                issues.add(
                    "missing_or_lost_record_marked_eligible",
                    hand_pose_id=record_id,
                    observation_state=observation,
                )
        if record.get("training_eligible") is not False:
            issues.add(
                "automatic_hand_record_training_eligible",
                hand_pose_id=record_id,
                value=record.get("training_eligible"),
            )
        if eligible:
            eligible_observations += 1
            eligible_frames.add(record.get("frame_index"))

    metrics = {
        "backend_state_counts": dict(sorted(backend_counts.items())),
        "quality_state_counts": dict(sorted(quality_counts.items())),
        "validation_state_counts": dict(sorted(validation_counts.items())),
        "action_feature_eligible_observation_count": eligible_observations,
        "action_feature_eligible_frame_count": len(eligible_frames),
    }
    return metrics, issues.payload()


def _runtime_quality_checks(
    runtime: dict[str, Any],
    derived: dict[str, Any],
) -> tuple[dict[str, bool], list[dict[str, Any]]]:
    field_map = {
        "hand_backend_state_counts": "backend_state_counts",
        "hand_quality_state_counts": "quality_state_counts",
        "hand_validation_state_counts": "validation_state_counts",
        "hand_action_feature_eligible_observation_count": (
            "action_feature_eligible_observation_count"
        ),
        "hand_action_feature_eligible_frame_count": (
            "action_feature_eligible_frame_count"
        ),
    }
    checks: dict[str, bool] = {}
    mismatches: list[dict[str, Any]] = []
    for runtime_field, derived_field in field_map.items():
        actual = runtime.get(runtime_field, {"__field_missing__": True})
        expected = derived[derived_field]
        matches = actual == expected
        checks[runtime_field] = matches
        if not matches:
            mismatches.append(
                {
                    "field": runtime_field,
                    "runtime_value": actual,
                    "derived_value": expected,
                }
            )
    return checks, mismatches


def _hand_model_checks(hand_model: Any) -> dict[str, bool]:
    if not isinstance(hand_model, dict):
        return {
            "hand_model_object_present": False,
            "backend_state_present": False,
            "backend_mode_present": False,
            "quality_gate_version_present": False,
        }
    return {
        "hand_model_object_present": True,
        "backend_state_present": bool(hand_model.get("backend_state")),
        "backend_mode_present": bool(hand_model.get("backend_mode")),
        "quality_gate_version_present": bool(
            hand_model.get("quality_gate_version")
        ),
    }


def _validation_flags_are_false(payload: dict[str, Any]) -> bool:
    flags = payload.get("validation_flags")
    return bool(
        isinstance(flags, dict)
        and VALIDATION_FLAG_NAMES.issubset(flags)
        and not any(bool(value) for value in flags.values())
    )


def _evaluation_is_not_evaluable(payload: dict[str, Any]) -> bool:
    evaluation = payload.get("evaluation")
    return bool(
        isinstance(evaluation, dict)
        and evaluation.get("status") == "not_evaluable"
    )


def compare_clip(
    clip_id: str,
    baseline_path: Path,
    candidate_path: Path,
) -> dict[str, Any]:
    baseline = _load_json(baseline_path)
    candidate = _load_json(candidate_path)
    baseline_hands = list(baseline.get("hand_pose_frames", []))
    candidate_hands = list(candidate.get("hand_pose_frames", []))
    baseline_actions = list(baseline.get("action_events", []))
    candidate_actions = list(candidate.get("action_events", []))
    baseline_runtime = baseline.get("runtime", {})
    candidate_runtime = candidate.get("runtime", {})
    baseline_counts = _hand_counts(baseline_hands)
    candidate_counts = _hand_counts(candidate_hands)
    hand_core = _hand_core_comparison(baseline_hands, candidate_hands)
    action_core = _action_core_comparison(
        baseline_actions,
        candidate_actions,
    )
    quality_metrics, quality_issues = _quality_contract(candidate_hands)
    runtime_checks, runtime_mismatches = _runtime_quality_checks(
        candidate_runtime,
        quality_metrics,
    )
    hand_model_checks = _hand_model_checks(candidate.get("hand_model"))

    baseline_source = baseline.get("source_video", {})
    candidate_source = candidate.get("source_video", {})
    gates = {
        "source_video_sha256_match": (
            baseline_source.get("sha256")
            == candidate_source.get("sha256")
            and bool(baseline_source.get("sha256"))
        ),
        "analysis_window_match": (
            baseline_source.get("analysis_window")
            == candidate_source.get("analysis_window")
        ),
        "processed_frame_count_match": (
            baseline_runtime.get("processed_frame_count")
            == candidate_runtime.get("processed_frame_count")
        ),
        "hand_inference_calls_match": (
            baseline_runtime.get("hand_inference_calls")
            == candidate_runtime.get("hand_inference_calls")
        ),
        "hand_core_sha256_match": hand_core["match"],
        "action_event_core_sha256_match": action_core["match"],
        "observation_counts_match": (
            baseline_counts["observation_counts"]
            == candidate_counts["observation_counts"]
        ),
        "association_warning_count_match": (
            baseline_counts["association_warning_count"]
            == candidate_counts["association_warning_count"]
        ),
        "phase_b2_quality_contract_passed": quality_issues["count"] == 0,
        "runtime_quality_metrics_match_derived": all(
            runtime_checks.values()
        ),
        "hand_model_quality_fields_present": all(
            hand_model_checks.values()
        ),
        "training_eligible_all_false": all(
            record.get("training_eligible") is False
            for record in candidate_hands
        ),
        "four_validation_flags_false": _validation_flags_are_false(candidate),
        "accuracy_not_evaluable": _evaluation_is_not_evaluable(candidate),
    }
    return {
        "clip_id": clip_id,
        "status": "passed" if all(gates.values()) else "failed",
        "paths": {
            "baseline": str(baseline_path.resolve()),
            "candidate": str(candidate_path.resolve()),
        },
        "source_video_sha256": {
            "baseline": baseline_source.get("sha256"),
            "candidate": candidate_source.get("sha256"),
        },
        "analysis_window": {
            "baseline": baseline_source.get("analysis_window"),
            "candidate": candidate_source.get("analysis_window"),
        },
        "runtime_invariants": {
            "processed_frame_count": {
                "baseline": baseline_runtime.get("processed_frame_count"),
                "candidate": candidate_runtime.get("processed_frame_count"),
            },
            "hand_inference_calls": {
                "baseline": baseline_runtime.get("hand_inference_calls"),
                "candidate": candidate_runtime.get("hand_inference_calls"),
            },
        },
        "baseline_hand_counts": baseline_counts,
        "candidate_hand_counts": candidate_counts,
        "quality_metrics": quality_metrics,
        "runtime_quality_checks": runtime_checks,
        "runtime_quality_mismatches": runtime_mismatches,
        "hand_model_checks": hand_model_checks,
        "hand_core": hand_core,
        "action_event_core": action_core,
        "quality_contract_issues": quality_issues,
        "gates": gates,
    }


def _aggregate_clip_counts(
    clips: list[dict[str, Any]],
    key: str,
) -> dict[str, Any]:
    observations: Counter[str] = Counter()
    warning_count = 0
    record_count = 0
    for clip in clips:
        counts = clip[key]
        observations.update(counts["observation_counts"])
        warning_count += int(counts["association_warning_count"])
        record_count += int(counts["record_count"])
    return {
        "record_count": record_count,
        "observation_counts": dict(sorted(observations.items())),
        "association_warning_count": warning_count,
    }


def _aggregate_quality(clips: list[dict[str, Any]]) -> dict[str, Any]:
    backend: Counter[str] = Counter()
    quality: Counter[str] = Counter()
    validation: Counter[str] = Counter()
    eligible_observations = 0
    eligible_frames = 0
    for clip in clips:
        metrics = clip["quality_metrics"]
        backend.update(metrics["backend_state_counts"])
        quality.update(metrics["quality_state_counts"])
        validation.update(metrics["validation_state_counts"])
        eligible_observations += int(
            metrics["action_feature_eligible_observation_count"]
        )
        eligible_frames += int(metrics["action_feature_eligible_frame_count"])
    return {
        "backend_state_counts": dict(sorted(backend.items())),
        "quality_state_counts": dict(sorted(quality.items())),
        "validation_state_counts": dict(sorted(validation.items())),
        "action_feature_eligible_observation_count": eligible_observations,
        "action_feature_eligible_frame_count": eligible_frames,
    }


def compare_roots(
    baseline_root: Path,
    candidate_root: Path,
    *,
    baseline_layout: str = DEFAULT_LAYOUT,
    candidate_layout: str = DEFAULT_LAYOUT,
    expected_observation_counts: dict[str, int] | None = None,
    expected_warning_count: int = EXPECTED_ASSOCIATION_WARNING_COUNT,
) -> dict[str, Any]:
    expected_observation_counts = (
        dict(EXPECTED_OBSERVATION_COUNTS)
        if expected_observation_counts is None
        else dict(expected_observation_counts)
    )
    baseline_paths = _discover(baseline_root, baseline_layout)
    candidate_paths = _discover(candidate_root, candidate_layout)
    if not baseline_paths:
        raise FileNotFoundError(
            f"No frozen baseline analyses found under {baseline_root}"
        )

    common_ids = sorted(baseline_paths.keys() & candidate_paths.keys())
    clips = [
        compare_clip(
            clip_id,
            baseline_paths[clip_id],
            candidate_paths[clip_id],
        )
        for clip_id in common_ids
    ]
    incomplete = [
        {
            "clip_id": clip_id,
            "baseline_present": clip_id in baseline_paths,
            "candidate_present": clip_id in candidate_paths,
        }
        for clip_id in sorted(baseline_paths.keys() ^ candidate_paths.keys())
    ]
    baseline_counts = _aggregate_clip_counts(
        clips,
        "baseline_hand_counts",
    )
    candidate_counts = _aggregate_clip_counts(
        clips,
        "candidate_hand_counts",
    )
    quality_counts = _aggregate_quality(clips)
    aggregate_gates = {
        "all_baseline_clips_have_candidates": not incomplete,
        "at_least_one_clip_compared": bool(clips),
        "all_clip_gates_passed": all(
            clip["status"] == "passed" for clip in clips
        ),
        "frozen_baseline_observation_counts_expected": (
            baseline_counts["observation_counts"]
            == expected_observation_counts
        ),
        "candidate_observation_counts_preserved": (
            candidate_counts["observation_counts"]
            == expected_observation_counts
        ),
        "frozen_baseline_warning_count_expected": (
            baseline_counts["association_warning_count"]
            == expected_warning_count
        ),
        "candidate_warning_count_preserved": (
            candidate_counts["association_warning_count"]
            == expected_warning_count
        ),
    }
    status = "passed" if all(aggregate_gates.values()) else "failed"
    return {
        "schema_version": "phase_b2_hand_quality_comparison_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "baseline_root": str(Path(baseline_root).resolve()),
        "candidate_root": str(Path(candidate_root).resolve()),
        "layouts": {
            "baseline": _normalise_layout(baseline_layout),
            "candidate": _normalise_layout(candidate_layout),
        },
        "evaluation": {
            "status": "not_evaluable",
            "accuracy_metrics_computed": False,
            "reason": (
                "No independently human-confirmed Hand association or "
                "quality ground truth; this report checks invariants only."
            ),
        },
        "expected_frozen_counts": {
            "observation_counts": expected_observation_counts,
            "association_warning_count": expected_warning_count,
        },
        "summary": {
            "clip_count": len(clips),
            "incomplete_clip_count": len(incomplete),
            "baseline_hand_counts": baseline_counts,
            "candidate_hand_counts": candidate_counts,
            "candidate_quality_counts": quality_counts,
            "aggregate_gates": aggregate_gates,
        },
        "clips": clips,
        "incomplete_clips": incomplete,
    }


def write_atomic_project_json(path: Path, payload: dict[str, Any]) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError("Comparison output must remain inside the project")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_suffix(resolved.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(resolved)
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument(
        "--baseline-layout",
        default=DEFAULT_LAYOUT,
        help="Relative template containing {clip_id}",
    )
    parser.add_argument(
        "--candidate-layout",
        default=DEFAULT_LAYOUT,
        help="Relative template containing {clip_id}",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    if Path.cwd().resolve() != EXPECTED_WORKSPACE.resolve():
        raise RuntimeError("Exact workspace gate failed")
    args = build_parser().parse_args()
    comparison = compare_roots(
        args.baseline_root,
        args.candidate_root,
        baseline_layout=args.baseline_layout,
        candidate_layout=args.candidate_layout,
    )
    output = write_atomic_project_json(args.output, comparison)
    print(
        json.dumps(
            {
                "status": comparison["status"],
                "output": str(output),
                **comparison["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if comparison["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
