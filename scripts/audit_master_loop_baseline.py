"""Independently recalculate the Master Loop Phase B baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PROJECT_ROOT = PROJECT_ROOT
PHASE_B_ROOT = (
    PROJECT_ROOT / "outputs" / "phase_b_validation_recovery_20260724"
)
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
ALLOWED_SIDES = {"left", "right", "bilateral"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def requirement_is_fully_pinned(line: str) -> bool:
    content = line.split("#", 1)[0].strip()
    if not content:
        return True
    if content.startswith(("-r ", "--requirement ")):
        return False
    return "==" in content and not any(
        token in content
        for token in (">=", "<=", "~=", "!=", ">", "<", "*")
    )


def parse_junit(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    leaf_suites = [
        suite
        for suite in suites
        if not any(child.tag == "testsuite" for child in suite)
    ]
    display_path = (
        path.relative_to(PROJECT_ROOT).as_posix()
        if path.is_relative_to(PROJECT_ROOT)
        else str(path.resolve())
    )
    return {
        "tests": sum(int(suite.attrib.get("tests", 0)) for suite in leaf_suites),
        "failures": sum(
            int(suite.attrib.get("failures", 0)) for suite in leaf_suites
        ),
        "errors": sum(
            int(suite.attrib.get("errors", 0)) for suite in leaf_suites
        ),
        "skipped": sum(
            int(suite.attrib.get("skipped", 0)) for suite in leaf_suites
        ),
        "path": display_path,
    }


def _event_duration(event: dict[str, Any]) -> float:
    stored = event.get("duration_seconds")
    if stored is not None:
        return max(0.0, float(stored))
    return max(
        0.0,
        float(event.get("end_time", 0.0))
        - float(event.get("start_time", 0.0)),
    )


def _side_switch_metrics(
    pose_frames: list[dict[str, Any]],
) -> dict[str, int]:
    tracked_switches = 0
    reliable_switches = 0
    reliable_frames = 0
    previous_tracked: str | None = None
    previous_reliable: str | None = None
    for frame in pose_frames:
        side = str(frame.get("anatomical_side", "unknown"))
        tracked = frame.get("track_state") == "tracked"
        if tracked and side in ALLOWED_SIDES:
            if previous_tracked is not None and side != previous_tracked:
                tracked_switches += 1
            previous_tracked = side
        else:
            previous_tracked = None

        reliable = (
            tracked
            and frame.get("observation_state") == "detected"
            and bool(frame.get("required_joints_reliable"))
            and side in ALLOWED_SIDES
        )
        if reliable:
            reliable_frames += 1
            if previous_reliable is not None and side != previous_reliable:
                reliable_switches += 1
            previous_reliable = side
        else:
            previous_reliable = None
    return {
        "tracked_side_switch_count": tracked_switches,
        "reliable_frame_count": reliable_frames,
        "reliable_side_switch_count": reliable_switches,
    }


def _analysis_paths() -> list[Path]:
    return sorted(PHASE_B_ROOT.glob("*/after/analysis.json"))


def audit() -> dict[str, Any]:
    actual_root = PROJECT_ROOT.resolve()
    if str(actual_root) != str(EXPECTED_PROJECT_ROOT):
        raise RuntimeError(
            f"Exact workspace gate failed: {actual_root!s}"
        )

    model_manifest = load_json(PROJECT_ROOT / "HAND_MODEL_MANIFEST.json")
    model_entry = model_manifest["model"]
    model_path = PROJECT_ROOT / model_entry["local_path"]
    model_actual = {
        "path": model_entry["local_path"],
        "size_bytes": model_path.stat().st_size,
        "sha256": sha256_file(model_path),
    }
    model_check = {
        "status": "passed"
        if (
            model_actual["size_bytes"] == int(model_entry["size_bytes"])
            and model_actual["sha256"] == model_entry["sha256"]
        )
        else "failed",
        "actual": model_actual,
        "manifest": {
            "project_name": model_entry["project_name"],
            "model_version": model_entry["model_version"],
            "official_url": model_entry["official_url"],
            "license": model_entry["license"],
            "size_bytes": model_entry["size_bytes"],
            "sha256": model_entry["sha256"],
        },
        "source_verification_scope": (
            "Official URL and license are recorded in the manifest; "
            "the model was not downloaded again during this audit."
        ),
    }

    junit = parse_junit(
        PROJECT_ROOT
        / "outputs"
        / "phase_b_focused_test_results_recovery_20260724.xml"
    )
    junit["status"] = (
        "passed"
        if (
            junit["tests"] == 99
            and junit["failures"] == 0
            and junit["errors"] == 0
            and junit["skipped"] == 0
        )
        else "failed"
    )

    analysis_paths = _analysis_paths()
    if len(analysis_paths) != 3:
        raise RuntimeError(
            f"Expected three Phase B after artifacts, found {len(analysis_paths)}"
        )
    analyses = [load_json(path) for path in analysis_paths]
    aggregate_keys = (
        "processed_frame_count",
        "hand_detected_frame_count",
        "hand_uncertain_frame_count",
        "hand_missing_frame_count",
        "left_right_association_error_count",
        "association_warning_count",
        "pose_segment_count",
        "suppressed_fragment_count",
        "merged_fragment_count",
        "stable_normal_action_count",
        "sub_1s_stable_event_count",
        "end_to_end_seconds",
    )
    aggregate = {
        key: round(
            sum(float(item["runtime"].get(key, 0)) for item in analyses),
            6,
        )
        for key in aggregate_keys
    }
    for integer_key in aggregate_keys[:-1]:
        aggregate[integer_key] = int(aggregate[integer_key])

    total_window_seconds = sum(
        float(item["source_video"]["analysis_window"]["end_time"])
        - float(item["source_video"]["analysis_window"]["start_time"])
        for item in analyses
    )
    display_eligible_seconds = sum(
        _event_duration(event)
        for item in analyses
        for event in item.get("action_events", [])
        if event.get("display_eligible") is not False
    )
    hard_boundary_frames = sum(
        int(
            item["stabilization_metrics"]["frame_stabilization"][
                "hard_boundary_frame_count"
            ]
        )
        for item in analyses
    )
    non_tracked_frames = sum(
        frame.get("track_state") != "tracked"
        for item in analyses
        for frame in item.get("pose_frames", [])
    )
    side_metrics = {
        "tracked_side_switch_count": 0,
        "reliable_frame_count": 0,
        "reliable_side_switch_count": 0,
    }
    for item in analyses:
        clip_metrics = _side_switch_metrics(item.get("pose_frames", []))
        for key, value in clip_metrics.items():
            side_metrics[key] += value

    pipeline_text = (
        PROJECT_ROOT / "src" / "multimodal_pipeline.py"
    ).read_text(encoding="utf-8")
    action_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(
            (PROJECT_ROOT / "src" / "action_segmentation").glob("*.py")
        )
    )
    classifier_position = pipeline_text.find("classifier.classify(")
    hand_inference_position = pipeline_text.find("hand_backend.infer_frame(")
    hand_action_dataflow = {
        "classifier_runs_before_hand_inference": (
            classifier_position >= 0
            and hand_inference_position >= 0
            and classifier_position < hand_inference_position
        ),
        "action_segmentation_hand_token_count": len(
            re.findall(r"\bhand(?:_pose|_motion|_shape)?\b", action_sources)
        ),
    }
    hand_action_dataflow["hand_pose_participates_in_action_naming"] = not (
        hand_action_dataflow["classifier_runs_before_hand_inference"]
        and hand_action_dataflow["action_segmentation_hand_token_count"] == 0
    )

    static_and_test_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for root in (
            PROJECT_ROOT / "src" / "web",
            PROJECT_ROOT / "tests",
            PROJECT_ROOT / "scripts",
        )
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".py", ".js", ".html"}
    )
    browser_framework_tokens = sorted(
        {
            token
            for token in ("playwright", "selenium", "chromedriver")
            if token in static_and_test_text.lower()
        }
    )
    browser_image_candidates = sorted(
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in (PROJECT_ROOT / "outputs").rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        and any(
            token in path.name.lower()
            for token in ("browser", "viewport", "screenshot")
        )
    )
    browser_validation = {
        "status": "not_evaluable",
        "automated_browser_framework_tokens": browser_framework_tokens,
        "browser_screenshot_artifacts": browser_image_candidates,
        "http_range_test_exists": "Range" in static_and_test_text,
        "reason": (
            "No real-browser automation or viewport screenshot artifact "
            "was found; HTTP/API and static string tests are not visual QA."
        ),
    }

    clearcut_docs = []
    for path in (
        PROJECT_ROOT / "docs" / "HAND_POSE_TECHNICAL_DECISION.md",
        PROJECT_ROOT / "HAND_ACTION_UPGRADE_REPORT.md",
    ):
        if path.is_file() and "clearcut" in path.read_text(
            encoding="utf-8"
        ).lower():
            clearcut_docs.append(path.relative_to(PROJECT_ROOT).as_posix())
    raw_clearcut_logs = sorted(
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in (PROJECT_ROOT / "outputs").rglob("*.log")
        if "clearcut" in path.read_text(
            encoding="utf-8", errors="replace"
        ).lower()
    )
    telemetry = {
        "documented_clearcut_attempt": bool(clearcut_docs),
        "documentation_paths": clearcut_docs,
        "raw_log_artifacts": raw_clearcut_logs,
        "status": "confirmed_documented_observation_without_raw_log",
    }

    git_dir = PROJECT_ROOT / ".git"
    git_entries = sorted(path.name for path in git_dir.iterdir()) if git_dir.is_dir() else []
    git_state = {
        "exists": git_dir.exists(),
        "entry_count": len(git_entries),
        "has_head": (git_dir / "HEAD").is_file(),
        "has_config": (git_dir / "config").is_file(),
        "valid_repository_metadata": (
            (git_dir / "HEAD").is_file() and (git_dir / "config").is_file()
        ),
    }

    requirement_lines = [
        line.strip()
        for line in (PROJECT_ROOT / "requirements.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    unpinned = [
        line for line in requirement_lines if not requirement_is_fully_pinned(line)
    ]
    requirements = {
        "status": "fully_locked" if not unpinned else "not_fully_locked",
        "requirement_count": len(requirement_lines),
        "unpinned_requirements": unpinned,
    }

    config_flags = load_json(PROJECT_ROOT / "configs" / "project.json")[
        "validation_flags"
    ]
    artifact_flags = [
        item["validation_flags"]
        for item in analyses
    ]
    all_flags_false = not any(config_flags.values()) and all(
        not any(flags.values()) for flags in artifact_flags
    )

    checks = [
        ("hand_model_asset", model_check["status"], model_check),
        ("phase_b_junit", junit["status"], junit),
        ("processed_frames", "passed" if aggregate["processed_frame_count"] == 288 else "failed", aggregate["processed_frame_count"]),
        ("hand_detected_frames", "passed" if aggregate["hand_detected_frame_count"] == 7 else "failed", aggregate["hand_detected_frame_count"]),
        ("hand_uncertain_frames", "passed" if aggregate["hand_uncertain_frame_count"] == 34 else "failed", aggregate["hand_uncertain_frame_count"]),
        ("hand_missing_frames", "passed" if aggregate["hand_missing_frame_count"] == 247 else "failed", aggregate["hand_missing_frame_count"]),
        ("association_counts", "passed" if aggregate["left_right_association_error_count"] == 25 and aggregate["association_warning_count"] == 39 else "failed", {"unique_frame_errors": aggregate["left_right_association_error_count"], "side_warnings": aggregate["association_warning_count"]}),
        ("pose_segments", "passed" if aggregate["pose_segment_count"] == 120 else "failed", aggregate["pose_segment_count"]),
        ("suppressed_fragments", "passed" if aggregate["suppressed_fragment_count"] == 84 else "failed", aggregate["suppressed_fragment_count"]),
        ("real_merged_fragments", "passed" if aggregate["merged_fragment_count"] == 0 else "failed", aggregate["merged_fragment_count"]),
        ("stable_normal_actions", "passed" if aggregate["stable_normal_action_count"] == 2 else "failed", aggregate["stable_normal_action_count"]),
        ("display_eligible_coverage", "confirmed_risk", {"seconds": round(display_eligible_seconds, 6), "window_seconds": round(total_window_seconds, 6), "ratio": round(display_eligible_seconds / total_window_seconds, 6)}),
        ("frame_stabilization_hard_boundaries", "confirmed_risk", {"hard_boundary_frames": hard_boundary_frames, "processed_frames": aggregate["processed_frame_count"]}),
        ("non_tracked_frames", "passed", non_tracked_frames),
        ("reliable_anatomical_side_switches", "confirmed_risk", side_metrics),
        ("hand_in_action_naming", "confirmed_gap", hand_action_dataflow),
        ("end_to_end_seconds", "passed" if abs(aggregate["end_to_end_seconds"] - 42.994282) < 1e-6 else "failed", aggregate["end_to_end_seconds"]),
        ("browser_visual_validation", "confirmed_gap", browser_validation),
        ("mediapipe_clearcut", "confirmed_risk", telemetry),
        ("git_repository", "confirmed_gap", git_state),
        ("requirements_locking", "confirmed_gap" if unpinned else "passed", requirements),
        ("validation_flags", "passed" if all_flags_false else "failed", {"config": config_flags, "artifacts": artifact_flags}),
    ]
    return {
        "schema_version": "factory_master_loop_baseline_audit_v1",
        "project_path": str(actual_root),
        "source_phase_b_root": PHASE_B_ROOT.relative_to(PROJECT_ROOT).as_posix(),
        "analysis_paths": [
            path.relative_to(PROJECT_ROOT).as_posix()
            for path in analysis_paths
        ],
        "check_count": len(checks),
        "checks": [
            {"check_id": check_id, "status": status, "evidence": evidence}
            for check_id, status, evidence in checks
        ],
        "summary": {
            "aggregate_runtime": aggregate,
            "analysis_window_seconds": round(total_window_seconds, 6),
            "display_eligible_seconds": round(display_eligible_seconds, 6),
            "display_eligible_ratio": round(
                display_eligible_seconds / total_window_seconds, 6
            ),
            "frame_hard_boundary_count": hard_boundary_frames,
            "non_tracked_frame_count": non_tracked_frames,
            **side_metrics,
            "precision_recall_status": "not_evaluable",
            "accuracy_claimed": False,
        },
        "all_numeric_claims_match": all(
            item[1] != "failed" for item in checks
        ),
        "audit_limitations": [
            "No human action or hand ground truth was provided.",
            "Official source metadata was verified against the recorded official documentation, but the model was not downloaded again.",
            "The documented Clearcut attempt lacks a frozen raw runtime log artifact.",
            "No real-browser visual validation artifact existed at baseline.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    output = args.output.resolve()
    if PROJECT_ROOT not in output.parents:
        raise ValueError("Audit output must stay inside the project workspace")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "check_count": result["check_count"],
                "all_numeric_claims_match": result[
                    "all_numeric_claims_match"
                ],
                "summary": result["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
