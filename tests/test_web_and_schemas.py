from __future__ import annotations

import json
import re
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

from scripts.validate_schemas import validate_all
from src.web.app import (
    AnalysisState,
    create_server,
    make_handler,
    parse_byte_range,
)
from src.web.resource_coordinator import (
    AnalysisResourceCoordinator,
    ResourceBusyError,
)


ROOT = Path(__file__).resolve().parents[1]


def test_all_schema_documents_are_valid_json():
    schemas = sorted((ROOT / "schemas").glob("*.schema.json"))
    assert len(schemas) >= 10
    assert (ROOT / "schemas" / "evidence_timeline.schema.json") in schemas
    assert all(json.loads(path.read_text("utf-8"))["$schema"] for path in schemas)


def test_recording_registry_passes_schema_validation():
    result = validate_all()
    assert result["status"] == "passed"
    assert result["schema_file_count"] >= 9


def test_byte_range_parser_supports_start_end_open_and_suffix():
    assert parse_byte_range("bytes=0-3", 10) == (0, 3)
    assert parse_byte_range("bytes=5-", 10) == (5, 9)
    assert parse_byte_range("bytes=-4", 10) == (6, 9)


def test_web_interface_contains_no_production_verdict_badges():
    static_root = ROOT / "src" / "web" / "static"
    text = "\n".join(
        (static_root / name).read_text("utf-8")
        for name in ("index.html", "app.js")
    )
    visible_words = set(re.findall(r"\b[A-Z]{2,5}\b", text))
    assert visible_words.isdisjoint({"OK", "NG", "PASS", "FAIL"})
    assert "100%" not in text
    assert "100分" not in text
    forbidden_identity_claims = (
        "员工",
        "姓名",
        "工号",
        "身份识别",
        "绩效",
        "face recognition",
        "employee id",
        "employee identity",
        "performance score",
    )
    assert all(term not in text.lower() for term in forbidden_identity_claims)


def test_web_ui_has_required_regions():
    text = (ROOT / "src" / "web" / "static" / "index.html").read_text("utf-8")
    for element_id in (
        "video",
        "pose-canvas",
        "body-pose-toggle",
        "hand-pose-toggle",
        "person-ref",
        "lock-state",
        "current-action",
        "current-action-duration",
        "current-evidence-state",
        "evidence-timeline",
        "evidence-track-source",
        "action-timeline",
        "process-timeline",
        "evidence-time",
        "hand-model-version",
        "source-segment-ids",
        "source-segment-detail",
        "body-points-used",
        "hand-evidence-detail",
        "event-visibility",
        "event-span",
        "event-observed-support",
        "event-bounded-gap",
        "bounded-gap-source-segment-ids",
        "event-merge-provenance",
        "stabilization-reason",
        "fragment-reason",
        "model-states",
        "pose-count",
        "hand-detected-count",
        "hand-uncertain-count",
        "hand-missing-count",
        "subsecond-action-count",
        "suppressed-count",
        "merged-count",
        "step-count",
        "timeline-coverage",
        "raw-switch-rate",
        "normal-action-coverage",
        "normal-observed-support-coverage",
        "unknown-uncertain-coverage",
        "hand-backend-state",
        "hand-backend-mode",
        "hand-quality-gate-version",
        "hand-action-feature-use",
        "left-hand-observation",
        "left-hand-quality",
        "left-hand-validation",
        "left-hand-eligible",
        "left-hand-binding",
        "left-hand-quality-reasons",
        "right-hand-observation",
        "right-hand-quality",
        "right-hand-validation",
        "right-hand-eligible",
        "right-hand-binding",
        "right-hand-quality-reasons",
        "hand-qualified-count",
        "hand-association-uncertain-count",
        "hand-insufficient-geometry-count",
        "hand-not-observed-count",
        "hand-eligible-observation-count",
    ):
        assert f'id="{element_id}"' in text


def test_web_hand_quality_ui_uses_separate_truth_axes() -> None:
    script = (ROOT / "src" / "web" / "static" / "app.js").read_text("utf-8")
    for element_id in (
        "hand-backend-state",
        "hand-backend-mode",
        "hand-quality-gate-version",
        "hand-action-feature-use",
    ):
        assert f'$("{element_id}")' in script
    for side_suffix in (
        "hand-observation",
        "hand-quality",
        "hand-validation",
        "hand-eligible",
        "hand-binding",
        "hand-quality-reasons",
    ):
        assert f"$(`${{side}}-{side_suffix}`)" in script
    for element_id in (
        "hand-qualified-count",
        "hand-association-uncertain-count",
        "hand-insufficient-geometry-count",
        "hand-not-observed-count",
        "hand-eligible-observation-count",
    ):
        assert f'"{element_id}"' in script
    for evidence_field in (
        "backend_state",
        "backend_mode",
        "quality_gate_version",
        "observation_state",
        "quality_state",
        "validation_state",
        "action_feature_eligible",
        "quality_reasons",
    ):
        assert evidence_field in script
    markup = (ROOT / "src" / "web" / "static" / "index.html").read_text(
        "utf-8"
    )
    assert "not_consumed_by_current_action_naming" in markup
    assert "not_consumed_by_current_action_naming" in script
    assert "consumed_by_current_action_naming" in script
    assert "recordFrameIndex !== frameIndex" in script
    assert "String(record.person_ref) !== String(frame.person_ref)" in script
    assert "String(record.lock_epoch) !== String(frame.lock_epoch)" in script
    assert "state.analysis?.hand_action_feature_use" in script
    assert "no bound record" in markup
    assert "no bound record" in script


def test_web_stable_action_display_uses_action_events_only():
    script = (ROOT / "src" / "web" / "static" / "app.js").read_text("utf-8")
    assert "state.allActionEvents = state.analysis.action_events || []" in script
    assert "event.display_eligible !== false" in script
    assert "state.poseSegments = state.analysis.pose_segments || []" in script
    assert '$("current-action").textContent = event?.action' in script
    assert '$("current-action").textContent = frame.action' not in script
    assert "sourceSegmentsFor(event)" in script
    assert "runtime.stable_normal_action_count" in script


def test_web_evidence_integrity_uses_explicit_provenance_and_same_lane() -> None:
    static_root = ROOT / "src" / "web" / "static"
    script = (static_root / "app.js").read_text("utf-8")
    markup = (static_root / "index.html").read_text("utf-8")

    for explicit_field in (
        "observed_support_seconds",
        "observed_support_ratio",
        "support_fragment_count",
        "bounded_gap_seconds",
        "maximum_bounded_gap_seconds",
        "bounded_uncertain_gaps",
        "bounded_gap_source_segment_ids",
        "pre_gate_merge_count",
        "pre_gate_aggregation_applied",
        "stable_normal_observed_support_seconds",
    ):
        assert explicit_field in script
    assert "sourceIds.length - 1" not in script
    assert "sourceEventIds.length - 1" not in script
    assert "source lineage is not merge proof" in script
    assert "item.anatomical_side || item.side" in script
    assert "itemSide !== lineageSide" in script
    assert "gapIdSet.has(itemId)" in script
    assert "absorbed bounded gaps" in script
    assert "same-lane suppressed" in script
    assert "direct-support interval union" in script
    assert "NORMAL EVENT SPAN" in markup
    assert "NORMAL OBSERVED SUPPORT" in markup


def test_web_uses_distinct_evidence_and_stable_tracks_with_window_geometry():
    script = (ROOT / "src" / "web" / "static" / "app.js").read_text("utf-8")
    styles = (ROOT / "src" / "web" / "static" / "styles.css").read_text(
        "utf-8"
    )

    assert "state.analysis.evidence_timeline || []" in script
    assert "deriveEvidenceTimeline(state.poseSegments)" in script
    assert '"pose_segments fallback"' in script
    assert "renderEvidenceTimeline()" in script
    assert "renderActionTimeline()" in script
    assert "state.allActionEvents = state.analysis.action_events || []" in script
    assert "event.event_kind === \"hard_boundary\"" in script
    assert "event.action === \"lost\"" in script
    assert "state.timelineStart = Number.isFinite(requestedStart)" in script
    assert "state.timelineEnd = (" in script
    assert "(start - state.timelineStart) / state.timelineDuration * 100" in script
    assert 'button.addEventListener("click", () => jumpToTime' in script
    assert 'video.addEventListener("loadedmetadata", () => {' in script
    assert "video.currentTime = clamp(" in script
    assert "@media (max-width: 1180px)" in styles
    assert "grid-template-columns: minmax(0, 1fr);" in styles
    assert "html, body { max-width: 100%; overflow-x: hidden; }" in styles
    assert ".model-version" in styles and "overflow-wrap: anywhere" in styles


def test_web_footer_labels_hand_counts_as_frame_counts():
    markup = (ROOT / "src" / "web" / "static" / "index.html").read_text("utf-8")
    script = (ROOT / "src" / "web" / "static" / "app.js").read_text("utf-8")
    assert "HAND DETECTED FRAMES" in markup
    assert "HAND UNCERTAIN FRAMES" in markup
    assert "HAND MISSING FRAMES" in markup
    assert "runtime.hand_detected_frame_count" in script
    assert "runtime.hand_uncertain_frame_count" in script
    assert "runtime.hand_missing_frame_count" in script


def test_web_hand_overlay_has_anatomical_colors_and_truthful_missing_guard():
    script = (ROOT / "src" / "web" / "static" / "app.js").read_text("utf-8")
    assert 'left: "#35b6d4"' in script
    assert 'right: "#9c8cff"' in script
    assert (
        'if (["missing", "lost", "unavailable"].includes(observation)) return;'
        in script
    )
    assert "if (!landmarks.length) return;" in script
    assert (
        'if (frame.track_state !== "tracked") return renderMetrics;'
        in script
    )
    assert "record.anatomical_side || record.side" in script
    assert "record.person_ref" in script
    assert "record.lock_epoch" in script


def test_web_status_and_video_range_are_accessible(tmp_path: Path):
    outputs = tmp_path / "outputs" / "run"
    outputs.mkdir(parents=True)
    video = outputs / "source_video.mp4"
    video.write_bytes(b"0123456789")
    analysis = outputs / "analysis.json"
    analysis.write_text(
        json.dumps(
            {
                "project": "Factory Multimodal Action Process Lab",
                "schema_version": "factory_multimodal_analysis_v1",
                "source_video": {
                    "path": "outputs/run/source_video.mp4",
                    "duration_seconds": 1.0,
                },
                "validation_flags": {
                    "factory_camera_validated": False,
                    "production_action_model_ready": False,
                    "external_factory_validated": False,
                    "production_process_model_ready": False,
                },
                "layer_states": [],
                "tracking_summary": {},
                "runtime": {
                    "hand_backend_state_counts": {
                        "available": 2,
                        "unavailable": 0,
                        "error": 1,
                        "unknown": 0,
                    },
                    "hand_quality_state_counts": {
                        "qualified": 1,
                        "association_uncertain": 1,
                        "insufficient_geometry": 0,
                        "not_observed": 1,
                        "lost": 0,
                        "unknown": 0,
                    },
                    "hand_validation_state_counts": {
                        "not_reviewed": 1,
                        "review_required": 1,
                        "not_evaluable": 1,
                        "unknown": 0,
                    },
                    "hand_action_feature_eligible_observation_count": 1,
                    "hand_action_feature_eligible_frame_count": 1,
                },
                "pose_frames": [],
                "pose_segments": [],
                "action_events": [],
                "hand_pose_frames": [
                    {
                        "frame_index": 0,
                        "backend_state": "available",
                        "observation_state": "detected",
                        "quality_state": "qualified",
                        "validation_state": "not_reviewed",
                        "action_feature_eligible": True,
                    },
                    {
                        "frame_index": 0,
                        "backend_state": "available",
                        "observation_state": "uncertain",
                        "quality_state": "association_uncertain",
                        "validation_state": "review_required",
                        "action_feature_eligible": False,
                    },
                    {
                        "frame_index": 1,
                        "backend_state": "error",
                        "observation_state": "missing",
                        "quality_state": "not_observed",
                        "validation_state": "not_evaluable",
                        "action_feature_eligible": False,
                    },
                ],
                "stabilization_metrics": {
                    "sub_1s_stable_event_count": 0,
                    "suppressed_fragment_count": 2,
                    "merged_fragment_count": 1,
                },
                "object_tracks": [],
                "interaction_events": [],
                "process_steps": [],
            }
        ),
        encoding="utf-8",
    )
    server = create_server(analysis, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        status = json.loads(
            urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/status", timeout=3
            ).read()
        )
        assert status["counts"]["pose_frames"] == 0
        assert status["counts"]["hand_pose_frames"] == 3
        assert status["counts"]["hand_detected"] == 1
        assert status["counts"]["hand_uncertain"] == 1
        assert status["counts"]["hand_missing"] == 1
        assert status["counts"]["sub_1s_stable_events"] == 0
        assert status["counts"]["suppressed_fragments"] == 2
        assert status["counts"]["merged_fragments"] == 1
        hand_quality = status["hand_quality_summary"]
        assert hand_quality["backend_state_counts"] == {
            "available": 2,
            "unavailable": 0,
            "error": 1,
            "unknown": 0,
        }
        assert hand_quality["quality_state_counts"] == {
            "qualified": 1,
            "association_uncertain": 1,
            "insufficient_geometry": 0,
            "not_observed": 1,
            "lost": 0,
            "unknown": 0,
        }
        assert hand_quality["validation_state_counts"] == {
            "not_reviewed": 1,
            "review_required": 1,
            "not_evaluable": 1,
            "unknown": 0,
        }
        assert hand_quality["action_feature_eligible_observation_count"] == 1
        assert hand_quality["action_feature_eligible_frame_count"] == 1
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/media/video",
            headers={"Range": "bytes=2-5"},
        )
        response = urllib.request.urlopen(request, timeout=3)
        assert response.status == 206
        assert response.read() == b"2345"
        head = urllib.request.Request(
            f"http://127.0.0.1:{port}/media/video",
            method="HEAD",
        )
        head_response = urllib.request.urlopen(head, timeout=3)
        assert head_response.status == 200
        assert head_response.headers["Accept-Ranges"] == "bytes"
        invalid = urllib.request.Request(
            f"http://127.0.0.1:{port}/media/video",
            headers={"Range": "bytes=100-200"},
        )
        try:
            urllib.request.urlopen(invalid, timeout=3)
        except urllib.error.HTTPError as exc:
            assert exc.code == 416
        else:
            raise AssertionError("Invalid byte range should return 416")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_web_status_hand_quality_summary_is_legacy_compatible(
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "outputs" / "legacy"
    outputs.mkdir(parents=True)
    (outputs / "source_video.mp4").write_bytes(b"legacy-video")
    analysis = outputs / "analysis.json"
    analysis.write_text(
        json.dumps(
            {
                "project": "Factory Multimodal Action Process Lab",
                "schema_version": "factory_multimodal_analysis_v1",
                "source_video": {
                    "path": "outputs/legacy/source_video.mp4",
                    "duration_seconds": 1.0,
                },
                "validation_flags": {
                    "factory_camera_validated": False,
                    "production_action_model_ready": False,
                    "external_factory_validated": False,
                    "production_process_model_ready": False,
                },
                "layer_states": [],
                "tracking_summary": {},
                "runtime": {},
                "action_events": [],
                "hand_pose_frames": [
                    {"frame_index": 0, "observation_state": "detected"},
                    {"frame_index": 0, "observation_state": "uncertain"},
                    {"frame_index": 1, "observation_state": "missing"},
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = AnalysisState(analysis).status_payload()["hand_quality_summary"]
    assert summary["backend_state_counts"] == {
        "available": 0,
        "unavailable": 0,
        "error": 0,
        "unknown": 3,
    }
    assert summary["quality_state_counts"] == {
        "qualified": 0,
        "association_uncertain": 0,
        "insufficient_geometry": 0,
        "not_observed": 0,
        "lost": 0,
        "unknown": 3,
    }
    assert summary["validation_state_counts"] == {
        "not_reviewed": 0,
        "review_required": 0,
        "not_evaluable": 0,
        "unknown": 3,
    }
    assert summary["action_feature_eligible_observation_count"] == 0
    assert summary["action_feature_eligible_frame_count"] == 0


def test_web_handler_quietly_handles_expected_client_disconnects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler_type = make_handler(object())
    handler = handler_type.__new__(handler_type)

    def disconnected(_handler: BaseHTTPRequestHandler) -> None:
        raise ConnectionResetError("browser cancelled a Range request")

    monkeypatch.setattr(BaseHTTPRequestHandler, "handle", disconnected)
    handler.handle()

    def unexpected(_handler: BaseHTTPRequestHandler) -> None:
        raise ValueError("real server defect")

    monkeypatch.setattr(BaseHTTPRequestHandler, "handle", unexpected)
    with pytest.raises(ValueError, match="real server defect"):
        handler.handle()


def test_camera_and_video_analysis_are_resource_mutually_exclusive():
    coordinator = AnalysisResourceCoordinator()
    lease = coordinator.acquire("video_analysis")
    assert coordinator.active_mode == "video_analysis"
    try:
        coordinator.acquire("camera")
    except ResourceBusyError:
        pass
    else:
        raise AssertionError("Camera must not overlap Video Analysis")
    coordinator.release(lease)
    camera = coordinator.acquire("camera")
    assert coordinator.active_mode == "camera"
    coordinator.release(camera)


def test_web_rejects_video_paths_outside_project_root(tmp_path: Path):
    outputs = tmp_path / "outputs" / "run"
    outputs.mkdir(parents=True)
    analysis = outputs / "analysis.json"
    analysis.write_text(
        json.dumps({"source_video": {"path": "../../outside.mp4"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="escapes"):
        AnalysisState(analysis)
