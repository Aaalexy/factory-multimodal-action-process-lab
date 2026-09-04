from __future__ import annotations

import hashlib
import json
import threading
import urllib.request
from pathlib import Path

import pytest

from src.web.app import create_server


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src" / "web" / "static"
ANALYSIS = (
    ROOT
    / "outputs"
    / "private_regression"
    / "replay"
    / "sample_video_C"
    / "candidate"
    / "analysis.json"
)


def test_classic_renderer_is_loaded_before_application_and_defaulted() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    classic_index = html.index('src="/classic_body_pose_renderer.js"')
    app_index = html.index('src="/app.js"')
    assert classic_index < app_index
    assert '<option value="classic" selected>' in html
    assert '<option value="evidence">' in html


def test_classic_renderer_has_fail_closed_evidence_states() -> None:
    source = (STATIC / "classic_body_pose_renderer.js").read_text(
        encoding="utf-8"
    )
    assert '"missing", "uncertain", "rejected", "lost"' in source
    assert 'status: "derived_visual_only"' in source
    assert 'evidence_type: "derived_visual_only"' in source
    assert "drew_fixed_geometry: false" in source
    assert "grasp" in source  # Explicitly documents the forbidden claim.
    assert "mock keypoint" not in source.lower()


def test_classic_and_evidence_body_modes_do_not_double_render() -> None:
    source = (STATIC / "app.js").read_text(encoding="utf-8")
    assert 'bodyRendererMode: "classic"' in source
    assert 'state.bodyRendererMode === "classic"' in source
    assert "window.ClassicBodyPoseRenderer.render(" in source
    assert "} else {\n      renderMetrics.body_segment_count = drawEvidenceBodyPose(" in source


def test_body_and_hand_visibility_controls_remain_independent() -> None:
    source = (STATIC / "app.js").read_text(encoding="utf-8")
    assert 'state.showBodyPose = event.target.checked' in source
    assert 'state.showHandPose = event.target.checked' in source
    assert "if (state.showBodyPose)" in source
    assert "if (state.showHandPose)" in source


@pytest.mark.private_artifacts
def test_classic_renderer_static_route_is_available() -> None:
    server = create_server(ANALYSIS, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        response = urllib.request.urlopen(
            f"http://127.0.0.1:{port}/classic_body_pose_renderer.js",
            timeout=5,
        )
        body = response.read().decode("utf-8")
        assert response.status == 200
        assert response.headers["Content-Type"].startswith("text/javascript")
        assert "ClassicBodyPoseRenderer" in body
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.private_artifacts
def test_stage_a_does_not_change_frozen_analysis() -> None:
    manifest = json.loads(
        (ROOT / "outputs/private_regression/fixture_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    expected = {
        clip_id: details["sha256"]
        for clip_id, details in manifest["analyses"].items()
    }
    replay_root = (
        ROOT
        / "outputs"
        / "private_regression"
        / "replay"
    )
    for clip_id, expected_hash in expected.items():
        path = replay_root / clip_id / "candidate" / "analysis.json"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
