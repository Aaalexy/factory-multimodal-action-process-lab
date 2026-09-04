"""Focused USB Camera lifecycle tests; no mock Pose evidence is created."""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.camera.contracts import CameraConfig, CameraState
from src.camera.controller import CameraController
from src.camera.frame_source import CameraOpenError, OpenCVUSBFrameSource
from src.camera.live_analysis import LiveFrameAnalyzer
from src.camera.worker import MonotonicCadenceGate
from src.web.app import AnalysisState, CameraHTTPServer, make_handler
from src.web.resource_coordinator import (
    AnalysisResourceCoordinator,
    ResourceBusyError,
)


ROOT = Path.cwd()
ANALYSIS = (
    ROOT
    / "outputs"
    / "private_regression"
    / "replay"
    / "sample_video_C"
    / "candidate"
    / "analysis.json"
)


class FakeCapture:
    def __init__(self, *, opens: bool = True) -> None:
        self.opens = opens
        self.opened = False
        self.released = False
        self.frame = np.full((48, 64, 3), 80, dtype=np.uint8)

    def set(self, *_: object) -> bool:
        return True

    def open(self, *_: object) -> bool:
        self.opened = self.opens
        return self.opened

    def isOpened(self) -> bool:
        return self.opened

    def get(self, prop: int) -> float:
        return {
            cv2.CAP_PROP_FRAME_WIDTH: 64.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 48.0,
            cv2.CAP_PROP_FPS: 8.0,
        }.get(prop, 0.0)

    def getBackendName(self) -> str:
        return "TEST"

    def read(self) -> tuple[bool, np.ndarray]:
        return True, self.frame.copy()

    def release(self) -> None:
        self.released = True
        self.opened = False


class ThreadProcess:
    sequence = 4000

    def __init__(self, *, target, args, name, daemon) -> None:
        del name, daemon
        type(self).sequence += 1
        self.pid = type(self).sequence
        self._target = target
        self._args = args
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._target,
            args=self._args,
            daemon=True,
        )
        self._thread.start()

    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    def terminate(self) -> None:
        self._args[3].set()


class ThreadContext:
    def Event(self) -> threading.Event:  # noqa: N802
        return threading.Event()

    def Queue(self, maxsize: int):  # noqa: N802
        return queue.Queue(maxsize=maxsize)

    def Process(self, **kwargs):  # noqa: N802
        return ThreadProcess(**kwargs)


def state_worker(
    project_root,
    config,
    session_id,
    stop_event,
    output_queue,
    command_queue,
):
    del project_root, config, command_queue
    output_queue.put(
        {
            "kind": "frame",
            "state": "live",
            "session_id": session_id,
            "sequence": 1,
            "jpeg": b"\xff\xd8\xff\xd9",
            "width": 64,
            "height": 48,
            "timestamp": 0.0,
            "evidence": {
                "frame": None,
                "hand_pose_frames": [],
                "stable_action": {
                    "status": "uncertain",
                    "training_eligible": False,
                },
            },
            "metrics": {"processed_frame_count": 1},
        }
    )
    stop_event.wait(2.0)
    output_queue.put(
        {
            "kind": "state",
            "state": "stopped",
            "session_id": session_id,
            "metrics": {"device_released": True},
        }
    )


def no_device_worker(
    project_root,
    config,
    session_id,
    stop_event,
    output_queue,
    command_queue,
):
    del project_root, config, stop_event, command_queue
    output_queue.put(
        {
            "kind": "error",
            "state": "no_device",
            "error_code": "usb_open_or_read_failed",
            "message": "Unable to open local USB Camera index 0",
            "session_id": session_id,
        }
    )


def controller(worker=state_worker):
    return CameraController(
        ROOT,
        AnalysisResourceCoordinator(),
        worker_target=worker,
        context=ThreadContext(),
    )


def test_camera_config_is_local_usb_only_and_flags_false():
    config = CameraConfig.load(ROOT / "configs" / "camera.json")
    assert config.device_index == 0
    assert config.requested_fps == 15.0
    assert config.pose_display_fps == 12.0
    assert config.analysis_fps == 8.0
    assert config.analysis_fps < config.pose_display_fps
    assert config.persist_recording is False
    payload = json.loads((ROOT / "configs" / "camera.json").read_text())
    assert all(
        payload[name] is False
        for name in (
            "factory_camera_validated",
            "production_action_model_ready",
            "external_factory_validated",
            "production_process_model_ready",
        )
    )
    assert "rtsp" not in payload


def test_pose_display_cadence_accumulates_deadlines_without_phase_reset():
    gate = MonotonicCadenceGate(12.0)
    camera_timestamps = [index / 15.0 for index in range(15)]
    accepted = [
        timestamp
        for timestamp in camera_timestamps
        if gate.should_run(timestamp)
    ]
    assert len(accepted) == 12
    assert accepted[-1] >= 0.9


def test_action_sampling_stays_at_eight_fps_and_forces_lost_boundary():
    analyzer = object.__new__(LiveFrameAnalyzer)
    analyzer.analysis_fps = 8.0
    analyzer._next_action_due = None
    analyzer._last_action_boundary = None
    sampled = [
        timestamp
        for timestamp in [index / 12.0 for index in range(12)]
        if analyzer._action_sample_due(
            timestamp=timestamp,
            person_ref="person-0001",
            lock_epoch=1,
            track_state="tracked",
            lock_state="locked",
        )
    ]
    assert len(sampled) == 8
    assert analyzer._action_sample_due(
        timestamp=1.01,
        person_ref="person-0001",
        lock_epoch=1,
        track_state="lost",
        lock_state="lost",
    )


def test_usb_source_uses_monotonic_timestamp_and_releases():
    capture = FakeCapture()
    moments = iter([10.0, 10.0, 10.125])
    source = OpenCVUSBFrameSource(
        0,
        requested_fps=8,
        open_timeout_seconds=1,
        read_timeout_seconds=1,
        capture_factory=lambda: capture,
        monotonic=lambda: next(moments),
    )
    source.open()
    packet = source.read_packet()
    source.close()
    assert packet.timestamp == pytest.approx(0.125)
    assert packet.image.shape == (48, 64, 3)
    assert capture.released


def test_usb_open_failure_releases_capture():
    capture = FakeCapture(opens=False)
    source = OpenCVUSBFrameSource(
        0,
        open_timeout_seconds=1,
        read_timeout_seconds=1,
        capture_factory=lambda: capture,
    )
    with pytest.raises(CameraOpenError):
        source.open()
    assert capture.released


def test_controller_repeat_start_has_one_owner_and_stop_releases():
    value = controller()
    first = value.start()
    live = value.wait_for_terminal_or_live(1.0)
    second = value.start()
    assert first["state"] in {"opening", "live"}
    assert live["state"] == "live"
    assert second["worker_pid"] == live["worker_pid"]
    assert value.coordinator.active_mode == "camera"
    stopped = value.stop()
    assert stopped["state"] == "stopped"
    assert stopped["worker_alive"] is False
    assert stopped["metrics"]["device_released"] is True
    assert value.coordinator.active_mode is None


def test_camera_live_rejects_video_owner_then_video_recovers():
    value = controller()
    value.start()
    assert value.wait_for_terminal_or_live(1.0)["state"] == "live"
    with pytest.raises(ResourceBusyError):
        value.coordinator.acquire("video_analysis")
    value.stop()
    lease = value.coordinator.acquire("video_analysis")
    value.coordinator.release(lease)


def test_camera_can_be_opened_again_after_stop():
    value = controller()
    value.start()
    first = value.wait_for_terminal_or_live(1.0)["worker_pid"]
    value.stop()
    value.start()
    second = value.wait_for_terminal_or_live(1.0)["worker_pid"]
    value.stop()
    assert first != second


def test_no_device_is_honest_and_does_not_expose_traceback():
    value = controller(no_device_worker)
    value.start()
    status = value.wait_for_terminal_or_live(1.0)
    assert status["state"] == "no_device"
    assert status["worker_alive"] is False
    assert "traceback" not in json.dumps(status).lower()
    assert value.coordinator.active_mode is None


def test_latest_only_frame_and_evidence_are_bounded():
    value = controller()
    value.start()
    value.wait_for_terminal_or_live(1.0)
    jpeg, sequence = value.latest_jpeg()
    evidence = value.latest_evidence()
    value.stop()
    assert jpeg == b"\xff\xd8\xff\xd9"
    assert sequence == 1
    assert evidence["evidence"]["stable_action"]["training_eligible"] is False


def test_exact_sequence_snapshot_never_returns_a_newer_jpeg():
    value = controller()
    value.start()
    value.wait_for_terminal_or_live(1.0)
    for sequence in (2, 3):
        value._output_queue.put(
            {
                "kind": "frame",
                "state": "live",
                "session_id": value._session_id,
                "sequence": sequence,
                "jpeg": f"jpeg-{sequence}".encode(),
                "width": 64,
                "height": 48,
                "timestamp": sequence / 8,
                "captured_at_monotonic": time.monotonic(),
                "evidence": {"frame_sequence": sequence},
                "metrics": {"processed_frame_count": sequence},
            }
        )
    value.status()
    old_jpeg, old_sequence = value.latest_jpeg(sequence=1)
    jpeg_2, sequence_2 = value.latest_jpeg(sequence=2)
    evidence_2 = value.latest_evidence(sequence=2)
    newest = value.latest_packet(after_sequence=2)
    acknowledgement = value.mark_displayed(3)
    metrics = value.status()["metrics"]
    value.stop()
    assert old_jpeg is None
    assert old_sequence is None
    assert jpeg_2 == b"jpeg-2"
    assert sequence_2 == 2
    assert evidence_2["sequence"] == 2
    assert newest[0]["sequence"] == 3
    assert newest[1] == b"jpeg-3"
    assert acknowledgement["accepted"] is True
    assert metrics["frame_evidence_sequence_mismatch_count"] == 0
    assert metrics["snapshot_buffer_size"] == 2
    assert metrics["snapshot_eviction_count"] >= 1


@pytest.mark.private_artifacts
def test_camera_http_api_start_status_frame_stop():
    state = AnalysisState(ANALYSIS)
    state.camera.close()
    state.camera = controller()
    server = CameraHTTPServer(("127.0.0.1", 0), make_handler(state))
    server.camera_controller = state.camera
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        request = urllib.request.Request(
            base + "/api/camera/start",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            assert response.status == 202
        state.camera.wait_for_terminal_or_live(1.0)
        with urllib.request.urlopen(base + "/api/camera/status") as response:
            assert json.load(response)["state"] == "live"
        with urllib.request.urlopen(base + "/api/camera/frame") as response:
            assert response.status == 200
            assert response.headers["Content-Type"] == "image/jpeg"
            assert response.headers["X-Camera-Frame-Sequence"] == "1"
        with urllib.request.urlopen(
            base + "/api/camera/frame?sequence=1"
        ) as response:
            assert response.read() == b"\xff\xd8\xff\xd9"
            assert response.headers["X-Camera-Frame-Sequence"] == "1"
        with pytest.raises(urllib.error.HTTPError) as stale:
            urllib.request.urlopen(
                base + "/api/camera/frame?sequence=999"
            )
        assert stale.value.code == 409
        with urllib.request.urlopen(
            base + "/api/camera/packet?after_sequence=0"
        ) as response:
            atomic_packet = json.load(response)
        assert atomic_packet["sequence"] == 1
        assert atomic_packet["transport"] == {
            "frame_sequence": 1,
            "evidence_sequence": 1,
            "atomic": True,
        }
        display_request = urllib.request.Request(
            base + "/api/camera/display-ack",
            data=b'{"sequence":1}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(display_request) as response:
            assert json.load(response)["accepted"] is True
        video_request = urllib.request.Request(
            base + "/api/video/activate",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as captured:
            urllib.request.urlopen(video_request)
        assert captured.value.code == 409
        stop_request = urllib.request.Request(
            base + "/api/camera/stop",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(stop_request) as response:
            assert json.load(response)["state"] == "stopped"
        range_request = urllib.request.Request(
            base + "/media/video",
            headers={"Range": "bytes=0-1023"},
        )
        with urllib.request.urlopen(range_request) as response:
            assert response.status == 206
            assert len(response.read()) == 1024
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_camera_ui_is_not_hardcoded_unavailable_and_keeps_classic_renderer():
    html = (ROOT / "src" / "web" / "static" / "index.html").read_text(
        encoding="utf-8"
    )
    javascript = (ROOT / "src" / "web" / "static" / "app.js").read_text(
        encoding="utf-8"
    )
    assert 'id="camera-mode-button"' in html
    assert "Camera · unavailable" not in html
    assert 'id="camera-view"' in html
    assert "ClassicBodyPoseRenderer.render" in javascript
    assert "/api/camera/start" in javascript
    assert "/api/camera/stop" in javascript
    assert "/api/camera/packet?after_sequence=" in javascript
    assert "/api/camera/display-ack" in javascript
    assert "requestVideoFrameCallback" in javascript
    assert "requestAnimationFrame" in javascript
    assert "setInterval(pollCamera, 350)" not in javascript
    assert '$("event-status")' not in javascript
    assert '$("event-sources")' not in javascript
    assert '$("event-reason")' not in javascript
    assert '$("source-segment-ids")' in javascript
    assert '$("stabilization-reason")' in javascript
    assert '$("fragment-reason")' in javascript
    assert "persist_recording" not in javascript


def test_camera_contract_declares_all_required_states():
    assert {
        "unavailable",
        "no_device",
        "permission_denied",
        "busy",
        "opening",
        "live",
        "stopping",
        "stopped",
        "error",
    } == {item.value for item in CameraState}


def test_no_mock_pose_or_identity_semantics_in_camera_package():
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "src" / "camera").glob("*.py")
    ).lower()
    assert "mock keypoint" not in text
    assert "face recognition" not in text
    assert "employee id" not in text
    assert "grasp_detected" not in text
    assert "pass/fail" not in text
    assert "deepseek" not in text
    assert "rtsp" not in text
