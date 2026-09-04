"""Dependency-free localhost server with byte-range video delivery."""

from __future__ import annotations

import base64
import argparse
import json
import mimetypes
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from src.camera.controller import CameraController
from src.web.job_controller import OfflineJobController
from src.web.preview_controller import PreviewController
from src.web.resource_coordinator import (
    AnalysisResourceCoordinator,
    ResourceBusyError,
)
from src.web.video_intake import (
    PreviewRegistry,
    VideoIntakeConfig,
    VideoIntakeManager,
)


STATIC_ROOT = Path(__file__).with_name("static")
HAND_BACKEND_STATES = ("available", "unavailable", "error", "unknown")
HAND_QUALITY_STATES = (
    "qualified",
    "association_uncertain",
    "insufficient_geometry",
    "not_observed",
    "lost",
    "unknown",
)
HAND_VALIDATION_STATES = (
    "not_reviewed",
    "review_required",
    "not_evaluable",
    "unknown",
)


def _nonnegative_count(value: Any, fallback: int = 0) -> int:
    if isinstance(value, bool):
        return fallback
    try:
        count = int(value)
    except (TypeError, ValueError, OverflowError):
        return fallback
    return max(0, count)


def _fixed_state_counts(
    *,
    runtime: dict[str, Any],
    runtime_key: str,
    records: list[dict[str, Any]],
    record_key: str,
    states: tuple[str, ...],
) -> dict[str, int]:
    """Return a stable enum-keyed count map for current and legacy analyses."""

    counts = {state: 0 for state in states}
    runtime_counts = runtime.get(runtime_key)
    if isinstance(runtime_counts, dict):
        for state in states:
            counts[state] = _nonnegative_count(runtime_counts.get(state))
        return counts

    known_states = set(states) - {"unknown"}
    for record in records:
        state = str(record.get(record_key, "unknown")).lower()
        counts[state if state in known_states else "unknown"] += 1
    return counts


def _hand_quality_summary(
    payload: dict[str, Any],
    hand_frames: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a fail-closed status summary without upgrading old evidence."""

    runtime_value = payload.get("runtime", {})
    runtime = runtime_value if isinstance(runtime_value, dict) else {}
    backend_counts = _fixed_state_counts(
        runtime=runtime,
        runtime_key="hand_backend_state_counts",
        records=hand_frames,
        record_key="backend_state",
        states=HAND_BACKEND_STATES,
    )
    quality_counts = _fixed_state_counts(
        runtime=runtime,
        runtime_key="hand_quality_state_counts",
        records=hand_frames,
        record_key="quality_state",
        states=HAND_QUALITY_STATES,
    )
    validation_counts = _fixed_state_counts(
        runtime=runtime,
        runtime_key="hand_validation_state_counts",
        records=hand_frames,
        record_key="validation_state",
        states=HAND_VALIDATION_STATES,
    )
    eligible_observations = sum(
        record.get("action_feature_eligible") is True
        for record in hand_frames
    )
    eligible_frames: set[int] = set()
    for record in hand_frames:
        if record.get("action_feature_eligible") is not True:
            continue
        try:
            frame_index = int(record.get("frame_index"))
        except (TypeError, ValueError, OverflowError):
            continue
        if frame_index >= 0:
            eligible_frames.add(frame_index)
    return {
        "backend_state_counts": backend_counts,
        "quality_state_counts": quality_counts,
        "validation_state_counts": validation_counts,
        "action_feature_eligible_observation_count": _nonnegative_count(
            runtime.get(
                "hand_action_feature_eligible_observation_count",
                eligible_observations,
            ),
            fallback=eligible_observations,
        ),
        "action_feature_eligible_frame_count": _nonnegative_count(
            runtime.get(
                "hand_action_feature_eligible_frame_count",
                len(eligible_frames),
            ),
            fallback=len(eligible_frames),
        ),
    }


def parse_byte_range(value: str, size: int) -> tuple[int, int]:
    if not value.startswith("bytes=") or "," in value:
        raise ValueError("Only one byte range is supported")
    start_text, end_text = value[6:].split("-", 1)
    if not start_text:
        suffix = int(end_text)
        if suffix <= 0:
            raise ValueError("Invalid suffix range")
        return max(0, size - suffix), size - 1
    start = int(start_text)
    end = size - 1 if not end_text else int(end_text)
    if start < 0 or start >= size or end < start:
        raise ValueError("Range is outside the file")
    return start, min(end, size - 1)


class AnalysisState:
    def __init__(self, analysis_path: str | Path) -> None:
        initial_analysis = Path(analysis_path).expanduser().resolve()
        project_root = initial_analysis
        while project_root.name != "outputs" and project_root.parent != project_root:
            project_root = project_root.parent
        if project_root.name == "outputs":
            project_root = project_root.parent
        else:
            project_root = initial_analysis.parent
        self.project_root = project_root
        self._lock = threading.RLock()
        self.analysis_path = initial_analysis
        self.payload: dict[str, Any] = {}
        self.video_path = initial_analysis
        self.activate_analysis(initial_analysis)
        self.resource_coordinator = AnalysisResourceCoordinator()
        self.camera = CameraController(
            self.project_root,
            self.resource_coordinator,
        )
        intake_config_path = (
            self.project_root / "configs" / "video_intake.json"
        )
        intake_config = (
            VideoIntakeConfig.load(intake_config_path)
            if intake_config_path.is_file()
            else VideoIntakeConfig()
        )
        self.intake = VideoIntakeManager(
            self.project_root,
            config=intake_config,
        )
        self.preview = PreviewController()
        self.preview_registry = PreviewRegistry(
            intake_config.preview_expiry_seconds
        )
        self.jobs = OfflineJobController(
            self.project_root,
            self.resource_coordinator,
            on_complete=self.activate_analysis,
        )

    def activate_analysis(self, analysis_path: str | Path) -> None:
        resolved = Path(analysis_path).expanduser().resolve()
        if not resolved.is_relative_to(self.project_root.resolve()):
            raise ValueError("Analysis path escapes the project root")
        if resolved.name != "analysis.json" or not resolved.is_file():
            raise FileNotFoundError("Analysis result is unavailable")
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        relative_video = Path(payload["source_video"]["path"])
        if relative_video.is_absolute():
            raise ValueError("Analysis video path must be project-relative")
        video_path = (self.project_root / relative_video).resolve()
        if not video_path.is_relative_to(self.project_root.resolve()):
            raise ValueError("Analysis video path escapes the project root")
        if not video_path.is_file():
            fallback = resolved.parent / relative_video.name
            if fallback.is_file():
                video_path = fallback.resolve()
            else:
                raise FileNotFoundError(
                    f"Analysis video is missing: {video_path}"
                )
        with self._lock:
            self.analysis_path = resolved
            self.payload = payload
            self.video_path = video_path

    def status_payload(self) -> dict[str, Any]:
        with self._lock:
            payload = self.payload
        hand_frames = [
            item
            for item in payload.get("hand_pose_frames", [])
            if isinstance(item, dict)
        ]
        hand_states = [
            str(item.get("observation_state", "missing")).lower()
            for item in hand_frames
            if isinstance(item, dict)
        ]
        stabilization = payload.get("stabilization_metrics", {})
        action_events = payload.get("action_events", [])
        sub_1s_count = stabilization.get("sub_1s_stable_event_count")
        if sub_1s_count is None:
            sub_1s_count = sum(
                1
                for item in action_events
                if item.get("action") != "lost"
                and float(
                    item.get(
                        "duration_seconds",
                        float(item.get("end_time", 0))
                        - float(item.get("start_time", 0)),
                    )
                )
                < 1.0
            )
        return {
            "project": payload["project"],
            "schema_version": payload["schema_version"],
            "source_video": payload["source_video"],
            "validation_flags": payload["validation_flags"],
            "layer_states": payload["layer_states"],
            "tracking_summary": payload["tracking_summary"],
            "runtime": payload["runtime"],
            "hand_quality_summary": _hand_quality_summary(
                self.payload,
                hand_frames,
            ),
            "counts": {
                "pose_frames": len(payload.get("pose_frames", [])),
                "pose_segments": len(payload.get("pose_segments", [])),
                "action_events": len(action_events),
                "hand_pose_frames": len(hand_frames),
                "hand_detected": sum(
                    state == "detected" for state in hand_states
                ),
                "hand_uncertain": sum(
                    state in {"uncertain", "predicted", "interpolated"}
                    for state in hand_states
                ),
                "hand_missing": sum(
                    state in {"missing", "lost", "off_frame"}
                    for state in hand_states
                ),
                "sub_1s_stable_events": int(sub_1s_count),
                "suppressed_fragments": int(
                    stabilization.get(
                        "suppressed_fragment_count",
                        stabilization.get("suppressed_count", 0),
                    )
                ),
                "merged_fragments": int(
                    stabilization.get(
                        "merged_fragment_count",
                        stabilization.get("merge_count", 0),
                    )
                ),
                "object_tracks": len(payload.get("object_tracks", [])),
                "interaction_events": len(
                    payload.get("interaction_events", [])
                ),
                "process_steps": len(payload.get("process_steps", [])),
            },
            "video_job": self.jobs.status(),
        }

    def close(self) -> None:
        self.jobs.close()
        self.preview.close()
        self.camera.close()


def make_handler(state: AnalysisState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "FactoryMultimodalLab/0.1"

        def handle(self) -> None:
            try:
                super().handle()
            except (
                BrokenPipeError,
                ConnectionAbortedError,
                ConnectionResetError,
            ):
                # Browsers routinely cancel speculative or superseded Range
                # requests while seeking. That is a client-side disconnect,
                # not an analysis or server failure.
                return

        def log_message(self, format: str, *args: object) -> None:
            return

        def _json(
            self,
            payload: Any,
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _static(self, name: str) -> None:
            path = (STATIC_ROOT / name).resolve()
            if path.parent != STATIC_ROOT.resolve() or not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            body = path.read_bytes()
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if path.suffix == ".js":
                content_type = "text/javascript; charset=utf-8"
            elif path.suffix == ".css":
                content_type = "text/css; charset=utf-8"
            elif path.suffix == ".html":
                content_type = "text/html; charset=utf-8"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def _video(self, *, send_body: bool = True) -> None:
            with state._lock:
                video_path = state.video_path
            size = video_path.stat().st_size
            range_header = self.headers.get("Range")
            start, end = 0, size - 1
            status = HTTPStatus.OK
            if range_header:
                try:
                    start, end = parse_byte_range(range_header, size)
                except (ValueError, TypeError):
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                status = HTTPStatus.PARTIAL_CONTENT
            length = end - start + 1
            content_type = mimetypes.guess_type(video_path.name)[0] or "video/mp4"
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if not send_body:
                return
            with video_path.open("rb") as handle:
                handle.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = handle.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def _query_integer(self, name: str) -> int | None:
            values = parse_qs(urlparse(self.path).query).get(name)
            if not values:
                return None
            try:
                value = int(values[-1])
            except ValueError as exc:
                raise ValueError(f"{name} must be an integer") from exc
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
            return value

        def _camera_frame(self) -> None:
            requested_sequence = self._query_integer("sequence")
            jpeg, sequence = state.camera.latest_jpeg(
                sequence=requested_sequence,
            )
            if jpeg is None:
                oldest, latest = state.camera.snapshot_bounds()
                self._json(
                    {
                        "state": state.camera.status()["state"],
                        "message": (
                            "Requested Camera sequence is no longer available."
                            if requested_sequence is not None and latest is not None
                            else "No real USB Camera frame is available."
                        ),
                        "requested_sequence": requested_sequence,
                        "oldest_sequence": oldest,
                        "latest_sequence": latest,
                    },
                    (
                        HTTPStatus.CONFLICT
                        if requested_sequence is not None and latest is not None
                        else HTTPStatus.SERVICE_UNAVAILABLE
                    ),
                )
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpeg)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Camera-Frame-Sequence", str(sequence))
            self.end_headers()
            self.wfile.write(jpeg)

        def _camera_packet(self) -> None:
            after_sequence = self._query_integer("after_sequence")
            snapshot = state.camera.latest_packet(
                after_sequence=after_sequence,
            )
            if snapshot is None:
                self.send_response(HTTPStatus.NO_CONTENT)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return
            evidence, jpeg = snapshot
            sequence = int(evidence["sequence"])
            self._json(
                {
                    "sequence": sequence,
                    "jpeg_base64": base64.b64encode(jpeg).decode("ascii"),
                    "evidence": evidence,
                    "transport": {
                        "frame_sequence": sequence,
                        "evidence_sequence": sequence,
                        "atomic": True,
                    },
                },
            )

        def _request_json(self) -> dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                raise ValueError("Invalid request length") from None
            if length < 0 or length > 65536:
                raise ValueError("Request body is too large")
            if length == 0:
                return {}
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("JSON request body must be an object")
            return payload

        def _video_upload(self) -> None:
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                raise ValueError("Invalid upload Content-Length") from None
            query = parse_qs(urlparse(self.path).query)
            original_filename = (
                query.get("filename", [None])[-1]
                or self.headers.get("X-Original-Filename")
            )
            if original_filename is None:
                raise ValueError("original filename is required")
            metadata = state.intake.receive(
                self.rfile,
                content_length=content_length,
                original_filename=original_filename,
            )
            self._json(metadata, HTTPStatus.CREATED)

        def _video_preview(self, payload: dict[str, Any]) -> None:
            upload_id = str(payload.get("upload_id", ""))
            intake = state.intake.get(upload_id)
            start_time = float(payload.get("start_time", 0.0))
            duration = float(intake["probe"]["duration_seconds"])
            if start_time < 0 or start_time >= duration:
                raise ValueError("preview start_time is outside the video")
            try:
                lease = state.resource_coordinator.acquire("video_analysis")
            except ResourceBusyError as exc:
                raise RuntimeError(
                    "Camera or Video Analysis currently owns the pose resource."
                ) from exc
            try:
                result = state.preview.run(
                    {
                        "video_path": str(state.intake.video_path(upload_id)),
                        "model_path": str(
                            state.project_root
                            / "models"
                            / "yolov8n-pose.onnx"
                        ),
                        "timestamp": start_time,
                    }
                )
            finally:
                state.resource_coordinator.release(lease)
            self._json(
                state.preview_registry.register(
                    upload_id=upload_id,
                    result=result,
                )
            )

        def _start_video_job(self, payload: dict[str, Any]) -> None:
            upload_id = str(payload.get("upload_id", ""))
            intake = state.intake.get(upload_id)
            duration_total = float(intake["probe"]["duration_seconds"])
            start_time = float(payload.get("start_time", 0.0))
            full_video = payload.get("full_video") is True
            if start_time < 0 or start_time >= duration_total:
                raise ValueError("analysis start_time is outside the video")
            if full_video:
                duration_seconds = duration_total - start_time
            else:
                duration_seconds = float(
                    payload.get(
                        "duration_seconds",
                        state.intake.config.default_analysis_seconds,
                    )
                )
                if duration_seconds <= 0:
                    raise ValueError(
                        "analysis duration_seconds must be positive"
                    )
                duration_seconds = min(
                    duration_seconds,
                    duration_total - start_time,
                )
            provider_policy = str(
                payload.get("body_provider_policy", "prefer_cuda")
            )
            if provider_policy not in {
                "auto",
                "prefer_cuda",
                "require_cuda",
                "cpu",
            }:
                raise ValueError("invalid Body Pose provider policy")
            recording_group_id = str(
                payload.get(
                    "recording_group_id",
                    "recording_group_unassigned",
                )
            ).strip()
            if not recording_group_id:
                recording_group_id = "recording_group_unassigned"
            if len(recording_group_id) > 128 or any(
                character in recording_group_id
                for character in ("/", "\\", "\x00")
            ):
                raise ValueError("recording_group_id is invalid")
            candidate = state.preview_registry.resolve(
                upload_id=upload_id,
                preview_id=str(payload.get("preview_id", "")),
                candidate_token=str(payload.get("candidate_token", "")),
            )
            result = state.jobs.start(
                {
                    "project_root": str(state.project_root),
                    "source_video": intake["relative_path"],
                    "start_time": start_time,
                    "duration_seconds": duration_seconds,
                    "sample_fps": 8.0,
                    "body_provider_policy": provider_policy,
                    "recording_group_id": recording_group_id,
                    "hand_enabled": payload.get("hand_enabled") is not False,
                    "manual_selection_seed": candidate,
                }
            )
            self._json(result, HTTPStatus.ACCEPTED)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path in {"/", "/index.html"}:
                self._static("index.html")
            elif path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self.send_header("Cache-Control", "public, max-age=86400")
                self.end_headers()
            elif path == "/app.js":
                self._static("app.js")
            elif path == "/classic_body_pose_renderer.js":
                self._static("classic_body_pose_renderer.js")
            elif path == "/styles.css":
                self._static("styles.css")
            elif path == "/api/status":
                self._json(state.status_payload())
            elif path == "/api/analysis":
                with state._lock:
                    payload = state.payload
                self._json(payload)
            elif path == "/api/video/job/status":
                self._json(state.jobs.status())
            elif path == "/api/video/intake/status":
                upload_id = parse_qs(urlparse(self.path).query).get(
                    "upload_id", [state.intake.latest_upload_id]
                )[-1]
                if not upload_id:
                    self._json(
                        {"state": "pending", "upload_id": None}
                    )
                else:
                    try:
                        self._json(state.intake.get(str(upload_id)))
                    except FileNotFoundError:
                        self._json(
                            {
                                "state": "failed",
                                "upload_id": upload_id,
                                "message": "Upload is not ready.",
                            },
                            HTTPStatus.NOT_FOUND,
                        )
            elif path == "/api/camera/status":
                self._json(state.camera.status())
            elif path == "/api/camera/evidence":
                evidence = state.camera.latest_evidence(
                    sequence=self._query_integer("sequence"),
                    after_sequence=self._query_integer("after_sequence"),
                )
                if evidence is None:
                    self._json(
                        {
                            "state": state.camera.status()["state"],
                            "evidence": None,
                        },
                        HTTPStatus.SERVICE_UNAVAILABLE,
                    )
                else:
                    self._json(evidence)
            elif path == "/api/camera/frame":
                self._camera_frame()
            elif path == "/api/camera/packet":
                self._camera_packet()
            elif path == "/health":
                with state._lock:
                    video_path = state.video_path
                    analysis_path = state.analysis_path
                self._json(
                    {
                        "status": "available",
                        "video_exists": video_path.is_file(),
                        "analysis_exists": analysis_path.is_file(),
                    }
                )
            elif path == "/media/video":
                self._video()
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            try:
                if path == "/api/video/upload":
                    self._video_upload()
                    return
                payload = self._request_json()
                if path == "/api/video/preview":
                    self._video_preview(payload)
                elif path == "/api/video/jobs/start":
                    self._start_video_job(payload)
                elif path == "/api/video/jobs/cancel":
                    self._json(state.jobs.cancel())
                elif path == "/api/camera/start":
                    raw_index = payload.get("device_index")
                    device_index = (
                        None if raw_index is None else int(raw_index)
                    )
                    if (
                        device_index is not None
                        and device_index not in {0, 1, 2, 3}
                    ):
                        raise ValueError(
                            "device_index must be between 0 and 3"
                        )
                    result = state.camera.start(device_index=device_index)
                    status = (
                        HTTPStatus.CONFLICT
                        if result["state"] == "busy"
                        else HTTPStatus.ACCEPTED
                    )
                    self._json(result, status)
                elif path == "/api/camera/stop":
                    self._json(state.camera.stop())
                elif path == "/api/camera/relock":
                    session_id = str(payload.get("session_id") or "")
                    candidate_token = str(
                        payload.get("candidate_token") or ""
                    )
                    raw_sequence = payload.get("frame_sequence")
                    if (
                        not session_id
                        or not candidate_token
                        or raw_sequence is None
                    ):
                        raise ValueError(
                            "session_id, frame_sequence and candidate_token "
                            "are required"
                        )
                    self._json(
                        state.camera.confirm_relock(
                            session_id=session_id,
                            frame_sequence=int(raw_sequence),
                            candidate_token=candidate_token,
                        ),
                        HTTPStatus.ACCEPTED,
                    )
                elif path == "/api/camera/relock/cancel":
                    session_id = str(payload.get("session_id") or "")
                    if not session_id:
                        raise ValueError("session_id is required")
                    self._json(
                        state.camera.cancel_relock(
                            session_id=session_id,
                            candidate_token=(
                                str(payload["candidate_token"])
                                if payload.get("candidate_token")
                                else None
                            ),
                        )
                    )
                elif path == "/api/camera/display-ack":
                    raw_sequence = payload.get("sequence")
                    if raw_sequence is None:
                        raise ValueError("sequence is required")
                    result = state.camera.mark_displayed(int(raw_sequence))
                    self._json(
                        result,
                        (
                            HTTPStatus.OK
                            if result["accepted"]
                            else HTTPStatus.CONFLICT
                        ),
                    )
                elif path == "/api/video/activate":
                    camera_status = state.camera.status()
                    if camera_status["worker_alive"]:
                        self._json(
                            {
                                "status": "busy",
                                "reason": (
                                    "Stop Camera before activating Video Analysis."
                                ),
                            },
                            HTTPStatus.CONFLICT,
                        )
                    else:
                        self._json({"status": "available"})
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
            except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
                self._json(
                    {
                        "status": "error",
                        "message": str(exc),
                    },
                    HTTPStatus.BAD_REQUEST,
                )

        def do_HEAD(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/media/video":
                self._video(send_body=False)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

    return Handler


class CameraHTTPServer(ThreadingHTTPServer):
    analysis_state: AnalysisState | None = None
    camera_controller: CameraController | None = None

    def server_close(self) -> None:
        if self.analysis_state is not None:
            self.analysis_state.close()
        elif self.camera_controller is not None:
            self.camera_controller.close()
        super().server_close()


def create_server(
    analysis_path: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> CameraHTTPServer:
    if host != "127.0.0.1":
        raise ValueError("Web intake server may only bind to 127.0.0.1")
    state = AnalysisState(analysis_path)
    server = CameraHTTPServer((host, port), make_handler(state))
    server.analysis_state = state
    server.camera_controller = state.camera
    return server


def check_web_state(analysis_path: str | Path) -> dict[str, Any]:
    state = AnalysisState(analysis_path)
    try:
        return {
            "status": "available",
            "analysis_path": str(state.analysis_path),
            "video_path": str(state.video_path),
            "video_size": state.video_path.stat().st_size,
            "api_status": state.status_payload(),
        }
    finally:
        state.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--check", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.check:
        print(json.dumps(check_web_state(args.analysis), ensure_ascii=False, indent=2))
        return
    server = create_server(args.analysis, host=args.host, port=args.port)
    print(f"Factory Multimodal Lab: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
