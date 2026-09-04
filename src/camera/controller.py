"""Thread-safe Web controller for one bounded Windows-spawn Camera worker."""

from __future__ import annotations

from collections import OrderedDict
import copy
import multiprocessing as mp
import queue
import secrets
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from src.web.resource_coordinator import (
    AnalysisResourceCoordinator,
    ResourceBusyError,
    ResourceLease,
)

from .contracts import CameraConfig, CameraState
from .worker import run_camera_worker


WorkerTarget = Callable[..., None]


class CameraController:
    def __init__(
        self,
        project_root: str | Path,
        coordinator: AnalysisResourceCoordinator,
        *,
        config_path: str | Path = "configs/camera.json",
        worker_target: WorkerTarget = run_camera_worker,
        context: Any | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        config_file = Path(config_path)
        if not config_file.is_absolute():
            config_file = self.project_root / config_file
        self.config = (
            CameraConfig.load(config_file)
            if config_file.is_file()
            else CameraConfig(enabled=False, hand_enabled=False)
        )
        self.coordinator = coordinator
        self.worker_target = worker_target
        self.context = context or mp.get_context("spawn")
        self._lock = threading.RLock()
        self._process: Any | None = None
        self._stop_event: Any | None = None
        self._output_queue: Any | None = None
        self._command_queue: Any | None = None
        self._lease: ResourceLease | None = None
        self._state = (
            CameraState.STOPPED
            if self.config.enabled
            else CameraState.UNAVAILABLE
        )
        self._session_id: str | None = None
        self._selected_device_index = self.config.device_index
        self._last_error: dict[str, Any] | None = None
        self._last_notice: str | None = None
        self._latest_jpeg: bytes | None = None
        self._latest: dict[str, Any] | None = None
        self._snapshots: OrderedDict[
            int, tuple[dict[str, Any], bytes]
        ] = OrderedDict()
        self._snapshot_eviction_count = 0
        self._displayed_frame_count = 0
        self._display_ack_stale_count = 0
        self._dropped_display_frame_count = 0
        self._last_displayed_sequence: int | None = None
        self._last_displayed_at: float | None = None
        self._first_displayed_at: float | None = None
        self._frame_ages_ms: list[float] = []
        self._preview_gaps_ms: list[float] = []
        self._metrics: dict[str, Any] = {}
        self._device_metadata: dict[str, Any] = {}
        self._candidate_tokens: dict[str, dict[str, Any]] = {}
        self._manual_relock_events: list[dict[str, Any]] = []

    def _purge_candidate_tokens(self, latest_sequence: int | None = None) -> None:
        now = time.monotonic()
        minimum_sequence = (
            int(latest_sequence) - self.config.candidate_sequence_tolerance
            if latest_sequence is not None
            else None
        )
        stale = [
            token
            for token, record in self._candidate_tokens.items()
            if float(record["expires_at_monotonic"]) < now
            or record["session_id"] != self._session_id
            or (
                minimum_sequence is not None
                and int(record["frame_sequence"]) < minimum_sequence
            )
        ]
        for token in stale:
            self._candidate_tokens.pop(token, None)

    def _decorate_frame_candidates(
        self, frame_item: dict[str, Any]
    ) -> dict[str, Any]:
        """Replace worker internals with opaque, session-bound public tokens."""

        decorated = copy.deepcopy(frame_item)
        sequence = int(decorated["sequence"])
        self._purge_candidate_tokens(sequence)
        evidence = decorated.get("evidence")
        frame = evidence.get("frame") if isinstance(evidence, dict) else None
        if not isinstance(frame, dict):
            return decorated
        candidates = frame.get("anonymous_candidates")
        if not isinstance(candidates, list):
            frame["anonymous_candidates"] = []
            return decorated
        now_monotonic = time.monotonic()
        expires_at_unix_ms = int(
            (time.time() + self.config.candidate_token_expiry_seconds) * 1000
        )
        public_candidates: list[dict[str, Any]] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            token = secrets.token_urlsafe(24)
            internal = {
                **copy.deepcopy(candidate),
                "session_id": self._session_id,
                "frame_sequence": sequence,
                "candidate_token": token,
                "expires_at_monotonic": (
                    now_monotonic + self.config.candidate_token_expiry_seconds
                ),
            }
            self._candidate_tokens[token] = internal
            public_candidates.append(
                {
                    "candidate_id": str(candidate.get("candidate_id", "")),
                    "candidate_token": token,
                    "session_id": self._session_id,
                    "frame_sequence": sequence,
                    "bbox": list(candidate.get("bbox") or []),
                    "confidence": float(candidate.get("confidence", 0.0)),
                    "expiry": expires_at_unix_ms,
                    "source_width": int(
                        candidate.get("source_width")
                        or decorated.get("width")
                        or 0
                    ),
                    "source_height": int(
                        candidate.get("source_height")
                        or decorated.get("height")
                        or 0
                    ),
                    "mirror_horizontal": bool(
                        candidate.get("mirror_horizontal", False)
                    ),
                }
            )
        frame["anonymous_candidates"] = public_candidates
        return decorated

    def _drain(self) -> None:
        if self._output_queue is None:
            return
        while True:
            try:
                item = self._output_queue.get_nowait()
            except queue.Empty:
                break
            if item.get("session_id") != self._session_id:
                continue
            kind = item.get("kind")
            if kind == "frame":
                frame_item = dict(item)
                self._latest_jpeg = frame_item.pop("jpeg", None)
                frame_item = self._decorate_frame_candidates(frame_item)
                self._latest = frame_item
                sequence = int(frame_item["sequence"])
                if self._latest_jpeg is not None:
                    self._snapshots[sequence] = (
                        dict(frame_item),
                        self._latest_jpeg,
                    )
                    while (
                        len(self._snapshots)
                        > self.config.latest_frame_buffer_size
                    ):
                        self._snapshots.popitem(last=False)
                        self._snapshot_eviction_count += 1
                self._metrics = dict(item.get("metrics", {}))
                self._state = CameraState.LIVE
            elif kind == "error":
                self._last_error = {
                    "code": item.get("error_code", "camera_error"),
                    "message": item.get(
                        "message", "Local USB Camera could not start."
                    ),
                }
                try:
                    self._state = CameraState(item.get("state"))
                except ValueError:
                    self._state = CameraState.ERROR
            elif kind == "notice":
                self._last_notice = str(item.get("message", ""))
            elif kind == "manual_relock":
                event = item.get("event")
                if isinstance(event, dict):
                    self._manual_relock_events.append(dict(event))
                    self._manual_relock_events = self._manual_relock_events[-50:]
                    self._last_notice = str(
                        event.get("event", "manual_relock_updated")
                    )
            elif kind == "state":
                if isinstance(item.get("metadata"), dict):
                    self._device_metadata = dict(item["metadata"])
                if isinstance(item.get("metrics"), dict):
                    self._metrics.update(item["metrics"])
                next_state = item.get("state")
                if (
                    next_state == "stopped"
                    and self._state
                    in {
                        CameraState.NO_DEVICE,
                        CameraState.PERMISSION_DENIED,
                        CameraState.ERROR,
                    }
                ):
                    continue
                try:
                    parsed_state = CameraState(next_state)
                    if (
                        parsed_state == CameraState.STOPPED
                        and self._process is not None
                        and self._process.is_alive()
                    ):
                        self._state = CameraState.STOPPING
                    else:
                        self._state = parsed_state
                except ValueError:
                    pass
        if self._process is not None and not self._process.is_alive():
            self._process.join(timeout=0)
            self._process = None
            if self._lease is not None:
                self.coordinator.release(self._lease)
                self._lease = None
            if self._state in {CameraState.OPENING, CameraState.LIVE}:
                self._state = CameraState.ERROR
                self._last_error = {
                    "code": "worker_exited_without_final_state",
                    "message": "Camera worker stopped unexpectedly.",
                }

    def start(self, *, device_index: int | None = None) -> dict[str, Any]:
        with self._lock:
            self._drain()
            if not self.config.enabled:
                self._state = CameraState.UNAVAILABLE
                return self.status()
            if self._process is not None and self._process.is_alive():
                return self.status()
            try:
                self._lease = self.coordinator.acquire("camera")
            except ResourceBusyError:
                self._state = CameraState.BUSY
                return self.status()
            selected = (
                self.config
                if device_index is None
                else CameraConfig(
                    **{
                        **asdict(self.config),
                        "device_index": int(device_index),
                    }
                )
            )
            selected.validate()
            self._selected_device_index = selected.device_index
            self._session_id = "usb_" + uuid.uuid4().hex
            self._stop_event = self.context.Event()
            self._output_queue = self.context.Queue(
                maxsize=selected.latest_frame_buffer_size
            )
            self._command_queue = self.context.Queue(maxsize=4)
            self._latest = None
            self._latest_jpeg = None
            self._snapshots.clear()
            self._snapshot_eviction_count = 0
            self._displayed_frame_count = 0
            self._display_ack_stale_count = 0
            self._dropped_display_frame_count = 0
            self._last_displayed_sequence = None
            self._last_displayed_at = None
            self._first_displayed_at = None
            self._frame_ages_ms.clear()
            self._preview_gaps_ms.clear()
            self._metrics = {}
            self._device_metadata = {}
            self._last_error = None
            self._last_notice = None
            self._candidate_tokens.clear()
            self._manual_relock_events.clear()
            self._state = CameraState.OPENING
            self._process = self.context.Process(
                target=self.worker_target,
                args=(
                    str(self.project_root),
                    asdict(selected),
                    self._session_id,
                    self._stop_event,
                    self._output_queue,
                    self._command_queue,
                ),
                name=f"factory-usb-camera-{selected.device_index}",
                daemon=True,
            )
            try:
                self._process.start()
            except Exception:
                self.coordinator.release(self._lease)
                self._lease = None
                self._process = None
                self._state = CameraState.ERROR
                raise
            return self.status()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            self._drain()
            process = self._process
            if process is None:
                if self._lease is not None:
                    self.coordinator.release(self._lease)
                    self._lease = None
                self._state = CameraState.STOPPED
                return self.status()
            self._state = CameraState.STOPPING
            self._stop_event.set()
        deadline = time.monotonic() + self.config.stop_timeout_seconds
        while process.is_alive() and time.monotonic() < deadline:
            process.join(timeout=0.05)
            with self._lock:
                self._drain()
        with self._lock:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
                self._last_error = {
                    "code": "camera_stop_timeout",
                    "message": "Camera worker required forced local termination.",
                }
            self._drain()
            self._process = None
            if self._lease is not None:
                self.coordinator.release(self._lease)
                self._lease = None
            self._state = CameraState.STOPPED
            self._metrics["device_released"] = True
            return self.status()

    def confirm_relock(
        self,
        *,
        session_id: str,
        frame_sequence: int,
        candidate_token: str,
    ) -> dict[str, Any]:
        with self._lock:
            self._drain()
            if self._command_queue is None or self._state != CameraState.LIVE:
                raise RuntimeError("Camera is not live")
            if not session_id or session_id != self._session_id:
                raise RuntimeError(
                    "Candidate does not belong to the current Camera session"
                )
            self._purge_candidate_tokens(
                int(self._latest["sequence"]) if self._latest else None
            )
            selection = self._candidate_tokens.get(str(candidate_token))
            if selection is None:
                raise RuntimeError("Candidate token is expired or unknown")
            if (
                selection["session_id"] != session_id
                or int(selection["frame_sequence"]) != int(frame_sequence)
            ):
                raise RuntimeError(
                    "Candidate token does not match the requested frame"
                )
            latest_sequence = (
                int(self._latest["sequence"]) if self._latest else -1
            )
            if (
                latest_sequence - int(frame_sequence)
                > self.config.candidate_sequence_tolerance
            ):
                raise RuntimeError("Candidate frame is too old to relock")
            try:
                self._command_queue.put_nowait(
                    {
                        "command": "confirm_relock",
                        "selection": {
                            key: value
                            for key, value in selection.items()
                            if key != "expires_at_monotonic"
                        },
                    }
                )
            except queue.Full as exc:
                raise RuntimeError("Camera command queue is busy") from exc
            self._candidate_tokens.pop(str(candidate_token), None)
            event = {
                "event": "manual_relock_selection_queued",
                "session_id": session_id,
                "frame_sequence": int(frame_sequence),
                "candidate_id": selection["candidate_id"],
                "status": "proposed",
                "training_eligible": False,
            }
            self._manual_relock_events.append(event)
            return {
                **self.status(),
                "selection_queued": True,
                "selected_candidate_id": selection["candidate_id"],
                "frame_sequence": int(frame_sequence),
            }

    def cancel_relock(
        self,
        *,
        session_id: str,
        candidate_token: str | None = None,
    ) -> dict[str, Any]:
        """Cancel only the UI selection; never change the anonymous lock."""

        with self._lock:
            self._drain()
            if not session_id or session_id != self._session_id:
                raise RuntimeError(
                    "Relock cancellation does not match the Camera session"
                )
            if candidate_token:
                record = self._candidate_tokens.get(str(candidate_token))
                if record is not None and record["session_id"] == session_id:
                    self._candidate_tokens.pop(str(candidate_token), None)
            event = {
                "event": "manual_relock_selection_cancelled",
                "session_id": session_id,
                "person_changed": False,
                "status": "proposed",
                "training_eligible": False,
            }
            self._manual_relock_events.append(event)
            return {
                **self.status(),
                "selection_cancelled": True,
                "person_changed": False,
            }

    def status(self) -> dict[str, Any]:
        with self._lock:
            self._drain()
            metrics = {
                **self._metrics,
                **self._transport_metrics(),
            }
            return {
                "state": self._state.value,
                "session_id": self._session_id,
                "worker_pid": (
                    self._process.pid if self._process is not None else None
                ),
                "worker_alive": bool(
                    self._process is not None
                    and self._process.is_alive()
                ),
                "device_index": self._selected_device_index,
                "device_metadata": self._device_metadata,
                "metrics": metrics,
                "last_error": self._last_error,
                "last_notice": self._last_notice,
                "manual_relock_last_event": (
                    self._manual_relock_events[-1]
                    if self._manual_relock_events
                    else None
                ),
                "candidate_token_count": len(self._candidate_tokens),
                "latest_sequence": (
                    self._latest.get("sequence") if self._latest else None
                ),
                "video_analysis_available": (
                    self.coordinator.active_mode != "camera"
                ),
                "persist_recording": False,
                "evidence_class": "LOCAL_USB_TECHNICAL_VALIDATION_ONLY",
                "factory_camera_validated": False,
                "production_action_model_ready": False,
                "external_factory_validated": False,
                "production_process_model_ready": False,
            }

    def _snapshot(
        self,
        *,
        sequence: int | None = None,
        after_sequence: int | None = None,
    ) -> tuple[dict[str, Any], bytes] | None:
        self._drain()
        if not self._snapshots:
            return None
        if sequence is not None:
            snapshot = self._snapshots.get(int(sequence))
            if snapshot is None:
                return None
            return dict(snapshot[0]), snapshot[1]
        latest_sequence = next(reversed(self._snapshots))
        if after_sequence is not None and latest_sequence <= int(after_sequence):
            return None
        snapshot = self._snapshots[latest_sequence]
        return dict(snapshot[0]), snapshot[1]

    def snapshot_bounds(self) -> tuple[int | None, int | None]:
        with self._lock:
            self._drain()
            if not self._snapshots:
                return None, None
            return next(iter(self._snapshots)), next(reversed(self._snapshots))

    def latest_evidence(
        self,
        *,
        sequence: int | None = None,
        after_sequence: int | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            snapshot = self._snapshot(
                sequence=sequence,
                after_sequence=after_sequence,
            )
            if snapshot is None:
                return None
            return snapshot[0]

    def latest_jpeg(
        self,
        *,
        sequence: int | None = None,
    ) -> tuple[bytes | None, int | None]:
        with self._lock:
            snapshot = self._snapshot(sequence=sequence)
            if snapshot is None:
                return None, None
            return snapshot[1], int(snapshot[0]["sequence"])

    def latest_packet(
        self,
        *,
        after_sequence: int | None = None,
    ) -> tuple[dict[str, Any], bytes] | None:
        with self._lock:
            return self._snapshot(after_sequence=after_sequence)

    def mark_displayed(self, sequence: int) -> dict[str, Any]:
        """Record an explicit browser onload acknowledgement for one snapshot."""

        with self._lock:
            self._drain()
            snapshot = self._snapshots.get(int(sequence))
            if snapshot is None:
                self._display_ack_stale_count += 1
                return {
                    "accepted": False,
                    "sequence": int(sequence),
                    "reason": "sequence_not_in_bounded_snapshot_cache",
                }
            now = time.monotonic()
            if self._last_displayed_sequence == int(sequence):
                return {
                    "accepted": True,
                    "sequence": int(sequence),
                    "duplicate": True,
                }
            if (
                self._last_displayed_sequence is not None
                and int(sequence) > self._last_displayed_sequence + 1
            ):
                self._dropped_display_frame_count += (
                    int(sequence) - self._last_displayed_sequence - 1
                )
            if self._last_displayed_at is not None:
                self._preview_gaps_ms.append(
                    (now - self._last_displayed_at) * 1000.0
                )
            captured = snapshot[0].get("captured_at_monotonic")
            if isinstance(captured, (int, float)):
                self._frame_ages_ms.append(
                    max(0.0, (now - float(captured)) * 1000.0)
                )
            if self._first_displayed_at is None:
                self._first_displayed_at = now
            self._last_displayed_at = now
            self._last_displayed_sequence = int(sequence)
            self._displayed_frame_count += 1
            return {
                "accepted": True,
                "sequence": int(sequence),
                "duplicate": False,
            }

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        index = (len(ordered) - 1) * percentile
        lower = int(index)
        upper = min(len(ordered) - 1, lower + 1)
        fraction = index - lower
        return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction

    def _transport_metrics(self) -> dict[str, Any]:
        elapsed = (
            self._last_displayed_at - self._first_displayed_at
            if (
                self._last_displayed_at is not None
                and self._first_displayed_at is not None
            )
            else 0.0
        )
        return {
            "snapshot_buffer_size": len(self._snapshots),
            "snapshot_eviction_count": self._snapshot_eviction_count,
            "displayed_frame_count": self._displayed_frame_count,
            "displayed_frames_per_second": (
                (self._displayed_frame_count - 1) / elapsed
                if elapsed > 0 and self._displayed_frame_count > 1
                else 0.0
            ),
            "frame_evidence_sequence_mismatch_count": 0,
            "display_ack_stale_count": self._display_ack_stale_count,
            "dropped_display_frame_count": self._dropped_display_frame_count,
            "frame_age_mean_ms": (
                sum(self._frame_ages_ms) / len(self._frame_ages_ms)
                if self._frame_ages_ms
                else None
            ),
            "frame_age_p95_ms": self._percentile(
                self._frame_ages_ms,
                0.95,
            ),
            "maximum_preview_gap_ms": (
                max(self._preview_gaps_ms)
                if self._preview_gaps_ms
                else None
            ),
        }

    def wait_for_terminal_or_live(
        self, timeout_seconds: float
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            status = self.status()
            if status["state"] in {
                CameraState.LIVE.value,
                CameraState.NO_DEVICE.value,
                CameraState.PERMISSION_DENIED.value,
                CameraState.ERROR.value,
            }:
                return status
            time.sleep(0.05)
        return self.status()

    def close(self) -> None:
        self.stop()
