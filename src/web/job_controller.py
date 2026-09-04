"""Bounded Windows-spawn controller for one uploaded-video analysis job."""

from __future__ import annotations

import json
import multiprocessing as mp
import queue
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.web.offline_worker import run_web_worker
from src.web.resource_coordinator import (
    AnalysisResourceCoordinator,
    ResourceBusyError,
    ResourceLease,
)


TERMINAL_STATES = {"completed", "cancelled", "failed"}
ACTIVE_STATES = {
    "queued",
    "gpu_warming_up",
    "analyzing",
    "writing",
    "cancelling",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


class OfflineJobController:
    def __init__(
        self,
        project_root: str | Path,
        coordinator: AnalysisResourceCoordinator,
        *,
        on_complete: Callable[[str], None] | None = None,
        worker_target: Callable[..., None] = run_web_worker,
        context: Any | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.output_root = (
            self.project_root / "outputs" / "analyses"
        ).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.coordinator = coordinator
        self.on_complete = on_complete
        self.worker_target = worker_target
        self.context = context or mp.get_context("spawn")
        self._lock = threading.RLock()
        self._process: Any | None = None
        self._channel: Any | None = None
        self._cancel_event: Any | None = None
        self._lease: ResourceLease | None = None
        self._job: dict[str, Any] = {
            "job_id": None,
            "state": "pending",
            "stage": "pending",
            "progress": 0.0,
            "message": "No uploaded-video analysis job has started.",
            "elapsed_seconds": 0.0,
            "worker_pid": None,
            "result": None,
            "public_error": None,
        }
        self._started_monotonic: float | None = None

    def start(self, config_values: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._drain_locked()
            if self._process is not None and self._process.is_alive():
                raise RuntimeError("an uploaded-video analysis job is active")
            try:
                self._lease = self.coordinator.acquire("video_analysis")
            except ResourceBusyError as exc:
                raise RuntimeError(
                    "Camera is live; stop Camera before starting Video Analysis."
                ) from exc
            job_id = "analysis_" + uuid.uuid4().hex
            output_dir = (self.output_root / job_id).resolve()
            if not output_dir.is_relative_to(self.output_root):
                self.coordinator.release(self._lease)
                self._lease = None
                raise ValueError("analysis output escapes the controlled root")
            values = dict(config_values)
            values["output_dir"] = output_dir.relative_to(
                self.project_root
            ).as_posix()
            self._channel = self.context.Queue(maxsize=16)
            self._cancel_event = self.context.Event()
            self._process = self.context.Process(
                target=self.worker_target,
                args=(values, self._channel, self._cancel_event),
                name=f"factory-offline-{job_id[-8:]}",
                daemon=True,
            )
            self._job = {
                "job_id": job_id,
                "state": "queued",
                "stage": "queued",
                "progress": 0.0,
                "message": "Analysis job queued.",
                "elapsed_seconds": 0.0,
                "worker_pid": None,
                "result": None,
                "public_error": None,
                "output_dir": output_dir.relative_to(
                    self.project_root
                ).as_posix(),
                "body_provider_policy": values.get(
                    "body_provider_policy", "prefer_cuda"
                ),
                "hand_enabled": bool(values.get("hand_enabled", True)),
                "recording_group_id": values.get(
                    "recording_group_id",
                    "recording_group_unassigned",
                ),
                "started_at": _utc_now(),
            }
            self._started_monotonic = time.monotonic()
            try:
                self._process.start()
            except BaseException:
                self._release_locked()
                self._process = None
                raise
            self._job["worker_pid"] = self._process.pid
            return self.status()

    def _write_diagnostic_locked(self, message: dict[str, Any]) -> None:
        output = self._job.get("output_dir")
        if not output:
            return
        path = self.project_root / str(output) / "worker_diagnostic.json"
        _atomic_json(
            path,
            {
                "schema_version": "factory_offline_worker_diagnostic_v1",
                "job_id": self._job.get("job_id"),
                "captured_at": _utc_now(),
                "error": message.get("error"),
                "traceback_tail": message.get("traceback_tail", []),
            },
        )

    def _release_locked(self) -> None:
        if self._lease is not None:
            self.coordinator.release(self._lease)
            self._lease = None

    def _drain_locked(self) -> None:
        if self._channel is not None:
            while True:
                try:
                    message = self._channel.get_nowait()
                except queue.Empty:
                    break
                if not isinstance(message, dict):
                    continue
                kind = str(message.get("kind", ""))
                if kind == "progress":
                    stage = str(message.get("stage", "analyzing"))
                    self._job.update(
                        state=stage,
                        stage=stage,
                        progress=max(
                            0.0,
                            min(1.0, float(message.get("progress", 0.0))),
                        ),
                        message=str(message.get("message", stage)),
                        worker_pid=message.get(
                            "worker_pid", self._job.get("worker_pid")
                        ),
                    )
                elif kind == "complete":
                    result = message.get("result")
                    self._job.update(
                        state="completed",
                        stage="completed",
                        progress=1.0,
                        message="Analysis completed.",
                        result=result,
                        public_error=None,
                    )
                    if isinstance(result, dict) and self.on_complete is not None:
                        analysis_path = result.get("analysis_path")
                        if analysis_path:
                            self.on_complete(str(analysis_path))
                    if self._process is not None:
                        self._process.join(timeout=0.25)
                        if not self._process.is_alive():
                            self._process = None
                            self._release_locked()
                elif kind == "cancelled":
                    self._job.update(
                        state="cancelled",
                        stage="cancelled",
                        message="Analysis cancelled; no partial result was loaded.",
                        public_error=None,
                    )
                elif kind == "failed":
                    self._write_diagnostic_locked(message)
                    self._job.update(
                        state="failed",
                        stage="failed",
                        message="Analysis failed. Inspect the local diagnostic record.",
                        public_error=str(
                            message.get("error", "analysis worker failed")
                        ).splitlines()[0][:300],
                    )
        if self._process is not None and not self._process.is_alive():
            self._process.join(timeout=0)
            if self._job.get("state") not in TERMINAL_STATES:
                self._job.update(
                    state="failed",
                    stage="failed",
                    message="Analysis worker exited unexpectedly.",
                    public_error="analysis worker exited unexpectedly",
                )
            self._process = None
            self._release_locked()

    def status(self) -> dict[str, Any]:
        with self._lock:
            self._drain_locked()
            if self._started_monotonic is not None:
                self._job["elapsed_seconds"] = round(
                    max(0.0, time.monotonic() - self._started_monotonic),
                    3,
                )
            return {
                **self._job,
                "worker_alive": bool(
                    self._process is not None and self._process.is_alive()
                ),
                "resource_owner": self.coordinator.active_mode,
            }

    def cancel(self) -> dict[str, Any]:
        with self._lock:
            self._drain_locked()
            if self._process is None or not self._process.is_alive():
                return self.status()
            self._job.update(
                state="cancelling",
                stage="cancelling",
                message="Cancellation requested; waiting for the frame loop.",
            )
            self._cancel_event.set()
            return self.status()

    def close(self) -> None:
        with self._lock:
            process = self._process
            if process is not None and process.is_alive():
                self._cancel_event.set()
        if process is not None and process.is_alive():
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=3.0)
        with self._lock:
            self._process = None
            self._release_locked()
            if self._channel is not None:
                try:
                    self._channel.close()
                except (AttributeError, OSError, ValueError):
                    pass
                self._channel = None
