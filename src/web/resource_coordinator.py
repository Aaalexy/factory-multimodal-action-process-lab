"""Single-owner guard shared by future Camera and current Video Analysis modes."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass


class ResourceBusyError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResourceLease:
    lease_id: str
    mode: str


class AnalysisResourceCoordinator:
    """Prevent camera and offline inference from owning the model concurrently."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: ResourceLease | None = None

    def acquire(self, mode: str) -> ResourceLease:
        if mode not in {"camera", "video_analysis"}:
            raise ValueError("mode must be camera or video_analysis")
        with self._lock:
            if self._active is not None:
                raise ResourceBusyError(
                    f"{self._active.mode} already owns the analysis resource"
                )
            self._active = ResourceLease(uuid.uuid4().hex, mode)
            return self._active

    def release(self, lease: ResourceLease) -> None:
        with self._lock:
            if self._active is None:
                return
            if self._active.lease_id != lease.lease_id:
                raise ResourceBusyError("Cannot release another owner's lease")
            self._active = None

    @property
    def active_mode(self) -> str | None:
        with self._lock:
            return self._active.mode if self._active is not None else None
