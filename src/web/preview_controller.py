"""Disposable Windows-spawn controller for real uploaded-video previews."""

from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import time
from typing import Any, Callable

from .preview_worker import run_preview_worker


class PreviewController:
    def __init__(
        self,
        *,
        worker_target: Callable[[dict[str, Any], Any], None] = run_preview_worker,
        timeout_seconds: float = 90.0,
        context: Any | None = None,
    ) -> None:
        self.context = context or mp.get_context("spawn")
        self.worker_target = worker_target
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self._guard = threading.Lock()
        self._process: Any | None = None
        self._channel: Any | None = None

    def run(self, request: dict[str, Any]) -> dict[str, Any]:
        if not self._guard.acquire(blocking=False):
            raise RuntimeError("a preview worker is already running")
        process = None
        channel = None
        try:
            channel = self.context.Queue(maxsize=2)
            process = self.context.Process(
                target=self.worker_target,
                args=(dict(request), channel),
                name="factory-video-preview",
                daemon=True,
            )
            self._process = process
            self._channel = channel
            process.start()
            deadline = time.monotonic() + self.timeout_seconds
            message: Any | None = None
            while time.monotonic() < deadline:
                try:
                    message = channel.get(timeout=0.2)
                    break
                except queue.Empty:
                    if process.is_alive():
                        continue
                    break
            if message is None:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=3.0)
                    raise TimeoutError("preview inference timed out")
                raise RuntimeError(
                    "preview worker exited without a terminal message"
                )
            process.join(timeout=3.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=3.0)
            if not isinstance(message, dict):
                raise RuntimeError("preview worker returned invalid data")
            if message.get("kind") == "failed":
                raise RuntimeError(str(message.get("error", "preview failed")))
            result = message.get("result")
            if message.get("kind") != "complete" or not isinstance(result, dict):
                raise RuntimeError("preview completion payload is invalid")
            return result
        finally:
            if process is not None and process.is_alive():
                process.terminate()
                process.join(timeout=3.0)
            if channel is not None:
                try:
                    channel.close()
                except (AttributeError, OSError, ValueError):
                    pass
            self._process = None
            self._channel = None
            self._guard.release()

    def close(self) -> None:
        process = self._process
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=3.0)
