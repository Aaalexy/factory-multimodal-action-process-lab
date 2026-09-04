"""Spawn-safe baseline processing entry point migrated from the Pose project."""

from __future__ import annotations

import os
import queue
import time
import traceback
from multiprocessing.synchronize import Event
from typing import Any

from src.multimodal_pipeline import BaselineConfig, run_baseline


def _publish(channel: Any, payload: dict[str, Any], *, terminal: bool = False) -> None:
    """Keep progress IPC bounded while never silently dropping terminal state."""

    payload["worker_pid"] = os.getpid()
    if terminal:
        channel.put(payload, timeout=10.0)
        return
    try:
        channel.put_nowait(payload)
    except queue.Full:
        # Per-frame progress is replaceable; the HTTP server remains responsive
        # and a later update will carry the current value.
        return


def run_web_worker(
    config_values: dict[str, Any],
    channel: Any,
    cancel_event: Event,
) -> None:
    """Construct ONNX Runtime inside the child and execute exactly one job."""

    last_progress_publish = 0.0
    try:
        config = BaselineConfig(**config_values)
        if cancel_event.is_set():
            raise InterruptedError("Baseline job cancelled before model loading")
        _publish(
            channel,
            {
                "kind": "progress",
                "progress": 0.05,
                "stage": "gpu_warming_up",
                "message": "loading real pose runtime",
            },
        )

        def progress(value: float, message: str, stage: str) -> None:
            nonlocal last_progress_publish
            now = time.monotonic()
            if value < 1.0 and now - last_progress_publish < 0.15:
                return
            last_progress_publish = now
            _publish(
                channel,
                {
                    "kind": "progress",
                    "progress": value,
                    "stage": stage,
                    "message": message,
                },
            )

        result = run_baseline(
            config,
            progress_callback=progress,
            cancel_check=cancel_event.is_set,
        )
        if cancel_event.is_set():
            raise InterruptedError("Baseline job cancelled after analysis")
        _publish(channel, {"kind": "complete", "result": result}, terminal=True)
    except InterruptedError as exc:
        _publish(
            channel,
            {"kind": "cancelled", "message": str(exc)},
            terminal=True,
        )
    except BaseException as exc:
        _publish(
            channel,
            {
                "kind": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback_tail": traceback.format_exc().splitlines()[-12:],
            },
            terminal=True,
        )
