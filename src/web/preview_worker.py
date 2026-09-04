"""Spawn-safe real Body Pose preview worker for uploaded MP4 files."""

from __future__ import annotations

import os
import traceback
from typing import Any


def run_preview_worker(request: dict[str, Any], channel: Any) -> None:
    try:
        from src.legacy_pose.manual_selection import detect_preview

        result = detect_preview(
            request["video_path"],
            request["model_path"],
            float(request.get("timestamp", 0.0)),
            float(request.get("person_confidence", 0.25)),
            float(request.get("keypoint_confidence", 0.25)),
            float(request.get("nms_iou", 0.45)),
        )
        channel.put(
            {
                "kind": "complete",
                "result": result,
                "worker_pid": os.getpid(),
            },
            timeout=10.0,
        )
    except BaseException as exc:
        channel.put(
            {
                "kind": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback_tail": traceback.format_exc().splitlines()[-12:],
                "worker_pid": os.getpid(),
            },
            timeout=10.0,
        )
