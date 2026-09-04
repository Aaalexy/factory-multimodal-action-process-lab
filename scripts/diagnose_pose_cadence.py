"""Isolate CUDA Body Pose latency under fixed, variable and gapped inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pose_core import PoseRuntime
from src.video_io import iter_video_frames


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "maximum": float(np.max(values)),
    }


def _measure(
    frames: list[np.ndarray],
    *,
    sleep_seconds: float,
) -> dict[str, object]:
    runtime = PoseRuntime(
        PROJECT_ROOT / "models" / "yolov8n-pose.onnx",
        provider_policy="require_cuda",
    )
    for _ in range(5):
        runtime.detect(frames[0])
    runtime.detector.inference_times_ms.clear()
    for frame in frames:
        if sleep_seconds:
            time.sleep(sleep_seconds)
        runtime.detect(frame)
    return {
        "provider_status": runtime.provider_status,
        "sleep_seconds": sleep_seconds,
        "frame_count": len(frames),
        "session_run_ms": _summary(runtime.detector.inference_times_ms),
        "samples_ms": runtime.detector.inference_times_ms,
    }


def _measure_streaming(video: Path, frame_count: int) -> dict[str, object]:
    runtime = PoseRuntime(
        PROJECT_ROOT / "models" / "yolov8n-pose.onnx",
        provider_policy="require_cuda",
    )
    seen = 0
    for packet in iter_video_frames(
        video,
        start_time=0.0,
        end_time=max(1.0, frame_count / 8.0),
        output_fps=8.0,
    ):
        runtime.detect(packet.image)
        seen += 1
        if seen >= frame_count:
            break
    return {
        "provider_status": runtime.provider_status,
        "frame_count": seen,
        "session_run_ms": _summary(runtime.detector.inference_times_ms),
        "samples_ms": runtime.detector.inference_times_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    frames = [
        packet.image.copy()
        for packet in iter_video_frames(
            Path(args.video),
            start_time=0.0,
            end_time=max(1.0, args.frames / 8.0),
            output_fps=8.0,
        )
    ][: args.frames]
    if not frames:
        raise RuntimeError("No diagnostic frames decoded")
    payload = {
        "schema_version": "factory_pose_cadence_diagnostic_v1",
        "streaming_open_decoder": _measure_streaming(
            Path(args.video),
            args.frames,
        ),
        "fixed_continuous": _measure(
            [frames[0]] * len(frames),
            sleep_seconds=0.0,
        ),
        "variable_predecoded_continuous": _measure(
            frames,
            sleep_seconds=0.0,
        ),
        "variable_predecoded_125ms_gap": _measure(
            frames,
            sleep_seconds=0.125,
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output)
    print(
        json.dumps(
            {
                name: value["session_run_ms"]
                for name, value in payload.items()
                if isinstance(value, dict) and "session_run_ms" in value
            }
        )
    )


if __name__ == "__main__":
    main()
