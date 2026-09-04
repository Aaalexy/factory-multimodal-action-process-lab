"""CPU/GPU same-frame Body Pose benchmark with evidence-preserving comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pose_core import PoseRuntime
from src.provenance import sha256_file


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
    }


def _read_frame(path: Path, timestamp_seconds: float) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000.0)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Unable to decode benchmark frame: {path}")
    return frame


def _run(
    model: Path,
    frame: np.ndarray,
    *,
    policy: str,
    warmup: int,
    iterations: int,
) -> tuple[dict[str, Any], list[Any]]:
    runtime = PoseRuntime(model, provider_policy=policy)
    for _ in range(warmup):
        runtime.detect(frame)
    detector = runtime.detector
    detector.preprocess_times_ms.clear()
    detector.inference_times_ms.clear()
    detector.postprocess_times_ms.clear()
    detector.detect_times_ms.clear()
    detections: list[Any] = []
    started = perf_counter()
    for _ in range(iterations):
        detections = runtime.detect(frame)
    elapsed = perf_counter() - started
    return (
        {
            "policy": policy,
            "provider_status": runtime.provider_status,
            "input_shape": list(detector.input_shape),
            "warmup_iterations": warmup,
            "iterations": iterations,
            "detection_count": len(detections),
            "preprocess_ms": _summary(detector.preprocess_times_ms),
            "session_run_ms": _summary(detector.inference_times_ms),
            "postprocess_ms": _summary(detector.postprocess_times_ms),
            "total_detect_ms": _summary(detector.detect_times_ms),
            "measured_frames_per_second": iterations / max(elapsed, 1e-9),
        },
        detections,
    )


def _numeric_comparison(
    cpu: list[Any],
    gpu: list[Any],
) -> dict[str, Any]:
    paired = min(len(cpu), len(gpu))
    bbox_delta = 0.0
    confidence_delta = 0.0
    keypoint_xy_delta = 0.0
    keypoint_confidence_delta = 0.0
    for index in range(paired):
        left = cpu[index]
        right = gpu[index]
        bbox_delta = max(
            bbox_delta,
            float(np.max(np.abs(left.bbox - right.bbox))),
        )
        confidence_delta = max(
            confidence_delta,
            abs(float(left.confidence) - float(right.confidence)),
        )
        keypoint_xy_delta = max(
            keypoint_xy_delta,
            float(
                np.nanmax(
                    np.abs(left.keypoints[:, :2] - right.keypoints[:, :2])
                )
            ),
        )
        keypoint_confidence_delta = max(
            keypoint_confidence_delta,
            float(
                np.nanmax(
                    np.abs(left.keypoints[:, 2] - right.keypoints[:, 2])
                )
            ),
        )
    return {
        "detection_count_equal": len(cpu) == len(gpu),
        "paired_detection_count": paired,
        "maximum_bbox_absolute_delta_px": bbox_delta,
        "maximum_person_confidence_absolute_delta": confidence_delta,
        "maximum_keypoint_xy_absolute_delta_px": keypoint_xy_delta,
        "maximum_keypoint_confidence_absolute_delta": (
            keypoint_confidence_delta
        ),
        "tolerance": {
            "bbox_px": 0.25,
            "keypoint_xy_px": 0.25,
            "confidence": 0.002,
        },
        "within_tolerance": bool(
            len(cpu) == len(gpu)
            and bbox_delta <= 0.25
            and keypoint_xy_delta <= 0.25
            and confidence_delta <= 0.002
            and keypoint_confidence_delta <= 0.002
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", default="models/yolov8n-pose.onnx")
    parser.add_argument("--timestamp", type=float, default=3.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    video = Path(args.video).resolve()
    model = Path(args.model).resolve()
    output = Path(args.output).resolve()
    frame = _read_frame(video, args.timestamp)
    cpu_metrics, cpu_detections = _run(
        model,
        frame,
        policy="cpu",
        warmup=args.warmup,
        iterations=args.iterations,
    )
    gpu_metrics, gpu_detections = _run(
        model,
        frame,
        policy="require_cuda",
        warmup=args.warmup,
        iterations=args.iterations,
    )
    payload = {
        "schema_version": "factory_gpu_cpu_pose_ab_v1",
        "video": str(video),
        "timestamp_seconds": args.timestamp,
        "frame_shape": list(frame.shape),
        "frame_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
        "model": str(model),
        "model_sha256": sha256_file(model),
        "onnxruntime_version": ort.__version__,
        "available_providers": ort.get_available_providers(),
        "cpu": cpu_metrics,
        "gpu": gpu_metrics,
        "numeric_comparison": _numeric_comparison(
            cpu_detections,
            gpu_detections,
        ),
        "acceptance": {
            "same_model_and_input": True,
            "gpu_session_active": (
                gpu_metrics["provider_status"]["active_provider"]
                == "CUDAExecutionProvider"
            ),
            "gpu_session_run_p95_at_most_100_ms": (
                gpu_metrics["session_run_ms"]["p95"] <= 100.0
            ),
            "gpu_mean_faster_than_cpu": (
                gpu_metrics["session_run_ms"]["mean"]
                < cpu_metrics["session_run_ms"]["mean"]
            ),
            "outputs_within_tolerance": False,
        },
    }
    payload["acceptance"]["outputs_within_tolerance"] = payload[
        "numeric_comparison"
    ]["within_tolerance"]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps(payload["acceptance"], ensure_ascii=False))


if __name__ == "__main__":
    main()
