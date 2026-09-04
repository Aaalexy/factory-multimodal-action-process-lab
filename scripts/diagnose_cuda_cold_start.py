"""Measure one fresh-process CUDA Body Pose cold start with explicit EP options."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np
import onnxruntime as ort


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.legacy_pose.pose_detector import _preload_cuda_runtime  # noqa: E402
from src.legacy_pose.pose_postprocess import letterbox  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", default="models/yolov8n-pose.onnx")
    parser.add_argument(
        "--algorithm-search",
        choices=("EXHAUSTIVE", "HEURISTIC", "DEFAULT"),
        required=True,
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    video = Path(args.video).expanduser().resolve()
    model = (PROJECT_ROOT / args.model).resolve()
    output = Path(args.output).expanduser().resolve()

    total_started = perf_counter()
    preload_started = perf_counter()
    _preload_cuda_runtime()
    preload_ms = (perf_counter() - preload_started) * 1000.0

    decode_started = perf_counter()
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open diagnostic video: {video}")
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Unable to decode diagnostic frame: {video}")
    decode_ms = (perf_counter() - decode_started) * 1000.0

    session_started = perf_counter()
    providers = [
        (
            "CUDAExecutionProvider",
            {
                "cudnn_conv_algo_search": args.algorithm_search,
                "do_copy_in_default_stream": "1",
            },
        ),
        "CPUExecutionProvider",
    ]
    session = ort.InferenceSession(str(model), providers=providers)
    session_create_ms = (perf_counter() - session_started) * 1000.0
    if not session.get_providers() or session.get_providers()[0] != (
        "CUDAExecutionProvider"
    ):
        raise RuntimeError(
            "Cold-start diagnostic requires an active CUDAExecutionProvider"
        )
    input_meta = session.get_inputs()[0]

    preprocess_started = perf_counter()
    padded, _ = letterbox(frame, (640, 640))
    tensor = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(
        tensor.transpose(2, 0, 1)[None],
        dtype=np.float32,
    )
    tensor /= np.float32(255.0)
    preprocess_ms = (perf_counter() - preprocess_started) * 1000.0

    runs_ms: list[float] = []
    output_shapes: list[list[int]] = []
    for _ in range(3):
        run_started = perf_counter()
        results = session.run(None, {input_meta.name: tensor})
        runs_ms.append((perf_counter() - run_started) * 1000.0)
        output_shapes = [list(item.shape) for item in results]

    payload = {
        "schema_version": "factory_cuda_cold_start_diagnostic_v1",
        "algorithm_search": args.algorithm_search,
        "video": str(video),
        "model": str(model),
        "available_providers": ort.get_available_providers(),
        "session_providers": session.get_providers(),
        "preload_ms": preload_ms,
        "decode_ms": decode_ms,
        "session_create_ms": session_create_ms,
        "preprocess_ms": preprocess_ms,
        "session_run_ms": runs_ms,
        "first_session_run_ms": runs_ms[0],
        "steady_session_run_mean_ms": float(np.mean(runs_ms[1:])),
        "output_shapes": output_shapes,
        "total_ms": (perf_counter() - total_started) * 1000.0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
