"""FP32 ONNX Runtime detector for YOLOv8n-Pose."""

from __future__ import annotations

import ctypes
from pathlib import Path
import sys
from time import perf_counter
from typing import Sequence

import cv2
import numpy as np
import onnxruntime as ort

from .pose_postprocess import PoseDetection, decode_pose_output, letterbox
from src.pose_provider_policy import (
    BodyProviderStatus,
    BodyProviderUnavailableError,
    build_body_provider_status,
    normalize_body_provider_policy,
    select_body_provider_request,
)


_NVIDIA_DLL_HANDLES: list[object] = []


def _preload_cuda_runtime() -> None:
    """Keep pip-installed cuDNN 9 split libraries loaded on Windows.

    ONNX Runtime's preload helper loads the main CUDA/cuDNN libraries, but
    current cuDNN 9 wheels also contain delayed split libraries. Without
    retaining explicit handles for those DLLs, the first convolution can
    fail and ONNX Runtime silently falls back to the CPU provider.
    """

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls(directory="")
    if sys.platform != "win32" or _NVIDIA_DLL_HANDLES:
        return

    cudnn_directory = (
        Path(sys.prefix)
        / "Lib"
        / "site-packages"
        / "nvidia"
        / "cudnn"
        / "bin"
    )
    if not cudnn_directory.is_dir():
        return

    for dll_path in cudnn_directory.glob("*.dll"):
        try:
            _NVIDIA_DLL_HANDLES.append(ctypes.WinDLL(str(dll_path.resolve())))
        except OSError as exc:
            raise RuntimeError(f"Unable to preload cuDNN library: {dll_path}") from exc


class PoseDetector:
    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.25,
        keypoint_threshold: float = 0.25,
        nms_iou_threshold: float = 0.45,
        providers: Sequence[str] | None = None,
        provider_policy: str = "prefer_cuda",
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Pose model does not exist: {self.model_path}")
        available = ort.get_available_providers()
        normalized_policy = normalize_body_provider_policy(provider_policy)
        fallback_reason: str | None = None
        if providers:
            selected_providers = list(providers)
            normalized_policy = (
                "cpu"
                if selected_providers == ["CPUExecutionProvider"]
                else normalized_policy
            )
        else:
            selected_providers, fallback_reason = select_body_provider_request(
                normalized_policy,
                available_providers=available,
            )
        requested_providers = list(selected_providers)
        if "CUDAExecutionProvider" in selected_providers:
            _preload_cuda_runtime()
        try:
            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=selected_providers,
            )
        except Exception as exc:
            if (
                normalized_policy not in {"auto", "prefer_cuda"}
                or "CUDAExecutionProvider" not in selected_providers
                or "CPUExecutionProvider" not in available
            ):
                raise
            fallback_reason = (
                f"cuda_session_initialization_failed:{type(exc).__name__}"
            )
            selected_providers = ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=selected_providers,
            )
        self.provider_status: BodyProviderStatus = build_body_provider_status(
            policy=normalized_policy,
            requested_providers=requested_providers,
            session_providers=self.session.get_providers(),
            fallback_reason=fallback_reason,
            available_providers=available,
        )
        if (
            normalized_policy == "require_cuda"
            and self.provider_status.active_provider != "CUDAExecutionProvider"
        ):
            raise BodyProviderUnavailableError(
                "CUDAExecutionProvider is required but the session is not using it"
            )
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise ValueError(f"Expected one image input, model exposes {len(inputs)}")
        self.input = inputs[0]
        if self.input.type != "tensor(float)":
            raise TypeError(
                "V1.0 requires a real FP32 ONNX input; "
                f"model input {self.input.name!r} is {self.input.type!r}"
            )
        shape = self.input.shape
        if len(shape) != 4 or shape[1] not in (3, "3"):
            raise ValueError(f"Expected NCHW RGB model input, got {shape}")
        if not isinstance(shape[2], int) or not isinstance(shape[3], int):
            raise ValueError("Dynamic spatial ONNX inputs are not supported in V1.0")
        self.input_shape = (int(shape[2]), int(shape[3]))
        self.confidence_threshold = float(confidence_threshold)
        self.keypoint_threshold = float(keypoint_threshold)
        self.nms_iou_threshold = float(nms_iou_threshold)
        self.last_inference_ms = 0.0
        self.inference_call_count = 0
        self.preprocess_times_ms: list[float] = []
        self.inference_times_ms: list[float] = []
        self.postprocess_times_ms: list[float] = []
        self.detect_times_ms: list[float] = []
        self.warmup_times_ms: list[float] = []
        self.warmup_call_count = 0

    @property
    def providers(self) -> list[str]:
        return self.session.get_providers()

    @property
    def active_provider(self) -> str | None:
        return self.provider_status.active_provider

    def detect(self, frame: np.ndarray) -> list[PoseDetection]:
        detect_started = perf_counter()
        preprocess_started = perf_counter()
        padded, transform = letterbox(frame, self.input_shape)
        tensor = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[None], dtype=np.float32)
        tensor /= np.float32(255.0)
        if tensor.dtype != np.float32:
            raise AssertionError("ONNX input preprocessing must remain FP32")
        self.preprocess_times_ms.append(
            (perf_counter() - preprocess_started) * 1000.0
        )
        started = perf_counter()
        outputs = self.session.run(None, {self.input.name: tensor})
        self.last_inference_ms = (perf_counter() - started) * 1000.0
        self.inference_call_count += 1
        self.inference_times_ms.append(self.last_inference_ms)
        if not outputs:
            raise RuntimeError("ONNX Runtime returned no outputs")
        postprocess_started = perf_counter()
        detections = decode_pose_output(
            outputs[0],
            transform,
            confidence_threshold=self.confidence_threshold,
            keypoint_threshold=self.keypoint_threshold,
            nms_iou_threshold=self.nms_iou_threshold,
        )
        self.postprocess_times_ms.append(
            (perf_counter() - postprocess_started) * 1000.0
        )
        self.detect_times_ms.append((perf_counter() - detect_started) * 1000.0)
        return detections

    def warmup(self, frame: np.ndarray, *, iterations: int = 1) -> list[float]:
        """Warm the selected session on a real frame without publishing evidence."""

        if iterations < 1:
            raise ValueError("warmup iterations must be positive")
        if self.inference_call_count:
            raise RuntimeError("Pose warmup must run before evidence inference")
        elapsed: list[float] = []
        for _ in range(iterations):
            started = perf_counter()
            self.detect(frame)
            elapsed.append((perf_counter() - started) * 1000.0)
        self.warmup_times_ms.extend(elapsed)
        self.warmup_call_count += iterations
        self.last_inference_ms = 0.0
        self.inference_call_count = 0
        self.preprocess_times_ms.clear()
        self.inference_times_ms.clear()
        self.postprocess_times_ms.clear()
        self.detect_times_ms.clear()
        return elapsed
