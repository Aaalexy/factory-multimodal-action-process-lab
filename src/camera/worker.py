"""Module-level Windows-spawn Camera worker entry."""

from __future__ import annotations

import queue
import time
import traceback
from multiprocessing.synchronize import Event
from pathlib import Path
from typing import Any

import cv2

from .contracts import CameraConfig
from .frame_source import CameraOpenError, OpenCVUSBFrameSource
from .live_analysis import LiveFrameAnalyzer


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = index - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


class MonotonicCadenceGate:
    """Select the newest available frame without resetting cadence phase."""

    def __init__(self, frames_per_second: float) -> None:
        if frames_per_second <= 0:
            raise ValueError("frames_per_second must be positive")
        self.interval_seconds = 1.0 / float(frames_per_second)
        self.next_due: float | None = None

    def should_run(self, timestamp: float) -> bool:
        current = float(timestamp)
        if self.next_due is None:
            self.next_due = current + self.interval_seconds
            return True
        if current + 1e-9 < self.next_due:
            return False
        while self.next_due <= current + 1e-9:
            self.next_due += self.interval_seconds
        return True


def _put_latest(target: Any, item: dict[str, Any]) -> int:
    dropped = 0
    deadline = time.monotonic() + 0.05
    while time.monotonic() < deadline:
        try:
            target.put_nowait(item)
            return dropped
        except queue.Full:
            try:
                target.get_nowait()
                dropped += 1
            except queue.Empty:
                time.sleep(0.001)
    # A stalled consumer must never keep a Camera device open. Dropping the
    # latest transport item is safer than blocking lifecycle finalization.
    return dropped + 1


def run_camera_worker(
    project_root: str,
    config_payload: dict[str, Any],
    session_id: str,
    stop_event: Event,
    output_queue: Any,
    command_queue: Any,
) -> None:
    """Open one USB source, emit latest-only real evidence, always release."""

    config = CameraConfig(**config_payload)
    source = OpenCVUSBFrameSource(
        config.device_index,
        backend=config.backend,
        requested_width=config.requested_width,
        requested_height=config.requested_height,
        requested_fps=config.requested_fps,
        open_timeout_seconds=config.open_timeout_seconds,
        read_timeout_seconds=config.read_timeout_seconds,
        mirror_horizontal=config.mirror_horizontal,
    )
    analyzer: LiveFrameAnalyzer | None = None
    emitted = 0
    captured = 0
    dropped_capture = 0
    dropped_result = 0
    started = time.monotonic()
    try:
        source.open()
        analyzer = LiveFrameAnalyzer(
            Path(project_root),
            body_model_path=config.body_model_path,
            hand_model_path=config.hand_model_path,
            hand_enabled=config.hand_enabled,
            session_id=session_id,
            analysis_fps=config.analysis_fps,
            body_provider_policy=config.body_provider_policy,
            mirror_horizontal=config.mirror_horizontal,
            candidate_history_size=(
                config.candidate_sequence_tolerance + 8
            ),
        )
        _put_latest(
            output_queue,
            {
                "kind": "state",
                "state": "opening",
                "session_id": session_id,
                "metadata": source.metadata.__dict__,
            },
        )
        pending_packet = source.read_packet()
        if pending_packet is None:
            raise CameraOpenError("USB Camera returned no warmup frame")
        captured += 1
        pending_captured_at = time.monotonic()
        pose_warmup_times_ms = analyzer.pose.warmup(pending_packet.image)
        live_started = time.monotonic()
        pose_gate = MonotonicCadenceGate(config.pose_display_fps)
        while not stop_event.is_set():
            try:
                command = command_queue.get_nowait()
            except queue.Empty:
                command = None
            if (
                isinstance(command, dict)
                and command.get("command") == "confirm_relock"
                and analyzer is not None
            ):
                try:
                    relock_event = analyzer.confirm_relock(
                        dict(command.get("selection") or {})
                    )
                    _put_latest(
                        output_queue,
                        {
                            "kind": "manual_relock",
                            "event": relock_event,
                            "session_id": session_id,
                        },
                    )
                except (RuntimeError, ValueError) as exc:
                    _put_latest(
                        output_queue,
                        {
                            "kind": "notice",
                            "message": str(exc),
                            "session_id": session_id,
                        },
                    )
            packet = pending_packet
            pending_packet = None
            if packet is None:
                packet = source.read_packet()
                if packet is not None:
                    captured += 1
                    packet_captured_at = time.monotonic()
            else:
                packet_captured_at = pending_captured_at
            if packet is None:
                raise CameraOpenError(
                    "USB Camera stopped returning frames"
                )
            if not pose_gate.should_run(packet.timestamp):
                dropped_capture += 1
                continue
            evidence = analyzer.analyze(
                packet.image,
                frame_index=packet.frame_index,
                timestamp=packet.timestamp,
            )
            ok, encoded = cv2.imencode(
                ".jpg",
                packet.image,
                [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality],
            )
            if not ok:
                raise RuntimeError("Unable to encode current USB frame")
            emitted += 1
            elapsed_seconds = max(1e-9, time.monotonic() - started)
            live_elapsed_seconds = max(
                1e-9,
                time.monotonic() - live_started,
            )
            pose_times = analyzer.pose.detector.inference_times_ms
            hand_times = analyzer.hand.inference_times_ms
            dropped_result += _put_latest(
                output_queue,
                {
                    "kind": "frame",
                    "state": "live",
                    "session_id": session_id,
                    "sequence": emitted,
                    "jpeg": encoded.tobytes(),
                    "width": int(packet.image.shape[1]),
                    "height": int(packet.image.shape[0]),
                    "timestamp": float(packet.timestamp),
                    "captured_at_monotonic": packet_captured_at,
                    "evidence": evidence,
                    "metrics": {
                        "captured_frame_count": captured,
                        "capture_frames_per_second": (
                            captured / live_elapsed_seconds
                        ),
                        "processed_frame_count": emitted,
                        "processed_frames_per_second": (
                            emitted / live_elapsed_seconds
                        ),
                        "pose_display_target_fps": config.pose_display_fps,
                        "action_analysis_target_fps": config.analysis_fps,
                        "action_sample_count": analyzer.action_sample_count,
                        "action_samples_per_second": (
                            analyzer.action_sample_count
                            / live_elapsed_seconds
                        ),
                        "dropped_capture_frame_count": dropped_capture,
                        "dropped_result_frame_count": dropped_result,
                        "dropped_frame_count": (
                            dropped_capture + dropped_result
                        ),
                        "pose_warmup_call_count": (
                            analyzer.pose.detector.warmup_call_count
                        ),
                        "pose_warmup_times_ms": pose_warmup_times_ms,
                        "elapsed_seconds": elapsed_seconds,
                        "live_elapsed_seconds": live_elapsed_seconds,
                        "mean_pose_inference_ms": (
                            sum(pose_times) / max(1, len(pose_times))
                        ),
                        "p50_pose_inference_ms": _percentile(
                            pose_times,
                            0.50,
                        ),
                        "p95_pose_inference_ms": _percentile(
                            pose_times,
                            0.95,
                        ),
                        "mean_hand_inference_ms": (
                            sum(hand_times) / max(1, len(hand_times))
                            if hand_times
                            else 0.0
                        ),
                        "p50_hand_inference_ms": _percentile(
                            hand_times,
                            0.50,
                        ),
                        "p95_hand_inference_ms": _percentile(
                            hand_times,
                            0.95,
                        ),
                        "body_pose_providers": analyzer.pose.providers,
                        "body_pose_provider_status": (
                            analyzer.pose.provider_status
                        ),
                        "hand_pose_provider": "CPU",
                    },
                },
            )
    except CameraOpenError as exc:
        _put_latest(
            output_queue,
            {
                "kind": "error",
                "state": "no_device",
                "error_code": "usb_open_or_read_failed",
                "message": str(exc),
                "session_id": session_id,
            },
        )
    except PermissionError as exc:
        _put_latest(
            output_queue,
            {
                "kind": "error",
                "state": "permission_denied",
                "error_code": "usb_permission_denied",
                "message": str(exc),
                "session_id": session_id,
            },
        )
    except Exception as exc:
        _put_latest(
            output_queue,
            {
                "kind": "error",
                "state": "error",
                "error_code": f"camera_worker_{type(exc).__name__}",
                "message": str(exc),
                "diagnostic": traceback.format_exc(limit=4),
                "session_id": session_id,
            },
        )
    finally:
        if analyzer is not None:
            analyzer.close()
        source.close()
        _put_latest(
            output_queue,
            {
                "kind": "state",
                "state": "stopped",
                "session_id": session_id,
                "metrics": {
                    "captured_frame_count": captured,
                    "processed_frame_count": emitted,
                    "dropped_capture_frame_count": dropped_capture,
                    "dropped_result_frame_count": dropped_result,
                    "dropped_frame_count": (
                        dropped_capture + dropped_result
                    ),
                    "elapsed_seconds": time.monotonic() - started,
                    "pose_display_target_fps": config.pose_display_fps,
                    "action_analysis_target_fps": config.analysis_fps,
                    "action_sample_count": (
                        analyzer.action_sample_count
                        if analyzer is not None
                        else 0
                    ),
                    "device_released": True,
                },
            },
        )
        try:
            output_queue.cancel_join_thread()
        except AttributeError:
            pass
