"""OpenCV USB frame source with monotonic timestamps and explicit release."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import cv2

from src.legacy_pose.frame_source import FramePacket


class CameraSourceError(RuntimeError):
    pass


class CameraOpenError(CameraSourceError):
    pass


class CameraReadError(CameraSourceError):
    pass


@dataclass(frozen=True)
class USBSourceMetadata:
    device_index: int
    width: int
    height: int
    fps: float
    backend: str
    mirror_horizontal: bool
    evidence_class: str = "LOCAL_USB_TECHNICAL_VALIDATION_ONLY"


class OpenCVUSBFrameSource:
    """Local-only USB source. It does not record, network or create evidence."""

    is_live = True

    def __init__(
        self,
        device_index: int,
        *,
        backend: str = "auto",
        requested_width: int = 640,
        requested_height: int = 480,
        requested_fps: float = 8.0,
        open_timeout_seconds: float = 4.0,
        read_timeout_seconds: float = 2.0,
        mirror_horizontal: bool = False,
        capture_factory: Callable[[], Any] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.device_index = int(device_index)
        self.backend_name = str(backend)
        self.requested_width = int(requested_width)
        self.requested_height = int(requested_height)
        self.requested_fps = float(requested_fps)
        self.open_timeout_seconds = float(open_timeout_seconds)
        self.read_timeout_seconds = float(read_timeout_seconds)
        self.mirror_horizontal = bool(mirror_horizontal)
        self._capture_factory = capture_factory or cv2.VideoCapture
        self._clock = monotonic
        self._capture: Any | None = None
        self._frame_index = 0
        self._opened_at = 0.0
        self.metadata = USBSourceMetadata(
            self.device_index, 0, 0, self.requested_fps,
            self.backend_name, self.mirror_horizontal,
        )

    def _backend(self) -> int:
        return {
            "auto": cv2.CAP_ANY,
            "dshow": getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),
            "msmf": getattr(cv2, "CAP_MSMF", cv2.CAP_ANY),
        }[self.backend_name]

    def open(self) -> None:
        if self._capture is not None:
            return
        capture = self._capture_factory()
        for prop, value in (
            (
                getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None),
                self.open_timeout_seconds * 1000.0,
            ),
            (
                getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None),
                self.read_timeout_seconds * 1000.0,
            ),
        ):
            if prop is not None:
                try:
                    capture.set(prop, value)
                except (AttributeError, cv2.error):
                    pass
        backend = self._backend()
        try:
            opened = bool(capture.open(self.device_index, backend))
        except TypeError:
            opened = bool(capture.open(self.device_index))
        if not opened or not capture.isOpened():
            capture.release()
            raise CameraOpenError(
                f"Unable to open local USB Camera index {self.device_index}"
            )
        for prop, value in (
            (cv2.CAP_PROP_FRAME_WIDTH, self.requested_width),
            (cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height),
            (cv2.CAP_PROP_FPS, self.requested_fps),
        ):
            if value > 0:
                try:
                    capture.set(prop, value)
                except (AttributeError, cv2.error):
                    pass
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or self.requested_fps)
        try:
            actual_backend = str(capture.getBackendName())
        except (AttributeError, cv2.error):
            actual_backend = self.backend_name
        self._capture = capture
        self._opened_at = self._clock()
        self._frame_index = 0
        self.metadata = USBSourceMetadata(
            self.device_index,
            width,
            height,
            fps if fps > 0 else self.requested_fps,
            actual_backend,
            self.mirror_horizontal,
        )

    def read_packet(self) -> FramePacket | None:
        if self._capture is None:
            raise CameraReadError("USB Camera source is not open")
        started = self._clock()
        ok, frame = self._capture.read()
        finished = self._clock()
        if finished - started > self.read_timeout_seconds:
            raise CameraReadError("USB Camera read exceeded the configured timeout")
        if not ok or frame is None:
            return None
        if self.mirror_horizontal:
            frame = cv2.flip(frame, 1)
        packet = FramePacket(
            frame_index=self._frame_index,
            source_frame_index=self._frame_index,
            timestamp=max(0.0, finished - self._opened_at),
            image=frame,
        )
        self._frame_index += 1
        return packet

    def close(self) -> None:
        capture, self._capture = self._capture, None
        if capture is not None:
            capture.release()

    def __enter__(self) -> "OpenCVUSBFrameSource":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
