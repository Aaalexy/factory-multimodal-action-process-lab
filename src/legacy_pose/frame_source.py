"""Unified causal frame-source contracts for offline and future live inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class FramePacket:
    """One causally available frame with source and presentation timing."""

    frame_index: int
    source_frame_index: int
    timestamp: float
    image: np.ndarray


@runtime_checkable
class FrameSource(Protocol):
    """Structural input contract shared by video files and future cameras."""

    metadata: object

    def iter_frames(
        self,
        start_time: float = 0.0,
        end_time: float | None = None,
        output_fps: float | None = None,
    ) -> Iterator[FramePacket]: ...


class USBCameraSource(FrameSource, Protocol):
    """Structural USB-source contract implemented by the Camera Technical RC."""

    device_index: int
    is_live: bool


@runtime_checkable
class LiveFrameSource(Protocol):
    """Explicit lifecycle contract shared by USB, RTSP and paced replay."""

    is_live: bool
    metadata: object

    def open(self) -> None: ...
    def read_packet(self) -> FramePacket | None: ...
    def close(self) -> None: ...
