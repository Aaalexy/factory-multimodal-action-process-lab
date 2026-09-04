"""Deterministic video metadata inspection and time-based frame sampling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .frame_source import FramePacket


@dataclass(frozen=True)
class VideoMetadata:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float


VideoFrame = FramePacket


class VideoFileSource:
    """Read a video without relying on locale-sensitive path conversions."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"Video does not exist: {self.path}")
        capture = cv2.VideoCapture(str(self.path))
        try:
            if not capture.isOpened():
                raise ValueError(f"OpenCV cannot decode video: {self.path}")
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0:
                raise ValueError(
                    "Invalid video metadata: "
                    f"size={width}x{height}, fps={fps}, frames={frame_count}"
                )
            self.metadata = VideoMetadata(
                path=self.path,
                width=width,
                height=height,
                fps=fps,
                frame_count=frame_count,
                duration_seconds=frame_count / fps,
            )
        finally:
            capture.release()

    def iter_frames(
        self,
        start_time: float = 0.0,
        end_time: float | None = None,
        output_fps: float | None = None,
    ) -> Iterator[VideoFrame]:
        """Yield nearest source frames at a stable requested cadence.

        ``frame_index`` is zero-based within the sampled clip, while
        ``source_frame_index`` identifies the decoded source frame.
        """

        start = max(0.0, float(start_time))
        end = self.metadata.duration_seconds if end_time is None else float(end_time)
        end = min(end, self.metadata.duration_seconds)
        if end <= start:
            raise ValueError(f"end_time ({end}) must be greater than start_time ({start})")
        target_fps = self.metadata.fps if output_fps is None else float(output_fps)
        if target_fps <= 0:
            raise ValueError("output_fps must be positive")
        target_fps = min(target_fps, self.metadata.fps)

        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            raise ValueError(f"OpenCV cannot decode video: {self.path}")
        first_source_index = int(np.floor(start * self.metadata.fps))
        capture.set(cv2.CAP_PROP_POS_FRAMES, first_source_index)
        next_sample_time = start
        sampled_index = 0
        source_index = first_source_index
        tolerance = 0.5 / self.metadata.fps
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp = source_index / self.metadata.fps
                if timestamp + tolerance >= end:
                    break
                if timestamp + tolerance >= next_sample_time:
                    yield VideoFrame(
                        frame_index=sampled_index,
                        source_frame_index=source_index,
                        timestamp=timestamp,
                        image=frame,
                    )
                    sampled_index += 1
                    next_sample_time = start + sampled_index / target_fps
                source_index += 1
        finally:
            capture.release()


# Backward-compatible name used by existing callers and tests.
VideoReader = VideoFileSource
