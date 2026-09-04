"""Video I/O adapter around the migrated, deterministic frame source."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import cv2

from src.legacy_pose.frame_source import FramePacket
from src.legacy_pose.video_reader import VideoFileSource
from src.provenance import sha256_file


@dataclass(frozen=True)
class VideoProbe:
    path: str
    sha256: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    codec: str
    decodable: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _fourcc_text(value: int) -> str:
    chars = [chr((value >> (8 * index)) & 0xFF) for index in range(4)]
    return "".join(char for char in chars if char.isprintable()).strip() or "unknown"


def probe_video(path: str | Path) -> VideoProbe:
    resolved = Path(path).expanduser().resolve()
    reader = VideoFileSource(resolved)
    capture = cv2.VideoCapture(str(resolved))
    try:
        codec = _fourcc_text(int(capture.get(cv2.CAP_PROP_FOURCC)))
        ok, frame = capture.read()
        decodable = bool(ok and frame is not None and frame.size > 0)
    finally:
        capture.release()
    metadata = reader.metadata
    return VideoProbe(
        path=str(resolved),
        sha256=sha256_file(resolved),
        width=metadata.width,
        height=metadata.height,
        fps=metadata.fps,
        frame_count=metadata.frame_count,
        duration_seconds=metadata.duration_seconds,
        codec=codec,
        decodable=decodable,
    )


def iter_video_frames(
    path: str | Path,
    *,
    start_time: float = 0.0,
    end_time: float | None = None,
    output_fps: float | None = None,
) -> Iterator[FramePacket]:
    yield from VideoFileSource(path).iter_frames(
        start_time=start_time,
        end_time=end_time,
        output_fps=output_fps,
    )
