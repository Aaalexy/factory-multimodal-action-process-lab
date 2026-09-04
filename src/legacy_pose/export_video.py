"""Browser-compatible MP4 export using OpenCV frames and bundled FFmpeg."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _ffmpeg_executable() -> str:
    try:
        import imageio_ffmpeg
    except ImportError as exc:  # pragma: no cover - exercised by deployment failure
        raise RuntimeError(
            "Browser-compatible MP4 export requires imageio-ffmpeg. "
            "Install this project's requirements in the active virtual environment."
        ) from exc
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - package/platform-specific
        raise RuntimeError(f"Bundled FFmpeg is unavailable: {exc}") from exc


def _top_level_mp4_atoms(path: Path) -> list[tuple[str, int]]:
    atoms: list[tuple[str, int]] = []
    file_size = path.stat().st_size
    offset = 0
    with path.open("rb") as stream:
        while offset + 8 <= file_size:
            stream.seek(offset)
            header = stream.read(8)
            atom_size = int.from_bytes(header[:4], "big")
            atom_type = header[4:8].decode("latin-1")
            header_size = 8
            if atom_size == 1:
                extended = stream.read(8)
                if len(extended) != 8:
                    break
                atom_size = int.from_bytes(extended, "big")
                header_size = 16
            elif atom_size == 0:
                atom_size = file_size - offset
            if atom_size < header_size or offset + atom_size > file_size:
                break
            atoms.append((atom_type, offset))
            offset += atom_size
    return atoms


def probe_browser_compatible_mp4(path: str | Path) -> dict[str, Any]:
    """Verify H.264, yuv420p, and faststart without trusting the extension."""

    video_path = Path(path).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    completed = subprocess.run(
        [_ffmpeg_executable(), "-hide_banner", "-i", str(video_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    description = completed.stderr
    codec_match = re.search(r"Video:\s*([A-Za-z0-9_]+)", description)
    codec = codec_match.group(1).strip().lower() if codec_match else "unknown"
    is_h264 = codec in {"h264", "avc", "avc1"}
    is_yuv420p = bool(re.search(r"\byuv420p(?:\([^)]*\))?", description))
    atoms = _top_level_mp4_atoms(video_path)
    offsets = {name: offset for name, offset in atoms if name in {"moov", "mdat"}}
    faststart = "moov" in offsets and "mdat" in offsets and offsets["moov"] < offsets["mdat"]
    result = {
        "path": str(video_path),
        "codec": codec,
        "pixel_format": "yuv420p" if is_yuv420p else "unknown",
        "faststart": faststart,
        "moov_offset": offsets.get("moov"),
        "mdat_offset": offsets.get("mdat"),
        "browser_compatible": is_h264 and is_yuv420p and faststart,
    }
    if not result["browser_compatible"]:
        raise ValueError(f"MP4 is not H.264/yuv420p/faststart: {result}")
    return result


def transcode_to_browser_mp4(
    source: str | Path,
    destination: str | Path,
    *,
    preset: str = "medium",
    threads: int | None = None,
) -> dict[str, Any]:
    """Transcode an existing video and atomically publish a browser MP4."""

    source_path = Path(source).resolve()
    destination_path = Path(destination).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    browser_temp = tempfile.NamedTemporaryFile(prefix="pose_stickman_h264_", suffix=".mp4", delete=False)
    browser_temp.close()
    browser_temp_path = Path(browser_temp.name)
    command = [
        _ffmpeg_executable(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source_path), "-map", "0:v:0", "-an",
        "-c:v", "libx264", "-preset", preset, "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
    ]
    if threads is not None:
        command.extend(["-threads", str(max(1, int(threads)))])
    command.append(str(browser_temp_path))
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if completed.returncode != 0 or not browser_temp_path.is_file() or browser_temp_path.stat().st_size == 0:
            detail = completed.stderr.strip() or f"exit code {completed.returncode}"
            raise RuntimeError(f"FFmpeg H.264 conversion failed for {destination_path}: {detail}")
        probe_browser_compatible_mp4(browser_temp_path)
        with tempfile.NamedTemporaryFile(
            prefix=".pose_stickman_publish_", suffix=".mp4", dir=destination_path.parent, delete=False
        ) as stage_file:
            stage_path = Path(stage_file.name)
        try:
            shutil.copyfile(browser_temp_path, stage_path)
            os.replace(stage_path, destination_path)
        finally:
            stage_path.unlink(missing_ok=True)
        return probe_browser_compatible_mp4(destination_path)
    finally:
        browser_temp_path.unlink(missing_ok=True)


class Mp4Writer:
    """Accept BGR frames and atomically publish an Edge-compatible MP4."""

    def __init__(self, path: str | Path, fps: float, size: tuple[int, int], *, realtime: bool = False) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.size = tuple(map(int, size))
        self.frame_count = 0
        self.realtime = bool(realtime)
        self._closed = False
        intermediate = tempfile.NamedTemporaryFile(prefix="pose_stickman_frames_", suffix=".mp4", delete=False)
        intermediate.close()
        self._intermediate_path = Path(intermediate.name)
        self.writer = cv2.VideoWriter(
            str(self._intermediate_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), self.size
        )
        if not self.writer.isOpened():
            self._cleanup_temporary_files()
            raise RuntimeError(f"OpenCV could not create intermediate MP4 for: {self.path}")

    def write(self, frame: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("Cannot write after MP4 exporter is closed")
        if frame.shape[1::-1] != self.size or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Unexpected output frame shape: {frame.shape}")
        self.writer.write(frame)
        self.frame_count += 1

    def _cleanup_temporary_files(self) -> None:
        for path in (self._intermediate_path,):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.writer.release()
        if self.frame_count <= 0:
            self._cleanup_temporary_files()
            raise RuntimeError(f"No frames were written to MP4: {self.path}")
        try:
            transcode_to_browser_mp4(
                self._intermediate_path,
                self.path,
                preset="veryfast" if self.realtime else "medium",
                threads=2 if self.realtime else None,
            )
        finally:
            self._cleanup_temporary_files()

    def __enter__(self) -> "Mp4Writer":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
