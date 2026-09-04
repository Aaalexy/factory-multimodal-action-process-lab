"""Project-contained, streamed MP4 intake with fail-closed path handling."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, BinaryIO, Callable

from src.video_io import probe_video


UPLOAD_ID_PATTERN = re.compile(r"^intake_[0-9a-f]{32}$")
PREVIEW_ID_PATTERN = re.compile(r"^preview_[0-9a-f]{32}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def sanitize_original_filename(value: str) -> str:
    """Return a display-only filename or reject every path-like form."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("original filename is required")
    candidate = value.strip()
    windows = PureWindowsPath(candidate)
    if (
        windows.is_absolute()
        or windows.drive
        or candidate.startswith(("\\\\", "//"))
        or "/" in candidate
        or "\\" in candidate
        or candidate in {".", ".."}
        or ".." in windows.parts
        or Path(candidate).name != candidate
    ):
        raise ValueError("filename must not contain a path")
    if Path(candidate).suffix.lower() != ".mp4":
        raise ValueError("only MP4 uploads are supported")
    if any(ord(character) < 32 for character in candidate):
        raise ValueError("filename contains a control character")
    return candidate


@dataclass(frozen=True)
class VideoIntakeConfig:
    maximum_upload_bytes: int = 2 * 1024 * 1024 * 1024
    upload_chunk_bytes: int = 1024 * 1024
    default_analysis_seconds: float = 12.0
    preview_expiry_seconds: float = 120.0
    preview_sequence_tolerance: int = 0

    @classmethod
    def load(cls, path: str | Path) -> "VideoIntakeConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        flags = payload.get("validation_flags", {})
        if not isinstance(flags, dict) or any(
            flags.get(name) is not False
            for name in (
                "factory_camera_validated",
                "production_action_model_ready",
                "external_factory_validated",
                "production_process_model_ready",
            )
        ):
            raise ValueError("video intake validation flags must remain false")
        values = {
            name: payload[name]
            for name in cls.__dataclass_fields__
            if name in payload
        }
        result = cls(**values)
        result.validate()
        return result

    def validate(self) -> None:
        if self.maximum_upload_bytes <= 0:
            raise ValueError("maximum_upload_bytes must be positive")
        if not 4096 <= self.upload_chunk_bytes <= 8 * 1024 * 1024:
            raise ValueError("upload_chunk_bytes is outside the safe range")
        if self.default_analysis_seconds <= 0:
            raise ValueError("default_analysis_seconds must be positive")
        if self.preview_expiry_seconds <= 0:
            raise ValueError("preview_expiry_seconds must be positive")
        if self.preview_sequence_tolerance < 0:
            raise ValueError("preview_sequence_tolerance cannot be negative")


class VideoIntakeManager:
    """Write one raw HTTP body to a unique `.part`, probe, then rename."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        config: VideoIntakeConfig | None = None,
        probe: Callable[[str | Path], Any] = probe_video,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.root = (self.project_root / "outputs" / "intake").resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_relative_to(self.project_root):
            raise ValueError("intake root escapes the project")
        self.config = config or VideoIntakeConfig()
        self.config.validate()
        self._probe = probe
        self._lock = threading.Lock()
        self._latest_upload_id: str | None = None

    def _directory(self, upload_id: str) -> Path:
        if not UPLOAD_ID_PATTERN.fullmatch(str(upload_id)):
            raise ValueError("invalid upload_id")
        directory = (self.root / upload_id).resolve()
        if not directory.is_relative_to(self.root):
            raise ValueError("upload_id escapes the intake root")
        return directory

    def receive(
        self,
        stream: BinaryIO,
        *,
        content_length: int,
        original_filename: str,
    ) -> dict[str, Any]:
        original = sanitize_original_filename(original_filename)
        if isinstance(content_length, bool) or content_length <= 0:
            raise ValueError("empty uploads are rejected")
        if content_length > self.config.maximum_upload_bytes:
            raise ValueError("upload exceeds the configured maximum size")

        upload_id = "intake_" + uuid.uuid4().hex
        directory = self._directory(upload_id)
        directory.mkdir(parents=False, exist_ok=False)
        part_path = directory / "source.mp4.part"
        final_path = directory / "source.mp4"
        digest = hashlib.sha256()
        written = 0
        try:
            with part_path.open("xb") as target:
                while written < content_length:
                    requested = min(
                        self.config.upload_chunk_bytes,
                        content_length - written,
                    )
                    chunk = stream.read(requested)
                    if not chunk:
                        raise ConnectionError(
                            "upload stream ended before Content-Length"
                        )
                    target.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
                target.flush()
                os.fsync(target.fileno())
            if written != content_length:
                raise ConnectionError("upload byte count mismatch")
            probed = self._probe(part_path)
            probe_payload = (
                probed.to_dict()
                if hasattr(probed, "to_dict")
                else asdict(probed)
                if hasattr(probed, "__dataclass_fields__")
                else vars(probed)
                if hasattr(probed, "__dict__")
                else dict(probed)
            )
            if not bool(probe_payload.get("decodable")):
                raise ValueError("uploaded MP4 is not decodable")
            part_path.replace(final_path)
            metadata = {
                "schema_version": "factory_video_intake_record_v1",
                "upload_id": upload_id,
                "state": "ready",
                "original_filename": original,
                "relative_path": final_path.relative_to(
                    self.project_root
                ).as_posix(),
                "size_bytes": written,
                "sha256": digest.hexdigest(),
                "uploaded_at": utc_now(),
                "probe": {
                    "duration_seconds": float(
                        probe_payload["duration_seconds"]
                    ),
                    "fps": float(probe_payload["fps"]),
                    "width": int(probe_payload["width"]),
                    "height": int(probe_payload["height"]),
                    "frame_count": int(probe_payload["frame_count"]),
                    "codec": str(probe_payload.get("codec", "unknown")),
                    "decodable": True,
                },
                "storage": {
                    "streamed": True,
                    "chunk_bytes": self.config.upload_chunk_bytes,
                    "atomic_part_rename": True,
                    "original_filename_used_for_storage": False,
                },
            }
            _atomic_json(directory / "intake.json", metadata)
            with self._lock:
                self._latest_upload_id = upload_id
            return metadata
        except BaseException:
            # The path is unique to this request. A failed request may retain
            # only a `.part` file and can never be mistaken for a ready MP4.
            failure = {
                "schema_version": "factory_video_intake_failure_v1",
                "upload_id": upload_id,
                "state": "failed",
                "original_filename": original,
                "received_bytes": written,
                "expected_bytes": content_length,
                "complete_video_present": final_path.is_file(),
                "part_present": part_path.is_file(),
                "failed_at": utc_now(),
            }
            _atomic_json(directory / "failure.json", failure)
            raise

    def get(self, upload_id: str) -> dict[str, Any]:
        path = self._directory(upload_id) / "intake.json"
        if not path.is_file():
            raise FileNotFoundError("upload is not ready")
        payload = json.loads(path.read_text(encoding="utf-8"))
        video = (self.project_root / payload["relative_path"]).resolve()
        if not video.is_relative_to(self.root) or not video.is_file():
            raise FileNotFoundError("uploaded MP4 is unavailable")
        return payload

    def video_path(self, upload_id: str) -> Path:
        payload = self.get(upload_id)
        return (self.project_root / payload["relative_path"]).resolve()

    @property
    def latest_upload_id(self) -> str | None:
        with self._lock:
            return self._latest_upload_id


class PreviewRegistry:
    """Keep opaque, short-lived candidate tokens in the localhost process."""

    def __init__(self, expiry_seconds: float = 120.0) -> None:
        self.expiry_seconds = float(expiry_seconds)
        self._lock = threading.Lock()
        self._documents: dict[str, dict[str, Any]] = {}

    def register(
        self,
        *,
        upload_id: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        preview_id = "preview_" + uuid.uuid4().hex
        now = time.time()
        expires_at = now + self.expiry_seconds
        candidates: list[dict[str, Any]] = []
        private: dict[str, dict[str, Any]] = {}
        for candidate in result.get("candidates", []):
            token = secrets.token_urlsafe(32)
            public = {
                **candidate,
                "candidate_token": token,
                "preview_id": preview_id,
                "expires_at": expires_at,
            }
            candidates.append(public)
            private[token] = dict(candidate)
        document = {
            "preview_id": preview_id,
            "upload_id": upload_id,
            "created_at_epoch": now,
            "expires_at_epoch": expires_at,
            "frame_index": int(result["frame_index"]),
            "timestamp": float(result["timestamp"]),
            "candidates": private,
        }
        with self._lock:
            self._documents[preview_id] = document
            self._purge_locked(now)
        return {
            **result,
            "preview_id": preview_id,
            "expires_at": expires_at,
            "candidates": candidates,
        }

    def _purge_locked(self, now: float) -> None:
        expired = [
            key
            for key, value in self._documents.items()
            if float(value["expires_at_epoch"]) < now
        ]
        for key in expired:
            self._documents.pop(key, None)

    def resolve(
        self,
        *,
        upload_id: str,
        preview_id: str,
        candidate_token: str,
    ) -> dict[str, Any]:
        if not PREVIEW_ID_PATTERN.fullmatch(str(preview_id)):
            raise ValueError("invalid preview_id")
        now = time.time()
        with self._lock:
            self._purge_locked(now)
            document = self._documents.get(preview_id)
            if document is None:
                raise ValueError("preview token is expired or unknown")
            if document["upload_id"] != upload_id:
                raise ValueError("preview token belongs to another upload")
            candidate = document["candidates"].get(str(candidate_token))
            if candidate is None:
                raise ValueError("candidate token is expired or unknown")
            return dict(candidate)
