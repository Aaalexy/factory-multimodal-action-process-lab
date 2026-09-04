"""Reusable, model-independent validators for V1.0 generated artifacts.

These helpers intentionally inspect decoded content rather than accepting a file
merely because it exists.  They are used by unit tests and can also be invoked
against the real end-to-end output directory during validation.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CSV_REQUIRED_COLUMNS = {
    "frame_index",
    "source_frame_index",
    "timestamp",
    "track_id",
    "track_state",
    "person_confidence",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "keypoint",
    "raw_x",
    "raw_y",
    "raw_confidence",
    "smoothed_x",
    "smoothed_y",
    "smoothed_confidence",
    "keypoint_state",
    "uncertain",
    "lost",
    "interpolation_used",
    "possible_switch",
    "source_type",
    "derived_sources",
}

FRAME_REQUIRED_KEYS = {
    "frame_index",
    "source_frame_index",
    "timestamp",
    "track_id",
    "track_state",
    "person_confidence",
    "bbox",
    "uncertain",
    "lost",
    "interpolation_used",
    "possible_switch",
    "keypoints",
    "derived_keypoints",
}

KEYPOINT_REQUIRED_KEYS = {"raw", "smoothed", "state"}
POINT_REQUIRED_KEYS = {"x", "y", "confidence"}
ALLOWED_KEYPOINT_STATES = {
    "detected",
    "interpolated",
    "derived",
    "uncertain",
    "missing",
}


def inspect_video(path: str | Path, expected_size: tuple[int, int] | None = None) -> dict[str, Any]:
    """Decode every frame and return objective MP4 properties.

    ``nonempty_frame_count`` is based on pixel range/energy rather than encoded
    file size, so an all-black or undecodable output cannot pass accidentally.
    """

    video_path = Path(path)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Video cannot be decoded: {video_path}")
    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    decoded_frames = 0
    nonempty_frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded_frames += 1
            if frame.size and (int(frame.max()) > int(frame.min()) or float(frame.mean()) > 1.0):
                nonempty_frames += 1
    finally:
        capture.release()

    if fps <= 0 or width <= 0 or height <= 0 or decoded_frames <= 0:
        raise ValueError(
            f"Invalid decoded video: fps={fps}, size={width}x{height}, frames={decoded_frames}"
        )
    if expected_size is not None and (width, height) != tuple(expected_size):
        raise ValueError(f"Unexpected video size {(width, height)}; expected {expected_size}")
    if nonempty_frames <= 0:
        raise ValueError("Every decoded frame is empty/black")
    return {
        "path": str(video_path.resolve()),
        "reported_frame_count": reported_frames,
        "decoded_frame_count": decoded_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_seconds": decoded_frames / fps,
        "nonempty_frame_count": nonempty_frames,
    }


def inspect_transparent_png(path: str | Path) -> dict[str, Any]:
    """Require a decodable BGRA PNG with both visible and transparent pixels."""

    png_path = Path(path)
    image = cv2.imdecode(np.frombuffer(png_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"PNG cannot be decoded: {png_path}")
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError(f"Expected BGRA PNG, got shape {image.shape}")
    alpha = image[:, :, 3]
    visible = int(np.count_nonzero(alpha))
    transparent = int(np.count_nonzero(alpha == 0))
    if visible == 0 or transparent == 0:
        raise ValueError(
            f"Alpha channel must contain visible and transparent pixels: visible={visible}, transparent={transparent}"
        )
    return {
        "path": str(png_path.resolve()),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "visible_alpha_pixels": visible,
        "transparent_alpha_pixels": transparent,
        "alpha_min": int(alpha.min()),
        "alpha_max": int(alpha.max()),
    }


def validate_keypoint_csv(path: str | Path) -> dict[str, Any]:
    """Validate the long-form CSV schema and state vocabulary."""

    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or ())
        missing = CSV_REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"CSV missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV contains no keypoint rows")
    states = {row["keypoint_state"] for row in rows}
    invalid_states = states - ALLOWED_KEYPOINT_STATES
    if invalid_states:
        raise ValueError(f"CSV contains invalid states: {sorted(invalid_states)}")
    for row in rows:
        int(row["frame_index"])
        int(row["source_frame_index"])
        float(row["timestamp"])
        float(row["person_confidence"])
        if row["source_type"] not in {"raw_and_smoothed", "derived"}:
            raise ValueError(f"Invalid source_type: {row['source_type']}")
    return {
        "row_count": len(rows),
        "frame_count": len({int(row["frame_index"]) for row in rows}),
        "states": sorted(states),
        "columns": sorted(columns),
    }


def validate_keypoint_json(path: str | Path) -> dict[str, Any]:
    """Validate per-frame raw/smoothed/status and derived-state structure."""

    json_path = Path(path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("metadata"), dict):
        raise ValueError("JSON metadata must be an object")
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("JSON frames must be a non-empty array")
    observed_states: set[str] = set()
    for frame in frames:
        missing = FRAME_REQUIRED_KEYS - set(frame)
        if missing:
            raise ValueError(f"JSON frame missing keys: {sorted(missing)}")
        if not isinstance(frame["keypoints"], dict) or not frame["keypoints"]:
            raise ValueError("JSON frame keypoints must be a non-empty object")
        for keypoint in frame["keypoints"].values():
            if KEYPOINT_REQUIRED_KEYS - set(keypoint):
                raise ValueError("JSON keypoint lacks raw/smoothed/state")
            state = keypoint["state"]
            if state not in ALLOWED_KEYPOINT_STATES - {"derived"}:
                raise ValueError(f"Invalid detected-keypoint state: {state}")
            observed_states.add(state)
            for representation in ("raw", "smoothed"):
                if POINT_REQUIRED_KEYS - set(keypoint[representation]):
                    raise ValueError(f"JSON {representation} point lacks x/y/confidence")
        for derived in frame["derived_keypoints"].values():
            if derived.get("state") != "derived":
                raise ValueError("Derived keypoints must remain labelled derived")
            if POINT_REQUIRED_KEYS - set(derived):
                raise ValueError("Derived point lacks x/y/confidence")
            observed_states.add("derived")
    return {
        "frame_count": len(frames),
        "states": sorted(observed_states),
        "metadata_keys": sorted(payload["metadata"]),
    }
