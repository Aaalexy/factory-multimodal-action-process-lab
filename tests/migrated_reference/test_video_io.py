import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.export_video import Mp4Writer
from src.video_reader import VideoReader
from tests.validation_tools import (
    CSV_REQUIRED_COLUMNS,
    inspect_video,
    validate_keypoint_csv,
    validate_keypoint_json,
)


def _write_sample_video(path: Path, frame_count: int = 8, fps: float = 8.0) -> None:
    with Mp4Writer(path, fps=fps, size=(96, 64)) as writer:
        for index in range(frame_count):
            frame = np.full((64, 96, 3), (5, 12, 20), np.uint8)
            cv2.rectangle(frame, (5 + index * 3, 12), (30 + index * 3, 52), (40, 180, 240), -1)
            writer.write(frame)


def test_chinese_and_space_path_video_round_trip_and_sampling(tmp_path: Path):
    video_path = tmp_path / "中文 空格目录" / "火柴人 输出.mp4"
    _write_sample_video(video_path)

    reader = VideoReader(video_path)
    assert reader.metadata.width == 96
    assert reader.metadata.height == 64
    assert reader.metadata.fps == pytest.approx(8.0, abs=0.1)
    assert reader.metadata.frame_count == 8
    assert reader.metadata.duration_seconds == pytest.approx(1.0, abs=0.05)

    sampled = list(reader.iter_frames(start_time=0.0, end_time=1.0, output_fps=4.0))
    assert len(sampled) == 4
    assert [frame.frame_index for frame in sampled] == [0, 1, 2, 3]
    assert all(frame.image.shape == (64, 96, 3) for frame in sampled)


def test_video_inspector_checks_frames_fps_size_duration_and_nonempty_pixels(tmp_path: Path):
    video_path = tmp_path / "inspect.mp4"
    _write_sample_video(video_path, frame_count=6, fps=12.0)
    result = inspect_video(video_path, expected_size=(96, 64))
    assert result["decoded_frame_count"] == 6
    assert result["reported_frame_count"] == 6
    assert result["fps"] == pytest.approx(12.0, abs=0.1)
    assert result["duration_seconds"] == pytest.approx(0.5, abs=0.05)
    assert result["nonempty_frame_count"] == 6


def _valid_csv_row() -> dict[str, object]:
    row = {column: "" for column in CSV_REQUIRED_COLUMNS}
    row.update(
        {
            "frame_index": 0,
            "source_frame_index": 10,
            "timestamp": 1.0,
            "track_id": 1,
            "track_state": "tracked",
            "person_confidence": 0.9,
            "bbox_x1": 1,
            "bbox_y1": 2,
            "bbox_x2": 20,
            "bbox_y2": 30,
            "keypoint": "left_wrist",
            "raw_x": 10,
            "raw_y": 11,
            "raw_confidence": 0.9,
            "smoothed_x": 10.5,
            "smoothed_y": 11,
            "smoothed_confidence": 0.9,
            "keypoint_state": "detected",
            "uncertain": False,
            "interpolation_used": False,
            "possible_switch": False,
            "source_type": "raw_and_smoothed",
            "derived_sources": "",
        }
    )
    return row


def test_csv_schema_validator_accepts_raw_smoothed_and_state_columns(tmp_path: Path):
    path = tmp_path / "keypoints.csv"
    fields = sorted(CSV_REQUIRED_COLUMNS)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(_valid_csv_row())
    summary = validate_keypoint_csv(path)
    assert summary["row_count"] == 1
    assert summary["frame_count"] == 1
    assert summary["states"] == ["detected"]


def test_json_schema_validator_preserves_raw_smoothed_and_derived_labels(tmp_path: Path):
    path = tmp_path / "keypoints.json"
    payload = {
        "metadata": {"version": "V1.0"},
        "frames": [
            {
                "frame_index": 0,
                "source_frame_index": 2,
                "timestamp": 0.2,
                "track_id": 1,
                "track_state": "tracked",
                "person_confidence": 0.9,
                "bbox": [1, 2, 30, 50],
                "uncertain": False,
                "lost": False,
                "interpolation_used": False,
                "possible_switch": False,
                "keypoints": {
                    "left_wrist": {
                        "raw": {"x": 10, "y": 12, "confidence": 0.9},
                        "smoothed": {"x": 10.5, "y": 12, "confidence": 0.9},
                        "state": "detected",
                    }
                },
                "derived_keypoints": {
                    "left_palm": {
                        "x": 13,
                        "y": 12,
                        "confidence": 0.7,
                        "state": "derived",
                        "sources": ["left_elbow", "left_wrist"],
                    }
                },
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    summary = validate_keypoint_json(path)
    assert summary["frame_count"] == 1
    assert summary["states"] == ["derived", "detected"]


def test_schema_validators_reject_missing_columns_and_mislabelled_derived_points(tmp_path: Path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("frame_index,keypoint\n0,nose\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        validate_keypoint_csv(csv_path)

    json_path = tmp_path / "bad.json"
    frame = {
        "frame_index": 0,
        "source_frame_index": 0,
        "timestamp": 0.0,
        "track_id": 1,
        "track_state": "tracked",
        "person_confidence": 0.9,
        "bbox": [0, 0, 1, 1],
        "uncertain": False,
        "lost": False,
        "interpolation_used": False,
        "possible_switch": False,
        "keypoints": {
            "nose": {
                "raw": {"x": 0, "y": 0, "confidence": 0.9},
                "smoothed": {"x": 0, "y": 0, "confidence": 0.9},
                "state": "detected",
            }
        },
        "derived_keypoints": {
            "left_palm": {"x": 1, "y": 1, "confidence": 0.5, "state": "detected"}
        },
    }
    json_path.write_text(json.dumps({"metadata": {}, "frames": [frame]}), encoding="utf-8")
    with pytest.raises(ValueError, match="labelled derived"):
        validate_keypoint_json(json_path)
