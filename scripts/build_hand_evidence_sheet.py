"""Build a representative Body/Hand Pose sheet from an existing real run."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.multimodal_pipeline import (  # noqa: E402
    _overlay_hand_evidence,
    _stable_event_at,
    _write_contact_sheet,
)


COCO_EDGES = (
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)
STATE_ORDER = ("detected", "uncertain", "missing")


def _normalized_hand_state(records: list[dict[str, Any]]) -> str:
    states = {
        str(record.get("observation_state", "missing")).lower()
        for record in records
    }
    if "detected" in states:
        return "detected"
    if states.intersection({"uncertain", "predicted", "interpolated"}):
        return "uncertain"
    return "missing"


def select_representative_frames(
    payload: dict[str, Any],
    *,
    per_state_limit: int = 4,
) -> list[dict[str, Any]]:
    """Select real sampled frames while prioritizing each hand evidence state."""

    hands_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in payload.get("hand_pose_frames", []):
        hands_by_frame[int(record["frame_index"])].append(record)

    buckets: dict[str, list[dict[str, Any]]] = {
        state: [] for state in STATE_ORDER
    }
    for pose_frame in payload.get("pose_frames", []):
        frame_index = int(pose_frame["source_frame_index"])
        hand_records = hands_by_frame.get(frame_index, [])
        state = _normalized_hand_state(hand_records)
        if len(buckets[state]) >= per_state_limit:
            continue
        buckets[state].append(
            {
                "source_frame_index": frame_index,
                "timestamp": float(pose_frame["timestamp"]),
                "hand_state": state,
                "pose_frame": pose_frame,
                "hand_records": hand_records,
            }
        )

    selected = [
        item
        for state in STATE_ORDER
        for item in buckets[state]
    ]
    return sorted(selected, key=lambda item: item["timestamp"])


def _draw_body_pose(frame: Any, pose_frame: dict[str, Any]) -> Any:
    rendered = frame.copy()
    keypoints = pose_frame.get("keypoints", [])
    statuses = pose_frame.get("keypoint_statuses", [])
    bbox = pose_frame.get("bbox")
    if bbox and len(bbox) == 4:
        cv2.rectangle(
            rendered,
            (int(round(bbox[0])), int(round(bbox[1]))),
            (int(round(bbox[2])), int(round(bbox[3]))),
            (154, 212, 53),
            2,
            cv2.LINE_AA,
        )

    def point(index: int) -> tuple[int, int] | None:
        if index >= len(keypoints):
            return None
        raw = keypoints[index]
        if not raw or raw[0] is None or raw[1] is None:
            return None
        state = str(statuses[index]) if index < len(statuses) else "missing"
        if state in {"missing", "uncertain", "rejected"}:
            return None
        return int(round(raw[0])), int(round(raw[1]))

    for first, second in COCO_EDGES:
        left = point(first)
        right = point(second)
        if left is not None and right is not None:
            cv2.line(rendered, left, right, (154, 212, 53), 3, cv2.LINE_AA)
    for index in range(len(keypoints)):
        location = point(index)
        if location is not None:
            cv2.circle(rendered, location, 3, (238, 253, 246), -1, cv2.LINE_AA)
    return rendered


def build_sheet(analysis_path: Path, output_path: Path) -> dict[str, Any]:
    analysis_path = analysis_path.resolve()
    payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    video_path = analysis_path.parent / "source_video.mp4"
    if not video_path.is_file():
        raise FileNotFoundError(f"Local validation video is missing: {video_path}")

    selected = select_representative_frames(payload)
    if not selected:
        raise ValueError("Analysis contains no sampled pose frames")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open validation video: {video_path}")
    rendered_frames = []
    rendered_records = []
    try:
        for item in selected:
            frame_index = int(item["source_frame_index"])
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            rendered = _draw_body_pose(frame, item["pose_frame"])
            rendered = _overlay_hand_evidence(
                rendered,
                item["hand_records"],
                stable_event=_stable_event_at(
                    payload.get("action_events", []),
                    float(item["timestamp"]),
                ),
            )
            cv2.putText(
                rendered,
                (
                    f"{item['hand_state']} | t={item['timestamp']:.3f}s | "
                    f"frame={frame_index} | "
                    f"{item['pose_frame'].get('person_ref')} / "
                    f"epoch {item['pose_frame'].get('lock_epoch')}"
                ),
                (18, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (238, 240, 244),
                3,
                cv2.LINE_AA,
            )
            rendered_frames.append(rendered)
            rendered_records.append(
                {
                    "source_frame_index": frame_index,
                    "timestamp": item["timestamp"],
                    "hand_state": item["hand_state"],
                    "person_ref": item["pose_frame"].get("person_ref"),
                    "lock_epoch": item["pose_frame"].get("lock_epoch"),
                    "hand_pose_ids": [
                        record.get("hand_pose_id")
                        for record in item["hand_records"]
                    ],
                }
            )
    finally:
        capture.release()

    if not rendered_frames:
        raise RuntimeError("No selected source frames could be decoded")
    _write_contact_sheet(rendered_frames, output_path)
    manifest = {
        "schema_version": "phase_b_representative_hand_evidence_v1",
        "analysis_path": str(analysis_path),
        "source_video_sha256": payload["source_video"]["sha256"],
        "output_path": str(output_path.resolve()),
        "selection_policy": "up_to_4_each_detected_uncertain_missing",
        "selected_frame_count": len(rendered_records),
        "state_counts": {
            state: sum(
                item["hand_state"] == state for item in rendered_records
            )
            for state in STATE_ORDER
        },
        "frames": rendered_records,
        "truthfulness": {
            "mock_hand_landmarks_used": False,
            "missing_landmarks_drawn": False,
            "model_outputs_reclassified_as_detected": False,
        },
    }
    manifest_path = output_path.with_suffix(".json")
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".part")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = build_sheet(args.analysis, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
