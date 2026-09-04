"""Real preview detection and anonymous manual-track seed handling."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from .pose_detector import PoseDetector
from .pose_postprocess import PoseDetection, box_iou
from .video_reader import VideoFileSource


TORSO_INDICES = (5, 6, 11, 12)


@dataclass(frozen=True)
class ManualSelectionSeed:
    candidate_id: str
    video_path: str
    selection_timestamp: float
    selection_frame_index: int
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    size: tuple[float, float]
    torso_keypoints: tuple[tuple[float, float, float], ...]
    person_confidence: float
    selection_source: str = "manual"
    manual_reselection: bool = False
    source_width: int = 0
    source_height: int = 0
    mirror_horizontal: bool = False
    normalized_bbox: tuple[float, float, float, float] = ()
    normalized_torso_keypoints: tuple[tuple[float, float, float], ...] = ()
    camera_backend: str = "unknown"
    preview_frame_hash: str = ""
    selected_candidate_fingerprint: str = ""

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "ManualSelectionSeed":
        if value.get("selection_source") != "manual":
            raise ValueError("manual selection seed must use selection_source=manual")
        bbox = tuple(float(item) for item in value["bbox"])
        center = tuple(float(item) for item in value.get("center", ()))
        size = tuple(float(item) for item in value.get("size", ()))
        if len(bbox) != 4:
            raise ValueError("manual selection bbox must contain four coordinates")
        if len(center) != 2:
            center = ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)
        if len(size) != 2:
            size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
        torso = tuple(
            tuple(float(component) for component in point)
            for point in value.get("torso_keypoints", ())
        )
        if len(torso) != 4 or any(len(point) != 3 for point in torso):
            raise ValueError("manual selection requires four shoulder/hip keypoints")
        normalized_bbox = tuple(float(item) for item in value.get("normalized_bbox", ()))
        normalized_torso = tuple(tuple(float(component) for component in point) for point in value.get("normalized_torso_keypoints", ()))
        return cls(
            candidate_id=str(value["candidate_id"]),
            video_path=str(Path(value["video_path"]).expanduser().resolve()),
            selection_timestamp=float(value["selection_timestamp"]),
            selection_frame_index=int(value["selection_frame_index"]),
            bbox=bbox, center=center, size=size, torso_keypoints=torso,
            person_confidence=float(value["person_confidence"]),
            selection_source="manual",
            manual_reselection=bool(value.get("manual_reselection", False)),
            source_width=max(0, int(value.get("source_width", value.get("preview_width", 0)) or 0)),
            source_height=max(0, int(value.get("source_height", value.get("preview_height", 0)) or 0)),
            mirror_horizontal=bool(value.get("mirror_horizontal", False)),
            normalized_bbox=normalized_bbox if len(normalized_bbox) == 4 else (),
            normalized_torso_keypoints=normalized_torso if len(normalized_torso) == 4 else (),
            camera_backend=str(value.get("camera_backend", "unknown")),
            preview_frame_hash=str(value.get("preview_frame_hash", "")),
            selected_candidate_fingerprint=str(value.get("selected_candidate_fingerprint", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "video_path": self.video_path,
            "selection_timestamp": self.selection_timestamp,
            "selection_frame_index": self.selection_frame_index,
            "bbox": list(self.bbox), "center": list(self.center), "size": list(self.size),
            "torso_keypoints": [list(point) for point in self.torso_keypoints],
            "person_confidence": self.person_confidence,
            "selection_source": self.selection_source,
            "manual_reselection": self.manual_reselection,
            "source_width": self.source_width, "source_height": self.source_height,
            "mirror_horizontal": self.mirror_horizontal,
            "normalized_bbox": list(self.normalized_bbox),
            "normalized_torso_keypoints": [list(point) for point in self.normalized_torso_keypoints],
            "camera_backend": self.camera_backend,
            "preview_frame_hash": self.preview_frame_hash,
            "selected_candidate_fingerprint": self.selected_candidate_fingerprint,
        }


def candidate_payload(
    detection: PoseDetection,
    index: int,
    *,
    video_path: str | Path,
    timestamp: float,
    frame_index: int,
    frame_width: int | None = None,
    frame_height: int | None = None,
    mirror_horizontal: bool = False,
    camera_backend: str = "unknown",
    preview_frame_hash: str = "",
) -> dict[str, Any]:
    bbox = detection.bbox.astype(float)
    size = bbox[2:] - bbox[:2]
    width=max(0,int(frame_width or 0));height=max(0,int(frame_height or 0))
    normalized_bbox=(bbox/np.array([width,height,width,height],dtype=np.float32)).tolist() if width and height else []
    torso=detection.keypoints[list(TORSO_INDICES)].astype(float)
    normalized_torso=torso.copy()
    if width and height:
        normalized_torso[:,0]/=width;normalized_torso[:,1]/=height
    else:normalized_torso=np.empty((0,3),dtype=np.float32)
    fingerprint_source=np.concatenate([bbox.astype(np.float32),torso[:,:2].reshape(-1)]).tobytes()
    return {
        "candidate_id": f"C{index + 1}",
        "video_path": str(Path(video_path).expanduser().resolve()),
        "selection_timestamp": round(float(timestamp), 6),
        "selection_frame_index": int(frame_index),
        "bbox": bbox.tolist(),
        "center": detection.center.astype(float).tolist(),
        "size": size.tolist(),
        "torso_keypoints": torso.tolist(),
        "person_confidence": round(float(detection.confidence), 6),
        "selection_source": "manual",
        "manual_reselection": False,
        "source_width":width,"source_height":height,
        "mirror_horizontal":bool(mirror_horizontal),
        "normalized_bbox":normalized_bbox,
        "normalized_torso_keypoints":normalized_torso.tolist(),
        "camera_backend":str(camera_backend),
        "preview_frame_hash":str(preview_frame_hash),
        "selected_candidate_fingerprint":__import__("hashlib").sha256(fingerprint_source).hexdigest(),
    }


def remap_manual_seed(
    seed: ManualSelectionSeed,
    *,
    target_width: int,
    target_height: int,
    target_mirror_horizontal: bool,
) -> tuple[ManualSelectionSeed, dict[str, Any]]:
    """Map a preview seed into the formal worker pixel coordinate system."""

    if target_width <= 0 or target_height <= 0:
        raise ValueError("target frame dimensions must be positive")
    if len(seed.normalized_bbox) == 4:
        normalized_bbox=np.asarray(seed.normalized_bbox,dtype=np.float32)
    elif seed.source_width > 0 and seed.source_height > 0:
        normalized_bbox=np.asarray(seed.bbox,dtype=np.float32)/np.asarray([seed.source_width,seed.source_height,seed.source_width,seed.source_height],dtype=np.float32)
    else:
        normalized_bbox=np.asarray(seed.bbox,dtype=np.float32)/np.asarray([target_width,target_height,target_width,target_height],dtype=np.float32)
    if len(seed.normalized_torso_keypoints) == 4:
        normalized_torso=np.asarray(seed.normalized_torso_keypoints,dtype=np.float32)
    else:
        normalized_torso=np.asarray(seed.torso_keypoints,dtype=np.float32).copy()
        source_width=seed.source_width or target_width;source_height=seed.source_height or target_height
        normalized_torso[:,0]/=source_width;normalized_torso[:,1]/=source_height
    mirror_changed=bool(seed.mirror_horizontal)!=bool(target_mirror_horizontal)
    if mirror_changed:
        normalized_bbox=np.asarray([1.0-normalized_bbox[2],normalized_bbox[1],1.0-normalized_bbox[0],normalized_bbox[3]],dtype=np.float32)
        normalized_torso=normalized_torso.copy();normalized_torso[:,0]=1.0-normalized_torso[:,0]
    bbox=normalized_bbox*np.asarray([target_width,target_height,target_width,target_height],dtype=np.float32)
    torso=normalized_torso.copy();torso[:,0]*=target_width;torso[:,1]*=target_height
    mapped=ManualSelectionSeed(
        candidate_id=seed.candidate_id,video_path=seed.video_path,
        selection_timestamp=seed.selection_timestamp,selection_frame_index=seed.selection_frame_index,
        bbox=tuple(float(value) for value in bbox),
        center=(float((bbox[0]+bbox[2])*.5),float((bbox[1]+bbox[3])*.5)),
        size=(float(bbox[2]-bbox[0]),float(bbox[3]-bbox[1])),
        torso_keypoints=tuple(tuple(float(component) for component in point) for point in torso),
        person_confidence=seed.person_confidence,selection_source="manual",
        manual_reselection=seed.manual_reselection,source_width=target_width,source_height=target_height,
        mirror_horizontal=bool(target_mirror_horizontal),normalized_bbox=tuple(float(value) for value in normalized_bbox),
        normalized_torso_keypoints=tuple(tuple(float(component) for component in point) for point in normalized_torso),
        camera_backend=seed.camera_backend,preview_frame_hash=seed.preview_frame_hash,
        selected_candidate_fingerprint=seed.selected_candidate_fingerprint,
    )
    diagnostic={
        "preview_resolution":[seed.source_width,seed.source_height],
        "formal_resolution":[target_width,target_height],
        "preview_mirror_horizontal":seed.mirror_horizontal,
        "formal_mirror_horizontal":bool(target_mirror_horizontal),
        "mirror_transform_applied":mirror_changed,
        "normalized_bbox":list(mapped.normalized_bbox),
        "mapped_bbox":list(mapped.bbox),
        "selection_mapping_valid":bool(np.isfinite(bbox).all() and bbox[2]>bbox[0] and bbox[3]>bbox[1]),
    }
    return mapped,diagnostic


def displayed_point_to_video(
    point_x: float,
    point_y: float,
    display_width: float,
    display_height: float,
    video_width: float,
    video_height: float,
) -> tuple[float, float] | None:
    """Map a contain-fitted display point through any letterbox bars."""

    if min(display_width, display_height, video_width, video_height) <= 0:
        raise ValueError("display and video dimensions must be positive")
    scale = min(display_width / video_width, display_height / video_height)
    content_width, content_height = video_width * scale, video_height * scale
    offset_x = (display_width - content_width) * 0.5
    offset_y = (display_height - content_height) * 0.5
    if not (
        offset_x <= point_x <= offset_x + content_width
        and offset_y <= point_y <= offset_y + content_height
    ):
        return None
    return (point_x - offset_x) / scale, (point_y - offset_y) / scale


def candidate_at_point(
    candidates: Sequence[dict[str, Any]], x: float, y: float
) -> dict[str, Any] | None:
    inside = [
        candidate for candidate in candidates
        if candidate["bbox"][0] <= x <= candidate["bbox"][2]
        and candidate["bbox"][1] <= y <= candidate["bbox"][3]
    ]
    if not inside:
        return None
    return min(
        inside,
        key=lambda candidate: (
            (candidate["bbox"][2] - candidate["bbox"][0])
            * (candidate["bbox"][3] - candidate["bbox"][1])
        ),
    )


def manual_seed_match_scores(
    seed: ManualSelectionSeed,
    detections: Sequence[PoseDetection],
    keypoint_threshold: float = 0.25,
) -> list[float]:
    seed_bbox = np.asarray(seed.bbox, dtype=np.float32)
    seed_center = np.asarray(seed.center, dtype=np.float32)
    diagonal = max(1.0, float(np.linalg.norm(seed_bbox[2:] - seed_bbox[:2])))
    seed_torso = np.asarray(seed.torso_keypoints, dtype=np.float32)
    scores: list[float] = []
    for detection in detections:
        iou = float(box_iou(seed_bbox, detection.bbox.reshape(1, 4))[0])
        center = max(0.0, 1.0 - float(np.linalg.norm(detection.center - seed_center) / diagonal))
        seed_area = max(1.0, (seed_bbox[2] - seed_bbox[0]) * (seed_bbox[3] - seed_bbox[1]))
        area = min(seed_area, detection.area) / max(seed_area, detection.area, 1.0)
        current_torso = detection.keypoints[list(TORSO_INDICES)]
        valid = (
            (seed_torso[:, 2] >= keypoint_threshold)
            & (current_torso[:, 2] >= keypoint_threshold)
            & np.isfinite(seed_torso[:, :2]).all(axis=1)
            & np.isfinite(current_torso[:, :2]).all(axis=1)
        )
        torso = (
            float(np.exp(-3.0 * np.mean(
                np.linalg.norm(seed_torso[valid, :2] - current_torso[valid, :2], axis=1)
                / diagonal
            ))) if np.any(valid) else 0.0
        )
        scores.append(float(0.36 * iou + 0.24 * center + 0.16 * area + 0.24 * torso))
    return scores


def choose_manual_seed_detection(
    seed: ManualSelectionSeed,
    detections: Sequence[PoseDetection],
    *,
    minimum_score: float = 0.52,
    ambiguity_margin: float = 0.04,
    keypoint_threshold: float = 0.25,
) -> tuple[int | None, float, bool]:
    scores = manual_seed_match_scores(seed, detections, keypoint_threshold)
    if not scores:
        return None, 0.0, False
    order = np.argsort(np.asarray(scores))[::-1]
    best_index, best = int(order[0]), float(scores[int(order[0])])
    ambiguous = len(order) > 1 and best - float(scores[int(order[1])]) < ambiguity_margin
    if best < minimum_score or ambiguous:
        return None, best, ambiguous
    return best_index, best, False


def detect_preview(
    video_path: str | Path,
    model_path: str | Path,
    timestamp: float,
    person_confidence: float = 0.25,
    keypoint_confidence: float = 0.25,
    nms_iou: float = 0.45,
) -> dict[str, Any]:
    source = VideoFileSource(video_path)
    start = float(np.clip(timestamp, 0.0, max(0.0, source.metadata.duration_seconds - 1.0 / source.metadata.fps)))
    packet = next(source.iter_frames(start, min(source.metadata.duration_seconds, start + 1.5 / source.metadata.fps), source.metadata.fps))
    detector = PoseDetector(model_path, person_confidence, keypoint_confidence, nms_iou)
    detections = detector.detect(packet.image)
    ok, encoded = cv2.imencode(".jpg", packet.image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise RuntimeError("Could not encode preview frame")
    candidates = [
        candidate_payload(
            detection, index, video_path=source.path,
            timestamp=packet.timestamp, frame_index=packet.source_frame_index,
            frame_width=source.metadata.width,
            frame_height=source.metadata.height,
        )
        for index, detection in enumerate(detections)
    ]
    return {
        "video_path": str(source.path),
        "frame_index": packet.source_frame_index,
        "timestamp": round(packet.timestamp, 6),
        "width": source.metadata.width, "height": source.metadata.height,
        "candidate_count": len(candidates), "candidates": candidates,
        "preview_image": "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii"),
        "real_model_inference": True,
        "model_input_type": detector.session.get_inputs()[0].type,
    }
