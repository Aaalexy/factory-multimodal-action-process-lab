"""YOLOv8 pose letterboxing, 56/57-channel decoding, and NMS."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxTransform:
    original_shape: tuple[int, int]
    input_shape: tuple[int, int]
    scale: float
    pad_x: float
    pad_y: float


@dataclass
class PoseDetection:
    bbox: np.ndarray
    confidence: float
    keypoints: np.ndarray

    def __post_init__(self) -> None:
        self.bbox = np.asarray(self.bbox, dtype=np.float32).reshape(4)
        self.keypoints = np.asarray(self.keypoints, dtype=np.float32).reshape(17, 3)
        self.confidence = float(self.confidence)

    @property
    def area(self) -> float:
        return float(
            max(0.0, self.bbox[2] - self.bbox[0])
            * max(0.0, self.bbox[3] - self.bbox[1])
        )

    @property
    def center(self) -> np.ndarray:
        return (self.bbox[:2] + self.bbox[2:]) * 0.5


def letterbox(
    image: np.ndarray,
    input_shape: tuple[int, int] = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, LetterboxTransform]:
    """Resize with unchanged aspect ratio and symmetric padding."""

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("letterbox expects an HxWx3 image")
    original_h, original_w = image.shape[:2]
    input_h, input_w = map(int, input_shape)
    if min(original_h, original_w, input_h, input_w) <= 0:
        raise ValueError("image and input dimensions must be positive")
    scale = min(input_w / original_w, input_h / original_h)
    resized_w = int(round(original_w * scale))
    resized_h = int(round(original_h * scale))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    left = (input_w - resized_w) // 2
    right = input_w - resized_w - left
    top = (input_h - resized_h) // 2
    bottom = input_h - resized_h - top
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return padded, LetterboxTransform(
        original_shape=(original_h, original_w),
        input_shape=(input_h, input_w),
        scale=scale,
        pad_x=float(left),
        pad_y=float(top),
    )


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    box = np.asarray(box, dtype=np.float32).reshape(4)
    intersection_tl = np.maximum(box[:2], boxes[:, :2])
    intersection_br = np.minimum(box[2:], boxes[:, 2:])
    intersection_wh = np.maximum(0.0, intersection_br - intersection_tl)
    intersection = intersection_wh[:, 0] * intersection_wh[:, 1]
    box_area = max(0.0, float(box[2] - box[0])) * max(
        0.0, float(box[3] - box[1])
    )
    boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    union = box_area + boxes_area - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def nms(detections: list[PoseDetection], iou_threshold: float) -> list[PoseDetection]:
    if not detections:
        return []
    ordered = sorted(detections, key=lambda item: item.confidence, reverse=True)
    kept: list[PoseDetection] = []
    while ordered:
        best = ordered.pop(0)
        kept.append(best)
        if ordered:
            overlaps = box_iou(best.bbox, np.stack([item.bbox for item in ordered]))
            ordered = [item for item, overlap in zip(ordered, overlaps) if overlap <= iou_threshold]
    return kept


def _prediction_matrix(output: np.ndarray) -> np.ndarray:
    prediction = np.asarray(output)
    while prediction.ndim > 2 and prediction.shape[0] == 1:
        prediction = prediction[0]
    if prediction.ndim != 2:
        raise ValueError(f"Expected a 2-D pose prediction, got {prediction.shape}")
    if prediction.shape[0] in (56, 57):
        prediction = prediction.T
    elif prediction.shape[1] not in (56, 57):
        raise ValueError(
            "Unsupported YOLO pose output. Expected 56 or 57 channels, "
            f"got {prediction.shape}"
        )
    return prediction.astype(np.float32, copy=False)


def decode_pose_output(
    output: np.ndarray,
    transform: LetterboxTransform,
    confidence_threshold: float = 0.25,
    keypoint_threshold: float = 0.25,
    nms_iou_threshold: float = 0.45,
) -> list[PoseDetection]:
    """Decode common YOLOv8 pose exports.

    56 channels are ``xywh, person_score, 17*(x,y,score)``. 57-channel
    exports include an objectness and a person-class score; the two scores are
    multiplied. Coordinates are mapped back to the original, unpadded frame.
    """

    rows = _prediction_matrix(output)
    channels = rows.shape[1]
    if channels == 56:
        person_scores = rows[:, 4]
        keypoint_offset = 5
    else:
        person_scores = rows[:, 4] * rows[:, 5]
        keypoint_offset = 6
    selected = np.flatnonzero(person_scores >= confidence_threshold)
    detections: list[PoseDetection] = []
    input_h, input_w = transform.input_shape
    original_h, original_w = transform.original_shape
    for index in selected:
        row = rows[index]
        cx, cy, width, height = map(float, row[:4])
        keypoints = row[keypoint_offset : keypoint_offset + 51].reshape(17, 3).copy()
        # Some custom exports normalize coordinates; official Ultralytics exports use pixels.
        if max(abs(cx), abs(cy), abs(width), abs(height)) <= 2.0:
            cx, width = cx * input_w, width * input_w
            cy, height = cy * input_h, height * input_h
            keypoints[:, 0] *= input_w
            keypoints[:, 1] *= input_h
        bbox = np.array(
            [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2],
            dtype=np.float32,
        )
        bbox[[0, 2]] = (bbox[[0, 2]] - transform.pad_x) / transform.scale
        bbox[[1, 3]] = (bbox[[1, 3]] - transform.pad_y) / transform.scale
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, original_w - 1)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, original_h - 1)
        keypoints[:, 0] = (keypoints[:, 0] - transform.pad_x) / transform.scale
        keypoints[:, 1] = (keypoints[:, 1] - transform.pad_y) / transform.scale
        keypoints[:, 0] = np.clip(keypoints[:, 0], 0, original_w - 1)
        keypoints[:, 1] = np.clip(keypoints[:, 1], 0, original_h - 1)
        # Confidence is retained. Consumers decide whether to draw/interpolate;
        # invalid coordinates are made explicit rather than fabricated.
        invalid = ~np.isfinite(keypoints).all(axis=1)
        keypoints[invalid, :2] = np.nan
        keypoints[invalid, 2] = 0.0
        keypoints[keypoints[:, 2] < keypoint_threshold, 2] = np.clip(
            keypoints[keypoints[:, 2] < keypoint_threshold, 2], 0.0, 1.0
        )
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        detections.append(PoseDetection(bbox, float(person_scores[index]), keypoints))
    return nms(detections, nms_iou_threshold)
