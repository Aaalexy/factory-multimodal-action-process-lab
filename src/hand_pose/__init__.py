"""Body-guided, real-model hand-pose integration."""

from .backend import (
    HAND_BACKEND_MODE,
    HAND_BACKEND_MODES,
    HAND_LANDMARK_COUNT,
    HAND_QUALITY_GATE_VERSION,
    MAX_MODEL_WRIST_DISTANCE_ROI_RATIO,
    DisabledHandBackend,
    HandPoseBackend,
    MediaPipeHandLandmarkerBackend,
    RealHandPoseBackend,
    finalize_hand_pose_record,
    map_normalized_landmarks,
)
from .video_backend import (
    VIDEO_BACKEND_MODE,
    MediaPipeHandLandmarkerVideoBackend,
)
from .roi import (
    COCO_HAND_GUIDE_INDICES,
    CropTransform,
    body_wrist_point,
    build_hand_crop_transform,
)

__all__ = [
    "COCO_HAND_GUIDE_INDICES",
    "HAND_BACKEND_MODE",
    "HAND_BACKEND_MODES",
    "HAND_LANDMARK_COUNT",
    "HAND_QUALITY_GATE_VERSION",
    "MAX_MODEL_WRIST_DISTANCE_ROI_RATIO",
    "CropTransform",
    "DisabledHandBackend",
    "HandPoseBackend",
    "MediaPipeHandLandmarkerBackend",
    "MediaPipeHandLandmarkerVideoBackend",
    "RealHandPoseBackend",
    "body_wrist_point",
    "build_hand_crop_transform",
    "finalize_hand_pose_record",
    "map_normalized_landmarks",
    "VIDEO_BACKEND_MODE",
]
