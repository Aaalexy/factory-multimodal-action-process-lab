# External Model Files

Binary model files are not distributed in this public portfolio repository. Supply model files that you are authorized to use and pass their paths explicitly or place them at the defaults below.

## Body pose

- Expected interface: YOLOv8n-Pose exported to ONNX.
- Default path: `models/yolov8n-pose.onnx`.
- Override: `--model <path>`.

The repository does not contain enough evidence to document a definitive upstream download/export URL or redistribution grant for the original ONNX artifact. Obtain or export a compatible model under terms you have independently reviewed.

## Optional hand landmarks

- Expected model: MediaPipe Hand Landmarker full, float16 version 1.
- Default path: `models/hand_pose/hand_landmarker.task`.
- Override: `--hand-model <path>`.
- Disable the optional layer: `--disable-hand`.

The recorded official artifact URL and model-card references are in [`../HAND_MODEL_MANIFEST.json`](../HAND_MODEL_MANIFEST.json). Confirm current upstream terms before downloading or redistributing it.

The pipeline raises a clear `FileNotFoundError` when the required body-pose model is absent. A missing optional hand model leaves the hand layer explicitly unavailable rather than fabricating hand evidence.
