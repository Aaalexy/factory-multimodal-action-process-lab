# Hand-Landmark Layer — Technical Decision

## Decision

Use MediaPipe Hand Landmarker as an optional evidence layer while retaining YOLOv8n-Pose COCO-17 wrists as the body-pose source.

## Why the layers remain separate

COCO-17 provides body keypoints including wrists. The hand model can provide 21 landmarks for a detected hand, but it does not identify a held object, prove contact, or establish grasp. Keeping the records separate prevents body-wrist availability from being reported as detailed hand-pose success.

## Failure and association rules

- Missing, lost, rejected, and uncertain body evidence cannot create hand geometry.
- Hand records remain associated with an anonymous person reference, side, frame, and lock epoch.
- Handedness output does not override the anatomical side determined by the body-pose association.
- Inference errors produce an explicit unavailable/error state.
- Disabling the optional layer leaves body pose usable.

## Model distribution

The model binary is not included in the public repository. The recorded upstream source and model-card references are retained in [../HAND_MODEL_MANIFEST.json](../HAND_MODEL_MANIFEST.json), but current license and notice obligations must be reviewed independently before redistribution.

## Validation boundary

Private replay checks covered association, missing-evidence handling, and non-fabrication behavior. The public suite retains source-controlled unit and contract tests; model-runtime and private replay checks are marked `private_artifacts`. No hand-pose accuracy, grasp accuracy, or production validation is claimed.
