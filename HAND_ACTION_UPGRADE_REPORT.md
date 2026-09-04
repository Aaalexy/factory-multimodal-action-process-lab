# Hand and Action Evidence Upgrade

This sanitized engineering summary documents the optional hand-landmark layer and the action-stabilization work. Private recordings and generated outputs are not included.

## Scope

- Added an optional MediaPipe Hand Landmarker backend alongside the COCO-17 body-pose pipeline.
- Kept body-pose wrist keypoints separate from 21-point hand-landmark observations.
- Preserved anonymous person references and lock epochs across evidence records.
- Required missing, lost, rejected, and uncertain states to avoid fabricated geometry.
- Kept object contact, grasp, and process correctness unavailable without supporting perception and review evidence.

## Validation design

Three privately held reference windows were used during the internship regression pass. They are referred to here only as `reference_clip_01`, `reference_clip_02`, and `reference_clip_03`. Each window contributed 96 sampled frames.

Across the 288 sampled frames, the recorded hand-evidence states were:

| State | Frames |
| --- | ---: |
| Detected | 7 |
| Uncertain | 34 |
| Missing | 247 |

These counts describe evidence availability under the tested settings. They are not hand-pose accuracy, grasp-detection accuracy, or an independent benchmark.

## Safety behavior

- Body-pose wrist visibility does not imply a detected hand.
- A hand result is associated only when body-track, side, quality, and temporal checks allow it.
- Missing or lost evidence never carries forward hand geometry.
- The action layer emits review candidates rather than confirmed process labels.
- Object, interaction, and process layers remain unavailable when their upstream evidence is not configured.

## Reproducibility boundary

The public test command excludes the private regression windows and generated analyses while retaining the original assertions in explicitly marked artifact-dependent tests. Externally supplied model files are also required for model-runtime checks. See [docs/PUBLIC_TESTING.md](docs/PUBLIC_TESTING.md) and [models/README.md](models/README.md).
