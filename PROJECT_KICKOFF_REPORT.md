# Project Kickoff and Baseline Validation

This is a sanitized public summary of the project baseline. Private recordings, source-workspace locations, file fingerprints, generated analyses, and workstation-specific details are intentionally excluded.

## Objective

Build an offline, review-oriented pipeline that converts authorized workstation video into traceable pose and temporal evidence without treating model output as verified process truth.

## Implemented baseline

- Local video decoding and provenance capture.
- YOLOv8n-Pose inference over COCO-17 body keypoints.
- Anonymous person selection and lock state; no identity recognition.
- Visibility-aware pose records and derived motion segments.
- Evidence-gated action candidates and a local browser review interface.
- Explicit unavailable states for object perception, interaction fusion, temporal models, and process reasoning when their required evidence is absent.

The architecture preserves a strict distinction between raw observations, derived signals, temporal interpretation, and higher-level process reasoning.

## Validation scope

The internship validation used private reference recordings and generated evidence that cannot be included in the public portfolio repository. The formal internship report records:

- 93.41% full-video availability for both COCO body-pose wrist keypoints.
- 71/71 Factory AI Camera tests passed.
- 224/224 focused pose tests passed.

Dual-wrist availability is a keypoint-visibility result. It is not hand-pose, grasp, object-contact, or action-recognition accuracy.

Publicly reproducible unit and contract tests are separated from regression checks that require private video, hardware, external source snapshots, or generated evidence. See [docs/PUBLIC_TESTING.md](docs/PUBLIC_TESTING.md).

## Data governance and limitations

- Private factory video and generated analyses are not distributed.
- Recording groups are represented only by public-safe aliases.
- Random frame splits across related recordings are prohibited.
- Model output remains candidate evidence until a human reviewer confirms it.
- Evaluation without independent labels remains `not_evaluable`.
- Production and external-factory validation flags remain false.

## Public release note

Source-lineage manifests in this repository retain only public target paths and high-level adaptation intent. Original source names, absolute paths, source hashes, timestamps, and binary-model fingerprints were removed because they are unnecessary for portfolio review.
