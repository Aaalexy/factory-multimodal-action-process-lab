# Private Short-Clip Intake Audit — Public Summary

This document preserves the validation method without exposing the original clip names, recording identifiers, hashes, sizes, or contact sheets.

## Method

- Opened 15 private MP4 clips with OpenCV.
- Decoded one frame near 15%, 50%, and 85% of each clip.
- Successfully decoded 45/45 requested sparse frames.
- Did not scan every frame, run pose inference, train a model, or infer process correctness during this intake check.

## Public aliases and split policy

The original files are represented only by generic aliases such as `sample_video_A`, `sample_video_B`, and `sample_video_C`. Related views belong to the same recording group and must not be split randomly across train, validation, or test sets. The available independent groups were insufficient for a defensible held-out test split, so the split status remains `not_available`.

## Observational limits

Sparse frames can support only coarse scene and visibility observations. They cannot confirm object identity, control interfaces, tool use, grasp, operation order, or task completion. Any future public demo should use newly created sanitized footage and a separate recording group.

## Excluded artifacts

Private videos, source fingerprints, local paths, contact sheets, and generated scan metadata are not part of the public repository.
