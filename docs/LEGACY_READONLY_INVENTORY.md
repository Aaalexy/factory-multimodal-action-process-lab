# Prior Pose Baseline Inventory — Public Summary

A prior project-owned pose implementation was inspected read-only before the current package was assembled. The original workspace name, file inventory, hashes, model binary, private media, generated outputs, and local paths are intentionally not published.

## Retained concepts

- Local video decoding and bounded frame sampling.
- YOLOv8 pose inference with explicit provider status.
- Anonymous candidate selection, person locking, and relock boundaries.
- Visibility validation before smoothing or temporal derivation.
- Explicit `detected`, `predicted`, `interpolated`, `uncertain`, `missing`, and `lost` states.
- Pose-derived segments and stable candidate events that retain source evidence.
- Local HTTP range playback and a small offline worker boundary.

## Excluded concepts and artifacts

- Private video, output folders, contact sheets, and evaluation artifacts.
- Model binaries without a complete public redistribution record.
- Training, automatic labeling, or automatic confirmation.
- Identity recognition or employee identifiers.
- Prior monolithic web applications and unrelated integrations.
- Environments, caches, build products, executables, and archives.

The target-only public lineage inventory is [`../SOURCE_IMPORT_MANIFEST.json`](../SOURCE_IMPORT_MANIFEST.json). It records public file paths and adaptation status without exposing the original source layout.
