# Reference Recording Analysis — Public Summary

The original read-only audit used a privately held reference video. The public repository calls it `reference_clip_01`; the file, original name, fingerprint, local path, contact sheet, and generated metadata are intentionally excluded.

## What was checked

- The container could be opened and representative frames could be decoded.
- The scene contained a workstation view suitable for pose-pipeline feasibility checks.
- Multiple-person and partial-visibility conditions were considered when defining anonymous person-lock behavior.
- Sparse visual inspection was kept separate from pose inference and from process interpretation.

## Evidence boundary

Directly observable image content can support body-location and visibility review. It does not by itself establish:

- detailed hand pose or grasp state;
- object identity or contact;
- operation correctness;
- process completion; or
- production readiness.

The public implementation therefore records raw pose observations first, derives motion signals second, and leaves unsupported object, interaction, temporal-model, and process conclusions unavailable.

## Public reproduction

Use an authorized, sanitized video and the commands in the main README. Generated outputs are written beneath ignored `outputs/` paths. Do not publish real facility imagery, employee identity, internal filenames, or workstation records.
