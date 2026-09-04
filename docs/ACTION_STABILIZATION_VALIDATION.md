# Action Stabilization Validation — Public Summary

This summary preserves the technical intent of the private regression work without publishing source-video identifiers, generated artifact paths, or file fingerprints.

## Change under test

The Phase B stabilization layer adds temporal confirmation, short-gap merging, visibility gates, identity boundaries, and minimum-duration rules on top of frame-level pose-derived motion candidates. It does not add object perception or convert candidates into ground-truth process labels.

## Private regression design

Three authorized reference windows—`sample_video_A`, `sample_video_B`, and `sample_video_C`—were replayed before and after the stabilization change. Related recordings remained in one recording group to prevent split leakage.

The regression checked:

- stable events do not cross `person_ref` or `lock_epoch` boundaries;
- missing and uncertain evidence cannot become confirmed geometry;
- short event fragments are not promoted solely because frame-level direction changes;
- event timing remains tied to the source analysis window; and
- unsupported object, interaction, and process layers remain unavailable.

## Result boundary

The engineering comparison showed improved temporal coverage and fewer short fragments on the private windows. Those observations are not an accuracy benchmark because no independent event-level ground truth was available. Training eligibility and production-validation flags therefore remained false.

The source recordings and generated before/after analyses are excluded from the public repository. Their regression assertions remain under the `private_artifacts` pytest marker.
