# Factory Multimodal Action Process Lab Technical Definition of Done

## Release position

The target is **Factory Multimodal Action Process Lab Technical Release
Candidate**. It is an independently runnable technical candidate, not a
factory-production certification and not evidence of production acceptance or
rejection capability.

All four validation flags remain `false` until separate external factory
validation explicitly changes the governing evidence.

## Mandatory acceptance gates

A Technical RC requires all of the following:

1. The project runs without any runtime dependency on the old Pose project.
2. Real MP4 analysis is stable, including Unicode and space-containing paths.
3. Anonymous person detection, explicit lock state, lost state, and manual
   relock boundaries remain observable and tested.
4. Body Pose uses the real YOLO ONNX asset and never uses mock keypoints.
5. Hand Pose uses a real backend and distinguishes backend availability,
   observation quality, validation state, `detected`, `uncertain`, `missing`,
   and `lost`.
6. Missing or lost hand evidence contains no synthetic hand geometry.
7. `evidence_timeline` covers the complete analysis window within one sampling
   interval and exposes normal, transition, unknown, uncertain, and lost
   evidence rather than hiding it as blank space.
8. `action_events` contains only stable actions and explicit hard boundaries,
   cites `source_segment_ids`, and does not manufacture duration.
9. Normal stable actions do not create a large sub-one-second event tail.
10. Person, `person_ref`, `lock_epoch`, lost/off-frame, severe occlusion, long
    missing, and explicit person-switch boundaries are never crossed.
11. Brief low-quality evidence uses a bounded and traceable uncertain gap
    rather than automatically clearing all temporal context.
12. Left, right, and bilateral temporal state are isolated and side switching
    does not silently change identity or erase unrelated context.
13. The original Web UI synchronizes video, Body Pose, Hand Pose, evidence
    timeline, stable events, source evidence, and explicit unavailable states.
14. Browser automation and screenshots validate 1280×720 and 1920×1080,
    seeking, Range, event navigation, overlays, empty/error states, and common
    scaling without exposing raw tracebacks.
15. Human review explicitly supports confirm, reject, correct, and comment
    while preserving original model output and an append-only audit record.
16. Training approval is a separate explicit action; ordinary review never
    sets `training_eligible=true`.
17. Future factory users can configure versioned process templates, object
    classes, ROI definitions, evidence requirements, uncertainty behavior, and
    sensor requirements without implying that a model detects those classes.
18. With no real object model, object outputs remain `not_configured` or
    `unavailable`; no object boxes or tracks are fabricated.
19. With no real object tracks or process evidence, interaction and process
    outputs remain empty and `unavailable` or `not_observed`.
20. MP4, USB, RTSP, PLC, scanner, MES, fixture, and sensor adapters have typed
    interfaces and safe lifecycle behavior; unconnected adapters remain
    `not_validated` and are not exercised without authorization.
21. Installation, offline operation, recovery, configuration, testing,
    privacy, telemetry, and troubleshooting are documented and verified.
22. Performance evidence separates decode, Body Pose, Hand Pose, temporal
    reasoning, rendering, and write time and includes a bounded long-run check.
23. Logs and reports contain no credentials, identity inference, employee
    performance conclusions, fake production verdicts, or hidden model claims.
24. Phase-specific tests, schema checks, browser checks, real-video replay, and
    `python -m compileall src` pass.
25. A full pytest run is performed exactly at the RC candidate phase, with all
    failures and skips reported rather than hidden.
26. Model, configuration, source-video, output, review, and checkpoint
    provenance is traceable through SHA256 and version fields.
27. The four validation flags remain `false`.

## Phase completion matrix

| Phase | Completion evidence | Initial state |
|---|---|---|
| Phase 0 | Baseline audit, frozen hashes, state/ledger/checkpoint, this DoD | in_progress |
| Phase B.1 | Full evidence timeline, bounded uncertain gaps, side-isolated temporal state, same-window A/B | pending |
| Phase B.2 | Hand quality states, association A/B, truthful UI, no synthetic geometry | pending |
| Phase C | Explainable Temporal Action Engine V3 and unavailable learned-model adapter | pending |
| Phase D | Original Web product UI, explicit human review, browser screenshots/automation | pending |
| Phase E | Object detector/tracker/config/manifest/import/evaluation interfaces | pending |
| Phase F | Evidence-gated `derived_interaction_candidate` fusion | pending |
| Phase G | Versioned configurable process templates and reviewable candidate state machine | pending |
| Phase H | Safe MP4/device/external-event adapter interfaces, no unauthorized connection | pending |
| Phase I | Performance, resilience, privacy, offline and telemetry validation | pending |
| Phase J | Integrated Technical RC evidence, full pytest and final report | pending |

## Per-iteration acceptance rule

A change is accepted only when its focused tests and compileall pass, the same
real-video window remains functional, truth and identity boundaries remain
intact, metrics have no unexplained regression, and the output cites video,
configuration, and model versions. Otherwise it is rejected and the
pre-change recovery copy is restored.

Accuracy metrics remain `not_evaluable` without independently human-confirmed
ground truth. Lower event count, higher suppression, or longer events are not
accepted as accuracy improvements by themselves.
