# Factory Multimodal Action Process Lab boundaries

- The only writable project workspace is
  `<repository-root>`.
- Before project work, `Get-Location` and `Resolve-Path -LiteralPath .` must
  both equal that path exactly. Stop on any mismatch.
- The Pose-to-Stickman project and all supplied videos are read-only sources.
  Never modify, move, delete, overwrite, or reverse-sync them.
- `<forbidden-unrelated-workspace>` is forbidden: do not read,
  write, enumerate, test, or use it as a fallback.
- This project may produce anonymous pose actions, technical object tracks,
  `derived_interaction_candidate` evidence, and proposed/uncertain process-step
  candidates. These layers must remain separately attributable.
- Never implement face recognition, names, employee IDs, clothing ReID,
  identity inference, or employee-performance conclusions.
- COCO-17 does not provide palms, fingers, joints of the hand, true grasp, or
  object segmentation. Wrist/object proximity is derived evidence only.
- Never create mock keypoints, preset actions, fixed missing skeletons, fake
  object boxes, fake process steps, or relabel predicted/interpolated evidence
  as detected.
- A person, `person_ref`, or `lock_epoch` boundary, lost/off-frame state,
  severe occlusion, or long missing interval is a hard action-event boundary.
- Automatic or model-assisted semantics default to `proposed` or `uncertain`
  and `training_eligible=false`. Only an explicit human review record may
  authorize training truth.
- Until independent factory validation is complete, never emit production
  acceptance/rejection verdicts or claim production qualification.
- Keep all four flags false:
  `factory_camera_validated`, `production_action_model_ready`,
  `external_factory_validated`, and `production_process_model_ready`.
- Do not call DeepSeek from the frame pipeline. Do not train a model unless a
  later user request explicitly authorizes a training phase.
- Do not run USB or RTSP sources during this kickoff validation.
- Do not automatically commit or push. Generated outputs remain local unless
  the user explicitly requests otherwise.
