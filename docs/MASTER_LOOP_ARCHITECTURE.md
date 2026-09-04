# Master Loop Architecture

## Purpose

The Master Loop makes engineering changes interruptible, attributable, and
recoverable without assuming a valid Git repository. It never broadens the
project's evidence or privacy permissions.

## State artifacts

- `MASTER_LOOP_STATE.json` is the single current recovery pointer. Exactly one
  phase may be `in_progress`.
- `MASTER_LOOP_LEDGER.jsonl` is append-only and records hypotheses, evidence,
  decisions, rejected attempts, and the next action.
- `outputs/master_loop/<run_id>/` stores immutable run-specific evidence.
- Each run contains `CHECKPOINT_MANIFEST.json`, whose file entries record
  pre-change hashes, recovery copies where applicable, post-change hashes,
  validation commands, and the accept/reject decision.
- `MASTER_TECHNICAL_RC_REPORT.md` is created and finalized only in Phase J.

Allowed phase states are `pending`, `in_progress`, `passed`, `blocked`, and
`not_evaluable`.

## Recovery protocol

1. Run the exact workspace gate and read `AGENTS.md`.
2. Read `MASTER_LOOP_STATE.json`.
3. Validate the last checkpoint's artifacts and post-change hashes.
4. Confirm the last test, compile, real-video, and browser evidence claimed by
   state actually exists.
5. Resume from `next_action`; never rerun an already passed full phase merely
   because a conversation was interrupted.
6. If a checkpoint is incomplete, inspect `.part` and temporary artifacts
   before any cleanup. Preserve valid outputs and use a new `run_id`.

## Change protocol without Git

Before modifying an existing file:

1. Calculate its SHA256.
2. Copy only that file into the current run's `recovery/` directory, preserving
   a workspace-relative path.
3. Record the original path, recovery path, and pre-change SHA256.
4. Apply the smallest attributable change.
5. Calculate the post-change SHA256.
6. Run focused tests, compileall, same-window real-video replay, and relevant
   browser/visual validation.
7. Accept the change only if all gates pass. If rejected, restore the recovery
   copy and record the rejection.

New files use `pre_change_state=absent`; large models, videos, `.venv`, caches,
and prior outputs are never duplicated into recovery storage.

## Iteration flow

```text
diagnose
  -> one attributable hypothesis
  -> pre-hash and minimal recovery copies
  -> minimum implementation
  -> focused tests
  -> compileall
  -> same real-video windows
  -> metric comparison
  -> browser/visual validation
  -> accept or restore/reject
  -> state + ledger + checkpoint
  -> next action
```

Each hypothesis has at most three attempts. Three unsuccessful attempts make
the hypothesis `blocked` or `rejected`; the last passed checkpoint is restored
and engineering moves to an independent phase.

## Truth and safety invariants

- Original Body/Hand/model evidence is immutable in review workflows.
- Automated suggestions remain `proposed` or `uncertain` and
  `training_eligible=false`.
- Object, interaction, temporal-model, and process layers fail closed when
  their real evidence is unavailable.
- No face, identity, clothing ReID, employee performance, device control, or
  production-verdict capability is introduced.
- No new model is downloaded or trained without later explicit authorization.
- The forbidden external project is never accessed.
- All four production validation flags remain `false`.
