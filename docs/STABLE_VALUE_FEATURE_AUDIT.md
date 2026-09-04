# Prior-Implementation Feature Audit — Public Summary

A prior project-owned implementation was inspected read-only to identify reusable concepts. Its workspace name, file fingerprints, run identifiers, and local paths are intentionally omitted.

## Adapted concepts

| Capability | Public-project decision |
| --- | --- |
| MP4 intake | Streamed, size-limited local intake with atomic temporary-file promotion |
| Video probe | Reuse the current metadata and decodability checks |
| Bounded analysis | Keep a bounded window as the default; require explicit full-video selection |
| Anonymous person selection | Reuse the current candidate and manual-lock contracts |
| Offline worker | Keep bounded progress, cancellation, and safe error reporting |
| Review playback | Link the source video, pose overlay, and evidence timeline |
| Camera relock | Use opaque candidate identifiers scoped to one local session |
| Runtime diagnostics | Expose provider, frame-rate, latency, drop, and framing state |
| Human review | Keep model events unconfirmed until a reviewer acts |

## Explicit exclusions

The audit did not import a prior web application, training workflow, LLM workflow, recording pipeline, packaged executable, environment, cache, output directory, or production language. It also did not add identity inference, automated confirmation, automatic training eligibility, RTSP control, or external-system actions.

MediaPipe hand inference remains an optional CPU evidence layer in the documented Windows setup. The public repository does not claim GPU hand inference or production validation.
