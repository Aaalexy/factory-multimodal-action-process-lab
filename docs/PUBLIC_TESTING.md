# Public and Artifact-Dependent Tests

The test collection contains two evidence levels.

## Public/core tests

These tests use source-controlled code, configuration, schemas, and synthetic fixtures only:

```powershell
python -m pytest tests -m "not private_artifacts" -q
```

This is the public portfolio test command. It should pass in a source-only clone after the development dependencies are installed.

## Private/artifact-dependent regression tests

Tests marked `private_artifacts` preserve assertions that require one or more of the following:

- private internship video;
- generated analysis or replay evidence beneath ignored `outputs/` paths;
- an external read-only source snapshot;
- USB camera hardware;
- externally supplied body-pose or hand-landmark model files.

Private frozen-analysis integrity checks also expect an ignored
`outputs/private_regression/fixture_manifest.json` with
`analyses.<public-safe-alias>.sha256` entries. The source-specific hashes remain
with the authorized private fixtures and are not published in this repository.

Run them only in an authorized environment that has those dependencies:

```powershell
python -m pytest tests -m private_artifacts -q
```

Absence of those artifacts is an expected public-release boundary, not evidence that the underlying assertions passed. Do not report the private command as passing unless it was actually run with the required authorized inputs.
