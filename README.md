# Factory Multimodal Action Process Lab

**Industrial Computer Vision / Human Action Analysis**

An offline Python prototype that turns workstation video into traceable pose and temporal evidence for human action review. I built the local pipeline around YOLOv8n-Pose, ONNX Runtime, OpenCV, anonymous person tracking, evidence-gated action segments, and a browser-based review interface. The system separates direct observations from derived signals and returns `unavailable` when an object, interaction, or process conclusion is not supported.

> This is a sanitized public portfolio version of a 2026 manufacturing AI internship project. It is a review-oriented prototype, not a production deployment or an automated process-decision system.

## Highlights

- Processes local workstation video offline; uploaded video is stored only under the repository's ignored `outputs/` directory.
- Preserves provenance with video hashes, model manifests, anonymous person references, and lock epochs.
- Distinguishes raw COCO-17 body-pose observations, derived motion/action signals, optional hand-landmark evidence, and higher-level reasoning.
- Fails closed: unconfigured object perception, interaction fusion, temporal-action models, and process reasoning do not produce substitute results.
- Exposes analysis, pose overlays, timelines, warnings, and evidence status in a local web interface.

### Verified internship validation

The following figures come from the formal internship report and describe the validated internship system, not a benchmark rerun performed from this public clone:

- **93.41% full-video dual-wrist availability** for the COCO body-pose wrist keypoints.
- **71/71 Factory AI Camera tests passed.**
- **224/224 focused pose tests passed.**

Dual-wrist availability is a keypoint-visibility measure. It is not detailed hand-pose, grasp, object-contact, or action-recognition accuracy.

## Tech stack

- **Language:** Python 3.12
- **Computer vision / ML:** YOLOv8n-Pose, ONNX Runtime, OpenCV, MediaPipe Hand Landmarker
- **Data and computation:** NumPy, JSON, SHA-256 provenance records
- **Application:** local HTML/CSS/JavaScript review UI served by Python
- **Engineering:** pytest, JSON Schema validation

## Evidence workflow

```text
Local video
  -> decode and SHA-256 provenance
  -> YOLOv8n-Pose inference (COCO-17 keypoints)
  -> anonymous candidate selection and person lock
  -> pose observations and visibility gates
  -> derived pose segments and stable action events
  -> optional MediaPipe hand-landmark evidence
  -> object perception       [not configured]
  -> interaction fusion      [unavailable without evidence]
  -> temporal/process models [unavailable without evidence]
  -> local review UI and JSON evidence outputs
```

The important boundary is deliberate:

1. **Raw observations** record model keypoints and visibility.
2. **Derived signals** summarize body motion and evidence-gated segments.
3. **Temporal interpretation** describes candidate action events across frames.
4. **Process reasoning** remains unavailable unless the required upstream evidence and models exist.

## Repository structure

| Path | Purpose |
| --- | --- |
| `src/pose_core/` | ONNX pose runtime and provider policy |
| `src/tracking/` | Anonymous person selection and lock state |
| `src/action_segmentation/` | Pose-derived segments and action-event logic |
| `src/hand_pose/` | Optional MediaPipe hand-landmark layer |
| `src/object_perception/` | Object-perception contract and unavailable state |
| `src/interaction_fusion/` | Evidence fusion boundary |
| `src/temporal_actions/` | Temporal-action contract and unavailable state |
| `src/process_reasoning/` | Higher-level process reasoning boundary |
| `src/web/` | Local review application and controlled video intake |
| `configs/`, `schemas/` | Runtime configuration and output contracts |
| `scripts/` | Analysis, validation, and bounded diagnostic commands |
| `tests/` | Unit, contract, camera, and artifact-dependent validation tests |
| `docs/` | Technical decisions and historical validation records |

## Install

The project was developed with Python 3.12 on Windows. The pinned ONNX Runtime package targets a CUDA-capable environment, while provider selection can fall back to CPU when available.

```powershell
git clone https://github.com/Aaalexy/factory-multimodal-action-process-lab.git
cd factory-multimodal-action-process-lab
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Two external model artifacts are referenced by the default configuration:

- `models/yolov8n-pose.onnx`
- `models/hand_pose/hand_landmarker.task`

The binaries are intentionally not included because this portfolio repository does not establish redistribution authorization for both artifacts. See [`models/README.md`](models/README.md) for the expected interfaces, placement, and the recorded MediaPipe source references. The body-pose model is required; the optional hand layer can be disabled with `--disable-hand`.

## Run a local analysis

Use only video that you are authorized to process. The input may be outside the repository; the run creates ignored output artifacts under `outputs/`.

```powershell
python scripts/run_baseline.py `
  --video "path\to\sanitized_demo.mp4" `
  --duration 12 `
  --sample-fps 2 `
  --recording-group-id sanitized_demo_group
```

The default run writes `analysis.json`, provenance/evaluation manifests, a review queue, a pose contact sheet, and a local source-video copy to `outputs/baseline_run/`.

Start the local review interface:

```powershell
python -m src.web.app `
  --analysis outputs/baseline_run/analysis.json `
  --host 127.0.0.1 `
  --port 8765
```

Then open `http://127.0.0.1:8765`.

## Validation

Public/core checks that require only source-controlled code and synthetic fixtures:

```powershell
python scripts/check_workspace.py
python scripts/verify_source_manifest.py --target-only
python -m compileall src scripts
python -m pytest tests -m "not private_artifacts" -q
```

After producing a local analysis, validate its schemas and evaluation record:

```powershell
python scripts/validate_schemas.py --analysis outputs/baseline_run/analysis.json
python scripts/evaluate.py --analysis outputs/baseline_run/analysis.json
```

Tests marked `private_artifacts` preserve regression assertions that require excluded video, generated evidence, hardware, external source snapshots, or model binaries. They are explicitly deselected by the public command; see [`docs/PUBLIC_TESTING.md`](docs/PUBLIC_TESTING.md). Evaluation without independent human ground truth must remain `not_evaluable`.

## Safety and limitations

- COCO wrist keypoints are body-pose observations, not detailed hand or grasp detection.
- MediaPipe hand landmarks are optional supporting evidence and do not establish object contact or process correctness.
- Person references are anonymous tracking identifiers; the project does not perform identity recognition.
- Object perception is currently `not_configured`; interaction, temporal-model, and process-reasoning outputs remain unavailable when evidence is absent.
- Results are candidate evidence for human review, not production decisions.
- Current configuration keeps `factory_camera_validated`, `production_action_model_ready`, `external_factory_validated`, and `production_process_model_ready` set to `false`.
- No confidential factory video or private internship dataset is included in this repository.

## Demo / screenshots

No public-safe screenshot or demo video is currently included. Future sanitized assets should be limited to:

1. A local input-and-status view using newly recorded non-factory footage.
2. A pose overlay showing anonymous tracking and visible COCO-17 keypoints.
3. An action timeline linked to the evidence and unavailable-state panel.

Do not show a real facility, employee identity, workstation record, or internal filename.

## Public portfolio note

The README is the English entry point; historical engineering summaries are sanitized for public review. This repository intentionally has no open-source license, and public visibility should not be interpreted as permission to reuse or redistribute the code. Model binaries are not included.
