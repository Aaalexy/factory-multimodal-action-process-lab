"""Validate project schemas and any generated baseline records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.schema_validation import validate_instance  # noqa: E402


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_all(analysis_path: Path | None = None) -> dict[str, object]:
    schema_root = PROJECT_ROOT / "schemas"
    schemas = {
        path.stem.replace(".schema", ""): _load(path)
        for path in sorted(schema_root.glob("*.schema.json"))
    }
    checks: list[str] = []
    registry = PROJECT_ROOT / "configs" / "recording_group_registry.json"
    validate_instance(_load(registry), schemas["recording_group_registry"])
    checks.append("recording_group_registry")

    if analysis_path is not None:
        analysis = _load(analysis_path)
        mapping = {
            "pose_segments": analysis.get("pose_segments", []),
            "action_events": analysis.get("action_events", []),
            "evidence_timeline": analysis.get("evidence_timeline", []),
            "hand_pose_frames": analysis.get("hand_pose_frames", []),
            "object_tracks": analysis.get("object_tracks", []),
            "interaction_events": analysis.get("interaction_events", []),
            "process_steps": analysis.get("process_steps", []),
        }
        for name, records in mapping.items():
            for index, record in enumerate(records):
                validate_instance(record, schemas[name], path=f"{name}[{index}]")
            checks.append(f"{name}:{len(records)}")
        output_dir = analysis_path.parent
        artifact_mapping = {
            "dataset_manifest": output_dir / "dataset_manifest.json",
            "model_evaluation_manifest": output_dir
            / "model_evaluation_manifest.json",
            "process_review_queue": output_dir / "process_review_queue.json",
        }
        for name, path in artifact_mapping.items():
            validate_instance(_load(path), schemas[name], path=name)
            checks.append(name)
    return {
        "status": "passed",
        "schema_file_count": len(schemas),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path)
    args = parser.parse_args()
    analysis = args.analysis.resolve() if args.analysis else None
    print(json.dumps(validate_all(analysis), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
