"""Evaluation entry point; refuses to fabricate metrics without human truth."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import METRIC_DEFINITIONS, not_evaluable_manifest  # noqa: E402


def evaluate(analysis_path: Path, ground_truth_path: Path | None) -> dict[str, object]:
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if ground_truth_path is None:
        return {
            "analysis": str(analysis_path),
            **not_evaluable_manifest(
                "No human-confirmed ground-truth manifest was supplied"
            ),
        }
    truth = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    if not truth.get("human_review_complete"):
        return {
            "analysis": str(analysis_path),
            "ground_truth": str(ground_truth_path),
            **not_evaluable_manifest(
                "Ground truth lacks an explicit human_review_complete record"
            ),
        }
    return {
        "status": "not_evaluable",
        "reason": (
            "Metric adapters are defined but calibrated temporal matching requires "
            "the next-phase reviewed annotation format"
        ),
        "available_prediction_counts": {
            "action_events": len(analysis.get("action_events", [])),
            "object_tracks": len(analysis.get("object_tracks", [])),
            "interaction_events": len(analysis.get("interaction_events", [])),
            "process_steps": len(analysis.get("process_steps", [])),
        },
        "metric_definitions": METRIC_DEFINITIONS,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.analysis.resolve(), args.ground_truth)
    body = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body, encoding="utf-8")
    print(body)


if __name__ == "__main__":
    main()
