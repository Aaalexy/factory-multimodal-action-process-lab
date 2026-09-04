"""CLI for the real-video, real-ONNX kickoff baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.multimodal_pipeline import BaselineConfig, run_baseline  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", default="models/yolov8n-pose.onnx")
    parser.add_argument(
        "--hand-model",
        default="models/hand_pose/hand_landmarker.task",
    )
    parser.add_argument("--output", default="outputs/baseline_run")
    parser.add_argument("--sample-fps", type=float, default=8.0)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument(
        "--provider",
        action="append",
        dest="providers",
        help="ONNX Runtime provider; may be repeated",
    )
    parser.add_argument(
        "--recording-group-id",
        default="recording_group_unassigned",
    )
    parser.add_argument(
        "--action-profile",
        choices=("phase_a", "phase_b"),
        default="phase_b",
    )
    parser.add_argument(
        "--disable-hand",
        action="store_true",
        help="Keep the hand layer unavailable without blocking Body Pose.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_baseline(
        BaselineConfig(
            project_root=str(PROJECT_ROOT),
            source_video=args.video,
            model_path=args.model,
            hand_model_path=args.hand_model,
            output_dir=args.output,
            sample_fps=args.sample_fps,
            start_time=args.start,
            duration_seconds=args.duration,
            providers=tuple(args.providers) if args.providers else None,
            recording_group_id=args.recording_group_id,
            hand_enabled=not args.disable_hand,
            action_profile=args.action_profile,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
