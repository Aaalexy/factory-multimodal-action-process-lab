"""Fail-closed checks for generated Phase B evidence artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.provenance import sha256_file  # noqa: E402


NORMAL_ACTIONS = {
    "idle",
    "reach",
    "retract",
    "lift",
    "lower",
    "move",
    "carry",
    "place",
    "hold",
    "release",
    "rotate",
    "push",
    "pull",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify(validation_root: Path) -> dict[str, Any]:
    root = validation_root.resolve()
    comparison = _load(root / "comparison.json")
    hand_manifest = _load(PROJECT_ROOT / "HAND_MODEL_MANIFEST.json")
    model_record = hand_manifest["model"]
    hand_model = PROJECT_ROOT / model_record["local_path"]
    checks: list[str] = []

    assert hand_model.is_file()
    assert hand_model.stat().st_size == int(model_record["size_bytes"])
    assert sha256_file(hand_model) == model_record["sha256"]
    checks.append("hand_model_size_and_sha256")

    assert comparison["evaluation"]["status"] == "not_evaluable"
    after_summary = comparison["summary"]["after"]
    assert after_summary["sub_1s_stable_event_count"] == 0
    assert after_summary["lost_normal_action_false_positive_count"] == 0
    assert after_summary["cross_identity_or_epoch_merge_count"] == 0
    checks.append("comparison_truthfulness_and_phase_b_targets")

    clip_results: list[dict[str, Any]] = []
    for clip in comparison["clips"]:
        before = _load(Path(clip["paths"]["before"]))
        after = _load(Path(clip["paths"]["after"]))
        assert before["source_video"]["sha256"] == after["source_video"]["sha256"]
        assert not any(after["validation_flags"].values())
        assert after["evaluation"]["status"] == "not_evaluable"
        assert after["runtime"]["mock_keypoints_used"] is False
        assert after["runtime"]["mock_hand_landmarks_used"] is False
        assert after["runtime"]["preset_actions_used"] is False
        assert after["runtime"]["model_training_performed"] is False
        assert after["runtime"]["deepseek_called"] is False
        assert after["runtime"]["usb_camera_used"] is False
        assert after["runtime"]["rtsp_used"] is False

        hand_records = after.get("hand_pose_frames", [])
        for record in hand_records:
            assert record["training_eligible"] is False
            assert record["status"] in {"proposed", "uncertain"}
            assert record["anatomical_side"] in {"left", "right"}
            if record["observation_state"] in {"missing", "lost"}:
                assert record["landmarks"] == []
                assert record["landmark_count"] == 0
            if record["observation_state"] == "detected":
                assert len(record["landmarks"]) == 21
                assert record["landmark_count"] == 21

        for event in after.get("action_events", []):
            assert event["source_segment_ids"]
            assert event["training_eligible"] is False
            if event.get("action") in NORMAL_ACTIONS:
                assert float(event["duration_seconds"]) >= 1.0 - 1e-9

        assert after.get("object_tracks", []) == []
        assert after.get("interaction_events", []) == []
        assert after.get("process_steps", []) == []
        clip_results.append(
            {
                "clip_id": clip["clip_id"],
                "source_hash_match": True,
                "hand_record_count": len(hand_records),
                "stable_event_count": len(after.get("action_events", [])),
                "sub_1s_stable_event_count": after["runtime"][
                    "sub_1s_stable_event_count"
                ],
            }
        )
    checks.append(f"clip_artifact_contracts:{len(clip_results)}")
    return {
        "schema_version": "phase_b_output_verification_v1",
        "status": "passed",
        "check_count": len(checks),
        "checks": checks,
        "clips": clip_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation-root",
        default="outputs/phase_b_validation",
        type=Path,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.validation_root)
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".part")
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
