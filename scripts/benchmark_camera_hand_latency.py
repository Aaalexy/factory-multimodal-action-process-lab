"""Run bounded real USB Camera Hand ON/OFF measurements without recording."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.camera import CameraConfig, CameraController
from src.web.resource_coordinator import AnalysisResourceCoordinator


EXPECTED_ROOT = PROJECT_ROOT


def _percentile(values: list[float], fraction: float) -> float | None:
    ordered = sorted(
        float(value) for value in values if math.isfinite(float(value))
    )
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return round(ordered[low], 6)
    weight = position - low
    return round(
        ordered[low] * (1.0 - weight) + ordered[high] * weight,
        6,
    )


def _run_one(
    *,
    root: Path,
    config_path: Path,
    duration_seconds: float,
) -> dict[str, Any]:
    coordinator = AnalysisResourceCoordinator()
    controller = CameraController(
        root,
        coordinator,
        config_path=config_path,
    )
    started = time.monotonic()
    unique_sequences: set[int] = set()
    records_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    receipt_ages_ms: list[float] = []
    try:
        start_status = controller.start(device_index=0)
        live_status = controller.wait_for_terminal_or_live(90.0)
        cold_start_seconds = time.monotonic() - started
        if live_status["state"] != "live":
            return {
                "status": "blocked",
                "start_status": start_status,
                "terminal_status": live_status,
                "cold_start_seconds": round(cold_start_seconds, 6),
            }
        deadline = time.monotonic() + float(duration_seconds)
        while time.monotonic() < deadline:
            packet = controller.latest_packet_after(
                max(unique_sequences, default=0)
            )
            if packet is not None:
                sequence = int(packet["sequence"])
                unique_sequences.add(sequence)
                captured_at = packet.get("captured_at_monotonic")
                if isinstance(captured_at, (int, float)):
                    receipt_ages_ms.append(
                        max(0.0, (time.monotonic() - float(captured_at)) * 1000)
                    )
                for record in (
                    (packet.get("evidence") or {}).get(
                        "hand_pose_frames", []
                    )
                ):
                    key = (
                        int(record.get("frame_index", -1)),
                        str(record.get("anatomical_side", "")),
                    )
                    records_by_key[key] = record
            time.sleep(0.02)
        live_end = controller.status()
        stop_status = controller.stop()
        lease = coordinator.acquire("video_analysis")
        coordinator.release(lease)
    finally:
        controller.close()

    records = list(records_by_key.values())
    failures: Counter[str] = Counter()
    warnings: Counter[str] = Counter()
    geometry_frames: set[int] = set()
    inference_calls = 0
    for record in records:
        reason = str(record.get("reason", ""))
        state = str(record.get("observation_state", "missing")).lower()
        landmarks = list(record.get("landmarks") or [])
        if state == "lost":
            category = "lost_or_person_boundary"
        elif reason == "body_tracking_not_reliable_for_hand_roi":
            category = "body_tracking_not_reliable"
        elif record.get("crop_bbox") is None:
            category = "body_guided_roi_unavailable"
        elif str(record.get("backend_state", "")).lower() == "error":
            category = "backend_error"
        elif len(landmarks) == 21 and state == "uncertain":
            category = "real_21_points_association_downgraded"
        elif len(landmarks) == 21:
            category = "real_21_points_qualified_or_detected"
        elif reason == "no_hand_detected_in_body_guided_roi":
            category = "roi_available_model_no_output"
        else:
            category = "other_no_geometry"
        failures[category] += 1
        if isinstance(record.get("inference_time_ms"), (int, float)):
            inference_calls += 1
        warnings.update(
            str(value)
            for value in (
                (record.get("association_checks") or {}).get("warnings", [])
            )
        )
        if len(landmarks) == 21:
            geometry_frames.add(int(record.get("frame_index", -1)))
    metrics = dict(live_end.get("metrics") or {})
    return {
        "status": "completed",
        "config": json.loads(config_path.read_text(encoding="utf-8")),
        "device_metadata": live_end.get("device_metadata"),
        "cold_start_seconds": round(cold_start_seconds, 6),
        "steady_state_seconds": float(duration_seconds),
        "unique_observed_sequence_count": len(unique_sequences),
        "hand_record_count": len(records),
        "hand_inference_call_count_from_records": inference_calls,
        "real_21_point_frame_count": len(geometry_frames),
        "failure_reason_counts": dict(sorted(failures.items())),
        "association_warning_counts": dict(sorted(warnings.items())),
        "receipt_age_mean_ms": (
            round(statistics.fmean(receipt_ages_ms), 6)
            if receipt_ages_ms else None
        ),
        "receipt_age_p50_ms": _percentile(receipt_ages_ms, 0.5),
        "receipt_age_p95_ms": _percentile(receipt_ages_ms, 0.95),
        "receipt_age_max_ms": (
            round(max(receipt_ages_ms), 6) if receipt_ages_ms else None
        ),
        "metrics": metrics,
        "stop_status": stop_status,
        "video_resource_reacquired_after_stop": True,
        "recording_persisted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--duration-seconds", type=float, default=15.0)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    if root != EXPECTED_ROOT:
        raise RuntimeError(f"Workspace gate mismatch: {root}")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    base = CameraConfig.load(root / "configs/camera.json")
    runs: list[dict[str, Any]] = []
    for hand_enabled in (False, True):
        values = asdict(base)
        values["hand_enabled"] = hand_enabled
        values.update(
            {
                "factory_camera_validated": False,
                "production_action_model_ready": False,
                "external_factory_validated": False,
                "production_process_model_ready": False,
            }
        )
        config_path = output.parent / (
            "camera_baseline_hand_on.json"
            if hand_enabled else "camera_baseline_hand_off.json"
        )
        config_path.write_text(
            json.dumps(values, indent=2) + "\n",
            encoding="utf-8",
        )
        runs.append(
            {
                "hand_enabled": hand_enabled,
                "result": _run_one(
                    root=root,
                    config_path=config_path,
                    duration_seconds=args.duration_seconds,
                ),
            }
        )
    payload = {
        "schema_version": "factory_camera_hand_on_off_baseline_v1",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "duration_seconds_per_run": args.duration_seconds,
        "runs": runs,
        "recording_persisted": False,
        "rtsp_used": False,
        "validation_flags": {
            "factory_camera_validated": False,
            "production_action_model_ready": False,
            "external_factory_validated": False,
            "production_process_model_ready": False,
        },
    }
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
