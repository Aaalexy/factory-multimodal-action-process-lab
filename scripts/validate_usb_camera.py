"""Bounded real USB Camera probe for the authorized Stage B validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.camera import CameraController
from src.web.resource_coordinator import AnalysisResourceCoordinator


EXPECTED_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_probe(output: Path) -> dict[str, Any]:
    root = Path.cwd().resolve()
    if str(root) != str(EXPECTED_ROOT):
        raise RuntimeError(f"Workspace gate mismatch: {root}")
    output.mkdir(parents=True, exist_ok=False)
    attempts: list[dict[str, Any]] = []
    selected: int | None = None
    live_result: dict[str, Any] | None = None
    for device_index in range(4):
        coordinator = AnalysisResourceCoordinator()
        controller = CameraController(root, coordinator)
        started_at = time.monotonic()
        try:
            started = controller.start(device_index=device_index)
            status = controller.wait_for_terminal_or_live(20.0)
            attempt = {
                "device_index": device_index,
                "start_state": started["state"],
                "terminal_state": status["state"],
                "worker_pid": status["worker_pid"],
                "elapsed_seconds": round(time.monotonic() - started_at, 6),
                "error": status["last_error"],
                "device_metadata": status["device_metadata"],
            }
            attempts.append(attempt)
            if status["state"] != "live":
                continue
            selected = device_index
            sequences: set[int] = set()
            deadline = time.monotonic() + 4.0
            last_evidence: dict[str, Any] | None = None
            while time.monotonic() < deadline:
                current = controller.status()
                if current["latest_sequence"] is not None:
                    sequences.add(int(current["latest_sequence"]))
                candidate = controller.latest_evidence()
                if candidate is not None:
                    last_evidence = candidate
                time.sleep(0.08)
            jpeg, sequence = controller.latest_jpeg()
            if jpeg is not None:
                still = output / "stage_b_real_camera_frame.jpg"
                still.write_bytes(jpeg)
            else:
                still = None
            stopped = controller.stop()
            video_lease = coordinator.acquire("video_analysis")
            coordinator.release(video_lease)

            reopened = controller.start(device_index=device_index)
            reopen_status = controller.wait_for_terminal_or_live(20.0)
            reopen_jpeg, _ = controller.latest_jpeg()
            reopen_stopped = controller.stop()
            live_result = {
                "selected_device_index": device_index,
                "unique_live_sequences": len(sequences),
                "last_sequence": sequence,
                "last_evidence": last_evidence,
                "still_path": (
                    still.relative_to(root).as_posix() if still else None
                ),
                "still_sha256": sha256(still) if still else None,
                "first_stop": stopped,
                "video_analysis_reacquired_after_stop": True,
                "reopen_start_state": reopened["state"],
                "reopen_terminal_state": reopen_status["state"],
                "reopen_real_frame_available": reopen_jpeg is not None,
                "reopen_stop": reopen_stopped,
            }
            break
        finally:
            controller.close()
    result = {
        "schema_version": "factory_usb_camera_real_validation_v1",
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(
            timespec="seconds"
        ),
        "project_path": str(root),
        "attempts": attempts,
        "hardware_status": (
            "passed_local_technical_validation"
            if selected is not None
            and live_result is not None
            and live_result["unique_live_sequences"] > 0
            and live_result["reopen_terminal_state"] == "live"
            and live_result["reopen_real_frame_available"]
            and live_result["first_stop"]["last_error"] is None
            and live_result["reopen_stop"]["last_error"] is None
            else "not_validated_no_available_device"
        ),
        "live_result": live_result,
        "recording_persisted": False,
        "rtsp_used": False,
        "factory_camera_validated": False,
        "production_action_model_ready": False,
        "external_factory_validated": False,
        "production_process_model_ready": False,
    }
    target = output / "STAGE_B_REAL_CAMERA_VALIDATION.json"
    target.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_probe(Path(args.output).resolve()),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
