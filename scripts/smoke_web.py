"""Bounded localhost smoke check for status and byte-range video endpoints."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.web.app import create_server  # noqa: E402


def smoke(analysis_path: Path) -> dict[str, object]:
    server = create_server(analysis_path, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        status = json.loads(
            urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/status",
                timeout=5,
            ).read()
        )
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/media/video",
            headers={"Range": "bytes=0-1023"},
        )
        response = urllib.request.urlopen(request, timeout=5)
        chunk = response.read()
        return {
            "status": "passed",
            "api_schema_version": status["schema_version"],
            "pose_frames": status["counts"]["pose_frames"],
            "action_events": status["counts"]["action_events"],
            "object_tracks": status["counts"]["object_tracks"],
            "process_steps": status["counts"]["process_steps"],
            "range_status": response.status,
            "range_bytes": len(chunk),
            "accept_ranges": response.headers.get("Accept-Ranges"),
            "content_range": response.headers.get("Content-Range"),
            "all_validation_flags_false": not any(
                status["validation_flags"].values()
            ),
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(smoke(args.analysis.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
