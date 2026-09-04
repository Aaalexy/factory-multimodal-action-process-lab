from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.build_master_loop_file_manifest import (
    PROJECT_ROOT,
    build_manifest,
    write_manifest,
)


def test_manifest_is_deterministic_sorted_and_excludes_transient_files(
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "outputs").mkdir()
    (tmp_path / "models").mkdir()
    (tmp_path / "AGENTS.md").write_text("boundary", encoding="utf-8")
    (tmp_path / "src" / "z.py").write_text("z = 1\n", encoding="utf-8")
    (tmp_path / "src" / "a.py").write_text("a = 1\n", encoding="utf-8")
    (tmp_path / "src" / "partial.part").write_bytes(b"incomplete")
    (tmp_path / "outputs" / "prior.json").write_text("{}", encoding="utf-8")
    (tmp_path / "models" / "model.bin").write_bytes(b"real-model")

    first = build_manifest(
        tmp_path,
        run_id="deterministic",
        generated_at="2026-07-27T00:00:00+08:00",
    )
    second = build_manifest(
        tmp_path,
        run_id="deterministic",
        generated_at="2026-07-27T00:00:00+08:00",
    )

    assert first == second
    paths = [entry["path"] for entry in first["files"]]
    assert paths == sorted(paths)
    assert paths == ["AGENTS.md", "models/model.bin", "src/a.py", "src/z.py"]
    assert first["files"][1]["sha256"] == hashlib.sha256(b"real-model").hexdigest()

    output = tmp_path / "manifest.json"
    write_manifest(output, first)
    assert json.loads(output.read_text(encoding="utf-8")) == first


def test_repository_manifest_covers_public_runtime_configuration_and_tests() -> None:
    manifest = build_manifest(
        PROJECT_ROOT,
        run_id="repository-coverage-test",
        generated_at="2026-07-27T00:00:00+08:00",
    )
    by_path = {entry["path"]: entry for entry in manifest["files"]}

    required = {
        "AGENTS.md",
        "configs/project.json",
        "models/README.md",
        "src/multimodal_pipeline.py",
        "src/action_segmentation/coarse.py",
        "src/web/app.py",
        "tests/test_phase_b_action_stabilization.py",
        "requirements.txt",
    }
    assert required <= set(by_path)
    assert "models/yolov8n-pose.onnx" not in by_path
    assert "models/hand_pose/hand_landmarker.task" not in by_path
    assert all("Factory AI Camera" not in path for path in by_path)
    assert not any(path.startswith("outputs/") for path in by_path)
    assert not any(path.startswith(".venv/") for path in by_path)
