"""Build a deterministic SHA256 inventory for the independently runnable baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = (
    ".gitignore",
    "AGENTS.md",
    "HAND_ACTION_UPGRADE_REPORT.md",
    "HAND_MODEL_MANIFEST.json",
    "PROJECT_KICKOFF_REPORT.md",
    "README.md",
    "requirements.txt",
    "SOURCE_IMPORT_MANIFEST.json",
)
TREE_ROOTS = ("configs", "docs", "models", "schemas", "scripts", "src", "tests")
EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "outputs",
}
EXCLUDED_SUFFIXES = {".part", ".pyc", ".pyo", ".tmp"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_allowed(path: Path, project_root: Path) -> bool:
    relative = path.relative_to(project_root)
    if any(part in EXCLUDED_DIRECTORY_NAMES for part in relative.parts):
        return False
    return path.suffix.lower() not in EXCLUDED_SUFFIXES


def iter_baseline_files(project_root: Path) -> Iterable[Path]:
    candidates: set[Path] = set()
    for relative in ROOT_FILES:
        path = project_root / relative
        if path.is_file():
            candidates.add(path)
    for relative in TREE_ROOTS:
        tree_root = project_root / relative
        if not tree_root.is_dir():
            continue
        for path in tree_root.rglob("*"):
            if path.is_file() and _is_allowed(path, project_root):
                candidates.add(path)
    yield from sorted(
        candidates,
        key=lambda item: item.relative_to(project_root).as_posix(),
    )


def build_manifest(
    project_root: Path,
    *,
    run_id: str,
    generated_at: str | None = None,
) -> dict:
    project_root = project_root.resolve()
    entries = []
    total_bytes = 0
    for path in iter_baseline_files(project_root):
        size = path.stat().st_size
        total_bytes += size
        entries.append(
            {
                "path": path.relative_to(project_root).as_posix(),
                "size_bytes": size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": "factory_master_loop_baseline_file_manifest_v1",
        "run_id": run_id,
        "project_path": str(project_root),
        "generated_at": generated_at or datetime.now().astimezone().isoformat(),
        "scope": {
            "root_files": list(ROOT_FILES),
            "tree_roots": list(TREE_ROOTS),
            "excluded_directory_names": sorted(EXCLUDED_DIRECTORY_NAMES),
            "excluded_suffixes": sorted(EXCLUDED_SUFFIXES),
            "dynamic_state_excluded": [
                "MASTER_LOOP_STATE.json",
                "MASTER_LOOP_LEDGER.jsonl",
                "outputs/",
            ],
        },
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "files": entries,
    }


def write_manifest(output: Path, manifest: dict) -> None:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_manifest(args.project_root, run_id=args.run_id)
    write_manifest(args.output, manifest)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "file_count": manifest["file_count"],
                "total_bytes": manifest["total_bytes"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
