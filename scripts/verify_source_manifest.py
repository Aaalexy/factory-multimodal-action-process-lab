"""Verify public target paths and, when present, private source hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.provenance import sha256_file  # noqa: E402


TEXT_SUFFIXES = {".py", ".json", ".yaml", ".yml", ".md", ".txt", ".csv"}


def _match_hash(path: Path, expected: str) -> tuple[bool, str | None]:
    """Match exact bytes, allowing Git's CRLF checkout normalization for text."""
    if not path.is_file():
        return False, None
    exact = sha256_file(path)
    if exact == expected:
        return True, "exact_bytes"
    if path.suffix.lower() in TEXT_SUFFIXES:
        normalized = path.read_bytes().replace(b"\r\n", b"\n")
        normalized_hash = hashlib.sha256(normalized).hexdigest()
        if normalized_hash == expected:
            return True, "normalized_lf"
    return False, "mismatch"


def verify(target_only: bool = False) -> dict[str, object]:
    manifest_path = PROJECT_ROOT / "SOURCE_IMPORT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = []
    for item in manifest.get("imports", []):
        target = PROJECT_ROOT / item["target_path"]
        target_hash = item.get("target_sha256")
        if target_hash:
            target_match, target_match_mode = _match_hash(target, target_hash)
        else:
            target_match = target.is_file()
            target_match_mode = "exists" if target_match else "missing"
        source_match = None
        source_match_mode = None
        if not target_only:
            source_path = item.get("source_path")
            source_hash = item.get("source_sha256")
            if source_path and source_hash:
                source = Path(source_path)
                source_match, source_match_mode = _match_hash(source, source_hash)
            else:
                source_match = False
                source_match_mode = "not_distributed"
        results.append(
            {
                "target_path": item["target_path"],
                "target_match": target_match,
                "target_match_mode": target_match_mode,
                "source_match": source_match,
                "source_match_mode": source_match_mode,
            }
        )
    return {
        "status": (
            "passed"
            if all(
                item["target_match"]
                and (target_only or item["source_match"])
                for item in results
            )
            else "failed"
        ),
        "entry_count": len(results),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-only", action="store_true")
    args = parser.parse_args()
    result = verify(args.target_only)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
