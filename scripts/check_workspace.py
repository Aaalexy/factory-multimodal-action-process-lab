"""Confirm that the command is running from this repository root."""

from __future__ import annotations

import json
from pathlib import Path


EXPECTED = str(Path(__file__).resolve().parents[1])


def check_workspace(path: str | Path = ".") -> dict[str, object]:
    current = str(Path(path).resolve())
    return {
        "expected": EXPECTED,
        "resolved": current,
        "exact_match": current == EXPECTED,
    }


if __name__ == "__main__":
    result = check_workspace()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["exact_match"]:
        raise SystemExit(2)
