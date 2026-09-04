from __future__ import annotations

from pathlib import Path

from scripts.audit_master_loop_baseline import (
    parse_junit,
    requirement_is_fully_pinned,
)


def test_requirement_locking_distinguishes_exact_and_range_pins() -> None:
    assert requirement_is_fully_pinned("mediapipe==0.10.35")
    assert requirement_is_fully_pinned("pkg==1.2.3 # documented")
    assert not requirement_is_fully_pinned("numpy>=2.0,<3")
    assert not requirement_is_fully_pinned("-r base.txt")


def test_junit_parser_aggregates_nested_leaf_suites(tmp_path: Path) -> None:
    report = tmp_path / "junit.xml"
    report.write_text(
        (
            '<testsuites><testsuite name="outer">'
            '<testsuite name="a" tests="2" failures="1" errors="0" skipped="0"/>'
            '<testsuite name="b" tests="3" failures="0" errors="1" skipped="1"/>'
            "</testsuite></testsuites>"
        ),
        encoding="utf-8",
    )
    result = parse_junit(report)
    assert result["tests"] == 5
    assert result["failures"] == 1
    assert result["errors"] == 1
    assert result["skipped"] == 1
