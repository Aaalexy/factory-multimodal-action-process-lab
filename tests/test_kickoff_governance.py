from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.check_workspace import EXPECTED, check_workspace
from scripts.verify_source_manifest import verify
from src.contracts import ValidationFlags


ROOT = Path(__file__).resolve().parents[1]


def test_workspace_path_hard_gate_is_exact():
    result = check_workspace(ROOT)
    assert result == {
        "expected": EXPECTED,
        "resolved": EXPECTED,
        "exact_match": True,
    }


def test_source_import_manifest_was_created_before_copy():
    manifest = json.loads((ROOT / "SOURCE_IMPORT_MANIFEST.json").read_text("utf-8"))
    assert manifest["generated_before_source_copy"] is True
    assert manifest["source_access"] == "not_distributed"
    assert manifest["status"] == "public_snapshot"


@pytest.mark.private_artifacts
def test_source_and_target_manifest_hashes_match_current_files():
    result = verify(target_only=False)
    assert result["status"] == "passed"
    assert result["entry_count"] >= 18


@pytest.mark.private_artifacts
def test_legacy_readonly_hash_snapshot_is_unchanged():
    inventory = json.loads(
        (ROOT / "outputs" / "legacy_readonly_inventory.json").read_text("utf-8")
    )
    integrity = inventory["integrity_verification"]
    assert integrity["matched"] is True
    assert integrity["mismatch_count"] == 0
    assert integrity["checked_file_count"] == 49


def test_all_validation_flags_remain_false():
    flags = ValidationFlags()
    flags.validate()
    assert not any(flags.__dict__.values())
    config = json.loads((ROOT / "configs" / "project.json").read_text("utf-8"))
    assert not any(config["validation_flags"].values())


def test_project_config_disables_identity_capabilities():
    config = json.loads((ROOT / "configs" / "project.json").read_text("utf-8"))
    assert not any(config["privacy"].values())


def test_recording_group_rules_prevent_leakage():
    registry = json.loads(
        (ROOT / "configs" / "recording_group_registry.json").read_text("utf-8")
    )
    policy = registry["split_policy"]
    assert policy["random_frame_split_allowed"] is False
    assert policy["minimum_independent_groups_before_test_split"] >= 3
    assert policy["current_test_split_status"] == "not_available"
    groups = {
        pattern: item["recording_group_id"]
        for item in registry["groups"]
        for pattern in item["video_name_patterns"]
    }
    assert groups["sample_video_A"] == groups["sample_video_B"]
    assert groups["sample_video_C"] != groups["sample_video_A"]
    assert all(item["split"] == "unassigned" for item in registry["groups"])


def test_no_forbidden_project_is_present_in_import_manifest():
    manifest_text = (ROOT / "SOURCE_IMPORT_MANIFEST.json").read_text("utf-8")
    assert r"<forbidden-unrelated-workspace>" not in manifest_text
