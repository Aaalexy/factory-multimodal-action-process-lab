"""Metric definitions and an honest not-evaluable result without human truth."""

from __future__ import annotations

from typing import Any


METRIC_DEFINITIONS: dict[str, list[str]] = {
    "action": [
        "macro_f1",
        "per_class_precision_recall_f1",
        "boundary_error_seconds",
        "stable_events_under_1s",
        "events_per_minute",
        "over_segmentation_count",
        "lost_off_frame_false_actions",
        "person_switch_count",
    ],
    "object": [
        "map",
        "per_class_precision_recall",
        "track_continuity",
        "occlusion_recovery_errors",
    ],
    "interaction": [
        "precision_recall_f1",
        "anatomical_side_error_rate",
        "wrong_person_or_object_associations",
    ],
    "process": [
        "step_accuracy",
        "boundary_error_seconds",
        "sequence_error_rate",
        "repeated_step_recognition",
        "unknown_uncertain_coverage",
        "false_production_conclusions",
    ],
}


def not_evaluable_manifest(reason: str) -> dict[str, Any]:
    return {
        "status": "not_evaluable",
        "reason": reason,
        "ground_truth_required": True,
        "metrics": {
            layer: {name: None for name in names}
            for layer, names in METRIC_DEFINITIONS.items()
        },
        "targets": {"false_production_conclusions": 0},
    }
