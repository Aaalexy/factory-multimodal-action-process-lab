"""Temporal model boundary; pose heuristic events remain a separate layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

from src.contracts import LayerState

from .contracts import TemporalActionCandidate, TemporalFeatureFrame


_REVIEW_TRUTH_KEYS = {
    "reviewer",
    "reviewed_at",
    "human_confirmed_semantic",
}


def _assert_no_automatic_truth_claims(value: Any, path: str) -> None:
    """Reject review/training truth hidden in diagnostics or change points."""

    if isinstance(value, dict):
        for key, item in value.items():
            child_path = f"{path}.{key}"
            if (
                key in _REVIEW_TRUTH_KEYS
                and item is not None
                and item is not False
                and item != ""
            ):
                raise ValueError(
                    f"{child_path} cannot claim human review"
                )
            if key == "training_eligible" and item is not False:
                raise ValueError(
                    f"{child_path} cannot authorize training"
                )
            if key in {"status", "review_state"} and item == "confirmed":
                raise ValueError(
                    f"{child_path} cannot claim confirmed state"
                )
            _assert_no_automatic_truth_claims(item, child_path)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_automatic_truth_claims(
                item,
                f"{path}[{index}]",
            )


def _id_set(values: Any, name: str) -> frozenset[str]:
    if not isinstance(values, (set, frozenset)):
        raise ValueError(f"{name} must be a set of IDs")
    if any(not isinstance(item, str) or not item.strip() for item in values):
        raise ValueError(f"{name} must contain non-empty string IDs")
    return frozenset(values)


@runtime_checkable
class TemporalActionProvider(Protocol):
    """Runtime-checkable boundary for optional temporal action providers."""

    model_version: str | None
    context_seconds: float
    rule_version: str | None
    parameter_profile_id: str | None
    parameter_profile_sha256: str | None

    def analyze(
        self,
        pose_events: list[dict[str, Any]],
        interaction_events: list[dict[str, Any]],
        *,
        feature_frames: list[
            TemporalFeatureFrame | dict[str, Any]
        ]
        | None = None,
    ) -> "TemporalActionOutput": ...


@dataclass
class TemporalActionOutput:
    """Provider result; the original two-field constructor remains valid."""

    state: LayerState
    action_candidates: list[
        TemporalActionCandidate | dict[str, Any]
    ] = field(default_factory=list)
    feature_frames: list[
        TemporalFeatureFrame | dict[str, Any]
    ] = field(default_factory=list)
    change_point_candidates: list[dict[str, Any]] = field(
        default_factory=list
    )
    diagnostics: dict[str, Any] = field(default_factory=dict)
    known_pose_segment_ids: frozenset[str] = field(
        default_factory=frozenset,
        repr=False,
    )
    known_hand_pose_ids: frozenset[str] = field(
        default_factory=frozenset,
        repr=False,
    )
    known_object_track_ids: frozenset[str] = field(
        default_factory=frozenset,
        repr=False,
    )
    known_interaction_ids: frozenset[str] = field(
        default_factory=frozenset,
        repr=False,
    )

    def validate(self) -> None:
        if not isinstance(self.state, LayerState):
            raise ValueError("state must be a LayerState")
        if self.state.evidence_count < 0:
            raise ValueError("LayerState evidence_count cannot be negative")
        if self.state.status == "unavailable" and (
            self.action_candidates
            or self.feature_frames
            or self.change_point_candidates
        ):
            raise ValueError(
                "Unavailable temporal providers cannot emit evidence"
            )
        known_pose_segment_ids = _id_set(
            self.known_pose_segment_ids,
            "known_pose_segment_ids",
        )
        known_hand_pose_ids = _id_set(
            self.known_hand_pose_ids,
            "known_hand_pose_ids",
        )
        known_object_track_ids = _id_set(
            self.known_object_track_ids,
            "known_object_track_ids",
        )
        known_interaction_ids = _id_set(
            self.known_interaction_ids,
            "known_interaction_ids",
        )
        typed_features: list[TemporalFeatureFrame] = []
        feature_by_id: dict[str, TemporalFeatureFrame] = {}
        for feature in self.feature_frames:
            if isinstance(feature, TemporalFeatureFrame):
                typed = feature
            elif isinstance(feature, dict):
                typed = TemporalFeatureFrame(**feature)
            else:
                raise ValueError(
                    "feature_frames must contain mappings or "
                    "TemporalFeatureFrame records"
                )
            typed.validate()
            if typed.temporal_feature_id in feature_by_id:
                raise ValueError("Duplicate temporal_feature_id")
            feature_by_id[typed.temporal_feature_id] = typed
            typed_features.append(typed)
            if not set(typed.source_segment_ids).issubset(
                known_pose_segment_ids
            ):
                raise ValueError(
                    "Feature references an unknown Pose segment"
                )
            if not set(typed.source_hand_pose_ids).issubset(
                known_hand_pose_ids
            ):
                raise ValueError(
                    "Feature references an unknown Hand observation"
                )
            if not set(typed.source_object_track_ids).issubset(
                known_object_track_ids
            ):
                raise ValueError(
                    "Feature references an unknown Object track"
                )
            if not set(typed.source_interaction_ids).issubset(
                known_interaction_ids
            ):
                raise ValueError(
                    "Feature references an unknown Interaction event"
                )
        typed_candidates: list[TemporalActionCandidate] = []
        for candidate in self.action_candidates:
            if isinstance(candidate, TemporalActionCandidate):
                typed = candidate
            elif isinstance(candidate, dict):
                typed = TemporalActionCandidate(**candidate)
            else:
                raise ValueError(
                    "action_candidates must contain mappings or "
                    "TemporalActionCandidate records"
                )
            typed.validate()
            typed_candidates.append(typed)
            missing_features = set(
                typed.source_temporal_feature_ids
            ) - feature_by_id.keys()
            if missing_features:
                raise ValueError(
                    "Candidate references unknown Temporal features"
                )
            sources = [
                feature_by_id[item]
                for item in typed.source_temporal_feature_ids
            ]
            for source in sources:
                for field_name in (
                    "person_ref",
                    "lock_epoch",
                    "anatomical_side",
                    "source_video_sha256",
                    "recording_group_id",
                    "rule_version",
                    "parameter_profile_id",
                    "parameter_profile_sha256",
                ):
                    if getattr(source, field_name) != getattr(
                        typed,
                        field_name,
                    ):
                        raise ValueError(
                            "Candidate crosses Temporal feature "
                            f"{field_name} boundary"
                        )
            source_frame_indices = sorted(
                {
                    index
                    for source in sources
                    for index in source.source_frame_indices
                }
            )
            if typed.source_frame_indices != source_frame_indices:
                raise ValueError(
                    "Candidate source_frame_indices must exactly match "
                    "referenced Temporal features"
                )
            source_timestamps = [source.timestamp for source in sources]
            if (
                abs(typed.start_time - min(source_timestamps)) > 1e-9
                or abs(typed.end_time - max(source_timestamps)) > 1e-9
            ):
                raise ValueError(
                    "Candidate time bounds must equal referenced Temporal "
                    "feature timestamps"
                )
            lineage_sets = {
                "source_segment_ids": {
                    item
                    for source in sources
                    for item in source.source_segment_ids
                },
                "source_hand_pose_ids": {
                    item
                    for source in sources
                    for item in source.source_hand_pose_ids
                },
                "source_object_track_ids": {
                    item
                    for source in sources
                    for item in source.source_object_track_ids
                },
                "source_interaction_ids": {
                    item
                    for source in sources
                    for item in source.source_interaction_ids
                },
            }
            for field_name, available_ids in lineage_sets.items():
                if not set(getattr(typed, field_name)).issubset(
                    available_ids
                ):
                    raise ValueError(
                        f"Candidate {field_name} lacks feature lineage"
                    )
            if not set(typed.source_segment_ids).issubset(
                known_pose_segment_ids
            ):
                raise ValueError(
                    "Candidate references an unknown Pose segment"
                )
            if not set(typed.source_hand_pose_ids).issubset(
                known_hand_pose_ids
            ):
                raise ValueError(
                    "Candidate references an unknown Hand observation"
                )
            if not set(typed.source_object_track_ids).issubset(
                known_object_track_ids
            ):
                raise ValueError(
                    "Candidate references an unknown Object track"
                )
            if not set(typed.source_interaction_ids).issubset(
                known_interaction_ids
            ):
                raise ValueError(
                    "Candidate references an unknown Interaction event"
                )
            if typed.hand_features_used and not any(
                source.hand_features_used for source in sources
            ):
                raise ValueError(
                    "Candidate claims Hand use without a used Hand feature"
                )
            if typed.object_features_used and not any(
                source.object_features_used for source in sources
            ):
                raise ValueError(
                    "Candidate claims Object use without a used Object feature"
                )
        if not isinstance(self.change_point_candidates, list):
            raise ValueError("change_point_candidates must be a list")
        if any(
            not isinstance(item, dict)
            for item in self.change_point_candidates
        ):
            raise ValueError(
                "change_point_candidates must contain mappings"
            )
        if not isinstance(self.diagnostics, dict):
            raise ValueError("diagnostics must be an object")
        _assert_no_automatic_truth_claims(
            self.change_point_candidates,
            "change_point_candidates",
        )
        _assert_no_automatic_truth_claims(
            self.diagnostics,
            "diagnostics",
        )
        expected_evidence_count = len(typed_features) + len(typed_candidates)
        if self.state.status == "available" and (
            self.state.evidence_count != expected_evidence_count
        ):
            raise ValueError(
                "LayerState evidence_count must match features + candidates"
            )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "state": asdict(self.state),
            "action_candidates": [
                item.to_dict()
                if isinstance(item, TemporalActionCandidate)
                else dict(item)
                for item in self.action_candidates
            ],
            "feature_frames": [
                item.to_dict()
                if isinstance(item, TemporalFeatureFrame)
                else dict(item)
                for item in self.feature_frames
            ],
            "change_point_candidates": [
                dict(item) for item in self.change_point_candidates
            ],
            "diagnostics": dict(self.diagnostics),
        }


class NotConfiguredTemporalActionModel:
    """Honest null provider; it never creates action or feature evidence."""

    model_version: str | None = None
    context_seconds: float = 0.0
    rule_version: str | None = None
    parameter_profile_id: str | None = None
    parameter_profile_sha256: str | None = None

    def analyze(
        self,
        pose_events: list[dict[str, Any]],
        interaction_events: list[dict[str, Any]],
        *,
        feature_frames: list[
            TemporalFeatureFrame | dict[str, Any]
        ]
        | None = None,
    ) -> TemporalActionOutput:
        del pose_events, interaction_events, feature_frames
        output = TemporalActionOutput(
            state=LayerState(
                layer="temporal_action_model",
                status="unavailable",
                reason=(
                    "not_configured: no trained seconds-context action model; "
                    "pose heuristic events remain available separately"
                ),
                model_version=None,
                evidence_count=0,
            ),
            action_candidates=[],
            feature_frames=[],
            change_point_candidates=[],
            diagnostics={
                "provider_state": "not_configured",
                "shadow_mode": True,
                "pose_event_count_consumed": 0,
                "interaction_event_count_consumed": 0,
                "accuracy_status": "not_evaluable",
            },
        )
        output.validate()
        return output
