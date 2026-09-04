"""Typed, evidence-bounded contracts for temporal action reasoning.

These records describe technical candidates only.  They do not confirm a
factory action, object interaction, process step, or production conclusion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
import re
from typing import Any, Mapping


TEMPORAL_ACTION_VOCABULARY = frozenset(
    {
        "idle",
        "reach",
        "retract",
        "lift",
        "lower",
        "move",
        "carry",
        "place",
        "hold",
        "release",
        "rotate",
        "push",
        "pull",
        "transition",
        "unknown",
        "lost",
    }
)
OBJECT_EVIDENCE_REQUIRED_ACTIONS = frozenset(
    {"carry", "place", "hold", "release"}
)
CHANGE_POINT_TYPES = frozenset(
    {
        "start_confirmation",
        "stop_confirmation",
        "direction_change",
        "amplitude_change",
        "quality_change",
        "bounded_gap",
        "hard_boundary",
    }
)
ANATOMICAL_SIDES = frozenset({"left", "right", "bilateral"})
AUTOMATIC_STATUSES = frozenset({"proposed", "uncertain"})
EVIDENCE_STATES = frozenset(
    {"normal", "transition", "unknown", "uncertain", "lost"}
)
BODY_FEATURE_STATES = frozenset(
    {"available", "uncertain", "missing", "lost", "unavailable"}
)
HAND_FEATURE_STATES = frozenset(
    {
        "qualified",
        "association_uncertain",
        "insufficient_geometry",
        "not_observed",
        "lost",
        "unavailable",
    }
)
OBJECT_FEATURE_STATES = frozenset(
    {"available", "not_observed", "unavailable", "not_configured"}
)
_UNCERTAIN_ACTIONS = frozenset({"transition", "unknown", "lost"})
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_sha256(value: Any, name: str) -> str:
    text = _require_text(value, name)
    if _SHA256_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA256")
    return text


def _require_non_negative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    integer = int(value)
    if integer < 0:
        raise ValueError(f"{name} cannot be negative")
    return integer


def _require_finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _validate_time_span(
    *,
    start_time: Any,
    end_time: Any,
    duration_seconds: Any,
) -> tuple[float, float, float]:
    start = _require_finite_number(start_time, "start_time")
    end = _require_finite_number(end_time, "end_time")
    duration = _require_finite_number(duration_seconds, "duration_seconds")
    if start < 0.0:
        raise ValueError("start_time cannot be negative")
    if end <= start:
        raise ValueError("end_time must be greater than start_time")
    if duration <= 0.0:
        raise ValueError("duration_seconds must be positive")
    expected = end - start
    tolerance = max(1e-9, expected * 1e-6)
    if abs(duration - expected) > tolerance:
        raise ValueError(
            "duration_seconds must equal end_time - start_time "
            f"within {tolerance:g} seconds"
        )
    return start, end, duration


def _validate_identity(
    *,
    person_ref: Any,
    lock_epoch: Any,
    anatomical_side: Any,
) -> None:
    _require_text(person_ref, "person_ref")
    _require_non_negative_integer(lock_epoch, "lock_epoch")
    if anatomical_side not in ANATOMICAL_SIDES:
        raise ValueError(
            "anatomical_side must be left, right, or bilateral"
        )


def _validate_provenance(
    *,
    source_video_sha256: Any,
    recording_group_id: Any,
    rule_version: Any,
    parameter_profile_id: Any,
    parameter_profile_sha256: Any,
) -> None:
    _require_sha256(source_video_sha256, "source_video_sha256")
    _require_text(recording_group_id, "recording_group_id")
    _require_text(rule_version, "rule_version")
    _require_text(parameter_profile_id, "parameter_profile_id")
    _require_sha256(
        parameter_profile_sha256,
        "parameter_profile_sha256",
    )


def _validate_string_ids(
    values: Any,
    name: str,
    *,
    required: bool = False,
) -> None:
    if not isinstance(values, list):
        raise ValueError(f"{name} must be a list")
    if required and not values:
        raise ValueError(f"{name} cannot be empty")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{name} must contain only non-empty strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} cannot contain duplicate IDs")


def _validate_feature_values(value: Any, path: str) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, Real):
        if not math.isfinite(float(value)):
            raise ValueError(f"{path} cannot contain NaN or infinity")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_text(key, f"{path} key")
            _validate_feature_values(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_feature_values(item, f"{path}[{index}]")
        return
    raise ValueError(f"{path} contains an unsupported feature value")


def _validate_boolean(value: Any, name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")


def _validate_automatic_review_fields(
    *,
    reviewer: Any,
    reviewed_at: Any,
    training_approval: Any,
    training_eligible: Any,
) -> None:
    if reviewer is not None or reviewed_at is not None:
        raise ValueError(
            "Automatic temporal records cannot claim human review"
        )
    if training_approval != "pending":
        raise ValueError(
            "Automatic temporal records require training_approval=pending"
        )
    if training_eligible is not False:
        raise ValueError(
            "Automatic temporal records cannot be training eligible"
        )


@dataclass(frozen=True)
class TemporalFeatureFrame:
    """One traceable shadow feature observation for an anatomical lane."""

    temporal_feature_id: str
    coarse_action: str
    evidence_state: str
    person_ref: str
    lock_epoch: int
    anatomical_side: str
    frame_index: int
    timestamp: float
    source_video_sha256: str
    recording_group_id: str
    rule_version: str
    parameter_profile_id: str
    parameter_profile_sha256: str
    body_feature_state: str
    hand_feature_state: str
    object_feature_state: str
    body_features_used: bool
    hand_features_used: bool
    object_features_used: bool
    body_motion_features: dict[str, Any] | None
    hand_motion_features: dict[str, Any] | None
    object_motion_features: dict[str, Any] | None
    source_frame_indices: list[int]
    source_segment_ids: list[str]
    source_hand_pose_ids: list[str]
    source_object_track_ids: list[str]
    source_interaction_ids: list[str]
    hard_boundary: bool
    boundary_reasons: list[str]
    status: str
    reviewer: None = None
    reviewed_at: None = None
    training_approval: str = "pending"
    training_eligible: bool = False

    def validate(self) -> None:
        _require_text(self.temporal_feature_id, "temporal_feature_id")
        if self.coarse_action not in TEMPORAL_ACTION_VOCABULARY:
            raise ValueError(
                f"Unsupported coarse action: {self.coarse_action}"
            )
        if self.evidence_state not in EVIDENCE_STATES:
            raise ValueError(
                f"Unsupported evidence_state: {self.evidence_state}"
            )
        _validate_identity(
            person_ref=self.person_ref,
            lock_epoch=self.lock_epoch,
            anatomical_side=self.anatomical_side,
        )
        _require_non_negative_integer(self.frame_index, "frame_index")
        timestamp = _require_finite_number(self.timestamp, "timestamp")
        if timestamp < 0.0:
            raise ValueError("timestamp cannot be negative")
        _validate_provenance(
            source_video_sha256=self.source_video_sha256,
            recording_group_id=self.recording_group_id,
            rule_version=self.rule_version,
            parameter_profile_id=self.parameter_profile_id,
            parameter_profile_sha256=self.parameter_profile_sha256,
        )
        if not isinstance(self.source_frame_indices, list):
            raise ValueError("source_frame_indices must be a list")
        if not self.source_frame_indices:
            raise ValueError("source_frame_indices cannot be empty")
        normalized_indices = [
            _require_non_negative_integer(value, "source_frame_indices item")
            for value in self.source_frame_indices
        ]
        if normalized_indices != sorted(set(normalized_indices)):
            raise ValueError(
                "source_frame_indices must be unique and ascending"
            )
        if self.frame_index not in normalized_indices:
            raise ValueError(
                "frame_index must be present in source_frame_indices"
            )
        for name, values in (
            ("source_segment_ids", self.source_segment_ids),
            ("source_hand_pose_ids", self.source_hand_pose_ids),
            ("source_object_track_ids", self.source_object_track_ids),
            ("source_interaction_ids", self.source_interaction_ids),
            ("boundary_reasons", self.boundary_reasons),
        ):
            _validate_string_ids(values, name)
        if self.body_feature_state not in BODY_FEATURE_STATES:
            raise ValueError("body_feature_state has unsupported state")
        if self.hand_feature_state not in HAND_FEATURE_STATES:
            raise ValueError("hand_feature_state has unsupported state")
        if self.object_feature_state not in OBJECT_FEATURE_STATES:
            raise ValueError("object_feature_state has unsupported state")
        for name, values in (
            ("body_motion_features", self.body_motion_features),
            ("hand_motion_features", self.hand_motion_features),
            ("object_motion_features", self.object_motion_features),
        ):
            if values is not None and not isinstance(values, dict):
                raise ValueError(f"{name} must be an object or null")
            if values is not None:
                _validate_feature_values(values, name)
        for name, value in (
            ("body_features_used", self.body_features_used),
            ("hand_features_used", self.hand_features_used),
            ("object_features_used", self.object_features_used),
            ("hard_boundary", self.hard_boundary),
        ):
            _validate_boolean(value, name)
        if self.body_features_used:
            if (
                self.body_feature_state != "available"
                or not self.body_motion_features
            ):
                raise ValueError(
                    "Used body features require available, non-empty "
                    "body_motion_features"
                )
        if (
            self.body_feature_state in {"missing", "lost", "unavailable"}
            and self.body_motion_features
        ):
            raise ValueError(
                "Missing/lost/unavailable Body state cannot carry motion "
                "features"
            )
        if (
            self.hand_feature_state != "qualified"
            and self.hand_motion_features
        ):
            raise ValueError(
                "Unqualified Hand state cannot carry Hand motion features"
            )
        if (
            self.object_feature_state != "available"
            and self.object_motion_features
        ):
            raise ValueError(
                "Unavailable Object state cannot carry Object motion features"
            )
        if self.hand_features_used:
            if (
                self.hand_feature_state != "qualified"
                or not self.hand_motion_features
                or not self.source_hand_pose_ids
            ):
                raise ValueError(
                    "Used Hand features require qualified real Hand evidence"
                )
        if self.object_features_used:
            if (
                self.object_feature_state != "available"
                or not self.object_motion_features
                or not self.source_object_track_ids
            ):
                raise ValueError(
                    "Used Object features require available real tracks"
                )
        if self.coarse_action in OBJECT_EVIDENCE_REQUIRED_ACTIONS:
            if (
                not self.object_features_used
                or self.object_feature_state != "available"
                or not self.source_object_track_ids
                or not self.source_interaction_ids
            ):
                raise ValueError(
                    f"{self.coarse_action} requires real Object and "
                    "Interaction lineage"
                )
        if self.status not in AUTOMATIC_STATUSES:
            raise ValueError("status must be proposed or uncertain")
        if self.status == "proposed":
            if (
                self.evidence_state != "normal"
                or self.coarse_action in _UNCERTAIN_ACTIONS
                or self.hard_boundary
                or self.body_feature_state != "available"
                or not self.body_features_used
            ):
                raise ValueError(
                    "Proposed features require normal, non-boundary, used "
                    "Body evidence"
                )
        if self.coarse_action in _UNCERTAIN_ACTIONS:
            if self.status != "uncertain":
                raise ValueError(
                    f"{self.coarse_action} features must remain uncertain"
                )
        if self.coarse_action == "lost":
            if (
                self.evidence_state != "lost"
                or self.body_feature_state != "lost"
                or not self.hard_boundary
            ):
                raise ValueError(
                    "lost features require lost Body state and hard boundary"
                )
        if self.evidence_state == "lost" and self.coarse_action != "lost":
            raise ValueError(
                "lost evidence_state requires coarse_action=lost"
            )
        if self.hard_boundary:
            if (
                self.evidence_state not in {"uncertain", "lost"}
                or self.status != "uncertain"
                or self.hand_features_used
                or self.object_features_used
                or not self.boundary_reasons
            ):
                raise ValueError(
                    "Hard-boundary features must be uncertain/lost, have a "
                    "reason, and cannot use Hand/Object features"
                )
        _validate_automatic_review_fields(
            reviewer=self.reviewer,
            reviewed_at=self.reviewed_at,
            training_approval=self.training_approval,
            training_eligible=self.training_eligible,
        )
        if self.hand_features_used and self.hand_feature_state != "qualified":
            raise ValueError(
                "Unqualified Hand evidence cannot be consumed"
            )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class TemporalActionCandidate:
    """A conservative, traceable temporal action hypothesis."""

    temporal_action_candidate_id: str
    action: str
    evidence_state: str
    person_ref: str
    lock_epoch: int
    anatomical_side: str
    start_frame_index: int
    end_frame_index: int
    start_time: float
    end_time: float
    duration_seconds: float
    source_video_sha256: str
    recording_group_id: str
    engine_kind: str
    shadow_mode: bool
    rule_version: str
    parameter_profile_id: str
    parameter_profile_sha256: str
    temporal_context_seconds: float
    body_feature_state: str
    hand_feature_state: str
    object_feature_state: str
    body_features_used: bool
    hand_features_used: bool
    object_features_used: bool
    source_temporal_feature_ids: list[str]
    source_segment_ids: list[str]
    source_frame_indices: list[int]
    source_hand_pose_ids: list[str]
    source_object_track_ids: list[str]
    source_interaction_ids: list[str]
    change_point_evidence: list[dict[str, Any]]
    hard_boundary: bool
    boundary_reasons: list[str]
    status: str
    reviewer: None = None
    reviewed_at: None = None
    training_approval: str = "pending"
    training_eligible: bool = False

    def validate(self) -> None:
        _require_text(
            self.temporal_action_candidate_id,
            "temporal_action_candidate_id",
        )
        if self.action not in TEMPORAL_ACTION_VOCABULARY:
            raise ValueError(f"Unsupported temporal action: {self.action}")
        if self.evidence_state not in EVIDENCE_STATES:
            raise ValueError(
                f"Unsupported evidence_state: {self.evidence_state}"
            )
        _validate_identity(
            person_ref=self.person_ref,
            lock_epoch=self.lock_epoch,
            anatomical_side=self.anatomical_side,
        )
        start_frame = _require_non_negative_integer(
            self.start_frame_index,
            "start_frame_index",
        )
        end_frame = _require_non_negative_integer(
            self.end_frame_index,
            "end_frame_index",
        )
        if end_frame < start_frame:
            raise ValueError(
                "end_frame_index must be >= start_frame_index"
            )
        _validate_time_span(
            start_time=self.start_time,
            end_time=self.end_time,
            duration_seconds=self.duration_seconds,
        )
        _validate_provenance(
            source_video_sha256=self.source_video_sha256,
            recording_group_id=self.recording_group_id,
            rule_version=self.rule_version,
            parameter_profile_id=self.parameter_profile_id,
            parameter_profile_sha256=self.parameter_profile_sha256,
        )
        if self.engine_kind != "interpretable_rules":
            raise ValueError("engine_kind must be interpretable_rules")
        if self.shadow_mode is not True:
            raise ValueError(
                "Temporal V3 candidates must remain shadow_mode=true"
            )
        context_seconds = _require_finite_number(
            self.temporal_context_seconds,
            "temporal_context_seconds",
        )
        if context_seconds <= 0.0:
            raise ValueError("temporal_context_seconds must be positive")
        for name, values in (
            (
                "source_temporal_feature_ids",
                self.source_temporal_feature_ids,
            ),
            ("source_segment_ids", self.source_segment_ids),
            ("source_hand_pose_ids", self.source_hand_pose_ids),
            ("source_object_track_ids", self.source_object_track_ids),
            ("source_interaction_ids", self.source_interaction_ids),
            ("boundary_reasons", self.boundary_reasons),
        ):
            _validate_string_ids(
                values,
                name,
                required=name
                in {"source_temporal_feature_ids", "source_segment_ids"},
            )
        if not isinstance(self.source_frame_indices, list):
            raise ValueError("source_frame_indices must be a list")
        if not self.source_frame_indices:
            raise ValueError("source_frame_indices cannot be empty")
        normalized_indices = [
            _require_non_negative_integer(value, "source_frame_indices item")
            for value in self.source_frame_indices
        ]
        if normalized_indices != sorted(set(normalized_indices)):
            raise ValueError(
                "source_frame_indices must be unique and ascending"
            )
        if (
            normalized_indices[0] < start_frame
            or normalized_indices[-1] > end_frame
        ):
            raise ValueError(
                "source_frame_indices must lie inside candidate frame bounds"
            )
        if (
            normalized_indices[0] != start_frame
            or normalized_indices[-1] != end_frame
        ):
            raise ValueError(
                "Candidate frame bounds must equal source frame endpoints"
            )
        if self.body_feature_state not in BODY_FEATURE_STATES:
            raise ValueError("body_feature_state has unsupported state")
        if self.hand_feature_state not in HAND_FEATURE_STATES:
            raise ValueError("hand_feature_state has unsupported state")
        if self.object_feature_state not in OBJECT_FEATURE_STATES:
            raise ValueError("object_feature_state has unsupported state")
        for name, value in (
            ("body_features_used", self.body_features_used),
            ("hand_features_used", self.hand_features_used),
            ("object_features_used", self.object_features_used),
            ("hard_boundary", self.hard_boundary),
        ):
            _validate_boolean(value, name)
        if self.body_features_used and self.body_feature_state != "available":
            raise ValueError(
                "Used Body features require body_feature_state=available"
            )
        if self.hand_features_used:
            if (
                self.hand_feature_state != "qualified"
                or not self.source_hand_pose_ids
            ):
                raise ValueError(
                    "Used Hand features require qualified real Hand lineage"
                )
        if self.object_features_used:
            if (
                self.object_feature_state != "available"
                or not self.source_object_track_ids
            ):
                raise ValueError(
                    "Used Object features require available real tracks"
                )
        if self.action in OBJECT_EVIDENCE_REQUIRED_ACTIONS:
            if (
                not self.object_features_used
                or self.object_feature_state != "available"
                or not self.source_object_track_ids
                or not self.source_interaction_ids
            ):
                raise ValueError(
                    f"{self.action} requires available object and interaction "
                    "layers with object-track and interaction lineage"
                )
        if not isinstance(self.change_point_evidence, list):
            raise ValueError("change_point_evidence must be a list")
        if not self.change_point_evidence:
            raise ValueError("change_point_evidence cannot be empty")
        for index, item in enumerate(self.change_point_evidence):
            if not isinstance(item, dict):
                raise ValueError(
                    f"change_point_evidence[{index}] must be an object"
                )
            for key in (
                "change_type",
                "timestamp",
                "reason",
                "source_temporal_feature_ids",
            ):
                if key not in item:
                    raise ValueError(
                        f"change_point_evidence[{index}] missing {key}"
                    )
            change_type = _require_text(
                item["change_type"],
                f"change_point_evidence[{index}].change_type",
            )
            if change_type not in CHANGE_POINT_TYPES:
                raise ValueError(
                    f"Unsupported change-point type: {change_type}"
                )
            change_timestamp = _require_finite_number(
                item["timestamp"],
                f"change_point_evidence[{index}].timestamp",
            )
            if not self.start_time <= change_timestamp <= self.end_time:
                raise ValueError(
                    "Change-point timestamp must lie inside candidate span"
                )
            _require_text(
                item["reason"],
                f"change_point_evidence[{index}].reason",
            )
            _validate_string_ids(
                item["source_temporal_feature_ids"],
                (
                    f"change_point_evidence[{index}]"
                    ".source_temporal_feature_ids"
                ),
                required=True,
            )
            if not set(item["source_temporal_feature_ids"]).issubset(
                self.source_temporal_feature_ids
            ):
                raise ValueError(
                    "Change-point lineage must reference candidate features"
                )
        if self.status not in AUTOMATIC_STATUSES:
            raise ValueError("status must be proposed or uncertain")
        if self.status == "proposed":
            if (
                self.evidence_state != "normal"
                or self.action in _UNCERTAIN_ACTIONS
                or self.hard_boundary
                or self.body_feature_state != "available"
                or not self.body_features_used
            ):
                raise ValueError(
                    "Proposed candidates require normal, non-boundary, used "
                    "Body evidence"
                )
        if self.action in _UNCERTAIN_ACTIONS:
            if self.status != "uncertain":
                raise ValueError(
                    f"{self.action} candidates must remain uncertain"
                )
        if self.action == "lost":
            if (
                self.evidence_state != "lost"
                or self.body_feature_state != "lost"
                or not self.hard_boundary
            ):
                raise ValueError(
                    "lost candidates require lost Body state and hard boundary"
                )
        if self.evidence_state == "lost" and self.action != "lost":
            raise ValueError("lost evidence_state requires action=lost")
        if self.hard_boundary:
            if (
                self.evidence_state not in {"uncertain", "lost"}
                or self.status != "uncertain"
                or self.hand_features_used
                or self.object_features_used
                or not self.boundary_reasons
            ):
                raise ValueError(
                    "Hard-boundary candidates must be uncertain/lost, have a "
                    "reason, and cannot use Hand/Object features"
                )
        _validate_automatic_review_fields(
            reviewer=self.reviewer,
            reviewed_at=self.reviewed_at,
            training_approval=self.training_approval,
            training_eligible=self.training_eligible,
        )
        if self.hand_features_used and self.hand_feature_state != "qualified":
            raise ValueError(
                "Unqualified Hand evidence cannot be consumed"
            )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)
