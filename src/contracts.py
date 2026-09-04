"""Shared, conservative data contracts for the multimodal pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


EvidenceStatus = Literal[
    "proposed",
    "uncertain",
    "not_observed",
    "unavailable",
]
ObservationState = Literal[
    "detected",
    "predicted",
    "interpolated",
    "missing",
    "lost",
    "not_configured",
    "unavailable",
]
HandBackendState = Literal["available", "unavailable", "error"]
HandQualityState = Literal[
    "qualified",
    "association_uncertain",
    "insufficient_geometry",
    "not_observed",
    "lost",
]
HandValidationState = Literal[
    "not_reviewed",
    "review_required",
    "not_evaluable",
]

ALLOWED_PROCESS_STATUSES = {
    "proposed",
    "uncertain",
    "not_observed",
    "unavailable",
}
FORBIDDEN_PRODUCTION_CONCLUSIONS = {"OK", "NG", "PASS", "FAIL"}


@dataclass(frozen=True)
class ValidationFlags:
    factory_camera_validated: bool = False
    production_action_model_ready: bool = False
    external_factory_validated: bool = False
    production_process_model_ready: bool = False

    def validate(self) -> None:
        if any(asdict(self).values()):
            raise ValueError("Kickoff baseline cannot assert production validation")


@dataclass
class LayerState:
    layer: str
    status: str
    reason: str
    model_version: str | None = None
    evidence_count: int = 0


@dataclass
class HandPoseFrame:
    """One anatomical hand observation bound to an anonymous lock epoch."""

    hand_pose_id: str
    person_ref: str
    lock_epoch: int
    anatomical_side: str
    frame_index: int
    timestamp: float
    crop_bbox: list[float] | None
    crop_transform: dict[str, Any] | None
    landmarks: list[dict[str, Any]]
    landmark_count: int
    confidence: float | None
    observation_state: str
    occlusion: str
    source_video_sha256: str
    recording_group_id: str
    source_model_version: str
    status: EvidenceStatus = "uncertain"
    reviewer: str | None = None
    reviewed_at: str | None = None
    training_approval: str = "pending"
    training_eligible: bool = False
    detection_confidence: float | None = None
    presence_confidence: float | None = None
    tracking_confidence: float | None = None
    raw_confidence_availability: dict[str, str] = field(default_factory=dict)
    runtime_version: str | None = None
    model_handedness_label: str | None = None
    model_handedness_score: float | None = None
    inference_time_ms: float | None = None
    reason: str = ""
    evidence_type: str = "no_hand_geometry"
    association_checks: dict[str, Any] = field(default_factory=dict)
    backend_state: HandBackendState = "unavailable"
    backend_mode: str = "image"
    quality_state: HandQualityState = "not_observed"
    quality_reasons: list[str] = field(
        default_factory=lambda: ["contract_default_not_finalized"]
    )
    validation_state: HandValidationState = "not_evaluable"
    action_feature_eligible: bool = False
    feature_eligibility_reasons: list[str] = field(
        default_factory=lambda: ["quality_gate_not_evaluated"]
    )
    quality_gate_version: str = "hand_quality_gate_v1"


@dataclass
class ObjectTrack:
    object_track_id: str
    object_class: str
    start_time: float
    end_time: float
    bbox: list[float]
    confidence: float
    observation_state: str
    occlusion: str
    source_video_sha256: str
    source_model_version: str
    status: str = "proposed"
    training_eligible: bool = False
    reviewer: str | None = None
    reviewed_at: str | None = None
    training_approval: str = "pending"
    recording_group_id: str | None = None


@dataclass
class InteractionEvent:
    interaction_id: str
    person_ref: str
    lock_epoch: int
    anatomical_side: str
    object_track_id: str
    interaction_type: str
    start_time: float
    end_time: float
    evidence_type: str
    evidence_confidence: float
    status: str = "proposed"
    training_eligible: bool = False
    source_video_sha256: str = ""
    source_model_version: str = ""
    reviewer: str | None = None
    reviewed_at: str | None = None
    training_approval: str = "pending"
    recording_group_id: str | None = None


@dataclass
class ProcessStep:
    process_step_id: str
    process_name: str
    involved_object_classes: list[str]
    source_action_event_ids: list[str]
    source_interaction_ids: list[str]
    start_time: float
    end_time: float
    predecessor_step_ids: list[str]
    completion_evidence: list[str]
    confidence: float
    status: EvidenceStatus = "proposed"
    review_state: str = "pending"
    training_eligible: bool = False
    source_video_sha256: str = ""
    source_model_version: str = ""
    reviewer: str | None = None
    reviewed_at: str | None = None
    training_approval: str = "pending"
    recording_group_id: str | None = None
    person_ref: str | None = None
    lock_epoch: int | None = None

    def validate(self) -> None:
        if self.status not in ALLOWED_PROCESS_STATUSES:
            raise ValueError(f"Unsupported process status: {self.status}")
        if self.process_name.upper() in FORBIDDEN_PRODUCTION_CONCLUSIONS:
            raise ValueError("Production OK/NG/PASS/FAIL conclusions are forbidden")
        if self.training_eligible and self.review_state != "human_confirmed":
            raise ValueError("Only an explicitly reviewed item can be training eligible")
        if self.status == "proposed" and not (
            self.source_action_event_ids or self.source_interaction_ids
        ):
            raise ValueError("A proposed process step requires traceable evidence")


@dataclass
class MultimodalResult:
    schema_version: str
    project: str
    source_video: dict[str, Any]
    validation_flags: ValidationFlags
    pose_segments: list[dict[str, Any]] = field(default_factory=list)
    action_events: list[dict[str, Any]] = field(default_factory=list)
    evidence_timeline: list[dict[str, Any]] = field(default_factory=list)
    evidence_timeline_metrics: dict[str, Any] = field(default_factory=dict)
    hand_pose_frames: list[dict[str, Any]] = field(default_factory=list)
    object_tracks: list[dict[str, Any]] = field(default_factory=list)
    interaction_events: list[dict[str, Any]] = field(default_factory=list)
    process_steps: list[dict[str, Any]] = field(default_factory=list)
    layer_states: list[LayerState] = field(default_factory=list)
    tracking_summary: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        self.validation_flags.validate()
        payload = asdict(self)
        automatic_collections = (
            self.pose_segments,
            self.action_events,
            self.evidence_timeline,
            self.hand_pose_frames,
            self.object_tracks,
            self.interaction_events,
            self.process_steps,
        )
        for records in automatic_collections:
            for record in records:
                if record.get("status") == "confirmed":
                    raise ValueError(
                        "Automatic analysis cannot create confirmed semantics"
                    )
                if bool(record.get("training_eligible", False)):
                    raise ValueError(
                        "Automatic analysis must keep training_eligible=false"
                    )
                if "human_confirmed_semantic" in record:
                    raise ValueError(
                        "Automatic analysis cannot set human_confirmed_semantic"
                    )
        for step in self.process_steps:
            status = str(step.get("status", ""))
            if status not in ALLOWED_PROCESS_STATUSES:
                raise ValueError(f"Invalid process step status: {status}")
        return payload
