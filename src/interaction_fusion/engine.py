"""Fuse real wrist observations with real object tracks when both exist."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.contracts import LayerState


@dataclass
class InteractionFusionOutput:
    state: LayerState
    interaction_events: list[dict[str, Any]] = field(default_factory=list)


class InteractionFusionEngine:
    def derive(
        self,
        *,
        pose_frames: list[dict[str, Any]],
        object_tracks: list[dict[str, Any]],
        object_layer_status: str,
    ) -> InteractionFusionOutput:
        if object_layer_status not in {"available", "detected"}:
            return InteractionFusionOutput(
                state=LayerState(
                    layer="interaction_fusion",
                    status="unavailable",
                    reason="real object tracks are unavailable; no interaction event generated",
                    evidence_count=0,
                )
            )
        if not object_tracks:
            return InteractionFusionOutput(
                state=LayerState(
                    layer="interaction_fusion",
                    status="not_observed",
                    reason="object model ran but produced no traceable tracks",
                    evidence_count=0,
                )
            )
        if not pose_frames:
            return InteractionFusionOutput(
                state=LayerState(
                    layer="interaction_fusion",
                    status="unavailable",
                    reason="pose evidence unavailable",
                    evidence_count=0,
                )
            )

        # A future provider may add a calibrated wrist-to-box association here.
        # The kickoff deliberately returns no event because no real object model
        # is configured and uncalibrated proximity must not become fake evidence.
        return InteractionFusionOutput(
            state=LayerState(
                layer="interaction_fusion",
                status="unavailable",
                reason=(
                    "derived_interaction_candidate association is not calibrated; "
                    "true grasp is outside COCO-17"
                ),
                evidence_count=0,
            )
        )
