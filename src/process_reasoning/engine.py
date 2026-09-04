"""Never promote incomplete multimodal evidence to a production conclusion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.contracts import LayerState


@dataclass
class ProcessReasoningOutput:
    state: LayerState
    process_steps: list[dict[str, Any]] = field(default_factory=list)


class ProcessReasoner:
    def infer(
        self,
        *,
        pose_action_events: list[dict[str, Any]],
        interaction_events: list[dict[str, Any]],
        temporal_action_candidates: list[dict[str, Any]],
        required_layers_available: bool,
    ) -> ProcessReasoningOutput:
        if not required_layers_available:
            return ProcessReasoningOutput(
                state=LayerState(
                    layer="process_reasoning",
                    status="unavailable",
                    reason=(
                        "real object/interaction/temporal evidence is incomplete; "
                        "no process step candidate generated"
                    ),
                    evidence_count=0,
                )
            )
        if not interaction_events or not temporal_action_candidates:
            return ProcessReasoningOutput(
                state=LayerState(
                    layer="process_reasoning",
                    status="not_observed",
                    reason="no evidence-qualified process step was observed",
                    evidence_count=0,
                )
            )
        return ProcessReasoningOutput(
            state=LayerState(
                layer="process_reasoning",
                status="uncertain",
                reason="process rule set is not yet human-confirmed",
                evidence_count=0,
            ),
            process_steps=[],
        )
