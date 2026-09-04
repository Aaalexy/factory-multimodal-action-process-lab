"""Explicitly unavailable object layer until a real versioned model is supplied."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from src.contracts import LayerState


@dataclass
class ObjectPerceptionOutput:
    state: LayerState
    object_tracks: list[dict[str, Any]] = field(default_factory=list)


class ObjectPerceptionProvider(Protocol):
    model_version: str | None

    def analyze(self, frames: list[dict[str, Any]]) -> ObjectPerceptionOutput: ...


class NotConfiguredObjectPerception:
    """Null provider with an honest state, never synthetic detections."""

    model_version: str | None = None

    def analyze(self, frames: list[dict[str, Any]]) -> ObjectPerceptionOutput:
        return ObjectPerceptionOutput(
            state=LayerState(
                layer="object_perception",
                status="unavailable",
                reason="not_configured: no real factory part detector is installed",
                model_version=None,
                evidence_count=0,
            ),
            object_tracks=[],
        )
