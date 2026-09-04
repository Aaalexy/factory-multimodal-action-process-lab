"""Object perception contracts; the kickoff ships with no object model."""

from .provider import (
    NotConfiguredObjectPerception,
    ObjectPerceptionOutput,
    ObjectPerceptionProvider,
)

__all__ = [
    "NotConfiguredObjectPerception",
    "ObjectPerceptionOutput",
    "ObjectPerceptionProvider",
]
