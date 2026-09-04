"""Fail-explicit ONNX Runtime provider selection for Body Pose."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import onnxruntime as ort


BODY_PROVIDER_POLICIES = frozenset(
    {"auto", "prefer_cuda", "require_cuda", "cpu"}
)


class BodyProviderUnavailableError(RuntimeError):
    """Raised when the requested Body Pose provider cannot be honored."""


@dataclass
class BodyProviderStatus:
    policy: str
    available_providers: list[str]
    requested_providers: list[str]
    session_providers: list[str]
    active_provider: str | None
    fallback_active: bool
    fallback_reason: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def normalize_body_provider_policy(value: str | None) -> str:
    policy = str(value or "prefer_cuda").strip().lower()
    if policy not in BODY_PROVIDER_POLICIES:
        raise ValueError(
            "Body Pose provider policy must be auto, prefer_cuda, "
            "require_cuda or cpu"
        )
    return policy


def select_body_provider_request(
    policy: str,
    *,
    available_providers: Sequence[str] | None = None,
) -> tuple[list[str], str | None]:
    """Return an ordered provider request and any pre-session fallback reason."""

    normalized = normalize_body_provider_policy(policy)
    available = list(
        available_providers
        if available_providers is not None
        else ort.get_available_providers()
    )
    has_cpu = "CPUExecutionProvider" in available
    has_cuda = "CUDAExecutionProvider" in available
    if normalized == "cpu":
        if not has_cpu:
            raise BodyProviderUnavailableError(
                "CPUExecutionProvider is not available"
            )
        return ["CPUExecutionProvider"], None
    if normalized == "require_cuda":
        if not has_cuda:
            raise BodyProviderUnavailableError(
                "CUDAExecutionProvider is required but unavailable"
            )
        providers = ["CUDAExecutionProvider"]
        if has_cpu:
            providers.append("CPUExecutionProvider")
        return providers, None
    if has_cuda:
        providers = ["CUDAExecutionProvider"]
        if has_cpu:
            providers.append("CPUExecutionProvider")
        return providers, None
    if not has_cpu:
        raise BodyProviderUnavailableError(
            "Neither CUDAExecutionProvider nor CPUExecutionProvider is available"
        )
    return (
        ["CPUExecutionProvider"],
        "cuda_execution_provider_unavailable",
    )


def build_body_provider_status(
    *,
    policy: str,
    requested_providers: Sequence[str],
    session_providers: Sequence[str],
    fallback_reason: str | None,
    available_providers: Sequence[str] | None = None,
) -> BodyProviderStatus:
    normalized = normalize_body_provider_policy(policy)
    available = list(
        available_providers
        if available_providers is not None
        else ort.get_available_providers()
    )
    actual = list(session_providers)
    active = actual[0] if actual else None
    requested_cuda = "CUDAExecutionProvider" in requested_providers
    fallback_active = bool(
        normalized in {"auto", "prefer_cuda"}
        and active != "CUDAExecutionProvider"
    )
    reason = fallback_reason
    if requested_cuda and active != "CUDAExecutionProvider" and reason is None:
        reason = "cuda_session_not_active"
    if normalized == "require_cuda" and active != "CUDAExecutionProvider":
        raise BodyProviderUnavailableError(
            "CUDAExecutionProvider was requested but is not active in the session"
        )
    return BodyProviderStatus(
        policy=normalized,
        available_providers=available,
        requested_providers=list(requested_providers),
        session_providers=actual,
        active_provider=active,
        fallback_active=fallback_active,
        fallback_reason=reason,
    )
