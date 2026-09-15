"""Task-local context for safe provider-attempt observability.

The context is deliberately limited to evaluation/call metadata.  It is not a
cache and it never carries prompts, credentials, provider responses, or other
request data.  ``ContextVar`` keeps concurrent async requests isolated while
allowing the existing gateway interface to remain provider-independent.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from .schemas import LLMLogicalStage


@dataclass(frozen=True, slots=True)
class LLMObservationContext:
    """Optional safe metadata attached to the current logical request."""

    case_id: str | None = None
    logical_stage: LLMLogicalStage | None = None


_CURRENT_CONTEXT: ContextVar[LLMObservationContext | None] = ContextVar(
    "knowledgescope_llm_observation_context",
    default=None,
)


def logical_stage_for_task(task_type: str) -> LLMLogicalStage:
    """Map the existing task labels to an observable stage when available."""
    if task_type == "nl2sql":
        return "generation"
    if task_type == "chatbi_analysis":
        return "analysis"
    return "other"


def current_observation_context(task_type: str) -> LLMObservationContext:
    """Return task-local metadata, with a safe task-type fallback."""
    current = _CURRENT_CONTEXT.get() or LLMObservationContext()
    return LLMObservationContext(
        case_id=current.case_id,
        logical_stage=current.logical_stage or logical_stage_for_task(task_type),
    )


@contextmanager
def provider_observation_context(
    *,
    case_id: str | None = None,
    logical_stage: LLMLogicalStage | None = None,
) -> Iterator[None]:
    """Temporarily set case/stage metadata and always restore the prior context."""
    current = _CURRENT_CONTEXT.get() or LLMObservationContext()
    token = _CURRENT_CONTEXT.set(
        LLMObservationContext(
            case_id=case_id if case_id is not None else current.case_id,
            logical_stage=logical_stage if logical_stage is not None else current.logical_stage,
        )
    )
    try:
        yield
    finally:
        _CURRENT_CONTEXT.reset(token)


__all__ = [
    "LLMObservationContext",
    "current_observation_context",
    "logical_stage_for_task",
    "provider_observation_context",
]
