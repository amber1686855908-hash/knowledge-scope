"""Small shared types used by linking prompts and sample review code."""

from __future__ import annotations

from dataclasses import dataclass

from knowledge_scope.graph.models import GraphEntity


@dataclass(frozen=True, slots=True)
class LocalEntityContext:
    """A local entity plus bounded human-review context, never a file path."""

    entity: GraphEntity
    source_excerpt: str = ""
    subject: str | None = None


__all__ = ["LocalEntityContext"]
