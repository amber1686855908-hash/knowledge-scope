"""Versioned prompt for ambiguous local-entity link adjudication."""

# ruff: noqa: RUF001

from __future__ import annotations

from knowledge_scope.llm.schemas import LLMMessage

from .models import EntityLinkCandidate
from .service_types import LocalEntityContext

LINKING_PROMPT_VERSION = "entity-linking-v1.0"
MAX_PROMPT_ALIASES = 12
MAX_PROMPT_ALIAS_CHARS = 80
MAX_PROMPT_EXCERPT_CHARS = 280


def _bounded(value: str, limit: int = MAX_PROMPT_EXCERPT_CHARS) -> str:
    normalized = " ".join(value.split())
    return normalized[:limit] + ("…" if len(normalized) > limit else "")


def _bounded_aliases(values: list[str]) -> str:
    """Keep provider input bounded even when extraction metadata is noisy."""

    aliases = sorted(
        (_bounded(value, MAX_PROMPT_ALIAS_CHARS) for value in values),
        key=lambda value: value.casefold(),
    )[:MAX_PROMPT_ALIASES]
    return ", ".join(aliases) or "无"


def build_link_adjudication_messages(
    candidate: EntityLinkCandidate,
    entity_a: LocalEntityContext,
    entity_b: LocalEntityContext,
) -> list[LLMMessage]:
    """Build a deterministic prompt without exposing internal identifiers."""

    del candidate
    system = (
        "你是知识库实体链接审核器。只能依据用户消息提供的两个实体和来源片段判断，"
        "不要使用外部常识。相同名称本身不足以建立链接；实体类型不兼容时必须 NO_LINK。"
        "只返回一个 JSON 对象，字段严格为 decision、confidence、reason；decision 只能是 "
        "LINK、NO_LINK 或 UNCERTAIN。不要输出任何 ID、别名之外的内部标识或额外字段。"
    )
    user = (
        "请判断实体 A 和实体 B 是否指向同一个知识库内的现实/概念对象。\n"
        "实体 A：\n"
        f"名称：{entity_a.entity.canonical_name}\n"
        f"类型：{entity_a.entity.entity_type}\n"
        f"别名：{_bounded_aliases(entity_a.entity.aliases)}\n"
        f"来源片段：{_bounded(entity_a.source_excerpt)}\n\n"
        "实体 B：\n"
        f"名称：{entity_b.entity.canonical_name}\n"
        f"类型：{entity_b.entity.entity_type}\n"
        f"别名：{_bounded_aliases(entity_b.entity.aliases)}\n"
        f"来源片段：{_bounded(entity_b.source_excerpt)}\n\n"
        "若证据不足，返回 UNCERTAIN；reason 使用简短中文说明。"
    )
    return [LLMMessage(role="system", content=system), LLMMessage(role="user", content=user)]


__all__ = [
    "LINKING_PROMPT_VERSION",
    "MAX_PROMPT_ALIASES",
    "MAX_PROMPT_ALIAS_CHARS",
    "MAX_PROMPT_EXCERPT_CHARS",
    "build_link_adjudication_messages",
]
