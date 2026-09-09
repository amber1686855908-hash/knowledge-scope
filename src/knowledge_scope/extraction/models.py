"""Strict, provider-independent schemas for LLM graph extraction output."""

# ruff: noqa: RUF001

from __future__ import annotations

import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

EntityType = Literal[
    "人物",
    "地点",
    "组织",
    "事件",
    "概念",
    "物质",
    "制度",
    "技术",
    "时间",
    "数值",
]
RelationType = Literal[
    "属于",
    "位于",
    "包括",
    "组成",
    "导致",
    "影响",
    "发生于",
    "先于",
    "用于",
    "体现",
    "具有",
    "相关",
]

ALLOWED_ENTITY_TYPES: tuple[EntityType, ...] = (
    "人物",
    "地点",
    "组织",
    "事件",
    "概念",
    "物质",
    "制度",
    "技术",
    "时间",
    "数值",
)
ENTITY_TYPE_DESCRIPTIONS: dict[EntityType, str] = {
    "人物": "具体的人或人物群体中的单个角色",
    "地点": "地理位置、区域、场所或空间对象",
    "组织": "学校、机构、团体、政党或其他组织",
    "事件": "已经发生或明确讨论的历史、社会、自然事件",
    "概念": "抽象术语、理论、原则、思想或现象",
    "物质": "具体物质、材料、化学物质或生物对象",
    "制度": "制度、规则、政策、法律或规范安排",
    "技术": "技术、方法、工具或工艺",
    "时间": "时间点、日期、时期或时代",
    "数值": "带有明确数值或量纲的数量对象",
}
ALLOWED_RELATION_TYPES: tuple[RelationType, ...] = (
    "属于",
    "位于",
    "包括",
    "组成",
    "导致",
    "影响",
    "发生于",
    "先于",
    "用于",
    "体现",
    "具有",
    "相关",
)
RELATION_TYPE_DESCRIPTIONS: dict[RelationType, str] = {
    "属于": "source 是 target 的成员、类别或组成范围中的一项",
    "位于": "source 的位置在 target",
    "包括": "source 包含 target",
    "组成": "source 由 target 构成，或 target 是 source 的组成部分",
    "导致": "source 直接导致 target",
    "影响": "source 对 target 产生影响",
    "发生于": "事件 source 发生在时间或地点 target",
    "先于": "source 在时间或顺序上先于 target",
    "用于": "source 被用于 target",
    "体现": "source 体现或表现出 target",
    "具有": "source 具有 target 这一属性或特征",
    "相关": "source 与 target 存在明确但未细分的关联",
}


class _ExtractionBaseModel(BaseModel):
    """Shared strict settings for untrusted model output."""

    model_config = ConfigDict(extra="forbid")


def normalize_label(value: str) -> str:
    """Normalize a label for comparison without changing the extracted spelling."""

    return " ".join(unicodedata.normalize("NFKC", value).strip().casefold().split())


def _clean_text(value: str, field_name: str) -> str:
    cleaned = " ".join(unicodedata.normalize("NFKC", value).strip().split())
    if not cleaned:
        raise ValueError(f"{field_name} must contain at least one non-whitespace character")
    return cleaned


class ExtractedEntity(_ExtractionBaseModel):
    """One entity mention proposed by the model, without application identity fields."""

    name: str = Field(min_length=1, max_length=200)
    entity_type: EntityType
    aliases: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _clean_text(value, "name")

    @field_validator("aliases")
    @classmethod
    def validate_aliases(cls, values: list[str]) -> list[str]:
        cleaned = [_clean_text(value, "alias") for value in values]
        keys = [normalize_label(value) for value in cleaned]
        if len(keys) != len(set(keys)):
            raise ValueError("aliases must be unique after normalization")
        return sorted(cleaned, key=normalize_label)


class ExtractedRelation(_ExtractionBaseModel):
    """One directed relation with a bounded source-text evidence quote."""

    source: str = Field(min_length=1, max_length=200)
    target: str = Field(min_length=1, max_length=200)
    relation_type: RelationType
    evidence: str = Field(
        min_length=1,
        max_length=500,
        description="A short, contiguous quote from the supplied chunk supporting this relation.",
    )

    @field_validator("source", "target", "evidence")
    @classmethod
    def validate_text(cls, value: str, info: ValidationInfo) -> str:
        return _clean_text(value, info.field_name)


class ExtractionOutput(_ExtractionBaseModel):
    """The only JSON shape accepted from the LLM."""

    entities: list[ExtractedEntity] = Field(default_factory=list, max_length=100)
    relations: list[ExtractedRelation] = Field(default_factory=list, max_length=200)


__all__ = [
    "ALLOWED_ENTITY_TYPES",
    "ALLOWED_RELATION_TYPES",
    "ENTITY_TYPE_DESCRIPTIONS",
    "RELATION_TYPE_DESCRIPTIONS",
    "EntityType",
    "ExtractedEntity",
    "ExtractedRelation",
    "ExtractionOutput",
    "RelationType",
    "normalize_label",
]
