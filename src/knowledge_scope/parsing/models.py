"""Parser-independent canonical document models."""

from __future__ import annotations

from pathlib import PureWindowsPath
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

CANONICAL_SCHEMA_VERSION = "1.0"


class _CanonicalBaseModel(BaseModel):
    """Shared validation settings for canonical data."""

    model_config = ConfigDict(extra="forbid")


class BoundingBox(_CanonicalBaseModel):
    """A normalized box in a page's top-left coordinate system."""

    x0: float = Field(ge=0, le=1)
    y0: float = Field(ge=0, le=1)
    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_dimensions(self) -> Self:
        if self.x0 >= self.x1:
            raise ValueError("x0 must be less than x1")
        if self.y0 >= self.y1:
            raise ValueError("y0 must be less than y1")
        return self


def _require_non_blank(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must contain at least one non-whitespace character")
    return value


class BlockBase(_CanonicalBaseModel):
    """Fields shared by every canonical block variant."""

    block_id: str = Field(min_length=1)
    reading_order: StrictInt = Field(ge=0)
    bbox: BoundingBox | None = None

    @field_validator("block_id")
    @classmethod
    def validate_block_id(cls, value: str) -> str:
        return _require_non_blank(value, "block_id")


class TitleBlock(BlockBase):
    """A document title or heading represented as text."""

    type: Literal["title"] = "title"
    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _require_non_blank(value, "text")


class TextBlock(BlockBase):
    """A paragraph or other ordinary document text."""

    type: Literal["text"] = "text"
    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _require_non_blank(value, "text")


class TableBlock(BlockBase):
    """A table represented by parser-independent Markdown."""

    type: Literal["table"] = "table"
    markdown: str = Field(min_length=1)
    caption: str | None = None

    @field_validator("markdown")
    @classmethod
    def validate_markdown(cls, value: str) -> str:
        return _require_non_blank(value, "markdown")


class FormulaBlock(BlockBase):
    """A formula represented as LaTeX without evaluation semantics."""

    type: Literal["formula"] = "formula"
    latex: str = Field(min_length=1)

    @field_validator("latex")
    @classmethod
    def validate_latex(cls, value: str) -> str:
        return _require_non_blank(value, "latex")


class ImageBlock(BlockBase):
    """A reference to an image asset managed by a future storage boundary."""

    type: Literal["image"] = "image"
    asset_ref: str = Field(min_length=1)
    caption: str | None = None

    @field_validator("asset_ref")
    @classmethod
    def validate_asset_ref(cls, value: str) -> str:
        value = _require_non_blank(value, "asset_ref")
        path_segments = value.replace("\\", "/").split("/")
        if any(segment in {".", ".."} for segment in path_segments):
            raise ValueError("asset_ref must not contain '.' or '..' path segments")
        windows_path = PureWindowsPath(value)
        if (
            value.startswith(("/", "\\"))
            or windows_path.drive
            or windows_path.is_absolute()
            or "://" in value
        ):
            raise ValueError("asset_ref must be an opaque non-absolute reference")
        return value


type CanonicalBlock = Annotated[
    TitleBlock | TextBlock | TableBlock | FormulaBlock | ImageBlock,
    Field(discriminator="type"),
]


class Page(_CanonicalBaseModel):
    """One 1-based page containing blocks in canonical reading order."""

    page_number: StrictInt = Field(ge=1)
    blocks: list[CanonicalBlock] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_reading_order(self) -> Self:
        orders = [block.reading_order for block in self.blocks]
        if orders != list(range(len(self.blocks))):
            raise ValueError("reading_order must be contiguous from zero within a page")
        return self


class CanonicalDocument(_CanonicalBaseModel):
    """The parser-independent representation of one uploaded document."""

    schema_version: Literal["1.0"] = CANONICAL_SCHEMA_VERSION
    document_id: UUID
    pages: list[Page] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_document_structure(self) -> Self:
        page_numbers = [page.page_number for page in self.pages]
        if page_numbers != list(range(1, len(self.pages) + 1)):
            raise ValueError("page_number values must be ordered, unique, and contiguous from 1")

        block_ids = [block.block_id for page in self.pages for block in page.blocks]
        if len(block_ids) != len(set(block_ids)):
            raise ValueError("block_id must be unique within a document")
        return self


__all__ = [
    "CANONICAL_SCHEMA_VERSION",
    "BlockBase",
    "BoundingBox",
    "CanonicalBlock",
    "CanonicalDocument",
    "FormulaBlock",
    "ImageBlock",
    "Page",
    "TableBlock",
    "TextBlock",
    "TitleBlock",
]
