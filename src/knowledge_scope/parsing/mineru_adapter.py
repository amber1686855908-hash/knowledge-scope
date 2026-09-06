"""Convert MinerU content-list JSON into the canonical document model."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from uuid import UUID

from pydantic import ValidationError

from .models import (
    BoundingBox,
    CanonicalBlock,
    CanonicalDocument,
    FormulaBlock,
    ImageBlock,
    Page,
    TableBlock,
    TextBlock,
    TitleBlock,
)

MINERU_BBOX_SCALE = 1000.0
MINERU_BBOX_TOLERANCE = 1.0
AUXILIARY_TYPES = frozenset({"discarded", "footer", "header", "page_footnote", "page_number"})


class MineruAdapterError(RuntimeError):
    """Raised when MinerU output cannot be mapped safely to canonical data."""


@dataclass(frozen=True, slots=True)
class AdapterStats:
    """Parsing counts for one adapter run; these are not accuracy metrics."""

    pages: int
    input_items: int
    canonical_blocks: int
    title_blocks: int
    text_blocks: int
    tables: int
    formulas: int
    images: int
    skipped_auxiliary: int
    unsupported_items: int
    bbox_clamped: int = 0
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, int]:
        """Return concise CLI-friendly counts."""
        return {
            "pages": self.pages,
            "mineru_input_items": self.input_items,
            "canonical_blocks": self.canonical_blocks,
            "title_blocks": self.title_blocks,
            "text_blocks": self.text_blocks,
            "tables": self.tables,
            "formulas": self.formulas,
            "images": self.images,
            "skipped_auxiliary": self.skipped_auxiliary,
            "unsupported_items": self.unsupported_items,
            "bbox_clamped": self.bbox_clamped,
            "warnings": len(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class AdaptedCanonicalDocument:
    """Canonical output together with the counts produced during adaptation."""

    document: CanonicalDocument
    stats: AdapterStats


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MineruAdapterError("MinerU JSON artifact could not be read") from error


def infer_page_count(output_dir: Path) -> int | None:
    """Read the page count from the optional MinerU middle artifact."""
    for middle_path in sorted(Path(output_dir).rglob("*_middle.json")):
        try:
            payload = _read_json(middle_path)
        except MineruAdapterError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("pdf_info"), list):
            page_count = len(payload["pdf_info"])
            if page_count > 0:
                return page_count
    return None


def _require_page_index(item: Mapping[str, Any], item_index: int) -> int:
    page_idx = item.get("page_idx")
    if isinstance(page_idx, bool) or not isinstance(page_idx, int) or page_idx < 0:
        raise MineruAdapterError(f"MinerU item {item_index} has an invalid page_idx")
    return page_idx


def _adapt_bbox(
    raw_bbox: Any,
    item_index: int,
    warnings: list[str],
    bbox_clamps: list[int],
) -> BoundingBox | None:
    if raw_bbox is None:
        return None
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        raise MineruAdapterError(f"MinerU item {item_index} has an invalid bbox shape")
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in raw_bbox):
        raise MineruAdapterError(f"MinerU item {item_index} has non-numeric bbox coordinates")

    coordinates = [float(value) for value in raw_bbox]
    if any(not isfinite(value) for value in coordinates):
        raise MineruAdapterError(f"MinerU item {item_index} has non-finite bbox coordinates")

    if any(
        value < -MINERU_BBOX_TOLERANCE or value > MINERU_BBOX_SCALE + MINERU_BBOX_TOLERANCE
        for value in coordinates
    ):
        raise MineruAdapterError(f"MinerU item {item_index} has an invalid bbox")

    if any(value < 0 or value > MINERU_BBOX_SCALE for value in coordinates):
        coordinates = [min(max(value, 0.0), MINERU_BBOX_SCALE) for value in coordinates]
        warnings.append(f"bbox_clamped:item={item_index}")
        bbox_clamps.append(item_index)

    normalized = [value / MINERU_BBOX_SCALE for value in coordinates]
    try:
        return BoundingBox(
            x0=normalized[0],
            y0=normalized[1],
            x1=normalized[2],
            y1=normalized[3],
        )
    except ValidationError as error:
        raise MineruAdapterError(f"MinerU item {item_index} has an invalid bbox") from error


def _optional_caption(raw_caption: Any, item_index: int) -> str | None:
    if raw_caption is None:
        return None
    if isinstance(raw_caption, str):
        return raw_caption if raw_caption.strip() else None
    if isinstance(raw_caption, list):
        parts = [part.strip() for part in raw_caption if isinstance(part, str) and part.strip()]
        return "\n".join(parts) or None
    raise MineruAdapterError(f"MinerU item {item_index} has an invalid caption")


def _flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(part for part in (_flatten_text(item) for item in value) if part.strip())
    if isinstance(value, dict):
        for key in ("text", "content"):
            if key in value:
                return _flatten_text(value[key])
    return ""


def _asset_reference(
    raw_path: Any,
    *,
    item_index: int,
    content_list_dir: Path,
    artifact_dir: Path,
) -> str:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise MineruAdapterError(f"MinerU item {item_index} has no usable image asset")

    path_text = raw_path.strip()
    windows_path = PureWindowsPath(path_text)
    if (
        path_text.startswith(("/", "\\"))
        or windows_path.drive
        or windows_path.is_absolute()
        or "://" in path_text
    ):
        raise MineruAdapterError(f"MinerU item {item_index} has an unsafe image asset path")

    normalized_text = path_text.replace("\\", "/")
    if any(segment in {".", ".."} for segment in normalized_text.split("/")):
        raise MineruAdapterError(f"MinerU item {item_index} has an unsafe image asset path")
    path_parts = PurePosixPath(normalized_text).parts
    if not path_parts:
        raise MineruAdapterError(f"MinerU item {item_index} has an unsafe image asset path")

    artifact_root = artifact_dir.resolve()
    candidate_bases = (content_list_dir.resolve(), artifact_root)
    for base in candidate_bases:
        candidate = (base.joinpath(*path_parts)).resolve()
        try:
            relative = candidate.relative_to(artifact_root)
        except ValueError:
            continue
        if candidate.is_file():
            return relative.as_posix()

    raise MineruAdapterError(f"MinerU item {item_index} image asset is missing or outside output")


def _validate_optional_asset(
    item: Mapping[str, Any],
    *,
    item_index: int,
    content_list_dir: Path,
    artifact_dir: Path,
) -> None:
    raw_path = item.get("img_path")
    if raw_path not in (None, ""):
        _asset_reference(
            raw_path,
            item_index=item_index,
            content_list_dir=content_list_dir,
            artifact_dir=artifact_dir,
        )


def _base_block_fields(
    item: Mapping[str, Any],
    *,
    item_index: int,
    block_id: str,
    reading_order: int,
    warnings: list[str],
    bbox_clamps: list[int],
) -> dict[str, Any]:
    return {
        "block_id": block_id,
        "reading_order": reading_order,
        "bbox": _adapt_bbox(item.get("bbox"), item_index, warnings, bbox_clamps),
    }


def _unsupported_warning(item_index: int, block_type: str) -> str:
    return f"item {item_index}: unsupported or empty MinerU block type {block_type!r}"


def _map_item(
    item: Mapping[str, Any],
    *,
    item_index: int,
    block_id: str,
    reading_order: int,
    content_list_dir: Path,
    artifact_dir: Path,
    warnings: list[str],
    bbox_clamps: list[int],
) -> CanonicalBlock | None:
    block_type = item["type"]
    base_fields = _base_block_fields(
        item,
        item_index=item_index,
        block_id=block_id,
        reading_order=reading_order,
        warnings=warnings,
        bbox_clamps=bbox_clamps,
    )

    if block_type == "text":
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            warnings.append(_unsupported_warning(item_index, block_type))
            return None
        text_level = item.get("text_level")
        if text_level is not None and (
            isinstance(text_level, bool) or not isinstance(text_level, int)
        ):
            warnings.append(f"item {item_index}: non-integer text_level mapped as ordinary text")
        block_class = TitleBlock if isinstance(text_level, int) and text_level > 0 else TextBlock
        return block_class(**base_fields, text=text)

    if block_type == "table":
        table_body = item.get("table_body")
        if not isinstance(table_body, str) or not table_body.strip():
            raise MineruAdapterError(f"MinerU table item {item_index} has no table_body")
        _validate_optional_asset(
            item,
            item_index=item_index,
            content_list_dir=content_list_dir,
            artifact_dir=artifact_dir,
        )
        caption = _optional_caption(item.get("table_caption"), item_index)
        if table_body.lstrip().lower().startswith("<table"):
            return TableBlock(**base_fields, html=table_body, caption=caption)
        return TableBlock(**base_fields, markdown=table_body, caption=caption)

    if block_type == "equation":
        latex = item.get("text")
        if not isinstance(latex, str) or not latex.strip():
            warnings.append(_unsupported_warning(item_index, block_type))
            return None
        _validate_optional_asset(
            item,
            item_index=item_index,
            content_list_dir=content_list_dir,
            artifact_dir=artifact_dir,
        )
        if item.get("text_format") not in (None, "latex"):
            warnings.append(f"item {item_index}: equation text_format was not latex")
        return FormulaBlock(**base_fields, latex=latex)

    if block_type == "image":
        if item.get("img_path") in (None, ""):
            warnings.append(_unsupported_warning(item_index, block_type))
            return None
        asset_ref = _asset_reference(
            item["img_path"],
            item_index=item_index,
            content_list_dir=content_list_dir,
            artifact_dir=artifact_dir,
        )
        return ImageBlock(
            **base_fields,
            asset_ref=asset_ref,
            caption=_optional_caption(item.get("image_caption"), item_index),
        )

    if block_type == "chart":
        if item.get("img_path") in (None, ""):
            warnings.append(_unsupported_warning(item_index, block_type))
            return None
        asset_ref = _asset_reference(
            item["img_path"],
            item_index=item_index,
            content_list_dir=content_list_dir,
            artifact_dir=artifact_dir,
        )
        return ImageBlock(
            **base_fields,
            asset_ref=asset_ref,
            caption=_optional_caption(item.get("chart_caption"), item_index),
        )

    if block_type == "list":
        text = _flatten_text(item.get("list_items")) or _flatten_text(item.get("text"))
        if not text.strip():
            warnings.append(_unsupported_warning(item_index, block_type))
            return None
        return TextBlock(**base_fields, text=text)

    warnings.append(_unsupported_warning(item_index, block_type))
    return None


def adapt_content_list(
    content_list_path: Path,
    document_id: UUID,
    artifact_dir: Path,
    *,
    page_count: int | None = None,
) -> AdaptedCanonicalDocument:
    """Adapt one stable MinerU ``*_content_list.json`` artifact."""
    content_path = Path(content_list_path)
    output_root = Path(artifact_dir)
    payload = _read_json(content_path)
    if not isinstance(payload, list):
        raise MineruAdapterError("MinerU content_list JSON must contain an array")
    if not all(isinstance(item, Mapping) for item in payload):
        raise MineruAdapterError("MinerU content_list items must be objects")

    if page_count is not None and (
        isinstance(page_count, bool) or not isinstance(page_count, int) or page_count < 1
    ):
        raise MineruAdapterError("MinerU page count must be a positive integer")

    pages: dict[int, list[CanonicalBlock]] = defaultdict(list)
    warnings: list[str] = []
    bbox_clamps: list[int] = []
    skipped_auxiliary = 0
    unsupported_items = 0
    for item_index, raw_item in enumerate(payload):
        item = raw_item
        block_type = item.get("type")
        if not isinstance(block_type, str) or not block_type:
            raise MineruAdapterError(f"MinerU item {item_index} has no valid type")
        page_idx = _require_page_index(item, item_index)
        if page_count is not None and page_idx >= page_count:
            raise MineruAdapterError(f"MinerU item {item_index} is outside the page count")
        if block_type in AUXILIARY_TYPES:
            skipped_auxiliary += 1
            continue

        reading_order = len(pages[page_idx])
        block = _map_item(
            item,
            item_index=item_index,
            block_id=f"p{page_idx + 1}-b{reading_order + 1}",
            reading_order=reading_order,
            content_list_dir=content_path.parent,
            artifact_dir=output_root,
            warnings=warnings,
            bbox_clamps=bbox_clamps,
        )
        if block is None:
            unsupported_items += 1
            continue
        pages[page_idx].append(block)

    if page_count is None:
        if not pages:
            raise MineruAdapterError("MinerU content_list contains no page content")
        page_count = max(pages) + 1

    canonical_pages = [
        Page(page_number=page_idx + 1, blocks=pages.get(page_idx, []))
        for page_idx in range(page_count)
    ]
    try:
        document = CanonicalDocument(document_id=document_id, pages=canonical_pages)
    except ValidationError as error:
        raise MineruAdapterError("adapted MinerU output failed canonical validation") from error

    block_types = [block.type for page in document.pages for block in page.blocks]
    stats = AdapterStats(
        pages=page_count,
        input_items=len(payload),
        canonical_blocks=len(block_types),
        title_blocks=block_types.count("title"),
        text_blocks=block_types.count("text"),
        tables=block_types.count("table"),
        formulas=block_types.count("formula"),
        images=block_types.count("image"),
        skipped_auxiliary=skipped_auxiliary,
        unsupported_items=unsupported_items,
        bbox_clamped=len(bbox_clamps),
        warnings=tuple(warnings),
    )
    return AdaptedCanonicalDocument(document=document, stats=stats)


__all__ = [
    "AUXILIARY_TYPES",
    "AdaptedCanonicalDocument",
    "AdapterStats",
    "MineruAdapterError",
    "adapt_content_list",
    "infer_page_count",
]
