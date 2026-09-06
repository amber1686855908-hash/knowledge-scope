import json
from pathlib import Path
from uuid import UUID

import pytest

from knowledge_scope.parsing.mineru_adapter import (
    MineruAdapterError,
    adapt_content_list,
    infer_page_count,
)

DOCUMENT_ID = UUID("11111111-1111-1111-1111-111111111111")


def _write_content_list(
    tmp_path: Path,
    items: list[dict[str, object]],
    *,
    page_count: int | None = None,
) -> tuple[Path, Path]:
    artifact_dir = tmp_path / "mineru"
    content_dir = artifact_dir / "sample" / "auto"
    image_dir = content_dir / "images"
    image_dir.mkdir(parents=True)
    for image_name in ("image.png", "chart.png", "table.png", "formula.png"):
        (image_dir / image_name).write_bytes(b"asset")

    content_list_path = content_dir / "sample_content_list.json"
    content_list_path.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    if page_count is not None:
        (content_dir / "sample_middle.json").write_text(
            json.dumps({"pdf_info": [{} for _ in range(page_count)]}),
            encoding="utf-8",
        )
    return content_list_path, artifact_dir


def _mixed_items() -> list[dict[str, object]]:
    return [
        {
            "type": "text",
            "page_idx": 0,
            "text": "章节标题",
            "text_level": 1,
            "bbox": [0, 0, 500, 100],
        },
        {"type": "text", "page_idx": 0, "text": "正文内容", "bbox": [0, 120, 900, 260]},
        {
            "type": "table",
            "page_idx": 0,
            "table_body": "<table><tr><td>项目</td></tr></table>",
            "table_caption": ["表格说明"],
            "img_path": "images/table.png",
            "bbox": [0, 280, 900, 500],
        },
        {
            "type": "equation",
            "page_idx": 0,
            "text": r"\frac{a}{b}",
            "text_format": "latex",
            "img_path": "images/formula.png",
            "bbox": [0, 520, 400, 600],
        },
        {
            "type": "image",
            "page_idx": 0,
            "img_path": "images/image.png",
            "image_caption": ["图片说明"],
            "bbox": [0, 620, 500, 900],
        },
        {
            "type": "chart",
            "page_idx": 0,
            "img_path": "images/chart.png",
            "chart_caption": ["图表说明"],
            "bbox": [500, 620, 1000, 900],
        },
        {
            "type": "list",
            "page_idx": 0,
            "list_items": ["第一项", {"text": "第二项"}],
            "bbox": [0, 910, 900, 1000],
        },
        {"type": "header", "page_idx": 0, "text": "页眉", "bbox": [0, 0, 1000, 20]},
        {"type": "page_number", "page_idx": 0, "text": "1", "bbox": [950, 950, 1000, 1000]},
        {"type": "code", "page_idx": 0, "code_body": "print('not canonical')"},
        {"type": "aside_text", "page_idx": 0, "text": "边栏内容"},
        {"type": "mystery", "page_idx": 0, "content": "meaningful unsupported content"},
        {"type": "text", "page_idx": 2, "text": "第三页内容", "bbox": [0, 0, 1000, 100]},
    ]


def test_adapter_maps_realistic_items_and_keeps_deterministic_order(tmp_path: Path) -> None:
    content_path, artifact_dir = _write_content_list(tmp_path, _mixed_items(), page_count=3)

    adapted = adapt_content_list(content_path, DOCUMENT_ID, artifact_dir, page_count=3)
    repeated = adapt_content_list(content_path, DOCUMENT_ID, artifact_dir, page_count=3)

    assert adapted.document.model_dump_json() == repeated.document.model_dump_json()
    assert adapted.stats.as_dict() == {
        "pages": 3,
        "mineru_input_items": 13,
        "canonical_blocks": 8,
        "title_blocks": 1,
        "text_blocks": 3,
        "tables": 1,
        "formulas": 1,
        "images": 2,
        "skipped_auxiliary": 2,
        "unsupported_items": 3,
        "bbox_clamped": 0,
        "warnings": 3,
    }

    first_page = adapted.document.pages[0]
    assert [block.reading_order for block in first_page.blocks] == list(range(7))
    assert [block.block_id for block in first_page.blocks] == [f"p1-b{i}" for i in range(1, 8)]
    assert adapted.document.pages[1].blocks == []
    assert adapted.document.pages[2].blocks[0].block_id == "p3-b1"
    assert first_page.blocks[0].type == "title"
    assert first_page.blocks[2].html == "<table><tr><td>项目</td></tr></table>"
    assert first_page.blocks[3].latex == r"\frac{a}{b}"
    assert first_page.blocks[4].asset_ref == "sample/auto/images/image.png"
    assert first_page.blocks[5].asset_ref == "sample/auto/images/chart.png"
    assert first_page.blocks[1].bbox is not None
    assert first_page.blocks[1].bbox.x1 == 0.9


def test_adapter_uses_markdown_when_table_body_is_markdown(tmp_path: Path) -> None:
    items = [
        {
            "type": "table",
            "page_idx": 0,
            "table_body": "| 项目 |\n| --- |\n| 正常 |",
        }
    ]
    content_path, artifact_dir = _write_content_list(tmp_path, items)

    table = adapt_content_list(content_path, DOCUMENT_ID, artifact_dir).document.pages[0].blocks[0]

    assert table.markdown == "| 项目 |\n| --- |\n| 正常 |"
    assert table.html is None


def test_adapter_can_preserve_empty_pages_from_middle_artifact(tmp_path: Path) -> None:
    content_path, artifact_dir = _write_content_list(tmp_path, [], page_count=2)

    adapted = adapt_content_list(
        content_path,
        DOCUMENT_ID,
        artifact_dir,
        page_count=infer_page_count(artifact_dir),
    )

    assert len(adapted.document.pages) == 2
    assert all(not page.blocks for page in adapted.document.pages)


@pytest.mark.parametrize(
    ("bbox", "expected"),
    [
        ([0, 699, 1000, 1001], (0.0, 0.699, 1.0, 1.0)),
        ([-1, 0, 500, 500], (0.0, 0.0, 0.5, 0.5)),
    ],
)
def test_adapter_clamps_one_unit_bbox_overshoot_and_reports_it(
    tmp_path: Path,
    bbox: list[int],
    expected: tuple[float, float, float, float],
) -> None:
    items = [
        {
            "type": "image",
            "page_idx": 0,
            "img_path": "images/image.png",
            "bbox": bbox,
        }
    ]
    content_path, artifact_dir = _write_content_list(tmp_path, items)

    adapted = adapt_content_list(content_path, DOCUMENT_ID, artifact_dir)

    bbox = adapted.document.pages[0].blocks[0].bbox
    assert bbox is not None
    assert (bbox.x0, bbox.y0, bbox.x1, bbox.y1) == expected
    assert adapted.stats.bbox_clamped == 1
    assert adapted.stats.warnings == ("bbox_clamped:item=0",)


def test_adapter_accepts_documented_bbox_boundary_without_clamping(tmp_path: Path) -> None:
    items = [{"type": "text", "page_idx": 0, "text": "正文", "bbox": [0, 0, 1000, 1000]}]
    content_path, artifact_dir = _write_content_list(tmp_path, items)

    adapted = adapt_content_list(content_path, DOCUMENT_ID, artifact_dir)

    assert adapted.stats.bbox_clamped == 0
    assert adapted.stats.warnings == ()


@pytest.mark.parametrize(
    "bbox",
    [
        [0, 0, 100],
        [-2, 0, 100, 100],
        [0, 0, 1002, 100],
        [0, 0, 1000, 1002],
        [1000, 0, 1001, 100],
        [500, 100, 400, 200],
    ],
)
def test_adapter_rejects_malformed_or_gross_bbox(tmp_path: Path, bbox: list[int]) -> None:
    items = [{"type": "text", "page_idx": 0, "text": "正文", "bbox": bbox}]
    content_path, artifact_dir = _write_content_list(tmp_path, items)

    with pytest.raises(MineruAdapterError, match="bbox"):
        adapt_content_list(content_path, DOCUMENT_ID, artifact_dir)


@pytest.mark.parametrize(
    "asset_path",
    [
        "../image.png",
        r"..\image.png",
        "assets/../image.png",
        "images/./image.png",
        r"images\.\image.png",
    ],
)
def test_adapter_rejects_unsafe_image_asset_paths(tmp_path: Path, asset_path: str) -> None:
    items = [{"type": "image", "page_idx": 0, "img_path": asset_path}]
    content_path, artifact_dir = _write_content_list(tmp_path, items)

    with pytest.raises(MineruAdapterError, match="unsafe image asset path"):
        adapt_content_list(content_path, DOCUMENT_ID, artifact_dir)


def test_adapter_rejects_table_without_preservable_content(tmp_path: Path) -> None:
    items = [{"type": "table", "page_idx": 0, "table_body": "   "}]
    content_path, artifact_dir = _write_content_list(tmp_path, items)

    with pytest.raises(MineruAdapterError, match="table_body"):
        adapt_content_list(content_path, DOCUMENT_ID, artifact_dir)
