import inspect
from uuid import UUID

import pytest
from pydantic import TypeAdapter, ValidationError

from knowledge_scope.parsing import models as canonical_models
from knowledge_scope.parsing.models import (
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

DOCUMENT_ID = UUID("11111111-1111-1111-1111-111111111111")


def make_mixed_document() -> CanonicalDocument:
    return CanonicalDocument(
        document_id=DOCUMENT_ID,
        pages=[
            Page(
                page_number=1,
                blocks=[
                    TitleBlock(block_id="p1-b1", reading_order=0, text="设备维护规范"),
                    TextBlock(
                        block_id="p1-b2",
                        reading_order=1,
                        text="请按照以下步骤进行检查。",
                        bbox=BoundingBox(x0=0.1, y0=0.2, x1=0.9, y1=0.3),
                    ),
                    TableBlock(
                        block_id="p1-b3",
                        reading_order=2,
                        markdown="| 项目 | 状态 |\n| --- | --- |\n| 温度 | 正常 |",
                        caption="检查结果",
                    ),
                    FormulaBlock(block_id="p1-b4", reading_order=3, latex=r"E = mc^2"),
                    ImageBlock(block_id="p1-b5", reading_order=4, asset_ref="asset-image-1"),
                ],
            ),
            Page(
                page_number=2,
                blocks=[TextBlock(block_id="p2-b1", reading_order=0, text="附录")],
            ),
        ],
    )


def test_valid_document_supports_all_v1_block_types() -> None:
    document = make_mixed_document()

    assert document.schema_version == "1.0"
    assert document.document_id == DOCUMENT_ID
    assert [block.type for block in document.pages[0].blocks] == [
        "title",
        "text",
        "table",
        "formula",
        "image",
    ]
    assert document.pages[0].blocks[0].text == "设备维护规范"
    assert document.pages[0].blocks[2].markdown.startswith("| 项目")
    assert document.pages[0].blocks[3].latex == r"E = mc^2"
    assert document.pages[0].blocks[4].asset_ref == "asset-image-1"


def test_json_round_trip_is_deterministic() -> None:
    document = make_mixed_document()

    serialized = document.model_dump_json()
    restored = CanonicalDocument.model_validate_json(serialized)

    assert restored == document
    assert restored.model_dump_json() == serialized


def test_block_union_uses_type_discriminator_and_rejects_unknown_types() -> None:
    adapter = TypeAdapter(CanonicalBlock)

    assert isinstance(
        adapter.validate_python(
            {"block_id": "b1", "reading_order": 0, "type": "title", "text": "标题"}
        ),
        TitleBlock,
    )
    with pytest.raises(ValidationError):
        adapter.validate_python(
            {"block_id": "b1", "reading_order": 0, "type": "unknown", "text": "内容"}
        )


def test_document_rejects_duplicate_block_ids() -> None:
    with pytest.raises(ValidationError):
        CanonicalDocument(
            document_id=DOCUMENT_ID,
            pages=[
                Page(
                    page_number=1,
                    blocks=[
                        TextBlock(block_id="same", reading_order=0, text="第一段"),
                        TextBlock(block_id="same", reading_order=1, text="第二段"),
                    ],
                )
            ],
        )


@pytest.mark.parametrize("page_numbers", [[0], [2], [1, 1], [1, 3], [2, 1]])
def test_document_rejects_invalid_page_number_sequences(page_numbers: list[int]) -> None:
    with pytest.raises(ValidationError):
        CanonicalDocument(
            document_id=DOCUMENT_ID,
            pages=[Page(page_number=page_number) for page_number in page_numbers],
        )


def test_document_requires_at_least_one_page() -> None:
    with pytest.raises(ValidationError):
        CanonicalDocument(document_id=DOCUMENT_ID, pages=[])


@pytest.mark.parametrize("reading_orders", [[1], [0, 2], [2, 3], [0, 0], [1, 0]])
def test_page_rejects_non_contiguous_reading_orders(reading_orders: list[int]) -> None:
    with pytest.raises(ValidationError):
        Page(
            page_number=1,
            blocks=[
                TextBlock(block_id=f"b{index}", reading_order=reading_order, text=f"内容{index}")
                for index, reading_order in enumerate(reading_orders)
            ],
        )


@pytest.mark.parametrize("reading_orders", [[], [0], [0, 1, 2]])
def test_page_accepts_zero_based_contiguous_reading_orders(reading_orders: list[int]) -> None:
    page = Page(
        page_number=1,
        blocks=[
            TextBlock(block_id=f"b{index}", reading_order=reading_order, text=f"内容{index}")
            for index, reading_order in enumerate(reading_orders)
        ],
    )

    assert [block.reading_order for block in page.blocks] == reading_orders


def test_negative_reading_order_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TextBlock(block_id="b1", reading_order=-1, text="内容")


@pytest.mark.parametrize(
    ("model", "field"),
    [
        (TitleBlock, "text"),
        (TextBlock, "text"),
        (TableBlock, "markdown"),
        (FormulaBlock, "latex"),
        (ImageBlock, "asset_ref"),
    ],
)
def test_required_content_cannot_be_blank(model: type[object], field: str) -> None:
    with pytest.raises(ValidationError):
        model(block_id="b1", reading_order=0, **{field: "   "})


@pytest.mark.parametrize(
    "bbox",
    [
        {"x0": -0.1, "y0": 0, "x1": 0.5, "y1": 0.5},
        {"x0": 0, "y0": 0, "x1": 1.1, "y1": 0.5},
        {"x0": 0.5, "y0": 0, "x1": 0.5, "y1": 0.5},
        {"x0": 0, "y0": 0.5, "x1": 0.5, "y1": 0.5},
    ],
)
def test_bounding_box_requires_normalized_positive_dimensions(
    bbox: dict[str, float],
) -> None:
    with pytest.raises(ValidationError):
        BoundingBox.model_validate(bbox)


@pytest.mark.parametrize(
    "asset_ref",
    ["/tmp/image.png", "C:\\tmp\\image.png", "file:///tmp/image.png"],
)
def test_image_asset_ref_rejects_an_absolute_path_or_uri(asset_ref: str) -> None:
    with pytest.raises(ValidationError):
        ImageBlock(block_id="b1", reading_order=0, asset_ref=asset_ref)


@pytest.mark.parametrize("asset_ref", ["image-1", "assets/image-1.png"])
def test_image_asset_ref_accepts_opaque_relative_references(asset_ref: str) -> None:
    image = ImageBlock(block_id="b1", reading_order=0, asset_ref=asset_ref)

    assert image.asset_ref == asset_ref


@pytest.mark.parametrize(
    "asset_ref",
    [
        "../image.png",
        "assets/../image.png",
        r"..\image.png",
        r"assets\..\image.png",
        "./image.png",
        "assets/./image.png",
        r"assets\.\image.png",
        ".",
        "..",
    ],
)
def test_image_asset_ref_rejects_dot_path_segments(asset_ref: str) -> None:
    with pytest.raises(ValidationError):
        ImageBlock(block_id="b1", reading_order=0, asset_ref=asset_ref)


def test_schema_version_is_explicit_and_rejects_unknown_versions() -> None:
    document = CanonicalDocument(document_id=DOCUMENT_ID, pages=[Page(page_number=1)])
    assert document.schema_version == "1.0"

    with pytest.raises(ValidationError):
        CanonicalDocument.model_validate(
            {
                "schema_version": "2.0",
                "document_id": str(DOCUMENT_ID),
                "pages": [{"page_number": 1}],
            }
        )


def test_canonical_models_have_no_parser_specific_dependency() -> None:
    source = inspect.getsource(canonical_models).lower()

    assert "mineru" not in source
    assert set(CanonicalDocument.model_fields) == {"schema_version", "document_id", "pages"}
