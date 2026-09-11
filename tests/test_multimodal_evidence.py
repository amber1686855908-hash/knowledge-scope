from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from knowledge_scope.documents.storage import restore_from_trash
from knowledge_scope.evidence import (
    EvidenceArtifactError,
    EvidenceRepresentation,
    EvidenceValidationError,
    MultimodalEvidenceDocument,
    RepresentationIndexPayload,
    StaleEvidenceError,
    assert_evidence_artifact_current,
    build_evidence_document,
    canonical_document_fingerprint,
    canonical_json_bytes,
    evidence_artifact_path,
    evidence_id_for,
    move_evidence_artifact_to_trash,
    rebuild_evidence_artifact,
    remove_evidence_artifact,
    representation_id_for,
    representation_index_payloads,
    sha256_fingerprint,
    validate_evidence_document,
    write_evidence_artifact,
)
from knowledge_scope.parsing.models import (
    CanonicalDocument,
    FormulaBlock,
    ImageBlock,
    Page,
    TableBlock,
    TextBlock,
    TitleBlock,
)

DOCUMENT_ID = UUID("11111111-1111-1111-1111-111111111111")
KNOWLEDGE_BASE_ID = UUID("22222222-2222-2222-2222-222222222222")


def _document(text: str = "设备需要定期维护。") -> CanonicalDocument:
    return CanonicalDocument(
        document_id=DOCUMENT_ID,
        pages=[
            Page(
                page_number=1,
                blocks=[
                    TitleBlock(block_id="title-1", reading_order=0, text="设备维护"),
                    TextBlock(block_id="text-1", reading_order=1, text=text),
                    TableBlock(
                        block_id="table-1",
                        reading_order=2,
                        markdown="| 项目 | 状态 |\n| --- | --- |\n| 温度 | 正常 |",
                        caption="检查项目表",
                        asset_ref="tables/table-1.png",
                    ),
                    FormulaBlock(block_id="formula-1", reading_order=3, latex="E = mc^2"),
                    ImageBlock(
                        block_id="image-1",
                        reading_order=4,
                        asset_ref="images/image-1.png",
                        caption="维护流程图",
                    ),
                ],
            ),
            Page(page_number=2),
        ],
    )


def test_identity_is_stable_versioned_and_representation_independent() -> None:
    first = build_evidence_document(_document(), KNOWLEDGE_BASE_ID)
    second = build_evidence_document(_document(), KNOWLEDGE_BASE_ID)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    text_evidence = first.evidence[1]
    assert text_evidence.evidence_id == evidence_id_for(
        text_evidence.lineage,
        text_evidence.modality,
    )
    assert first.evidence[2].evidence_id != first.evidence[1].evidence_id
    assert len({item.evidence_id for item in first.evidence}) == 5

    with_context = build_evidence_document(
        _document(),
        KNOWLEDGE_BASE_ID,
        context_by_block_id={"text-1": "维护规范的上下文"},
    )
    assert with_context.evidence[1].evidence_id == text_evidence.evidence_id
    assert with_context.evidence[1].lineage.source_block_ids == ["text-1"]
    assert len(with_context.evidence[1].representations) == 2


def test_identity_ignores_section_path_but_tracks_source_identity_changes() -> None:
    artifact = build_evidence_document(_document(), KNOWLEDGE_BASE_ID)
    evidence = artifact.evidence[1]
    display_changed = evidence.lineage.model_copy(update={"section_path": ["新章节"]})
    source_block_changed = evidence.lineage.model_copy(update={"source_block_ids": ["title-1"]})
    content_fingerprint_changed = evidence.lineage.model_copy(
        update={"source_fingerprint": "b" * 64}
    )

    assert evidence_id_for(display_changed, evidence.modality) == evidence.evidence_id
    assert evidence_id_for(source_block_changed, evidence.modality) != evidence.evidence_id
    assert evidence_id_for(content_fingerprint_changed, evidence.modality) != evidence.evidence_id

    display_changed_artifact = artifact.model_copy(deep=True)
    display_changed_artifact.evidence[1].lineage.section_path = ["新章节"]
    validate_evidence_document(
        _document(),
        display_changed_artifact,
        KNOWLEDGE_BASE_ID,
    )


def test_identity_serialization_is_unambiguous_for_nul_and_unicode() -> None:
    first = {"left": "甲\x00乙", "right": "丙\n丁"}
    second = {"left": "甲", "right": "乙\x00丙\n丁"}

    assert canonical_json_bytes(first) != canonical_json_bytes(second)
    assert sha256_fingerprint(first) != sha256_fingerprint(second)
    assert sha256_fingerprint({"b": "值", "a": "🙂"}) == sha256_fingerprint({"a": "🙂", "b": "值"})


def test_real_canonical_blocks_produce_one_evidence_with_multiple_representations() -> None:
    artifact = build_evidence_document(_document(), KNOWLEDGE_BASE_ID)
    by_block = {item.lineage.source_block_ids[0]: item for item in artifact.evidence}

    assert by_block["title-1"].modality == "text"
    assert by_block["text-1"].representations[0].representation_type == "text"
    assert {rep.representation_type for rep in by_block["table-1"].representations} == {
        "markdown",
        "caption",
        "asset_ref",
    }
    assert {rep.representation_type for rep in by_block["formula-1"].representations} == {
        "latex",
    }
    assert {rep.representation_type for rep in by_block["image-1"].representations} == {
        "caption",
        "asset_ref",
    }
    assert len(artifact.evidence) == 5
    assert len(representation_index_payloads(artifact)) == 6
    assert len(representation_index_payloads(artifact, searchable_only=False)) == 8


def test_invalid_modality_representation_combination_is_rejected() -> None:
    evidence_id = "evidence_v1_" + "a" * 64
    representation_id = representation_id_for(
        evidence_id,
        "text",
        "caption",
        content="不应作为正文 caption",
    )

    with pytest.raises(ValidationError, match="not valid for modality"):
        EvidenceRepresentation(
            representation_id=representation_id,
            evidence_id=evidence_id,
            modality="text",
            representation_type="caption",
            content="不应作为正文 caption",
            searchable=True,
        )

    with pytest.raises(ValidationError, match="opaque non-absolute"):
        EvidenceRepresentation(
            representation_id=representation_id_for(
                evidence_id,
                "image",
                "asset_ref",
                reference="../image.png",
            ),
            evidence_id=evidence_id,
            modality="image",
            representation_type="asset_ref",
            reference="../image.png",
            searchable=False,
        )


def test_block_ids_reject_surrounding_whitespace_at_each_evidence_boundary() -> None:
    with pytest.raises(ValidationError, match="block_id must not have surrounding whitespace"):
        TextBlock(block_id=" block-1 ", reading_order=0, text="正文")

    artifact = build_evidence_document(_document(), KNOWLEDGE_BASE_ID)
    lineage = artifact.evidence[1].lineage.model_dump(mode="python")
    with pytest.raises(
        ValidationError,
        match="source_block_id must not have surrounding whitespace",
    ):
        type(artifact.evidence[1].lineage)(**lineage | {"source_block_ids": [" text-1 "]})

    payload = representation_index_payloads(artifact)[0].model_dump(mode="python")
    with pytest.raises(
        ValidationError,
        match="source_block_id must not have surrounding whitespace",
    ):
        RepresentationIndexPayload(**payload | {"source_block_ids": [" text-1 "]})


def test_lineage_validation_rejects_changed_canonical_source() -> None:
    document = _document()
    artifact = build_evidence_document(document, KNOWLEDGE_BASE_ID)
    validate_evidence_document(document, artifact, KNOWLEDGE_BASE_ID)

    changed_document = _document("正文内容已经改变。")
    with pytest.raises(StaleEvidenceError, match="stale"):
        validate_evidence_document(changed_document, artifact, KNOWLEDGE_BASE_ID)

    changed_artifact = artifact.model_copy(deep=True)
    changed_artifact.evidence[1].lineage.source_fingerprint = "b" * 64
    with pytest.raises(EvidenceValidationError, match="structure"):
        validate_evidence_document(document, changed_artifact, KNOWLEDGE_BASE_ID)


def test_index_payload_preserves_authoritative_lineage() -> None:
    artifact = build_evidence_document(_document(), KNOWLEDGE_BASE_ID)
    payloads = representation_index_payloads(artifact)
    table_payload = next(payload for payload in payloads if payload.modality == "table")

    assert table_payload.evidence_id == artifact.evidence[2].evidence_id
    assert table_payload.document_id == DOCUMENT_ID
    assert table_payload.knowledge_base_id == KNOWLEDGE_BASE_ID
    assert table_payload.page_start == 1
    assert table_payload.page_end == 1
    assert table_payload.source_block_ids == ["table-1"]
    assert table_payload.asset_refs == ["tables/table-1.png"]
    assert table_payload.text is not None
    assert table_payload.reference is None


def test_index_payload_rejects_stale_identity_or_non_searchable_text() -> None:
    artifact = build_evidence_document(_document(), KNOWLEDGE_BASE_ID)
    payload = representation_index_payloads(artifact)[0]

    with pytest.raises(ValidationError, match="representation_id"):
        RepresentationIndexPayload(
            **payload.model_dump(mode="python")
            | {"representation_id": "representation_v1_" + "a" * 64}
        )

    with pytest.raises(ValidationError, match="cannot contain text"):
        RepresentationIndexPayload(
            **payload.model_dump(mode="python")
            | {
                "representation_type": "asset_ref",
                "reference": "images/image-1.png",
                "text": "不应出现在 opaque reference 中",
                "searchable": False,
                "representation_id": representation_id_for(
                    payload.evidence_id,
                    "text",
                    "asset_ref",
                    reference="images/image-1.png",
                ),
            }
        )


def test_index_payload_rejects_mismatched_evidence_lineage_and_assets() -> None:
    artifact = build_evidence_document(_document(), KNOWLEDGE_BASE_ID)
    payload = representation_index_payloads(artifact)[0]
    payload_data = payload.model_dump(mode="python")

    for update in (
        {"knowledge_base_id": UUID("33333333-3333-3333-3333-333333333333")},
        {"document_id": UUID("44444444-4444-4444-4444-444444444444")},
        {"page_start": 2, "page_end": 2},
        {"source_block_ids": ["other-block"]},
        {"source_fingerprint": "b" * 64},
    ):
        with pytest.raises(ValidationError, match="evidence_id"):
            RepresentationIndexPayload(**payload_data | update)

    display_changed = RepresentationIndexPayload(
        **payload_data | {"section_path": ["新的展示路径"]}
    )
    assert display_changed.evidence_id == payload.evidence_id

    asset_payload = next(
        item
        for item in representation_index_payloads(artifact, searchable_only=False)
        if item.representation_type == "asset_ref"
    )
    with pytest.raises(ValidationError):
        RepresentationIndexPayload(
            **asset_payload.model_dump(mode="python") | {"asset_refs": ["other/asset.png"]}
        )


def test_evidence_artifact_rebuild_and_deletion_are_source_scoped(tmp_path: Path) -> None:
    document = _document()
    artifact = build_evidence_document(document, KNOWLEDGE_BASE_ID)
    path = write_evidence_artifact(tmp_path, artifact)

    assert path == evidence_artifact_path(tmp_path, DOCUMENT_ID)
    assert assert_evidence_artifact_current(tmp_path, document, KNOWLEDGE_BASE_ID).document_id == (
        DOCUMENT_ID
    )
    with pytest.raises(EvidenceArtifactError, match="stale"):
        assert_evidence_artifact_current(
            tmp_path,
            _document("当前 canonical 内容不同。"),
            KNOWLEDGE_BASE_ID,
        )

    rebuilt_path = rebuild_evidence_artifact(
        tmp_path,
        _document("当前 canonical 内容不同。"),
        KNOWLEDGE_BASE_ID,
    )
    assert rebuilt_path == path
    assert remove_evidence_artifact(tmp_path, DOCUMENT_ID) is True
    assert remove_evidence_artifact(tmp_path, DOCUMENT_ID) is False


def test_evidence_storage_rejects_symlinked_ancestors_and_artifacts(tmp_path: Path) -> None:
    artifact = build_evidence_document(_document(), KNOWLEDGE_BASE_ID)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    (data_dir / "evidence").symlink_to(outside, target_is_directory=True)
    with pytest.raises(EvidenceArtifactError, match="symlink"):
        write_evidence_artifact(data_dir, artifact)
    with pytest.raises(EvidenceArtifactError, match="symlink"):
        remove_evidence_artifact(data_dir, DOCUMENT_ID)

    (data_dir / "evidence").unlink()
    (data_dir / "evidence").mkdir()
    (data_dir / "evidence" / str(DOCUMENT_ID)).symlink_to(outside, target_is_directory=True)
    with pytest.raises(EvidenceArtifactError, match="symlink"):
        write_evidence_artifact(data_dir, artifact)

    (data_dir / "evidence" / str(DOCUMENT_ID)).unlink()
    evidence_dir = data_dir / "evidence" / str(DOCUMENT_ID)
    evidence_dir.mkdir(parents=True)
    outside_file = outside / "evidence.json"
    outside_file.write_text("{}", encoding="utf-8")
    (evidence_dir / "evidence.json").symlink_to(outside_file)
    with pytest.raises(EvidenceArtifactError, match="symlink"):
        write_evidence_artifact(data_dir, artifact)
    with pytest.raises(EvidenceArtifactError, match="symlink"):
        remove_evidence_artifact(data_dir, DOCUMENT_ID)


def test_evidence_atomic_write_does_not_follow_preexisting_temp_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = build_evidence_document(_document(), KNOWLEDGE_BASE_ID)
    data_dir = tmp_path / "data"
    evidence_dir = data_dir / "evidence" / str(DOCUMENT_ID)
    evidence_dir.mkdir(parents=True)
    outside_file = tmp_path / "outside.json"
    outside_file.write_text("outside", encoding="utf-8")
    monkeypatch.setattr(
        "knowledge_scope.evidence.lifecycle.uuid4",
        lambda: UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
    )
    temp_path = evidence_dir / ".evidence.json.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.tmp"
    temp_path.symlink_to(outside_file)

    with pytest.raises(EvidenceArtifactError, match="could not be saved"):
        write_evidence_artifact(data_dir, artifact)

    assert outside_file.read_text(encoding="utf-8") == "outside"
    assert not temp_path.exists()


def test_evidence_trash_round_trip_does_not_require_documents_root(tmp_path: Path) -> None:
    artifact = build_evidence_document(_document(), KNOWLEDGE_BASE_ID)
    path = write_evidence_artifact(tmp_path, artifact)

    trashed = move_evidence_artifact_to_trash(tmp_path, DOCUMENT_ID)
    assert trashed is not None
    assert not path.exists()

    restore_from_trash(trashed)
    assert path.exists()


def test_document_fingerprint_changes_when_canonical_block_changes() -> None:
    assert canonical_document_fingerprint(_document()) != canonical_document_fingerprint(
        _document("不同正文")
    )


def test_evidence_document_rejects_duplicate_evidence_ids() -> None:
    artifact = build_evidence_document(_document(), KNOWLEDGE_BASE_ID)
    with pytest.raises(ValidationError, match="evidence IDs must be unique"):
        MultimodalEvidenceDocument(
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            document_id=DOCUMENT_ID,
            canonical_document_fingerprint=artifact.canonical_document_fingerprint,
            evidence=[artifact.evidence[0], artifact.evidence[0]],
        )


def test_evidence_validation_rejects_missing_canonical_block_coverage() -> None:
    document = _document()
    artifact = build_evidence_document(document, KNOWLEDGE_BASE_ID)
    incomplete = artifact.model_copy(update={"evidence": artifact.evidence[:-1]})

    with pytest.raises(StaleEvidenceError, match="cover every canonical source block"):
        validate_evidence_document(document, incomplete, KNOWLEDGE_BASE_ID)
