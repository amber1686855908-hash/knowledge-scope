import hashlib
import json
from pathlib import Path
from uuid import UUID

import pytest
from qdrant_client import QdrantClient

from knowledge_scope.evidence import evidence_artifact_path, rebuild_evidence_artifact
from knowledge_scope.evidence.lifecycle import load_evidence_artifact
from knowledge_scope.parsing.mineru_adapter import AdaptedCanonicalDocument, AdapterStats
from knowledge_scope.parsing.mineru_runner import MineruRunnerError, MineruRunResult
from knowledge_scope.parsing.models import CanonicalDocument, Page, TextBlock, TitleBlock
from knowledge_scope.parsing.service import (
    MAX_MANIFEST_WARNING_COUNT,
    MAX_MANIFEST_WARNING_LENGTH,
    DocumentParseError,
    _manifest,
    parse_document_file,
)
from knowledge_scope.retrieval.embedding import embedding_config_fingerprint
from knowledge_scope.retrieval.representation_index import (
    QDRANT_VECTOR_DIMENSION,
    QWEN_EMBEDDING_MODEL_ID,
    MultimodalRepresentationRetrievalService,
    QdrantRepresentationStore,
    RepresentationIndexError,
    index_canonical_document,
)
from knowledge_scope.shared.config import Settings

DOCUMENT_ID = UUID("11111111-1111-1111-1111-111111111111")
KNOWLEDGE_BASE_ID = UUID("22222222-2222-2222-2222-222222222222")


class _RepresentationEncoder:
    model_id = QWEN_EMBEDDING_MODEL_ID

    def __init__(self, *, fail: bool = False) -> None:
        settings = Settings(_env_file=None)
        self.model_revision = settings.embedding_model_revision
        self.config_fingerprint = embedding_config_fingerprint(settings)
        self.fail = fail

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        if self.fail:
            raise RuntimeError("injected embedding failure")
        return [self._vector() for _ in texts]

    def encode_query(self, _query: str) -> list[float]:
        return self._vector()

    @staticmethod
    def _vector() -> list[float]:
        return [1.0, *([0.0] * (QDRANT_VECTOR_DIMENSION - 1))]


def _text_document(text: str) -> CanonicalDocument:
    return CanonicalDocument(
        document_id=DOCUMENT_ID,
        pages=[
            Page(
                page_number=1,
                blocks=[
                    TitleBlock(block_id="title-1", reading_order=0, text="章节"),
                    TextBlock(block_id="text-1", reading_order=1, text=text),
                ],
            )
        ],
    )


def _adapted(document: CanonicalDocument) -> AdaptedCanonicalDocument:
    return AdaptedCanonicalDocument(
        document=document,
        stats=AdapterStats(
            pages=len(document.pages),
            input_items=sum(len(page.blocks) for page in document.pages),
            canonical_blocks=sum(len(page.blocks) for page in document.pages),
            title_blocks=1,
            text_blocks=1,
            tables=0,
            formulas=0,
            images=0,
            skipped_auxiliary=0,
            unsupported_items=0,
        ),
    )


def _prepare_indexed_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[
    Path,
    Settings,
    dict[str, CanonicalDocument],
    QdrantRepresentationStore,
    _RepresentationEncoder,
]:
    source_path = tmp_path / "source.pdf"
    source_path.write_bytes(b"%PDF-real-source")
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")
    documents = {"current": _text_document("旧版本内容")}
    monkeypatch.setattr("knowledge_scope.parsing.service.run_mineru", _fake_mineru_run)
    monkeypatch.setattr(
        "knowledge_scope.parsing.service.adapt_content_list",
        lambda *_args, **_kwargs: _adapted(documents["current"]),
    )
    parse_document_file(DOCUMENT_ID, source_path, _sha256(source_path), settings)

    store = QdrantRepresentationStore(settings, client=QdrantClient(":memory:"))
    encoder = _RepresentationEncoder()
    canonical_path = settings.data_dir / "parsing" / str(DOCUMENT_ID) / "canonical.json"
    index_canonical_document(
        canonical_path,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        settings=settings,
        store=store,
        embedder=encoder,
    )
    chunking_path = settings.data_dir / "chunking" / str(DOCUMENT_ID) / "chunks.json"
    chunking_path.parent.mkdir(parents=True)
    chunking_path.write_bytes(b"old chunks")
    return source_path, settings, documents, store, encoder


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_mineru_run(
    source_pdf: Path,
    output_dir: Path,
    command: str,
    *,
    timeout_seconds: int,
) -> MineruRunResult:
    del source_pdf, command, timeout_seconds
    content_dir = output_dir / "sample" / "auto"
    image_dir = content_dir / "images"
    image_dir.mkdir(parents=True)
    (image_dir / "image.png").write_bytes(b"image")
    (content_dir / "sample_content_list.json").write_text(
        json.dumps(
            [
                {"type": "text", "page_idx": 0, "text": "正文", "bbox": [0, 0, 1000, 100]},
                {"type": "image", "page_idx": 0, "img_path": "images/image.png"},
                {"type": "code", "page_idx": 0, "code_body": "unsupported"},
            ]
        ),
        encoding="utf-8",
    )
    (content_dir / "sample_middle.json").write_text(
        json.dumps({"pdf_info": [{}]}),
        encoding="utf-8",
    )
    return MineruRunResult(
        output_dir=output_dir,
        version="3.4.5",
        backend="pipeline",
        elapsed_seconds=1.25,
        stdout="mineru stdout",
        stderr="",
    )


def test_parse_document_file_promotes_complete_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.pdf"
    source_path.write_bytes(b"%PDF-real-source")
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")
    monkeypatch.setattr("knowledge_scope.parsing.service.run_mineru", _fake_mineru_run)

    result = parse_document_file(DOCUMENT_ID, source_path, _sha256(source_path), settings)

    final_dir = settings.data_dir / "parsing" / str(DOCUMENT_ID)
    canonical = json.loads((final_dir / "canonical.json").read_text(encoding="utf-8"))
    manifest = json.loads((final_dir / "manifest.json").read_text(encoding="utf-8"))
    assert result.stats.canonical_blocks == 2
    assert canonical["document_id"] == str(DOCUMENT_ID)
    assert manifest["parser_version"] == "3.4.5"
    assert manifest["canonical_ref"] == f"parsing/{DOCUMENT_ID}/canonical.json"
    assert manifest["raw_ref"] == f"parsing/{DOCUMENT_ID}/mineru"
    assert manifest["parse_stats"] == {
        "elapsed_seconds": 1.25,
        "pages": 1,
        "mineru_input_items": 3,
        "canonical_blocks": 2,
        "title_blocks": 0,
        "text_blocks": 1,
        "tables": 0,
        "formulas": 0,
        "images": 1,
        "skipped_auxiliary": 0,
        "unsupported_items": 1,
        "bbox_clamped": 0,
        "table_asset_only": 0,
        "table_missing_content": 0,
        "warning_count": 1,
        "warnings": ["item 2: unsupported or empty MinerU block type 'code'"],
    }
    assert str(tmp_path) not in json.dumps(manifest)
    assert (final_dir / "mineru" / "stdout.log").read_text(encoding="utf-8") == "mineru stdout"
    assert not list((settings.data_dir / "parsing").glob(".*"))
    assert not (settings.data_dir / "chunking" / str(DOCUMENT_ID)).exists()


def test_successful_reparse_removes_existing_chunk_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.pdf"
    source_path.write_bytes(b"%PDF-real-source")
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")
    monkeypatch.setattr("knowledge_scope.parsing.service.run_mineru", _fake_mineru_run)

    parse_document_file(DOCUMENT_ID, source_path, _sha256(source_path), settings)
    chunking_dir = settings.data_dir / "chunking" / str(DOCUMENT_ID)
    chunking_dir.mkdir(parents=True)
    (chunking_dir / "chunks.json").write_text("old chunks", encoding="utf-8")

    parse_document_file(DOCUMENT_ID, source_path, _sha256(source_path), settings)

    assert not chunking_dir.exists()
    assert not list((settings.data_dir / "parsing").glob(".*"))


def test_successful_reparse_removes_existing_evidence_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.pdf"
    source_path.write_bytes(b"%PDF-real-source")
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")
    monkeypatch.setattr("knowledge_scope.parsing.service.run_mineru", _fake_mineru_run)

    parse_document_file(DOCUMENT_ID, source_path, _sha256(source_path), settings)
    canonical = CanonicalDocument.model_validate_json(
        (settings.data_dir / "parsing" / str(DOCUMENT_ID) / "canonical.json").read_bytes()
    )
    rebuild_evidence_artifact(
        settings.data_dir,
        canonical,
        UUID("22222222-2222-2222-2222-222222222222"),
    )
    assert evidence_artifact_path(settings.data_dir, DOCUMENT_ID).exists()

    parse_document_file(DOCUMENT_ID, source_path, _sha256(source_path), settings)

    assert not evidence_artifact_path(settings.data_dir, DOCUMENT_ID).exists()


def test_failed_reparse_leaves_previous_parsing_and_chunking_artifacts_untouched(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.pdf"
    source_path.write_bytes(b"%PDF-real-source")
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")
    monkeypatch.setattr("knowledge_scope.parsing.service.run_mineru", _fake_mineru_run)
    parse_document_file(DOCUMENT_ID, source_path, _sha256(source_path), settings)

    parsing_dir = settings.data_dir / "parsing" / str(DOCUMENT_ID)
    canonical_before = (parsing_dir / "canonical.json").read_bytes()
    canonical = CanonicalDocument.model_validate_json(canonical_before)
    rebuild_evidence_artifact(
        settings.data_dir,
        canonical,
        UUID("22222222-2222-2222-2222-222222222222"),
    )
    evidence_path = evidence_artifact_path(settings.data_dir, DOCUMENT_ID)
    chunking_dir = settings.data_dir / "chunking" / str(DOCUMENT_ID)
    chunking_dir.mkdir(parents=True)
    chunks_path = chunking_dir / "chunks.json"
    chunks_path.write_bytes(b"old chunks")

    def fail(*_: object, **__: object) -> MineruRunResult:
        raise MineruRunnerError("MinerU exited with code 3")

    monkeypatch.setattr("knowledge_scope.parsing.service.run_mineru", fail)

    with pytest.raises(DocumentParseError, match="code 3"):
        parse_document_file(DOCUMENT_ID, source_path, _sha256(source_path), settings)

    assert (parsing_dir / "canonical.json").read_bytes() == canonical_before
    assert chunks_path.read_bytes() == b"old chunks"
    assert evidence_path.exists()
    assert not list((settings.data_dir / "parsing").glob(".*"))


def test_failed_evidence_invalidation_restores_complete_previous_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.pdf"
    source_path.write_bytes(b"%PDF-real-source")
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")
    monkeypatch.setattr("knowledge_scope.parsing.service.run_mineru", _fake_mineru_run)
    parse_document_file(DOCUMENT_ID, source_path, _sha256(source_path), settings)

    parsing_dir = settings.data_dir / "parsing" / str(DOCUMENT_ID)
    canonical_before = (parsing_dir / "canonical.json").read_bytes()
    canonical = CanonicalDocument.model_validate_json(canonical_before)
    rebuild_evidence_artifact(
        settings.data_dir,
        canonical,
        UUID("22222222-2222-2222-2222-222222222222"),
    )
    evidence_path = evidence_artifact_path(settings.data_dir, DOCUMENT_ID)
    evidence_before = evidence_path.read_bytes()
    chunking_dir = settings.data_dir / "chunking" / str(DOCUMENT_ID)
    chunking_dir.mkdir(parents=True)
    chunks_path = chunking_dir / "chunks.json"
    chunks_path.write_bytes(b"old chunks")

    def fail_evidence_invalidation(*_: object, **__: object) -> None:
        raise DocumentParseError("injected evidence invalidation failure")

    monkeypatch.setattr(
        "knowledge_scope.parsing.service._invalidate_evidence_artifacts",
        fail_evidence_invalidation,
    )

    with pytest.raises(DocumentParseError, match="injected evidence invalidation failure"):
        parse_document_file(DOCUMENT_ID, source_path, _sha256(source_path), settings)

    assert (parsing_dir / "canonical.json").read_bytes() == canonical_before
    assert chunks_path.read_bytes() == b"old chunks"
    assert evidence_path.read_bytes() == evidence_before
    assert not list((settings.data_dir / "parsing").glob(".*"))
    assert not list((settings.data_dir / "documents").glob(".delete-*"))
    assert not list((settings.data_dir / "evidence").glob(".delete-*"))


def test_failed_parse_does_not_leave_a_successful_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.pdf"
    source_path.write_bytes(b"%PDF-real-source")
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")

    def fail(*_: object, **__: object) -> MineruRunResult:
        raise MineruRunnerError("MinerU exited with code 3")

    monkeypatch.setattr("knowledge_scope.parsing.service.run_mineru", fail)

    with pytest.raises(DocumentParseError, match="code 3"):
        parse_document_file(DOCUMENT_ID, source_path, _sha256(source_path), settings)

    parsing_root = settings.data_dir / "parsing"
    assert not (parsing_root / str(DOCUMENT_ID)).exists()
    assert not list(parsing_root.glob(".*"))


def test_manifest_bounds_persisted_adapter_warnings(tmp_path: Path) -> None:
    warnings = tuple(
        f"warning-{index}-" + ("x" * MAX_MANIFEST_WARNING_LENGTH)
        for index in range(MAX_MANIFEST_WARNING_COUNT + 7)
    )
    manifest = _manifest(
        document_id=DOCUMENT_ID,
        source_sha256="a" * 64,
        run_result=MineruRunResult(
            output_dir=tmp_path,
            version="3.4.5",
            backend="pipeline",
            elapsed_seconds=1.0,
            stdout="",
            stderr="",
        ),
        stats=AdapterStats(
            pages=1,
            input_items=1,
            canonical_blocks=1,
            title_blocks=0,
            text_blocks=1,
            tables=0,
            formulas=0,
            images=0,
            skipped_auxiliary=0,
            unsupported_items=0,
            warnings=warnings,
        ),
    )

    parse_stats = manifest["parse_stats"]
    assert parse_stats["warning_count"] == MAX_MANIFEST_WARNING_COUNT + 7
    assert len(parse_stats["warnings"]) == MAX_MANIFEST_WARNING_COUNT
    assert all(len(warning) == MAX_MANIFEST_WARNING_LENGTH for warning in parse_stats["warnings"])


def test_parse_document_file_rejects_changed_source_before_running_mineru(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.pdf"
    source_path.write_bytes(b"%PDF-real-source")
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")
    called = False

    def unexpected_run(*_: object, **__: object) -> MineruRunResult:
        nonlocal called
        called = True
        raise AssertionError("MinerU should not run for a changed source")

    monkeypatch.setattr("knowledge_scope.parsing.service.run_mineru", unexpected_run)

    with pytest.raises(DocumentParseError, match="SHA-256"):
        parse_document_file(DOCUMENT_ID, source_path, "0" * 64, settings)

    assert called is False


def _capture_indexed_generation(
    settings: Settings,
    store: QdrantRepresentationStore,
) -> tuple[bytes, bytes, tuple[object, ...], bytes]:
    canonical_path = settings.data_dir / "parsing" / str(DOCUMENT_ID) / "canonical.json"
    evidence_path = evidence_artifact_path(settings.data_dir, DOCUMENT_ID)
    chunks_path = settings.data_dir / "chunking" / str(DOCUMENT_ID) / "chunks.json"
    return (
        canonical_path.read_bytes(),
        evidence_path.read_bytes(),
        store.list_payloads(knowledge_base_id=KNOWLEDGE_BASE_ID),
        chunks_path.read_bytes(),
    )


def _assert_indexed_generation(
    settings: Settings,
    store: QdrantRepresentationStore,
    expected: tuple[bytes, bytes, tuple[object, ...], bytes],
    encoder: _RepresentationEncoder,
) -> None:
    canonical_bytes, evidence_bytes, payloads, chunks_bytes = expected
    canonical_path = settings.data_dir / "parsing" / str(DOCUMENT_ID) / "canonical.json"
    evidence_path = evidence_artifact_path(settings.data_dir, DOCUMENT_ID)
    chunks_path = settings.data_dir / "chunking" / str(DOCUMENT_ID) / "chunks.json"
    assert canonical_path.read_bytes() == canonical_bytes
    assert evidence_path.read_bytes() == evidence_bytes
    assert store.list_payloads(knowledge_base_id=KNOWLEDGE_BASE_ID) == payloads
    assert chunks_path.read_bytes() == chunks_bytes
    result = MultimodalRepresentationRetrievalService(store, encoder).search(
        "旧版本内容",
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        top_k=5,
    )
    assert result.items


def test_coordinated_reparse_commits_one_consistent_representation_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path, settings, documents, store, encoder = _prepare_indexed_generation(
        monkeypatch,
        tmp_path,
    )
    documents["current"] = _text_document("新版本内容")

    parse_document_file(
        DOCUMENT_ID,
        source_path,
        _sha256(source_path),
        settings,
        representation_store=store,
        representation_embedder=encoder,
        representation_knowledge_base_id=KNOWLEDGE_BASE_ID,
    )

    canonical = CanonicalDocument.model_validate_json(
        (settings.data_dir / "parsing" / str(DOCUMENT_ID) / "canonical.json").read_bytes()
    )
    assert canonical.pages[0].blocks[1].text == "新版本内容"
    assert load_evidence_artifact(settings.data_dir, DOCUMENT_ID).canonical_document_fingerprint
    assert store.list_payloads(knowledge_base_id=KNOWLEDGE_BASE_ID)
    assert not (settings.data_dir / "chunking" / str(DOCUMENT_ID)).exists()
    result = MultimodalRepresentationRetrievalService(store, encoder).search(
        "新版本内容",
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        top_k=5,
    )
    assert result.items
    store.close()


def test_coordinated_reparse_embedding_failure_keeps_last_known_good_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path, settings, documents, store, encoder = _prepare_indexed_generation(
        monkeypatch,
        tmp_path,
    )
    expected = _capture_indexed_generation(settings, store)
    documents["current"] = _text_document("新版本内容")

    with pytest.raises(DocumentParseError, match="embedded"):
        parse_document_file(
            DOCUMENT_ID,
            source_path,
            _sha256(source_path),
            settings,
            representation_store=store,
            representation_embedder=_RepresentationEncoder(fail=True),
            representation_knowledge_base_id=KNOWLEDGE_BASE_ID,
        )

    _assert_indexed_generation(settings, store, expected, encoder)
    store.close()


def test_coordinated_reparse_qdrant_upsert_failure_keeps_last_known_good_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path, settings, documents, store, encoder = _prepare_indexed_generation(
        monkeypatch,
        tmp_path,
    )
    expected = _capture_indexed_generation(settings, store)
    documents["current"] = _text_document("新版本内容")

    def fail_upsert(*_: object, **__: object) -> None:
        raise RuntimeError("injected Qdrant upsert failure")

    monkeypatch.setattr(store, "_upsert", fail_upsert)
    with pytest.raises(DocumentParseError, match="replacement failed"):
        parse_document_file(
            DOCUMENT_ID,
            source_path,
            _sha256(source_path),
            settings,
            representation_store=store,
            representation_embedder=encoder,
            representation_knowledge_base_id=KNOWLEDGE_BASE_ID,
        )

    _assert_indexed_generation(settings, store, expected, encoder)
    store.close()


def test_coordinated_reparse_validation_failure_keeps_last_known_good_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path, settings, documents, store, encoder = _prepare_indexed_generation(
        monkeypatch,
        tmp_path,
    )
    expected = _capture_indexed_generation(settings, store)
    documents["current"] = _text_document("新版本内容")

    def fail_materialization(*_: object, **__: object) -> object:
        raise RepresentationIndexError("injected representation validation failure")

    monkeypatch.setattr(
        "knowledge_scope.retrieval.representation_index.build_indexable_evidence",
        fail_materialization,
    )
    with pytest.raises(DocumentParseError, match="validation failure"):
        parse_document_file(
            DOCUMENT_ID,
            source_path,
            _sha256(source_path),
            settings,
            representation_store=store,
            representation_embedder=encoder,
            representation_knowledge_base_id=KNOWLEDGE_BASE_ID,
        )

    _assert_indexed_generation(settings, store, expected, encoder)
    store.close()


def test_coordinated_reparse_stale_cleanup_failure_rolls_back_the_new_index(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path, settings, documents, store, encoder = _prepare_indexed_generation(
        monkeypatch,
        tmp_path,
    )
    expected = _capture_indexed_generation(settings, store)
    documents["current"] = _text_document("新版本内容")
    original_delete = store._delete_ids
    call_count = 0

    def fail_first_delete(point_ids: object) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("injected stale cleanup failure")
        original_delete(point_ids)

    monkeypatch.setattr(store, "_delete_ids", fail_first_delete)
    with pytest.raises(DocumentParseError, match="replacement failed"):
        parse_document_file(
            DOCUMENT_ID,
            source_path,
            _sha256(source_path),
            settings,
            representation_store=store,
            representation_embedder=encoder,
            representation_knowledge_base_id=KNOWLEDGE_BASE_ID,
        )

    _assert_indexed_generation(settings, store, expected, encoder)
    store.close()


def test_coordinated_reparse_failure_after_canonical_switch_restores_everything(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path, settings, documents, store, encoder = _prepare_indexed_generation(
        monkeypatch,
        tmp_path,
    )
    expected = _capture_indexed_generation(settings, store)
    documents["current"] = _text_document("新版本内容")

    def fail_chunk_invalidation(*_: object, **__: object) -> None:
        raise DocumentParseError("injected post-switch cleanup failure")

    monkeypatch.setattr(
        "knowledge_scope.parsing.service._invalidate_chunking_artifacts",
        fail_chunk_invalidation,
    )
    with pytest.raises(DocumentParseError, match="post-switch cleanup failure"):
        parse_document_file(
            DOCUMENT_ID,
            source_path,
            _sha256(source_path),
            settings,
            representation_store=store,
            representation_embedder=encoder,
            representation_knowledge_base_id=KNOWLEDGE_BASE_ID,
        )

    _assert_indexed_generation(settings, store, expected, encoder)
    store.close()
