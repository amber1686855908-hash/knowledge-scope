import hashlib
import json
from pathlib import Path
from uuid import UUID

import pytest

from knowledge_scope.parsing.mineru_adapter import AdapterStats
from knowledge_scope.parsing.mineru_runner import MineruRunnerError, MineruRunResult
from knowledge_scope.parsing.service import (
    MAX_MANIFEST_WARNING_COUNT,
    MAX_MANIFEST_WARNING_LENGTH,
    DocumentParseError,
    _manifest,
    parse_document_file,
)
from knowledge_scope.shared.config import Settings

DOCUMENT_ID = UUID("11111111-1111-1111-1111-111111111111")


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
