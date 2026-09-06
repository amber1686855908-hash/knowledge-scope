from pathlib import Path
from uuid import UUID

import pytest

from knowledge_scope.cli import main
from knowledge_scope.parsing.mineru_adapter import AdapterStats
from knowledge_scope.parsing.service import ParseResult
from knowledge_scope.shared.config import Settings


def test_health_command_reports_project_and_configuration(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("KNOWLEDGE_SCOPE_ENVIRONMENT", "test")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_DATA_DIR", "var/data")

    exit_code = main(["health"])
    output = capsys.readouterr()

    assert exit_code == 0
    assert "project_name: KnowledgeScope" in output.out
    assert "config_status: ok" in output.out
    assert "environment: test" in output.out
    assert "data_dir: var/data" in output.out


def test_health_command_returns_failure_for_invalid_configuration(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("KNOWLEDGE_SCOPE_LOG_LEVEL", "not-a-level")

    exit_code = main(["health"])
    output = capsys.readouterr()

    assert exit_code == 1
    assert output.out == ""
    assert "config_status: invalid" in output.err


def test_parse_document_command_reports_non_sensitive_statistics(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    document_id = UUID("11111111-1111-1111-1111-111111111111")
    monkeypatch.setenv("KNOWLEDGE_SCOPE_ENVIRONMENT", "test")

    async def fake_parse_document_by_id(document_id: UUID, _settings: object) -> ParseResult:
        return ParseResult(
            document_id=document_id,
            source_sha256="a" * 64,
            parser_version="3.4.5",
            backend="pipeline",
            elapsed_seconds=1.25,
            canonical_ref=f"parsing/{document_id}/canonical.json",
            raw_ref=f"parsing/{document_id}/mineru",
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
            ),
        )

    monkeypatch.setattr("knowledge_scope.cli.parse_document_by_id", fake_parse_document_by_id)

    exit_code = main(["parse-document", str(document_id)])
    output = capsys.readouterr()

    assert exit_code == 0
    assert "parse_status: ok" in output.out
    assert "parser_version: 3.4.5" in output.out
    assert "canonical_validation: ok" in output.out
    assert "source_sha256" not in output.out


def test_benchmark_inventory_only_requires_explicit_corpus(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "数学教材.pdf").write_bytes(b"%PDF-test")
    workspace = tmp_path / "workspace"
    monkeypatch.setattr(
        "knowledge_scope.cli.get_settings",
        lambda: Settings(_env_file=None, data_dir=tmp_path / "data"),
    )

    exit_code = main(
        [
            "benchmark-parsing",
            "--corpus",
            str(corpus),
            "--workspace",
            str(workspace),
            "--inventory-only",
        ]
    )
    output = capsys.readouterr()

    assert exit_code == 0
    assert "inventory_status: ok" in output.out
    assert '"pdfs": 1' in output.out
    assert (workspace / "corpus-manifest.jsonl").is_file()
