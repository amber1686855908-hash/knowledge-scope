import json
from pathlib import Path
from uuid import UUID

import pytest

from knowledge_scope.chunking.models import ChunkedDocument
from knowledge_scope.cli import _redact_sql_display, build_parser, main
from knowledge_scope.graph.neo4j import Neo4jReadiness
from knowledge_scope.parsing.mineru_adapter import AdapterStats
from knowledge_scope.parsing.models import CanonicalDocument, Page, TextBlock
from knowledge_scope.parsing.service import ParseResult
from knowledge_scope.shared.config import Settings


class _FakeNeo4jStore:
    def __init__(self, _settings: Settings) -> None:
        self.schema_called = False
        self.closed = False

    def readiness(self) -> Neo4jReadiness:
        return Neo4jReadiness(status="ready", database="neo4j")

    def ensure_schema(self) -> Neo4jReadiness:
        self.schema_called = True
        return Neo4jReadiness(status="ready", database="neo4j")

    def close(self) -> None:
        self.closed = True


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


def test_chatbi_schema_parser_requires_datasource_and_supports_budget() -> None:
    args = build_parser().parse_args(
        [
            "chatbi",
            "schema",
            "11111111-1111-1111-1111-111111111111",
            "--max-chars",
            "12000",
        ]
    )

    assert args.chatbi_action == "schema"
    assert args.datasource_id == UUID("11111111-1111-1111-1111-111111111111")
    assert args.max_chars == 12_000


def test_chatbi_nl2sql_parser_requires_question_and_supports_limits() -> None:
    args = build_parser().parse_args(
        [
            "chatbi",
            "nl2sql",
            "11111111-1111-1111-1111-111111111111",
            "按客户统计销售额",
            "--max-chars",
            "12000",
            "--max-tokens",
            "256",
            "--model",
            "deepseek-chat",
        ]
    )

    assert args.chatbi_action == "nl2sql"
    assert args.datasource_id == UUID("11111111-1111-1111-1111-111111111111")
    assert args.question == "按客户统计销售额"
    assert args.max_chars == 12_000
    assert args.max_tokens == 256
    assert args.model == "deepseek-chat"


def test_chatbi_execute_parser_accepts_datasource_and_sql() -> None:
    args = build_parser().parse_args(
        [
            "chatbi",
            "execute",
            "11111111-1111-1111-1111-111111111111",
            "SELECT 1",
        ]
    )

    assert args.chatbi_action == "execute"
    assert args.datasource_id == UUID("11111111-1111-1111-1111-111111111111")
    assert args.sql == "SELECT 1"


def test_chatbi_ask_parser_accepts_question_and_model() -> None:
    args = build_parser().parse_args(
        [
            "chatbi",
            "ask",
            "11111111-1111-1111-1111-111111111111",
            "统计销售额",
            "--max-chars",
            "12000",
            "--model",
            "deepseek-chat",
        ]
    )

    assert args.chatbi_action == "ask"
    assert args.datasource_id == UUID("11111111-1111-1111-1111-111111111111")
    assert args.question == "统计销售额"
    assert args.max_chars == 12_000
    assert args.model == "deepseek-chat"


def test_chatbi_eval_offline_output_states_quality_boundary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("knowledge_scope.cli.get_settings", lambda: Settings(_env_file=None))

    assert main(["chatbi", "eval", "--output", str(tmp_path / "run.json")]) == 0
    output = capsys.readouterr().out

    assert "offline_verification: infrastructure-only; not a model-quality benchmark" in output
    assert '"quality_claim": "offline_infrastructure_only"' in output
    assert (
        json.loads(output[output.index("{\n") :])["aggregates"]["all"]["provider_quality"] is None
    )


def test_mcp_serve_parser_uses_local_stdio_command() -> None:
    args = build_parser().parse_args(["mcp", "serve"])

    assert args.command == "mcp"
    assert args.mcp_action == "serve"


def test_chatbi_sql_cli_display_redacts_nested_sql_literals() -> None:
    displayed = _redact_sql_display(
        {
            "candidate": {"sql": "SELECT * FROM public.sales WHERE id = 'secret'"},
            "validated_sql": {
                "original_sql": "SELECT 1",
                "normalized_sql": "SELECT * FROM public.sales LIMIT 1000",
            },
        }
    )

    assert displayed == {
        "candidate": {"sql": "SELECT * FROM public.sales WHERE id = '<redacted>'"},
        "validated_sql": {
            "original_sql": "SELECT 0",
            "normalized_sql": "SELECT * FROM public.sales LIMIT 0",
        },
    }


def test_neo4j_commands_report_readiness_and_schema(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "knowledge_scope.cli.get_settings",
        lambda: Settings(_env_file=None, environment="test"),
    )
    monkeypatch.setattr("knowledge_scope.cli.Neo4jGraphStore", _FakeNeo4jStore)

    assert main(["neo4j", "check"]) == 0
    check_output = capsys.readouterr()
    assert '"status": "ready"' in check_output.out
    assert "password" not in check_output.out

    assert main(["neo4j", "schema"]) == 0
    schema_output = capsys.readouterr()
    assert '"schema_version": "1.0"' in schema_output.out


def test_graph_extraction_sample_parser_has_explicit_runtime_defaults() -> None:
    args = build_parser().parse_args(["graph-extraction-sample"])

    assert args.sample_per_subject == 2
    assert args.sample_offset == 0
    assert args.persist is False
    assert args.output == Path("data/evaluation/a3-2")


def test_graph_retrieval_commands_have_bounded_typed_arguments() -> None:
    search_args = build_parser().parse_args(
        [
            "graph-search",
            "查找起点",
            "--knowledge-base-id",
            "11111111-1111-4111-8111-111111111111",
            "--max-hops",
            "1",
        ]
    )
    sample_args = build_parser().parse_args(["graph-retrieval-sample"])

    assert search_args.max_hops == 1
    assert search_args.knowledge_base_id == UUID("11111111-1111-4111-8111-111111111111")
    assert sample_args.output == Path("data/evaluation/a3-4")


def test_hybrid_search_parser_requires_kb_and_exposes_bounded_overrides() -> None:
    args = build_parser().parse_args(
        [
            "hybrid-search",
            "查找起点",
            "--knowledge-base-id",
            "11111111-1111-4111-8111-111111111111",
            "--vector-candidate-limit",
            "20",
            "--vector-rerank-limit",
            "10",
            "--graph-limit",
            "8",
            "--limit",
            "6",
            "--rrf-k",
            "60",
            "--failure-mode",
            "strict",
        ]
    )

    assert args.knowledge_base_id == UUID("11111111-1111-4111-8111-111111111111")
    assert args.vector_candidate_limit == 20
    assert args.vector_rerank_limit == 10
    assert args.graph_limit == 8
    assert args.limit == 6
    assert args.rrf_k == 60
    assert args.failure_mode == "strict"


def test_graph_corpus_commands_have_explicit_scope_and_safe_defaults() -> None:
    audit_args = build_parser().parse_args(
        [
            "graph-corpus-audit",
            "--knowledge-base-id",
            "11111111-1111-4111-8111-111111111111",
        ]
    )
    estimate_args = build_parser().parse_args(["graph-corpus-estimate"])
    build_args = build_parser().parse_args(
        [
            "graph-corpus-build",
            "--knowledge-base-id",
            "11111111-1111-4111-8111-111111111111",
            "--sample-per-subject",
            "2",
            "--persist",
        ]
    )

    assert audit_args.chunk_index == Path("data/evaluation/a2-1/chunk_index.jsonl")
    assert estimate_args.target_chunks is None
    assert estimate_args.input_snapshot == Path("docs/benchmarks/a3-6-corpus-input-snapshot.json")
    assert build_args.sample_per_subject == 2
    assert build_args.persist is True
    assert build_args.resume is True


def test_qdrant_attribution_parser_defaults_to_audit_and_supports_apply() -> None:
    audit_args = build_parser().parse_args(["qdrant", "audit-kb"])
    apply_args = build_parser().parse_args(["qdrant", "audit-kb", "--apply"])

    assert audit_args.apply is False
    assert apply_args.apply is True


def test_multimodal_index_parser_has_explicit_scope_and_filters() -> None:
    audit_args = build_parser().parse_args(
        [
            "multimodal-index",
            "audit",
            "--knowledge-base-id",
            "11111111-1111-1111-1111-111111111111",
        ]
    )
    search_args = build_parser().parse_args(
        [
            "multimodal-index",
            "search",
            "温度表",
            "--knowledge-base-id",
            "11111111-1111-1111-1111-111111111111",
            "--modality",
            "table",
            "--limit",
            "3",
        ]
    )

    assert audit_args.multimodal_index_action == "audit"
    assert search_args.modality == "table"
    assert search_args.limit == 3

    all_args = build_parser().parse_args(
        [
            "multimodal-index",
            "search",
            "设备维护",
            "--knowledge-base-id",
            "11111111-1111-1111-1111-111111111111",
            "--modality",
            "all",
        ]
    )
    assert all_args.modality == "all"


def test_sparse_index_parser_has_explicit_scope_and_bounded_query() -> None:
    build_args = build_parser().parse_args(
        [
            "sparse-index",
            "build",
            "--knowledge-base-id",
            "11111111-1111-1111-1111-111111111111",
            "--limit",
            "2",
            "--index-path",
            "tmp/sparse.sqlite3",
        ]
    )
    query_args = build_parser().parse_args(
        [
            "sparse-index",
            "query",
            "温度表",
            "--knowledge-base-id",
            "11111111-1111-1111-1111-111111111111",
            "--top-k",
            "5",
        ]
    )

    assert build_args.sparse_index_action == "build"
    assert build_args.knowledge_base_id == UUID("11111111-1111-1111-1111-111111111111")
    assert build_args.limit == 2
    assert build_args.index_path == Path("tmp/sparse.sqlite3")
    assert query_args.sparse_index_action == "query"
    assert query_args.top_k == 5


def test_corpus_registration_parser_requires_explicit_knowledge_base() -> None:
    args = build_parser().parse_args(
        [
            "corpus",
            "register",
            "--knowledge-base-id",
            "11111111-1111-4111-8111-111111111111",
        ]
    )

    assert args.knowledge_base_id == UUID("11111111-1111-4111-8111-111111111111")
    assert args.expected_document_count == 255
    assert args.expected_chunk_count == 7_524


def test_llm_smoke_test_fails_cleanly_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "knowledge_scope.cli.get_settings",
        lambda: Settings(_env_file=None, llm_api_key=None),
    )

    exit_code = main(["llm-smoke-test", "hello"])
    output = capsys.readouterr()

    assert exit_code == 1
    assert output.out == ""
    assert "llm_status: failed" in output.err
    assert "KNOWLEDGE_SCOPE_LLM_API_KEY" in output.err


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


def test_chunk_document_command_persists_chunks_from_existing_canonical_artifact(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    document_id = UUID("11111111-1111-1111-1111-111111111111")
    data_dir = tmp_path / "data"
    canonical_path = data_dir / "parsing" / str(document_id) / "canonical.json"
    canonical_path.parent.mkdir(parents=True)
    document = CanonicalDocument(
        document_id=document_id,
        pages=[
            Page(
                page_number=1,
                blocks=[TextBlock(block_id="p1-b1", reading_order=0, text="正文")],
            )
        ],
    )
    canonical_path.write_text(document.model_dump_json(indent=2), encoding="utf-8")
    monkeypatch.setattr(
        "knowledge_scope.cli.get_settings",
        lambda: Settings(_env_file=None, data_dir=data_dir),
    )

    exit_code = main(["chunk-document", str(document_id)])
    output = capsys.readouterr()
    stored = ChunkedDocument.model_validate_json(
        (data_dir / "chunking" / str(document_id) / "chunks.json").read_text(encoding="utf-8")
    )

    assert exit_code == 0
    assert "chunk_status: ok" in output.out
    assert "chunks: 1" in output.out
    assert len(stored.chunks) == 1


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
