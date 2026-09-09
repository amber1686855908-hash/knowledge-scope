"""Command-line entry point for the current KnowledgeScope foundation."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from uuid import UUID

from pydantic import ValidationError

from knowledge_scope import __version__
from knowledge_scope.chunking.service import ChunkingError, chunk_document_by_id
from knowledge_scope.evaluation import reranker_benchmark
from knowledge_scope.evaluation.embedding_benchmark import (
    DEFAULT_CHUNK_INDEX,
    DEFAULT_DATASET,
    DEFAULT_MATERIALIZED,
    DEFAULT_OUTPUT,
    MODEL_KEYS,
    EmbeddingBenchmarkError,
    EmbeddingBenchmarkProtocol,
    run_embedding_benchmark,
)
from knowledge_scope.evaluation.entity_linking_sample import (
    DEFAULT_INPUT as DEFAULT_LINKING_INPUT,
)
from knowledge_scope.evaluation.entity_linking_sample import (
    DEFAULT_OUTPUT as DEFAULT_LINKING_OUTPUT,
)
from knowledge_scope.evaluation.entity_linking_sample import (
    run_linking_review_sample,
)
from knowledge_scope.evaluation.graph_extraction_sample import (
    DEFAULT_CANONICAL_ROOT as DEFAULT_EXTRACTION_CANONICAL_ROOT,
)
from knowledge_scope.evaluation.graph_extraction_sample import (
    DEFAULT_CORPUS_MANIFEST as DEFAULT_EXTRACTION_CORPUS_MANIFEST,
)
from knowledge_scope.evaluation.graph_extraction_sample import (
    DEFAULT_OUTPUT as DEFAULT_EXTRACTION_OUTPUT,
)
from knowledge_scope.evaluation.graph_extraction_sample import (
    DEFAULT_SAMPLE_KNOWLEDGE_BASE_ID,
    run_sample_evaluation,
    select_sample_chunks,
)
from knowledge_scope.evaluation.parsing_benchmark import (
    RAW_RETENTION_VALUES,
    BenchmarkConfig,
    BenchmarkError,
    inventory_corpus,
    run_benchmark,
)
from knowledge_scope.evaluation.retrieval_eval import (
    QUERY_TYPES,
    RetrievalEvalError,
    apply_review_action,
    finalize_retrieval_eval_dataset,
    materialize_retrieval_eval_set,
    regenerate_candidate_review_pack,
    validate_runtime_evaluation,
)
from knowledge_scope.evaluation.retrieval_system_benchmark import (
    DEFAULT_CHUNK_INDEX as DEFAULT_SYSTEM_CHUNK_INDEX,
)
from knowledge_scope.evaluation.retrieval_system_benchmark import (
    DEFAULT_DATASET as DEFAULT_SYSTEM_DATASET,
)
from knowledge_scope.evaluation.retrieval_system_benchmark import (
    DEFAULT_MATERIALIZED as DEFAULT_SYSTEM_MATERIALIZED,
)
from knowledge_scope.evaluation.retrieval_system_benchmark import (
    DEFAULT_OUTPUT as DEFAULT_SYSTEM_OUTPUT,
)
from knowledge_scope.evaluation.retrieval_system_benchmark import (
    RetrievalSystemBenchmarkError,
    run_retrieval_system_benchmark,
)
from knowledge_scope.extraction.service import ExtractionError
from knowledge_scope.graph.neo4j import GraphStoreError, Neo4jGraphStore
from knowledge_scope.linking.service import LinkingValidationError
from knowledge_scope.llm import LLMGateway, LLMMessage, LLMRequest, LLMResult, create_llm_provider
from knowledge_scope.llm.errors import LLMError
from knowledge_scope.llm.schemas import LLM_TASK_TYPES
from knowledge_scope.llm.usage import DatabaseUsageRecorder
from knowledge_scope.parsing.service import DocumentParseError, parse_document_by_id
from knowledge_scope.retrieval.embedding import EmbeddingModelError, QwenEmbeddingModel
from knowledge_scope.retrieval.indexing import (
    IndexingError,
    index_canonical_corpus,
    index_document_by_id,
)
from knowledge_scope.retrieval.qdrant import QdrantVectorStore, VectorStoreError
from knowledge_scope.retrieval.reranking import (
    RerankerError,
    RerankingService,
    create_local_reranker,
)
from knowledge_scope.retrieval.service import DenseRetrievalService, RetrievalError
from knowledge_scope.shared import build_health_report, get_settings
from knowledge_scope.shared.database import create_database_engine, create_session_factory


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="knowledgescope",
        description="Inspect the KnowledgeScope project foundation.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("health", help="report project, runtime, and configuration health")
    llm_smoke_test = subparsers.add_parser(
        "llm-smoke-test",
        help="call the configured OpenAI-compatible LLM provider once",
    )
    llm_smoke_test.add_argument(
        "prompt",
        nargs="?",
        default="Reply with exactly: KnowledgeScope gateway ok.",
    )
    llm_smoke_test.add_argument("--system")
    llm_smoke_test.add_argument("--model")
    llm_smoke_test.add_argument("--temperature", type=float, default=0.0)
    llm_smoke_test.add_argument("--max-tokens", type=_positive_int, default=64)
    llm_smoke_test.add_argument(
        "--task-type",
        choices=LLM_TASK_TYPES,
        default="evaluation",
    )
    parse_document = subparsers.add_parser(
        "parse-document",
        help="parse one uploaded PDF into canonical artifacts with MinerU",
    )
    parse_document.add_argument("document_id", type=UUID)
    chunk_document = subparsers.add_parser(
        "chunk-document",
        help="chunk an existing canonical document into structural chunks",
    )
    chunk_document.add_argument("document_id", type=UUID)

    benchmark = subparsers.add_parser(
        "benchmark-parsing",
        help="inventory and benchmark the existing MinerU parsing pipeline",
    )
    benchmark.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="read-only corpus root; it must be supplied explicitly",
    )
    benchmark.add_argument(
        "--workspace",
        type=Path,
        default=Path("data/benchmarks/a1-5"),
        help="ignored benchmark workspace",
    )
    benchmark.add_argument("--resume", action="store_true", help="resume an existing run")
    benchmark.add_argument(
        "--retry-failed",
        action="store_true",
        help="retry failed and timed-out representatives while resuming",
    )
    benchmark.add_argument(
        "--raw-retention",
        choices=RAW_RETENTION_VALUES,
        default="failures",
        help="retain raw MinerU output for failures, all items, or none",
    )
    benchmark.add_argument(
        "--limit", type=_positive_int, help="benchmark at most N inventory entries"
    )
    benchmark.add_argument("--subject", help="benchmark one exact classified subject")
    benchmark.add_argument(
        "--inventory-only",
        action="store_true",
        help="scan and persist inventory without starting MinerU",
    )

    retrieval_eval = subparsers.add_parser(
        "retrieval-eval",
        help="build and review the canonical-evidence text retrieval evaluation set",
    )
    retrieval_actions = retrieval_eval.add_subparsers(dest="retrieval_action", required=True)
    retrieval_build = retrieval_actions.add_parser(
        "build",
        help="materialize ignored chunks and candidate review artifacts",
    )
    retrieval_build.add_argument(
        "--canonical-root",
        type=Path,
        default=Path("data/benchmarks/a1-5/canonical"),
    )
    retrieval_build.add_argument(
        "--corpus-manifest",
        type=Path,
        default=Path("data/benchmarks/a1-5/corpus-manifest.jsonl"),
    )
    retrieval_build.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation/a2-1"),
    )
    retrieval_refresh = retrieval_actions.add_parser(
        "refresh-candidates",
        help="refresh candidates and review artifacts while reusing the existing chunk index",
    )
    retrieval_refresh.add_argument(
        "--canonical-root",
        type=Path,
        default=Path("data/benchmarks/a1-5/canonical"),
    )
    retrieval_refresh.add_argument(
        "--corpus-manifest",
        type=Path,
        default=Path("data/benchmarks/a1-5/corpus-manifest.jsonl"),
    )
    retrieval_refresh.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation/a2-1"),
    )
    retrieval_validate = retrieval_actions.add_parser(
        "validate",
        help="validate an existing ignored evaluation package against canonical evidence",
    )
    retrieval_validate.add_argument(
        "--canonical-root",
        type=Path,
        default=Path("data/benchmarks/a1-5/canonical"),
    )
    retrieval_validate.add_argument(
        "--corpus-manifest",
        type=Path,
        default=Path("data/benchmarks/a1-5/corpus-manifest.jsonl"),
    )
    retrieval_validate.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation/a2-1"),
    )
    retrieval_review = retrieval_actions.add_parser(
        "review",
        help="accept, reject, or edit one local evaluation candidate",
    )
    retrieval_review.add_argument("--output", type=Path, default=Path("data/evaluation/a2-1"))
    retrieval_review.add_argument("--item-id", required=True)
    retrieval_review.add_argument("--action", choices=("accept", "reject", "edit"), required=True)
    retrieval_review.add_argument("--query")
    retrieval_review.add_argument("--query-type", choices=QUERY_TYPES)
    retrieval_finalize = retrieval_actions.add_parser(
        "finalize",
        help="apply a complete human-review JSONL and build retrieval-eval-v1",
    )
    retrieval_finalize.add_argument(
        "--canonical-root",
        type=Path,
        default=Path("data/benchmarks/a1-5/canonical"),
    )
    retrieval_finalize.add_argument(
        "--corpus-manifest",
        type=Path,
        default=Path("data/benchmarks/a1-5/corpus-manifest.jsonl"),
    )
    retrieval_finalize.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation/a2-1"),
    )
    retrieval_finalize.add_argument(
        "--recommendations",
        type=Path,
        default=Path("data/evaluation/a2-1/A2.1_人工审核建议.jsonl"),
    )
    retrieval_finalize.add_argument(
        "--final-output",
        type=Path,
        default=Path("data/evaluation/a2-1/retrieval-eval-v1"),
    )
    retrieval_finalize.add_argument(
        "--repository-safe-output",
        type=Path,
        default=Path("docs/benchmarks/a2-1-retrieval-eval-v1.jsonl"),
    )

    embedding_benchmark = subparsers.add_parser(
        "embedding-benchmark",
        help="benchmark local dense embedding models on the frozen A2.1 set",
    )
    embedding_benchmark.add_argument(
        "--split",
        choices=("dev", "test", "both"),
        default="dev",
        help="benchmark dev first, then run the unchanged protocol on test",
    )
    embedding_benchmark.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_KEYS,
        default=list(MODEL_KEYS),
        help="candidate model keys; use the same list for dev and test",
    )
    embedding_benchmark.add_argument("--chunk-index", type=Path, default=DEFAULT_CHUNK_INDEX)
    embedding_benchmark.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    embedding_benchmark.add_argument(
        "--materialized",
        type=Path,
        default=DEFAULT_MATERIALIZED,
    )
    embedding_benchmark.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    embedding_benchmark.add_argument("--batch-size", type=_positive_int, default=4)
    embedding_benchmark.add_argument("--max-seq-length", type=_positive_int, default=512)
    embedding_benchmark.add_argument("--device", default="cuda")
    embedding_benchmark.add_argument(
        "--dtype",
        choices=("float16", "float32", "bfloat16"),
        default="float16",
    )

    reranker_benchmark_parser = subparsers.add_parser(
        "reranker-benchmark",
        help="benchmark local rerankers on the frozen A2.1 set",
    )
    reranker_benchmark_parser.add_argument(
        "--split",
        choices=("dev", "test", "both"),
        default="both",
        help="run the same reranking protocol on the selected frozen split(s)",
    )
    reranker_benchmark_parser.add_argument(
        "--models",
        nargs="+",
        choices=reranker_benchmark.RERANKER_MODEL_KEYS,
        default=list(reranker_benchmark.RERANKER_MODEL_KEYS),
    )
    reranker_benchmark_parser.add_argument(
        "--chunk-index",
        type=Path,
        default=reranker_benchmark.DEFAULT_CHUNK_INDEX,
    )
    reranker_benchmark_parser.add_argument(
        "--dataset",
        type=Path,
        default=reranker_benchmark.DEFAULT_DATASET,
    )
    reranker_benchmark_parser.add_argument(
        "--materialized",
        type=Path,
        default=reranker_benchmark.DEFAULT_MATERIALIZED,
    )
    reranker_benchmark_parser.add_argument(
        "--output",
        type=Path,
        default=reranker_benchmark.DEFAULT_OUTPUT,
    )
    reranker_benchmark_parser.add_argument("--batch-size", type=_positive_int, default=4)
    reranker_benchmark_parser.add_argument("--max-seq-length", type=_positive_int, default=512)
    reranker_benchmark_parser.add_argument(
        "--dense-max-seq-length",
        type=_positive_int,
        default=512,
    )
    reranker_benchmark_parser.add_argument(
        "--candidate-sizes",
        nargs="+",
        type=_positive_int,
        default=reranker_benchmark.CANDIDATE_SIZES,
    )
    reranker_benchmark_parser.add_argument("--device", default="cuda")
    reranker_benchmark_parser.add_argument(
        "--dtype",
        choices=("float16", "float32", "bfloat16"),
        default="float16",
    )

    system_benchmark = subparsers.add_parser(
        "retrieval-system-benchmark",
        help="benchmark exact dense, Qdrant dense, and fixed BGE Top-10 retrieval",
    )
    system_benchmark.add_argument(
        "--split",
        choices=("dev", "test", "both"),
        default="both",
        help="benchmark the frozen dev split, test split, or both",
    )
    system_benchmark.add_argument(
        "--chunk-index",
        type=Path,
        default=DEFAULT_SYSTEM_CHUNK_INDEX,
    )
    system_benchmark.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_SYSTEM_DATASET,
    )
    system_benchmark.add_argument(
        "--materialized",
        type=Path,
        default=DEFAULT_SYSTEM_MATERIALIZED,
    )
    system_benchmark.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SYSTEM_OUTPUT,
        help="ignored runtime output directory",
    )

    qdrant = subparsers.add_parser(
        "qdrant",
        help="create, index, and search the local Qdrant chunk vector store",
    )
    qdrant_actions = qdrant.add_subparsers(dest="qdrant_action", required=True)
    qdrant_actions.add_parser("check", help="check Qdrant connectivity and collection schema")
    qdrant_actions.add_parser("create", help="create or validate the versioned chunk collection")
    qdrant_index_document = qdrant_actions.add_parser(
        "index-document",
        help="index an existing document chunk artifact with Qwen3-Embedding-0.6B",
    )
    qdrant_index_document.add_argument("document_id", type=UUID)
    qdrant_index_corpus = qdrant_actions.add_parser(
        "index-corpus",
        help="index existing canonical artifacts without rerunning MinerU",
    )
    qdrant_index_corpus.add_argument(
        "--canonical-root",
        type=Path,
        default=Path("data/benchmarks/a1-5/canonical"),
    )
    qdrant_index_corpus.add_argument("--limit", type=_positive_int)
    qdrant_search = qdrant_actions.add_parser(
        "search",
        help="run dense top-k search over indexed chunks",
    )
    qdrant_search.add_argument("query")
    qdrant_search.add_argument("--knowledge-base-id", type=UUID)
    qdrant_search.add_argument("--document-id", type=UUID)
    qdrant_search.add_argument("--limit", type=_positive_int, default=10)

    neo4j = subparsers.add_parser(
        "neo4j",
        help="check and initialize the local Neo4j graph infrastructure",
    )
    neo4j_actions = neo4j.add_subparsers(dest="neo4j_action", required=True)
    neo4j_actions.add_parser("check", help="check Neo4j connectivity")
    neo4j_actions.add_parser("schema", help="create or validate graph constraints and indexes")

    graph_extraction = subparsers.add_parser(
        "graph-extraction-sample",
        help="extract a small stratified chunk sample into the A3.1 graph model",
    )
    graph_extraction.add_argument(
        "--canonical-root",
        type=Path,
        default=DEFAULT_EXTRACTION_CANONICAL_ROOT,
    )
    graph_extraction.add_argument(
        "--corpus-manifest",
        type=Path,
        default=DEFAULT_EXTRACTION_CORPUS_MANIFEST,
    )
    graph_extraction.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EXTRACTION_OUTPUT,
        help="ignored runtime sample and summary output directory",
    )
    graph_extraction.add_argument(
        "--sample-per-subject",
        type=_positive_int,
        default=2,
        help="number of canonical documents sampled for each subject",
    )
    graph_extraction.add_argument(
        "--sample-offset",
        type=int,
        default=0,
        help="skip this many eligible chunks within each subject before sampling",
    )
    graph_extraction.add_argument(
        "--knowledge-base-id",
        type=UUID,
        default=DEFAULT_SAMPLE_KNOWLEDGE_BASE_ID,
    )
    graph_extraction.add_argument(
        "--persist",
        action="store_true",
        help="persist the validated sample through the local Neo4j adapter",
    )

    entity_linking = subparsers.add_parser(
        "entity-linking-sample",
        help="review a small explicit local-entity linking sample",
    )
    entity_linking.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=list(DEFAULT_LINKING_INPUT),
        help="ignored A3.2 accepted-extraction JSONL file(s)",
    )
    entity_linking.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_LINKING_OUTPUT,
        help="ignored runtime review output directory",
    )
    entity_linking.add_argument(
        "--max-candidates",
        type=_positive_int,
        default=200,
        help="maximum deterministic candidates retained in the review sample",
    )
    entity_linking.add_argument(
        "--max-block-size",
        type=_positive_int,
        default=64,
        help="skip generic exact/prefix blocks larger than this size",
    )
    entity_linking.add_argument(
        "--adjudicate",
        action="store_true",
        help="use the configured LLM only for ambiguous candidates",
    )
    entity_linking.add_argument(
        "--persist",
        action="store_true",
        help="persist this explicit sample through the local Neo4j adapter",
    )

    rerank_search = subparsers.add_parser(
        "rerank-search",
        help="run dense Qdrant search followed by the configured local reranker",
    )
    rerank_search.add_argument("query")
    rerank_search.add_argument(
        "--model",
        choices=reranker_benchmark.RERANKER_MODEL_KEYS,
        help="local reranker model key; defaults to KNOWLEDGE_SCOPE_RERANKER_MODEL_KEY",
    )
    rerank_search.add_argument("--candidate-limit", type=_positive_int, default=20)
    rerank_search.add_argument("--limit", type=_positive_int, default=10)
    rerank_search.add_argument("--knowledge-base-id", type=UUID)
    rerank_search.add_argument("--document-id", type=UUID)
    return parser


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least one")
    return parsed


def _run_health() -> int:
    """Validate settings and print a non-sensitive health report."""
    try:
        settings = get_settings()
    except ValidationError:
        print("config_status: invalid", file=sys.stderr)
        return 1

    report = build_health_report(settings)
    for key, value in report.as_dict().items():
        print(f"{key}: {value}")
    return 0


async def _call_llm_smoke_test(
    args: argparse.Namespace,
) -> LLMResult:
    """Call the configured provider and persist its usage observation."""
    settings = get_settings()
    engine = create_database_engine(settings)
    provider = create_llm_provider(settings)
    try:
        gateway = LLMGateway(
            provider,
            DatabaseUsageRecorder(create_session_factory(engine)),
            settings,
        )
        messages: list[LLMMessage] = []
        if args.system:
            messages.append(LLMMessage(role="system", content=args.system))
        messages.append(LLMMessage(role="user", content=args.prompt))
        return await gateway.complete(
            LLMRequest(
                messages=messages,
                task_type=args.task_type,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                model=args.model,
            )
        )
    finally:
        await provider.aclose()
        await engine.dispose()


def _run_llm_smoke_test(args: argparse.Namespace) -> int:
    """Run one real configured-provider call without printing credentials."""
    try:
        result = asyncio.run(_call_llm_smoke_test(args))
    except (LLMError, ValidationError, ValueError) as error:
        print("llm_status: failed", file=sys.stderr)
        print(f"error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("llm_status: interrupted", file=sys.stderr)
        return 130

    print("llm_status: ok")
    print(f"provider: {result.provider}")
    print(f"model: {result.model}")
    print(f"input_tokens: {result.input_tokens}")
    print(f"output_tokens: {result.output_tokens}")
    print(f"latency_ms: {result.latency_ms:.2f}")
    print(f"finish_reason: {result.finish_reason}")
    print(f"text: {result.text}")
    return 0


def _run_parse_document(document_id: UUID) -> int:
    """Run the developer-only parse workflow and print non-sensitive facts."""
    try:
        settings = get_settings()
    except ValidationError:
        print("config_status: invalid", file=sys.stderr)
        return 1

    try:
        result = asyncio.run(parse_document_by_id(document_id, settings))
    except DocumentParseError as error:
        print("parse_status: failed", file=sys.stderr)
        print(f"error: {error}", file=sys.stderr)
        return 1

    print("parse_status: ok")
    print(f"document_id: {result.document_id}")
    print("parser: mineru")
    print(f"parser_version: {result.parser_version}")
    print(f"backend: {result.backend}")
    print(f"elapsed_seconds: {result.elapsed_seconds:.2f}")
    print(f"canonical_ref: {result.canonical_ref}")
    print(f"raw_ref: {result.raw_ref}")
    for key, value in result.stats.as_dict().items():
        print(f"{key}: {value}")
    print("canonical_validation: ok")
    return 0


def _run_chunk_document(document_id: UUID) -> int:
    """Run the developer-only canonical chunking workflow."""
    try:
        settings = get_settings()
    except ValidationError:
        print("config_status: invalid", file=sys.stderr)
        return 1

    try:
        result = chunk_document_by_id(document_id, settings)
    except ChunkingError as error:
        print("chunk_status: failed", file=sys.stderr)
        print(f"error: {error}", file=sys.stderr)
        return 1

    print("chunk_status: ok")
    print(f"document_id: {result.document_id}")
    print(f"chunks: {result.chunk_count}")
    print(f"config_fingerprint: {result.config_fingerprint}")
    print(f"chunks_ref: {result.chunks_ref}")
    print(f"manifest_ref: {result.manifest_ref}")
    return 0


def _run_benchmark(args: argparse.Namespace) -> int:
    """Run inventory-only or resumable benchmark execution."""
    try:
        settings = get_settings()
        if args.inventory_only:
            if args.resume or args.retry_failed:
                raise BenchmarkError("inventory-only cannot be combined with resume options")
            inventory = inventory_corpus(args.corpus, args.workspace)
            print("inventory_status: ok")
            print(f"inventory_fingerprint: {inventory.fingerprint}")
            print(json.dumps(inventory.summary, ensure_ascii=False, indent=2))
            return 0

        outcome = run_benchmark(
            BenchmarkConfig(
                corpus_root=args.corpus,
                workspace=args.workspace,
                raw_retention=args.raw_retention,
                resume=args.resume,
                retry_failed=args.retry_failed,
                limit=args.limit,
                subject=args.subject,
            ),
            settings,
        )
    except (BenchmarkError, ValueError) as error:
        print("benchmark_status: failed", file=sys.stderr)
        print(f"error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("benchmark_status: interrupted", file=sys.stderr)
        return 130

    benchmark = outcome.aggregate["benchmark"]
    if not isinstance(benchmark, dict):
        print("benchmark_status: failed", file=sys.stderr)
        print("error: aggregate benchmark section is invalid", file=sys.stderr)
        return 1
    status = "complete" if benchmark.get("all_unique_pdfs_terminal") else "partial"
    print(f"benchmark_status: {status}")
    print(f"run_id: {outcome.run_id}")
    print(f"workspace: {outcome.workspace}")
    print(f"inventory_pdfs: {len(outcome.inventory.items)}")
    print(f"selected_entries: {len(outcome.selected_items)}")
    print(f"completed_unique_pdfs: {benchmark.get('completed_unique_pdfs')}")
    print(f"successful_unique_pdfs: {benchmark.get('successful_unique_pdfs')}")
    print(f"failed_unique_pdfs: {benchmark.get('failed_unique_pdfs')}")
    print(f"aggregate_ref: {outcome.workspace / 'aggregate.json'}")
    return 0


def _run_retrieval_eval(args: argparse.Namespace) -> int:
    """Build, validate, or edit the local A2.1 evaluation package."""
    try:
        if args.retrieval_action == "build":
            manifest = materialize_retrieval_eval_set(
                args.canonical_root,
                args.corpus_manifest,
                args.output,
            )
            print("retrieval_eval_status: built")
            print(json.dumps(manifest, ensure_ascii=False, indent=2))
            return 0
        if args.retrieval_action == "validate":
            report = validate_runtime_evaluation(
                args.output,
                args.canonical_root,
                args.corpus_manifest,
            )
            print(
                "retrieval_eval_status: valid" if report.valid else "retrieval_eval_status: invalid"
            )
            print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
            return 0 if report.valid else 1
        if args.retrieval_action == "refresh-candidates":
            manifest = regenerate_candidate_review_pack(
                args.canonical_root,
                args.corpus_manifest,
                args.output,
            )
            print("retrieval_eval_status: refreshed")
            print(json.dumps(manifest, ensure_ascii=False, indent=2))
            return 0
        if args.retrieval_action == "review":
            updated = apply_review_action(
                args.output,
                args.item_id,
                args.action,
                query=args.query,
                query_type=args.query_type,
            )
            print("retrieval_eval_status: reviewed")
            print(json.dumps(updated.model_dump(mode="json"), ensure_ascii=False, indent=2))
            return 0
        if args.retrieval_action == "finalize":
            manifest = finalize_retrieval_eval_dataset(
                args.output,
                args.recommendations,
                args.canonical_root,
                args.corpus_manifest,
                final_output_dir=args.final_output,
                repository_safe_path=args.repository_safe_output,
            )
            print("retrieval_eval_status: finalized")
            print(json.dumps(manifest, ensure_ascii=False, indent=2))
            return 0
    except (RetrievalEvalError, ValueError) as error:
        print("retrieval_eval_status: failed", file=sys.stderr)
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 1


def _run_embedding_benchmark(args: argparse.Namespace) -> int:
    """Run the read-only local embedding benchmark."""
    try:
        outcome = run_embedding_benchmark(
            split=args.split,
            model_keys=args.models,
            protocol=EmbeddingBenchmarkProtocol(
                batch_size=args.batch_size,
                max_seq_length=args.max_seq_length,
                dtype=args.dtype,
                device=args.device,
            ),
            chunk_index_path=args.chunk_index,
            dataset_path=args.dataset,
            materialized_path=args.materialized,
            output_dir=args.output,
        )
    except (EmbeddingBenchmarkError, ValueError) as error:
        print("embedding_benchmark_status: failed", file=sys.stderr)
        print(f"error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("embedding_benchmark_status: interrupted", file=sys.stderr)
        return 130

    manifest = outcome["manifest"]
    if not isinstance(manifest, dict):
        print("embedding_benchmark_status: failed", file=sys.stderr)
        print("error: benchmark manifest is invalid", file=sys.stderr)
        return 1
    status_counts = manifest.get("status_counts", {})
    status = "complete" if status_counts.get("failed", 0) == 0 else "complete_with_failures"
    print(f"embedding_benchmark_status: {status}")
    print(f"output: {args.output}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _run_reranker_benchmark(args: argparse.Namespace) -> int:
    """Run the read-only local reranker benchmark."""
    try:
        outcome = reranker_benchmark.run_reranker_benchmark(
            split=args.split,
            model_keys=args.models,
            protocol=reranker_benchmark.RerankerBenchmarkProtocol(
                batch_size=args.batch_size,
                max_seq_length=args.max_seq_length,
                dense_max_seq_length=args.dense_max_seq_length,
                dtype=args.dtype,
                device=args.device,
                candidate_sizes=tuple(args.candidate_sizes),
            ),
            chunk_index_path=args.chunk_index,
            dataset_path=args.dataset,
            materialized_path=args.materialized,
            output_dir=args.output,
        )
    except (reranker_benchmark.RerankerBenchmarkError, ValueError) as error:
        print("reranker_benchmark_status: failed", file=sys.stderr)
        print(f"error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("reranker_benchmark_status: interrupted", file=sys.stderr)
        return 130

    manifest = outcome["manifest"]
    if not isinstance(manifest, dict):
        print("reranker_benchmark_status: failed", file=sys.stderr)
        print("error: benchmark manifest is invalid", file=sys.stderr)
        return 1
    status_counts = manifest.get("status_counts", {})
    status = "complete" if status_counts.get("failed", 0) == 0 else "complete_with_failures"
    print(f"reranker_benchmark_status: {status}")
    print(f"output: {args.output}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _run_retrieval_system_benchmark(args: argparse.Namespace) -> int:
    """Run the frozen A2.5 retrieval-stage system benchmark."""
    try:
        outcome = run_retrieval_system_benchmark(
            split=args.split,
            chunk_index_path=args.chunk_index,
            dataset_path=args.dataset,
            materialized_path=args.materialized,
            output_dir=args.output,
        )
    except (RetrievalSystemBenchmarkError, ValueError) as error:
        print("retrieval_system_benchmark_status: failed", file=sys.stderr)
        print(f"error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("retrieval_system_benchmark_status: interrupted", file=sys.stderr)
        return 130

    manifest = outcome["manifest"]
    if not isinstance(manifest, dict):
        print("retrieval_system_benchmark_status: failed", file=sys.stderr)
        print("error: benchmark manifest is invalid", file=sys.stderr)
        return 1
    print("retrieval_system_benchmark_status: complete")
    print(f"output: {args.output}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _run_rerank_search(args: argparse.Namespace) -> int:
    """Run the developer dense-then-rerank search workflow."""
    try:
        settings = get_settings()
        store = QdrantVectorStore(settings)
        try:
            dense_result = DenseRetrievalService(
                store,
                QwenEmbeddingModel(settings),
            ).search(
                args.query,
                limit=args.candidate_limit,
                knowledge_base_id=args.knowledge_base_id,
                document_id=args.document_id,
            )
            reranker = create_local_reranker(settings, model_key=args.model)
            reranked = RerankingService(reranker).rerank(
                args.query,
                dense_result.items,
                limit=args.limit,
            )
            print(
                json.dumps(
                    {
                        "query": args.query,
                        "dense_model": dense_result.model_id,
                        "dense_limit": dense_result.limit,
                        "reranker_model": reranker.model_id,
                        "items": [
                            {
                                "dense_rank": item.dense_rank,
                                "reranker_score": item.reranker_score,
                                "chunk": item.chunk.model_dump(mode="json"),
                            }
                            for item in reranked
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        finally:
            store.close()
    except (
        EmbeddingModelError,
        RerankerError,
        RetrievalError,
        ValidationError,
        VectorStoreError,
        ValueError,
    ) as error:
        print("rerank_search_status: failed", file=sys.stderr)
        print(f"error: {error}", file=sys.stderr)
        return 1


def _run_qdrant(args: argparse.Namespace) -> int:
    """Run the small local Qdrant developer workflow."""
    try:
        settings = get_settings()
        store = QdrantVectorStore(settings)
        try:
            if args.qdrant_action == "check":
                readiness = store.readiness()
                print(json.dumps(readiness.model_dump(mode="json"), ensure_ascii=False, indent=2))
                return 0 if readiness.status != "unavailable" else 1
            if args.qdrant_action == "create":
                readiness = store.ensure_collection()
                print(json.dumps(readiness.model_dump(mode="json"), ensure_ascii=False, indent=2))
                return 0
            if args.qdrant_action == "index-document":
                result = asyncio.run(
                    index_document_by_id(
                        args.document_id,
                        settings,
                        store=store,
                        embedder=QwenEmbeddingModel(settings),
                    )
                )
                print(json.dumps(asdict(result), ensure_ascii=False, indent=2, default=str))
                return 0
            if args.qdrant_action == "index-corpus":
                results = index_canonical_corpus(
                    args.canonical_root,
                    settings,
                    limit=args.limit,
                    store=store,
                    embedder=QwenEmbeddingModel(settings),
                )
                print(
                    json.dumps(
                        {"documents": len(results), "results": [asdict(item) for item in results]},
                        ensure_ascii=False,
                        indent=2,
                        default=str,
                    )
                )
                return 0
            if args.qdrant_action == "search":
                result = DenseRetrievalService(
                    store,
                    QwenEmbeddingModel(settings),
                ).search(
                    args.query,
                    limit=args.limit,
                    knowledge_base_id=args.knowledge_base_id,
                    document_id=args.document_id,
                )
                print(
                    json.dumps(
                        {
                            "query": result.query,
                            "model": result.model_id,
                            "collection": result.collection_name,
                            "items": [item.model_dump(mode="json") for item in result.items],
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                return 0
        finally:
            store.close()
    except (
        EmbeddingModelError,
        IndexingError,
        RetrievalError,
        ValidationError,
        VectorStoreError,
        ValueError,
    ) as error:
        print("qdrant_status: failed", file=sys.stderr)
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 1


def _run_neo4j(args: argparse.Namespace) -> int:
    """Run the small local Neo4j developer workflow."""
    store = None
    try:
        settings = get_settings()
        store = Neo4jGraphStore(settings)
        if args.neo4j_action == "check":
            readiness = store.readiness()
            print(json.dumps(readiness.model_dump(mode="json"), ensure_ascii=False, indent=2))
            return 0 if readiness.status == "ready" else 1
        if args.neo4j_action == "schema":
            readiness = store.ensure_schema()
            print(json.dumps(readiness.model_dump(mode="json"), ensure_ascii=False, indent=2))
            return 0
    except (GraphStoreError, ValidationError, ValueError) as error:
        print("neo4j_status: failed", file=sys.stderr)
        print(f"error: {error}", file=sys.stderr)
        return 1
    finally:
        if store is not None:
            store.close()
    return 1


async def _run_graph_extraction_sample_async(args: argparse.Namespace) -> dict[str, object]:
    """Run the explicit, small A3.2 sample workflow and close its resources."""

    settings = get_settings()
    samples = select_sample_chunks(
        args.canonical_root,
        args.corpus_manifest,
        sample_per_subject=args.sample_per_subject,
        sample_offset=args.sample_offset,
    )
    engine = create_database_engine(settings)
    provider = create_llm_provider(settings)
    store = Neo4jGraphStore(settings) if args.persist else None
    try:
        if store is not None:
            await asyncio.to_thread(store.ensure_schema)
        gateway = LLMGateway(
            provider,
            DatabaseUsageRecorder(create_session_factory(engine)),
            settings,
        )
        return await run_sample_evaluation(
            samples,
            gateway=gateway,
            settings=settings,
            output_dir=args.output,
            knowledge_base_id=args.knowledge_base_id,
            store=store,
        )
    finally:
        if store is not None:
            store.close()
        await provider.aclose()
        await engine.dispose()


def _run_graph_extraction_sample(args: argparse.Namespace) -> int:
    """Run the developer-only A3.2 extraction sample without exposing secrets."""

    try:
        summary = asyncio.run(_run_graph_extraction_sample_async(args))
    except (
        ExtractionError,
        GraphStoreError,
        LLMError,
        RetrievalEvalError,
        ValidationError,
        ValueError,
    ) as error:
        print("graph_extraction_status: failed", file=sys.stderr)
        print(f"error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("graph_extraction_status: interrupted", file=sys.stderr)
        return 130

    print("graph_extraction_status: complete")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


async def _run_entity_linking_sample_async(args: argparse.Namespace) -> dict[str, object]:
    """Run the bounded A3.3 sample and close optional provider/database resources."""

    settings = get_settings()
    provider = None
    engine = None
    store = None
    gateway = None
    if args.adjudicate:
        if settings.llm_api_key is None or not settings.llm_api_key.get_secret_value().strip():
            raise ValueError("KNOWLEDGE_SCOPE_LLM_API_KEY is required with --adjudicate")
        engine = create_database_engine(settings)
        provider = create_llm_provider(settings)
        gateway = LLMGateway(
            provider,
            DatabaseUsageRecorder(create_session_factory(engine)),
            settings,
        )
    if args.persist:
        store = Neo4jGraphStore(settings)
        await asyncio.to_thread(store.ensure_schema)
    try:
        return await run_linking_review_sample(
            args.input,
            settings=settings,
            output_dir=args.output,
            gateway=gateway,
            store=store,
            max_candidates=args.max_candidates,
            max_block_size=args.max_block_size,
        )
    finally:
        if store is not None:
            store.close()
        if provider is not None:
            await provider.aclose()
        if engine is not None:
            await engine.dispose()


def _run_entity_linking_sample(args: argparse.Namespace) -> int:
    """Run the developer-only A3.3 linking sample without exposing secrets."""

    try:
        summary = asyncio.run(_run_entity_linking_sample_async(args))
    except (
        GraphStoreError,
        LLMError,
        LinkingValidationError,
        ValidationError,
        ValueError,
    ) as error:
        print("entity_linking_status: failed", file=sys.stderr)
        print(f"error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("entity_linking_status: interrupted", file=sys.stderr)
        return 130

    print("entity_linking_status: complete")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "health":
        return _run_health()
    if args.command == "llm-smoke-test":
        return _run_llm_smoke_test(args)
    if args.command == "parse-document":
        return _run_parse_document(args.document_id)
    if args.command == "chunk-document":
        return _run_chunk_document(args.document_id)
    if args.command == "benchmark-parsing":
        return _run_benchmark(args)
    if args.command == "retrieval-eval":
        return _run_retrieval_eval(args)
    if args.command == "embedding-benchmark":
        return _run_embedding_benchmark(args)
    if args.command == "reranker-benchmark":
        return _run_reranker_benchmark(args)
    if args.command == "retrieval-system-benchmark":
        return _run_retrieval_system_benchmark(args)
    if args.command == "qdrant":
        return _run_qdrant(args)
    if args.command == "neo4j":
        return _run_neo4j(args)
    if args.command == "graph-extraction-sample":
        return _run_graph_extraction_sample(args)
    if args.command == "entity-linking-sample":
        return _run_entity_linking_sample(args)
    if args.command == "rerank-search":
        return _run_rerank_search(args)

    parser.print_help()
    return 0
