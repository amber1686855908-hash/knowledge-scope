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
from sqlalchemy import select

from knowledge_scope import __version__
from knowledge_scope.chunking.service import ChunkingError, chunk_document_by_id
from knowledge_scope.documents.models import DOCUMENT_STATUS_REGISTERED, Document
from knowledge_scope.documents.registration import (
    DEFAULT_CANONICAL_ROOT as DEFAULT_REGISTRATION_CANONICAL_ROOT,
)
from knowledge_scope.documents.registration import (
    DEFAULT_CHUNK_INDEX as DEFAULT_REGISTRATION_CHUNK_INDEX,
)
from knowledge_scope.documents.registration import (
    DEFAULT_CORPUS_MANIFEST,
    DEFAULT_EVAL_DATASET,
    DEFAULT_EVAL_KB_MAPPING,
    CorpusRegistrationError,
    build_frozen_eval_kb_mapping,
    build_registration_spec,
    register_corpus,
    write_frozen_eval_kb_mapping,
)
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
from knowledge_scope.evaluation.graph_corpus_build import (
    DEFAULT_CHUNK_INDEX as DEFAULT_GRAPH_CORPUS_CHUNK_INDEX,
)
from knowledge_scope.evaluation.graph_corpus_build import (
    DEFAULT_CORPUS_MANIFEST as DEFAULT_GRAPH_CORPUS_MANIFEST,
)
from knowledge_scope.evaluation.graph_corpus_build import (
    DEFAULT_ESTIMATE_SUMMARIES,
    GraphCorpusBuildError,
    GraphCorpusRunner,
    audit_corpus_files,
    build_corpus_input_snapshot,
    estimate_full_run,
    iter_corpus_documents,
    load_corpus_input_snapshot,
    read_graph_coverage,
    select_document_ids,
    validate_corpus_input_snapshot,
)
from knowledge_scope.evaluation.graph_corpus_build import (
    DEFAULT_EVAL_DATASET as DEFAULT_GRAPH_CORPUS_EVAL_DATASET,
)
from knowledge_scope.evaluation.graph_corpus_build import (
    DEFAULT_INPUT_SNAPSHOT as DEFAULT_GRAPH_CORPUS_INPUT_SNAPSHOT,
)
from knowledge_scope.evaluation.graph_corpus_build import (
    DEFAULT_OUTPUT as DEFAULT_GRAPH_CORPUS_OUTPUT,
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
from knowledge_scope.evaluation.graph_retrieval_sample import (
    DEFAULT_INPUT as DEFAULT_GRAPH_RETRIEVAL_INPUT,
)
from knowledge_scope.evaluation.graph_retrieval_sample import (
    DEFAULT_OUTPUT as DEFAULT_GRAPH_RETRIEVAL_OUTPUT,
)
from knowledge_scope.evaluation.graph_retrieval_sample import (
    DEFAULT_SAMPLE_KNOWLEDGE_BASE_ID as DEFAULT_GRAPH_RETRIEVAL_KNOWLEDGE_BASE_ID,
)
from knowledge_scope.evaluation.graph_retrieval_sample import (
    run_graph_retrieval_review_sample,
)
from knowledge_scope.evaluation.hybrid_evaluation import (
    DEFAULT_CHUNK_INDEX as DEFAULT_HYBRID_EVAL_CHUNK_INDEX,
)
from knowledge_scope.evaluation.hybrid_evaluation import (
    DEFAULT_DATASET as DEFAULT_HYBRID_EVAL_DATASET,
)
from knowledge_scope.evaluation.hybrid_evaluation import (
    DEFAULT_GRAPH_EXCLUSIONS,
    DEFAULT_GRAPH_RUN_MANIFEST,
    DEFAULT_KB_MAPPING,
    HybridEvaluationError,
    run_hybrid_evaluation,
)
from knowledge_scope.evaluation.hybrid_evaluation import (
    DEFAULT_MATERIALIZED as DEFAULT_HYBRID_EVAL_MATERIALIZED,
)
from knowledge_scope.evaluation.hybrid_evaluation import (
    DEFAULT_OUTPUT as DEFAULT_HYBRID_EVAL_OUTPUT,
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
    BGE_RERANKER_MODEL_REVISION,
    RetrievalSystemBenchmarkError,
    run_retrieval_system_benchmark,
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
from knowledge_scope.extraction.service import ExtractionError, ExtractionService
from knowledge_scope.graph.neo4j import GraphStoreError, Neo4jGraphStore
from knowledge_scope.graph.retrieval import GraphRetrievalConfig
from knowledge_scope.graph.retrieval_service import GraphRetrievalError, GraphRetrievalService
from knowledge_scope.linking.service import LinkingValidationError
from knowledge_scope.llm import LLMGateway, LLMMessage, LLMRequest, LLMResult, create_llm_provider
from knowledge_scope.llm.errors import LLMError
from knowledge_scope.llm.schemas import LLM_TASK_TYPES
from knowledge_scope.llm.usage import DatabaseUsageRecorder
from knowledge_scope.parsing.service import DocumentParseError, parse_document_by_id
from knowledge_scope.retrieval.embedding import EmbeddingModelError, QwenEmbeddingModel
from knowledge_scope.retrieval.hybrid import (
    HybridRetrievalConfig,
    HybridRetrievalError,
    HybridRetrievalService,
)
from knowledge_scope.retrieval.indexing import (
    IndexingError,
    index_canonical_corpus,
    index_document_by_id,
)
from knowledge_scope.retrieval.qdrant import QdrantVectorStore, VectorStoreError
from knowledge_scope.retrieval.qdrant_attribution import (
    QdrantAttributionError,
    apply_attribution_repair,
    build_attribution_audit,
    load_authoritative_document_mappings,
    point_identity_signature,
)
from knowledge_scope.retrieval.reranking import (
    RerankerError,
    RerankingService,
    create_local_reranker,
)
from knowledge_scope.retrieval.service import DenseRetrievalService, RetrievalError
from knowledge_scope.shared import build_health_report, get_settings
from knowledge_scope.shared.config import Settings
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

    corpus = subparsers.add_parser(
        "corpus",
        help="register an existing parsed corpus as explicit reference-backed documents",
    )
    corpus_actions = corpus.add_subparsers(dest="corpus_action", required=True)
    register_corpus_parser = corpus_actions.add_parser(
        "register",
        help="register existing canonical/chunk artifacts in one supplied Knowledge Base",
    )
    register_corpus_parser.add_argument("--knowledge-base-id", type=UUID, required=True)
    register_corpus_parser.add_argument(
        "--corpus-manifest",
        type=Path,
        default=DEFAULT_CORPUS_MANIFEST,
    )
    register_corpus_parser.add_argument(
        "--canonical-root",
        type=Path,
        default=DEFAULT_REGISTRATION_CANONICAL_ROOT,
    )
    register_corpus_parser.add_argument(
        "--chunk-index",
        type=Path,
        default=DEFAULT_REGISTRATION_CHUNK_INDEX,
    )
    register_corpus_parser.add_argument(
        "--expected-document-count",
        type=_non_negative_int,
        default=255,
    )
    register_corpus_parser.add_argument(
        "--expected-chunk-count",
        type=_non_negative_int,
        default=7_524,
    )
    register_corpus_parser.add_argument(
        "--evaluation-dataset",
        type=Path,
        default=DEFAULT_EVAL_DATASET,
        help="frozen A2.1 dataset used to derive an ignored item-to-KB mapping",
    )
    register_corpus_parser.add_argument(
        "--evaluation-mapping-output",
        type=Path,
        default=DEFAULT_EVAL_KB_MAPPING,
        help="ignored runtime path for the derived A2.1 item-to-KB mapping",
    )

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

    hybrid_evaluation = subparsers.add_parser(
        "hybrid-evaluation",
        help="evaluate frozen A2.1 vector-only and vector-plus-graph retrieval",
    )
    hybrid_evaluation.add_argument("--knowledge-base-id", type=UUID, required=True)
    hybrid_evaluation.add_argument(
        "--split",
        choices=("dev", "test", "both"),
        default="both",
        help="evaluate the frozen dev split, test split, or both",
    )
    hybrid_evaluation.add_argument(
        "--chunk-index",
        type=Path,
        default=DEFAULT_HYBRID_EVAL_CHUNK_INDEX,
    )
    hybrid_evaluation.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_HYBRID_EVAL_DATASET,
    )
    hybrid_evaluation.add_argument(
        "--materialized",
        type=Path,
        default=DEFAULT_HYBRID_EVAL_MATERIALIZED,
    )
    hybrid_evaluation.add_argument(
        "--kb-mapping",
        type=Path,
        default=DEFAULT_KB_MAPPING,
    )
    hybrid_evaluation.add_argument(
        "--graph-manifest",
        type=Path,
        default=DEFAULT_GRAPH_RUN_MANIFEST,
    )
    hybrid_evaluation.add_argument(
        "--graph-exclusions",
        type=Path,
        default=DEFAULT_GRAPH_EXCLUSIONS,
    )
    hybrid_evaluation.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_HYBRID_EVAL_OUTPUT,
        help="ignored per-query and aggregate output directory",
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
    qdrant_audit_kb = qdrant_actions.add_parser(
        "audit-kb",
        help="audit or explicitly repair authoritative Knowledge Base payloads",
    )
    qdrant_audit_kb.add_argument(
        "--apply",
        action="store_true",
        help="apply payload-only repair after a complete, unambiguous preflight",
    )
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

    graph_search = subparsers.add_parser(
        "graph-search",
        help="run bounded evidence-first graph retrieval for one knowledge base",
    )
    graph_search.add_argument("query")
    graph_search.add_argument("--knowledge-base-id", type=UUID, required=True)
    graph_search.add_argument("--max-seed-entities", type=_positive_int)
    graph_search.add_argument("--max-hops", type=_positive_int, choices=(1, 2))
    graph_search.add_argument("--max-neighbors", type=_positive_int)
    graph_search.add_argument("--max-relations", type=_positive_int)
    graph_search.add_argument("--max-evidence", type=_positive_int)

    graph_retrieval_sample = subparsers.add_parser(
        "graph-retrieval-sample",
        help="review a small graph-retrieval sample over an existing local graph",
    )
    graph_retrieval_sample.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=list(DEFAULT_GRAPH_RETRIEVAL_INPUT),
        help="ignored A3.2 accepted-extraction JSONL file(s)",
    )
    graph_retrieval_sample.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_GRAPH_RETRIEVAL_OUTPUT,
        help="ignored runtime review output directory",
    )
    graph_retrieval_sample.add_argument(
        "--knowledge-base-id",
        type=UUID,
        default=DEFAULT_GRAPH_RETRIEVAL_KNOWLEDGE_BASE_ID,
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

    hybrid_search = subparsers.add_parser(
        "hybrid-search",
        help="run Qwen dense+BGE reranking and bounded graph retrieval with RRF",
    )
    hybrid_search.add_argument("query")
    hybrid_search.add_argument("--knowledge-base-id", type=UUID, required=True)
    hybrid_search.add_argument("--document-id", type=UUID)
    hybrid_search.add_argument("--vector-candidate-limit", type=_positive_int)
    hybrid_search.add_argument("--vector-rerank-limit", type=_positive_int)
    hybrid_search.add_argument("--graph-limit", type=_positive_int)
    hybrid_search.add_argument("--limit", type=_positive_int)
    hybrid_search.add_argument("--rrf-k", type=_positive_int)
    hybrid_search.add_argument(
        "--failure-mode",
        choices=("strict", "degraded"),
        help="strict fails if either branch fails; degraded returns the surviving branch",
    )

    graph_corpus_audit = subparsers.add_parser(
        "graph-corpus-audit",
        help="audit registered corpus, chunk coverage, and current Neo4j graph coverage",
    )
    graph_corpus_audit.add_argument("--knowledge-base-id", type=UUID, required=True)
    graph_corpus_audit.add_argument(
        "--corpus-manifest",
        type=Path,
        default=DEFAULT_GRAPH_CORPUS_MANIFEST,
    )
    graph_corpus_audit.add_argument(
        "--chunk-index",
        type=Path,
        default=DEFAULT_GRAPH_CORPUS_CHUNK_INDEX,
    )
    graph_corpus_audit.add_argument(
        "--eval-dataset",
        type=Path,
        default=DEFAULT_GRAPH_CORPUS_EVAL_DATASET,
    )
    graph_corpus_audit.add_argument(
        "--input-snapshot",
        type=Path,
        default=DEFAULT_GRAPH_CORPUS_INPUT_SNAPSHOT,
        help="repository-safe authoritative corpus/chunk completeness snapshot",
    )

    graph_corpus_estimate = subparsers.add_parser(
        "graph-corpus-estimate",
        help="estimate full-corpus extraction cost from existing A3.2 samples",
    )
    graph_corpus_estimate.add_argument(
        "--summary",
        type=Path,
        nargs="+",
        default=list(DEFAULT_ESTIMATE_SUMMARIES),
    )
    graph_corpus_estimate.add_argument(
        "--target-chunks",
        type=_positive_int,
        help="override the target chunk count; defaults to the authoritative input snapshot",
    )
    graph_corpus_estimate.add_argument(
        "--input-snapshot",
        type=Path,
        default=DEFAULT_GRAPH_CORPUS_INPUT_SNAPSHOT,
        help="authoritative snapshot used for the default target chunk count",
    )
    graph_corpus_estimate.add_argument("--max-concurrency", type=_positive_int, default=1)

    graph_corpus_build = subparsers.add_parser(
        "graph-corpus-build",
        help=(
            "run bounded, resumable extraction and optional A3.3 linking over registered documents"
        ),
    )
    graph_corpus_build.add_argument("--knowledge-base-id", type=UUID, required=True)
    graph_corpus_build.add_argument(
        "--corpus-manifest",
        type=Path,
        default=DEFAULT_GRAPH_CORPUS_MANIFEST,
    )
    graph_corpus_build.add_argument(
        "--chunk-index",
        type=Path,
        default=DEFAULT_GRAPH_CORPUS_CHUNK_INDEX,
    )
    graph_corpus_build.add_argument(
        "--input-snapshot",
        type=Path,
        default=DEFAULT_GRAPH_CORPUS_INPUT_SNAPSHOT,
        help="repository-safe authoritative corpus/chunk completeness snapshot",
    )
    graph_corpus_build.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_GRAPH_CORPUS_OUTPUT,
        help="ignored runtime checkpoint and manifest directory",
    )
    graph_corpus_scope = graph_corpus_build.add_mutually_exclusive_group()
    graph_corpus_scope.add_argument(
        "--full",
        action="store_true",
        help="select all registered corpus documents",
    )
    graph_corpus_scope.add_argument(
        "--sample-per-subject",
        type=_positive_int,
        default=1,
        help="select a stable bounded number of documents per subject",
    )
    graph_corpus_build.add_argument("--sample-offset", type=_non_negative_int, default=0)
    graph_corpus_build.add_argument("--max-documents", type=_positive_int)
    graph_corpus_build.add_argument("--max-concurrency", type=_positive_int, default=1)
    graph_corpus_build.add_argument("--link-batch-documents", type=_positive_int, default=8)
    graph_corpus_build.add_argument("--retry-failed", action="store_true")
    graph_corpus_build.add_argument(
        "--reprocess",
        action="store_true",
        help="delete the existing graph state for each selected document before rebuilding",
    )
    graph_corpus_build.add_argument(
        "--adjudicate",
        action="store_true",
        help="use the configured LLM only for ambiguous A3.3 linking candidates",
    )
    graph_corpus_build.add_argument(
        "--persist",
        action="store_true",
        help="required: persist extraction/linking results through Neo4j",
    )
    graph_corpus_build.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        default=True,
        help="refuse an existing output directory instead of resuming it",
    )
    return parser


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least one")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must not be negative")
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


async def _register_corpus_async(args: argparse.Namespace, settings: Settings) -> dict[str, object]:
    """Validate corpus artifacts, then register them in one database transaction."""
    spec = build_registration_spec(
        args.knowledge_base_id,
        corpus_manifest=args.corpus_manifest,
        canonical_root=args.canonical_root,
        chunk_index=args.chunk_index,
        expected_document_count=args.expected_document_count,
        expected_chunk_count=args.expected_chunk_count,
    )
    evaluation_mapping = build_frozen_eval_kb_mapping(
        args.evaluation_dataset,
        registered_document_ids={document.document_id for document in spec.documents},
        knowledge_base_id=args.knowledge_base_id,
    )
    engine = create_database_engine(settings)
    try:
        session_factory = create_session_factory(engine)
        async with session_factory() as session:
            result = await register_corpus(session, spec)
    finally:
        await engine.dispose()
    write_frozen_eval_kb_mapping(args.evaluation_mapping_output, evaluation_mapping)
    return {
        "schema_version": "1.0",
        "status": "registered",
        "manifest_rows": spec.manifest_rows,
        "duplicate_manifest_rows": spec.duplicate_manifest_rows,
        "unique_document_count": len(spec.documents),
        "chunk_count": spec.chunk_count,
        "documents_attempted": result.documents_attempted,
        "documents_inserted": result.documents_inserted,
        "documents_already_valid": result.documents_already_valid,
        "conflicts": result.conflicts,
        "failures": result.failures,
        "knowledge_base_id": str(result.knowledge_base_id),
        "evaluation_item_count": len(evaluation_mapping),
        "evaluation_mapping_output": str(args.evaluation_mapping_output),
    }


def _run_corpus(args: argparse.Namespace) -> int:
    """Run the explicit external-corpus registration workflow."""
    try:
        settings = get_settings()
        output = asyncio.run(_register_corpus_async(args, settings))
    except (CorpusRegistrationError, ValidationError, ValueError) as error:
        print("corpus_registration_status: failed", file=sys.stderr)
        print(f"error: {error}", file=sys.stderr)
        return 1
    print("corpus_registration_status: complete")
    print(json.dumps(output, ensure_ascii=False, indent=2))
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


def _run_hybrid_evaluation(args: argparse.Namespace) -> int:
    """Run the read-only A3.7 vector-versus-hybrid evaluation."""
    try:
        aggregate = run_hybrid_evaluation(
            knowledge_base_id=args.knowledge_base_id,
            split=args.split,
            chunk_index_path=args.chunk_index,
            dataset_path=args.dataset,
            materialized_path=args.materialized,
            mapping_path=args.kb_mapping,
            graph_manifest_path=args.graph_manifest,
            exclusions_path=args.graph_exclusions,
            output_dir=args.output,
        )
    except (HybridEvaluationError, GraphStoreError, ValidationError, ValueError) as error:
        print("hybrid_evaluation_status: failed", file=sys.stderr)
        print(f"error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("hybrid_evaluation_status: interrupted", file=sys.stderr)
        return 130
    print("hybrid_evaluation_status: complete")
    print(f"output: {args.output}")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
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


def _hybrid_retrieval_config(
    settings: Settings,
    args: argparse.Namespace,
) -> HybridRetrievalConfig:
    """Build hybrid bounds from settings and explicit CLI overrides."""

    values = {
        "rrf_k": settings.hybrid_rrf_k,
        "vector_candidate_limit": settings.hybrid_vector_candidate_limit,
        "vector_rerank_limit": settings.hybrid_vector_rerank_limit,
        "graph_result_limit": settings.hybrid_graph_result_limit,
        "result_limit": settings.hybrid_result_limit,
        "failure_mode": settings.hybrid_failure_mode,
    }
    for argument, setting in (
        ("rrf_k", "rrf_k"),
        ("vector_candidate_limit", "vector_candidate_limit"),
        ("vector_rerank_limit", "vector_rerank_limit"),
        ("graph_limit", "graph_result_limit"),
        ("limit", "result_limit"),
        ("failure_mode", "failure_mode"),
    ):
        value = getattr(args, argument, None)
        if value is not None:
            values[setting] = value
    return HybridRetrievalConfig(**values)


def _run_hybrid_search(args: argparse.Namespace) -> int:
    """Run the independent vector-plus-graph developer workflow."""

    qdrant_store = None
    graph_store = None
    try:
        settings = get_settings()
        qdrant_store = QdrantVectorStore(settings)
        graph_store = Neo4jGraphStore(settings)
        vector_retrieval = DenseRetrievalService(
            qdrant_store,
            QwenEmbeddingModel(settings),
        )
        reranker_settings = settings
        if reranker_settings.reranker_model_revision is None:
            reranker_settings = settings.model_copy(
                update={"reranker_model_revision": BGE_RERANKER_MODEL_REVISION}
            )
        reranking = RerankingService(
            create_local_reranker(reranker_settings, model_key="bge-reranker-v2-m3")
        )
        graph_retrieval = GraphRetrievalService(
            graph_store,
            config=_graph_retrieval_config(settings),
        )
        result = asyncio.run(
            HybridRetrievalService(
                vector_retrieval,
                reranking,
                graph_retrieval,
                config=_hybrid_retrieval_config(settings, args),
            ).search(
                args.query,
                args.knowledge_base_id,
                document_id=args.document_id,
            )
        )
        output = result.model_dump(mode="json")
        output["source_distribution"] = result.source_distribution
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0
    except (
        EmbeddingModelError,
        GraphRetrievalError,
        GraphStoreError,
        HybridRetrievalError,
        RerankerError,
        RetrievalError,
        ValidationError,
        VectorStoreError,
        ValueError,
    ) as error:
        print("hybrid_search_status: failed", file=sys.stderr)
        print(f"error: {error}", file=sys.stderr)
        return 1
    finally:
        if qdrant_store is not None:
            qdrant_store.close()
        if graph_store is not None:
            graph_store.close()


def _run_qdrant_attribution(
    store: QdrantVectorStore,
    settings: Settings,
    *,
    apply: bool,
) -> int:
    """Audit or safely repair Qdrant Knowledge Base payloads."""
    before_points = store.list_point_metadata()
    mappings = asyncio.run(
        load_authoritative_document_mappings(
            [point.document_id for point in before_points],
            settings,
        )
    )
    before = build_attribution_audit(before_points, mappings)
    output: dict[str, object] = {
        "mode": "apply" if apply else "audit",
        "before": before.as_dict(),
    }
    if not apply:
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0
    if not before.can_apply:
        output["message"] = (
            "repair refused: every Qdrant document must have one non-null, "
            "unambiguous PostgreSQL Knowledge Base mapping"
        )
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 1

    updated_points = apply_attribution_repair(store, before)
    after_points = store.list_point_metadata()
    after = build_attribution_audit(after_points, mappings)
    if point_identity_signature(before_points) != point_identity_signature(after_points):
        raise QdrantAttributionError("Qdrant repair changed point, document, or chunk identity")
    if after.point_count != before.point_count:
        raise QdrantAttributionError("Qdrant repair changed the collection point count")
    if not after.can_apply or after.points_with_null_kb or after.planned_update_points:
        raise QdrantAttributionError("Qdrant repair did not produce complete KB attribution")
    output["updated_points"] = updated_points
    output["after"] = after.as_dict()
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


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
            if args.qdrant_action == "audit-kb":
                return _run_qdrant_attribution(store, settings, apply=args.apply)
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
        QdrantAttributionError,
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


def _graph_retrieval_config(
    settings: Settings,
    args: argparse.Namespace | None = None,
) -> GraphRetrievalConfig:
    """Build the small graph-retrieval bounds from settings and CLI overrides."""

    values = {
        "max_seed_entities": settings.graph_retrieval_max_seed_entities,
        "max_hops": settings.graph_retrieval_max_hops,
        "max_neighbors": settings.graph_retrieval_max_neighbors,
        "max_relations": settings.graph_retrieval_max_relations,
        "max_evidence": settings.graph_retrieval_max_evidence,
        "max_entity_scan": settings.graph_retrieval_max_entity_scan,
        "lexical_threshold": settings.graph_retrieval_lexical_threshold,
    }
    if args is not None:
        for name in (
            "max_seed_entities",
            "max_hops",
            "max_neighbors",
            "max_relations",
            "max_evidence",
        ):
            value = getattr(args, name, None)
            if value is not None:
                values[name] = value
    return GraphRetrievalConfig(**values)


def _run_graph_search(args: argparse.Namespace) -> int:
    """Run the developer-only bounded graph retrieval workflow."""

    store = None
    try:
        settings = get_settings()
        store = Neo4jGraphStore(settings)
        result = GraphRetrievalService(
            store,
            config=_graph_retrieval_config(settings, args),
        ).search(args.query, args.knowledge_base_id)
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return 0
    except (GraphRetrievalError, GraphStoreError, ValidationError, ValueError) as error:
        print("graph_retrieval_status: failed", file=sys.stderr)
        print(f"error: {error}", file=sys.stderr)
        return 1
    finally:
        if store is not None:
            store.close()


def _run_graph_retrieval_sample(args: argparse.Namespace) -> int:
    """Run the bounded read-only graph retrieval review sample."""

    store = None
    try:
        settings = get_settings()
        store = Neo4jGraphStore(settings)
        readiness = store.readiness()
        if readiness.status != "ready":
            raise GraphStoreError("Neo4j is not ready; run `knowledgescope neo4j check` first")
        summary = run_graph_retrieval_review_sample(
            args.input,
            settings=settings,
            output_dir=args.output,
            knowledge_base_id=args.knowledge_base_id,
            store=store,
        )
        print("graph_retrieval_status: complete")
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0
    except (
        GraphRetrievalError,
        GraphStoreError,
        RetrievalEvalError,
        ValidationError,
        ValueError,
    ) as error:
        print("graph_retrieval_status: failed", file=sys.stderr)
        print(f"error: {error}", file=sys.stderr)
        return 1
    finally:
        if store is not None:
            store.close()


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


async def _registered_document_ids(settings: Settings, knowledge_base_id: UUID) -> set[UUID]:
    """Read registered document IDs for one Knowledge Base without loading documents."""

    engine = create_database_engine(settings)
    try:
        try:
            session_factory = create_session_factory(engine)
            async with session_factory() as session:
                result = await session.execute(
                    select(Document.id).where(
                        Document.knowledge_base_id == knowledge_base_id,
                        Document.status == DOCUMENT_STATUS_REGISTERED,
                    )
                )
                return set(result.scalars().all())
        except Exception as error:
            raise GraphCorpusBuildError(
                "registered document audit could not read PostgreSQL"
            ) from error
    finally:
        await engine.dispose()


async def _run_graph_corpus_audit_async(args: argparse.Namespace) -> dict[str, object]:
    """Audit corpus files and best-effort current Neo4j coverage."""

    settings = get_settings()
    registered = await _registered_document_ids(settings, args.knowledge_base_id)
    store = Neo4jGraphStore(settings)
    graph_snapshot = None
    graph_available = False
    try:
        readiness = await asyncio.to_thread(store.readiness)
        if readiness.status == "ready":
            try:
                graph_snapshot = await asyncio.to_thread(
                    read_graph_coverage,
                    store,
                    args.knowledge_base_id,
                )
                graph_available = True
            except GraphStoreError:
                graph_snapshot = None
        audit = audit_corpus_files(
            args.knowledge_base_id,
            corpus_manifest=args.corpus_manifest,
            chunk_index=args.chunk_index,
            eval_dataset=args.eval_dataset,
            registered_document_ids=registered,
            graph_snapshot=graph_snapshot,
            expected_snapshot=load_corpus_input_snapshot(args.input_snapshot),
        )
        if not graph_available:
            audit = audit.model_copy(update={"graph_status": "unavailable"})
        return audit.model_dump(mode="json")
    finally:
        store.close()


def _run_graph_corpus_audit(args: argparse.Namespace) -> int:
    """Run the read-only A3.6 corpus and graph coverage audit."""

    try:
        output = asyncio.run(_run_graph_corpus_audit_async(args))
    except (GraphCorpusBuildError, GraphStoreError, ValidationError, ValueError) as error:
        print("graph_corpus_audit_status: failed", file=sys.stderr)
        print(f"error: {error}", file=sys.stderr)
        return 1
    print("graph_corpus_audit_status: complete")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def _run_graph_corpus_estimate(args: argparse.Namespace) -> int:
    """Print a transparent estimate based only on existing A3.2 sample summaries."""

    try:
        target_chunks = args.target_chunks
        if target_chunks is None:
            target_chunks = load_corpus_input_snapshot(args.input_snapshot).chunk_count
        estimate = estimate_full_run(
            args.summary,
            target_chunks=target_chunks,
            settings=get_settings(),
            max_concurrency=args.max_concurrency,
        )
    except (GraphCorpusBuildError, OSError, ValidationError, ValueError) as error:
        print("graph_corpus_estimate_status: failed", file=sys.stderr)
        print(f"error: {error}", file=sys.stderr)
        return 1
    print("graph_corpus_estimate_status: complete")
    print(json.dumps(estimate.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


async def _run_graph_corpus_build_async(args: argparse.Namespace) -> dict[str, object]:
    """Run one explicit, persistent A3.6 corpus build and close resources."""

    if not args.persist:
        raise GraphCorpusBuildError(
            "graph-corpus-build requires --persist; use graph-corpus-audit for read-only checks"
        )
    settings = get_settings()
    if settings.llm_api_key is None or not settings.llm_api_key.get_secret_value().strip():
        raise ValueError("KNOWLEDGE_SCOPE_LLM_API_KEY is required for graph-corpus-build")
    registered = await _registered_document_ids(settings, args.knowledge_base_id)
    selected = select_document_ids(
        args.corpus_manifest,
        full=args.full,
        sample_per_subject=args.sample_per_subject,
        sample_offset=args.sample_offset,
        max_documents=args.max_documents,
    )
    documents_for_snapshot = iter_corpus_documents(
        args.corpus_manifest,
        args.chunk_index,
        selected_document_ids=selected,
        registered_document_ids=registered,
    )
    actual_snapshot = build_corpus_input_snapshot(
        documents_for_snapshot,
        knowledge_base_id=args.knowledge_base_id,
        corpus_manifest=args.corpus_manifest,
        chunk_index=args.chunk_index,
    )
    validate_corpus_input_snapshot(
        actual_snapshot,
        load_corpus_input_snapshot(args.input_snapshot),
        require_full=args.full and args.max_documents is None,
    )
    documents = iter_corpus_documents(
        args.corpus_manifest,
        args.chunk_index,
        selected_document_ids=selected,
        registered_document_ids=registered,
    )
    engine = None
    provider = None
    store = None
    try:
        engine = create_database_engine(settings)
        provider = create_llm_provider(settings)
        store = Neo4jGraphStore(settings)
        await asyncio.to_thread(store.ensure_schema)
        gateway = LLMGateway(
            provider,
            DatabaseUsageRecorder(create_session_factory(engine)),
            settings,
        )
        runner = GraphCorpusRunner(
            ExtractionService(gateway, settings),
            settings,
            knowledge_base_id=args.knowledge_base_id,
            output_dir=args.output,
            store=store,
            linking_gateway=gateway if args.adjudicate else None,
            max_concurrency=args.max_concurrency,
            link_batch_documents=args.link_batch_documents,
            resume=args.resume,
            retry_failed=args.retry_failed,
            reprocess=args.reprocess,
            input_snapshot=actual_snapshot,
        )
        manifest = await runner.run(documents)
        return manifest.model_dump(mode="json")
    finally:
        if store is not None:
            store.close()
        if provider is not None:
            await provider.aclose()
        if engine is not None:
            await engine.dispose()


def _run_graph_corpus_build(args: argparse.Namespace) -> int:
    """Run the explicit resumable A3.6 graph-corpus build."""

    try:
        output = asyncio.run(_run_graph_corpus_build_async(args))
    except (
        ExtractionError,
        GraphCorpusBuildError,
        GraphStoreError,
        LLMError,
        ValidationError,
        ValueError,
    ) as error:
        print("graph_corpus_build_status: failed", file=sys.stderr)
        print(f"error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("graph_corpus_build_status: interrupted", file=sys.stderr)
        return 130
    print("graph_corpus_build_status: complete")
    print(json.dumps(output, ensure_ascii=False, indent=2))
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
    if args.command == "corpus":
        return _run_corpus(args)
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
    if args.command == "hybrid-evaluation":
        return _run_hybrid_evaluation(args)
    if args.command == "qdrant":
        return _run_qdrant(args)
    if args.command == "neo4j":
        return _run_neo4j(args)
    if args.command == "graph-extraction-sample":
        return _run_graph_extraction_sample(args)
    if args.command == "entity-linking-sample":
        return _run_entity_linking_sample(args)
    if args.command == "graph-search":
        return _run_graph_search(args)
    if args.command == "graph-retrieval-sample":
        return _run_graph_retrieval_sample(args)
    if args.command == "rerank-search":
        return _run_rerank_search(args)
    if args.command == "hybrid-search":
        return _run_hybrid_search(args)
    if args.command == "graph-corpus-audit":
        return _run_graph_corpus_audit(args)
    if args.command == "graph-corpus-estimate":
        return _run_graph_corpus_estimate(args)
    if args.command == "graph-corpus-build":
        return _run_graph_corpus_build(args)

    parser.print_help()
    return 0
