"""Command-line entry point for the current KnowledgeScope foundation."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from uuid import UUID

from pydantic import ValidationError

from knowledge_scope import __version__
from knowledge_scope.evaluation.parsing_benchmark import (
    RAW_RETENTION_VALUES,
    BenchmarkConfig,
    BenchmarkError,
    inventory_corpus,
    run_benchmark,
)
from knowledge_scope.parsing.service import DocumentParseError, parse_document_by_id
from knowledge_scope.shared import build_health_report, get_settings


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
    parse_document = subparsers.add_parser(
        "parse-document",
        help="parse one uploaded PDF into canonical artifacts with MinerU",
    )
    parse_document.add_argument("document_id", type=UUID)

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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "health":
        return _run_health()
    if args.command == "parse-document":
        return _run_parse_document(args.document_id)
    if args.command == "benchmark-parsing":
        return _run_benchmark(args)

    parser.print_help()
    return 0
