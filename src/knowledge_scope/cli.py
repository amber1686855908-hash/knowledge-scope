"""Command-line entry point for the current KnowledgeScope foundation."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from uuid import UUID

from pydantic import ValidationError

from knowledge_scope import __version__
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
    return parser


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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "health":
        return _run_health()
    if args.command == "parse-document":
        return _run_parse_document(args.document_id)

    parser.print_help()
    return 0
