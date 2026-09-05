"""Command-line entry point for the current KnowledgeScope foundation."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from pydantic import ValidationError

from knowledge_scope import __version__
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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "health":
        return _run_health()

    parser.print_help()
    return 0
