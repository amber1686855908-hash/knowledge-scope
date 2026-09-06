"""Subprocess boundary for the externally managed MinerU runtime."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

MINERU_BACKEND = "pipeline"
MINERU_OUTPUT_MODE = "auto"
MINERU_VERSION_TIMEOUT_SECONDS = 30


class MineruRunnerError(RuntimeError):
    """Raised when the external MinerU process cannot produce an output."""


@dataclass(frozen=True, slots=True)
class MineruRunResult:
    """Captured facts from one successful MinerU invocation."""

    output_dir: Path
    version: str
    backend: str
    elapsed_seconds: float
    stdout: str
    stderr: str


def _validated_command(command: str) -> str:
    normalized = command.strip()
    if not normalized or "\x00" in normalized:
        raise MineruRunnerError("KNOWLEDGE_SCOPE_MINERU_COMMAND must be a valid executable")
    return normalized


def _runtime_environment(command: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment["MINERU_MODEL_SOURCE"] = "local"

    if "MINERU_TOOLS_CONFIG_JSON" not in environment:
        executable = shutil.which(command)
        if executable is not None:
            config_path = Path(executable).resolve().parent.parent / "mineru.json"
            if config_path.is_file():
                environment["MINERU_TOOLS_CONFIG_JSON"] = str(config_path)
    return environment


def _run_process(
    arguments: list[str],
    *,
    timeout_seconds: int | float,
    environment: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            arguments,
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
            env=environment,
            shell=False,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as error:
        raise MineruRunnerError("MinerU executable was not found") from error
    except subprocess.TimeoutExpired as error:
        raise MineruRunnerError(f"MinerU timed out after {timeout_seconds} seconds") from error
    except OSError as error:
        raise MineruRunnerError("MinerU process could not be started") from error


def get_mineru_version(command: str) -> str:
    """Read the version from the configured external executable."""
    normalized_command = _validated_command(command)
    completed = _run_process(
        [normalized_command, "--version"],
        timeout_seconds=MINERU_VERSION_TIMEOUT_SECONDS,
        environment=_runtime_environment(normalized_command),
    )
    if completed.returncode != 0:
        raise MineruRunnerError("MinerU version check failed")

    first_line = next((line.strip() for line in completed.stdout.splitlines() if line.strip()), "")
    match = re.search(r"\bversion\s+([^\s,]+)", first_line, flags=re.IGNORECASE)
    version = match.group(1) if match else first_line
    if not version:
        raise MineruRunnerError("MinerU version check returned no version")
    return version[:128]


def run_mineru(
    source_pdf: Path,
    output_dir: Path,
    command: str,
    *,
    timeout_seconds: int,
) -> MineruRunResult:
    """Run the fixed pipeline backend against an application-controlled PDF."""
    normalized_command = _validated_command(command)
    source_path = Path(source_pdf)
    target_dir = Path(output_dir)
    if not source_path.is_file():
        raise MineruRunnerError("the application-controlled source PDF does not exist")
    if not target_dir.is_dir():
        raise MineruRunnerError("the application-controlled MinerU output directory is invalid")
    if any(target_dir.iterdir()):
        raise MineruRunnerError("the application-controlled MinerU output directory is not empty")
    if timeout_seconds < 1:
        raise MineruRunnerError("MinerU timeout must be at least one second")

    environment = _runtime_environment(normalized_command)
    version = get_mineru_version(normalized_command)
    arguments = [
        normalized_command,
        "-p",
        str(source_path),
        "-o",
        str(target_dir),
        "-b",
        MINERU_BACKEND,
        "-m",
        MINERU_OUTPUT_MODE,
    ]
    started_at = perf_counter()
    completed = _run_process(
        arguments,
        timeout_seconds=timeout_seconds,
        environment=environment,
    )
    elapsed_seconds = perf_counter() - started_at
    if completed.returncode != 0:
        raise MineruRunnerError(f"MinerU exited with code {completed.returncode}")

    return MineruRunResult(
        output_dir=target_dir,
        version=version,
        backend=MINERU_BACKEND,
        elapsed_seconds=elapsed_seconds,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def find_content_list(output_dir: Path) -> Path:
    """Find the stable v1 content-list artifact in one MinerU output tree."""
    candidates = sorted(Path(output_dir).rglob("*_content_list.json"))
    if not candidates:
        raise MineruRunnerError("MinerU did not emit a content_list JSON artifact")
    if len(candidates) > 1:
        raise MineruRunnerError("MinerU emitted more than one content_list JSON artifact")
    return candidates[0]


__all__ = [
    "MINERU_BACKEND",
    "MineruRunResult",
    "MineruRunnerError",
    "find_content_list",
    "get_mineru_version",
    "run_mineru",
]
