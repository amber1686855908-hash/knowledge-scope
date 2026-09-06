import subprocess
from pathlib import Path

import pytest

from knowledge_scope.parsing.mineru_runner import (
    MineruRunnerError,
    find_content_list,
    run_mineru,
)


def test_runner_uses_argument_sequence_and_local_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source_path = tmp_path / "source.pdf"
    output_dir = tmp_path / "output"
    source_path.write_bytes(b"%PDF-test")
    output_dir.mkdir()
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((arguments, kwargs))
        if arguments[-1] == "--version":
            return subprocess.CompletedProcess(arguments, 0, "mineru, version 3.4.5\n", "")
        return subprocess.CompletedProcess(arguments, 0, "stdout", "stderr")

    monkeypatch.setattr("knowledge_scope.parsing.mineru_runner.subprocess.run", fake_run)

    result = run_mineru(source_path, output_dir, "mineru", timeout_seconds=600)

    assert calls[0][0] == ["mineru", "--version"]
    assert calls[1][0] == [
        "mineru",
        "-p",
        str(source_path),
        "-o",
        str(output_dir),
        "-b",
        "pipeline",
        "-m",
        "auto",
    ]
    assert all(call[1]["shell"] is False for call in calls)
    assert all(call[1]["capture_output"] is True for call in calls)
    assert calls[1][1]["env"]["MINERU_MODEL_SOURCE"] == "local"
    assert result.version == "3.4.5"
    assert result.backend == "pipeline"
    assert result.stdout == "stdout"


def test_runner_reports_non_zero_exit_without_exposing_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.pdf"
    output_dir = tmp_path / "output"
    source_path.write_bytes(b"%PDF-test")
    output_dir.mkdir()

    def fake_run(arguments: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if arguments[-1] == "--version":
            return subprocess.CompletedProcess(arguments, 0, "mineru, version 3.4.5\n", "")
        return subprocess.CompletedProcess(arguments, 7, "", f"failed at {source_path}")

    monkeypatch.setattr("knowledge_scope.parsing.mineru_runner.subprocess.run", fake_run)

    with pytest.raises(MineruRunnerError, match="code 7") as error:
        run_mineru(source_path, output_dir, "mineru", timeout_seconds=600)

    assert str(source_path) not in str(error.value)


def test_runner_converts_timeout_to_controlled_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.pdf"
    output_dir = tmp_path / "output"
    source_path.write_bytes(b"%PDF-test")
    output_dir.mkdir()

    def fake_run(arguments: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if arguments[-1] == "--version":
            return subprocess.CompletedProcess(arguments, 0, "mineru, version 3.4.5\n", "")
        raise subprocess.TimeoutExpired(arguments, 1)

    monkeypatch.setattr("knowledge_scope.parsing.mineru_runner.subprocess.run", fake_run)

    with pytest.raises(MineruRunnerError, match="timed out"):
        run_mineru(source_path, output_dir, "mineru", timeout_seconds=1)


def test_runner_rejects_stale_output_directory(tmp_path: Path) -> None:
    source_path = tmp_path / "source.pdf"
    output_dir = tmp_path / "output"
    source_path.write_bytes(b"%PDF-test")
    output_dir.mkdir()
    (output_dir / "stale.json").write_text("{}", encoding="utf-8")

    with pytest.raises(MineruRunnerError, match="not empty"):
        run_mineru(source_path, output_dir, "mineru", timeout_seconds=600)


def test_find_content_list_ignores_experimental_v2_artifact(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    stable = output_dir / "document_content_list.json"
    experimental = output_dir / "document_content_list_v2.json"
    stable.write_text("[]", encoding="utf-8")
    experimental.write_text("[]", encoding="utf-8")

    assert find_content_list(output_dir) == stable
