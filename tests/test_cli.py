import pytest

from knowledge_scope.cli import main


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
