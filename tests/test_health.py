from pathlib import Path
from platform import python_version

from knowledge_scope.shared.config import Settings
from knowledge_scope.shared.health import build_health_report


def test_health_report_contains_validated_runtime_details() -> None:
    settings = Settings(_env_file=None, environment="test", data_dir=Path("var/data"))

    report = build_health_report(settings)

    assert report.config_status == "ok"
    assert report.environment == "test"
    assert report.python_version == python_version()
    assert report.data_dir == "var/data"
    assert report.as_dict()["project_name"] == "KnowledgeScope"
