"""Health reporting for the currently implemented project foundation."""

from __future__ import annotations

from dataclasses import dataclass
from platform import python_version

from knowledge_scope import __version__
from knowledge_scope.shared.config import Settings


@dataclass(frozen=True, slots=True)
class HealthReport:
    """Non-sensitive project and configuration health details."""

    project_name: str
    version: str
    python_version: str
    config_status: str
    environment: str
    log_level: str
    data_dir: str

    def as_dict(self) -> dict[str, str]:
        """Return the report in a stable, CLI-friendly shape."""
        return {
            "project_name": self.project_name,
            "version": self.version,
            "python_version": self.python_version,
            "config_status": self.config_status,
            "environment": self.environment,
            "log_level": self.log_level,
            "data_dir": self.data_dir,
        }


def build_health_report(settings: Settings) -> HealthReport:
    """Build a report from already validated settings."""
    return HealthReport(
        project_name=settings.project_name,
        version=__version__,
        python_version=python_version(),
        config_status="ok",
        environment=settings.environment,
        log_level=settings.log_level,
        data_dir=str(settings.data_dir),
    )
