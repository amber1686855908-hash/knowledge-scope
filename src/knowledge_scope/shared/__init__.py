"""Shared application infrastructure used by the current package."""

from knowledge_scope.shared.config import Settings, get_settings
from knowledge_scope.shared.health import HealthReport, build_health_report

__all__ = ["HealthReport", "Settings", "build_health_report", "get_settings"]
