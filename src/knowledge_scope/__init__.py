"""KnowledgeScope application package."""

from importlib.metadata import PackageNotFoundError, version
from typing import Final

_FALLBACK_VERSION: Final = "0.1.0"

try:
    __version__ = version("knowledgescope")
except PackageNotFoundError:
    __version__ = _FALLBACK_VERSION

__all__ = ["__version__"]
