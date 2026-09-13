"""Safe credential-resolution boundaries for external ChatBI datasources."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol

from sqlalchemy.engine import make_url

from .errors import ChatBIError, ChatBIErrorCategory
from .schemas import validate_connection_ref


class SecretReferenceResolver(Protocol):
    """Explicit backend contract for resolving a ``secret:NAME`` reference."""

    def resolve(self, name: str) -> str:
        """Return a connection URL for one secret name without exposing it to callers."""


class CredentialResolver(Protocol):
    """Provider-independent credential resolution contract."""

    def resolve(self, connection_ref: str) -> ResolvedDatabaseCredentials:
        """Resolve an opaque connection reference into in-memory credentials."""


@dataclass(frozen=True, repr=False)
class ResolvedDatabaseCredentials:
    """In-memory credentials; the DSN is deliberately hidden from representations."""

    _connection_url: str = field(repr=False)

    @property
    def connection_url(self) -> str:
        """Return the private URL for the adapter that owns this object."""
        return self._connection_url

    def __repr__(self) -> str:
        return "ResolvedDatabaseCredentials(<redacted>)"


def _validate_postgresql_url(raw_url: object) -> ResolvedDatabaseCredentials:
    if not isinstance(raw_url, str) or not raw_url.strip():
        raise ChatBIError(
            ChatBIErrorCategory.CREDENTIAL_RESOLUTION_FAILED,
            "the configured database credential is missing",
        )
    try:
        parsed = make_url(raw_url.strip())
    except Exception:
        raise ChatBIError(
            ChatBIErrorCategory.CREDENTIAL_RESOLUTION_FAILED,
            "the configured database credential is not a valid PostgreSQL URL",
        ) from None
    if parsed.drivername != "postgresql" and not parsed.drivername.startswith("postgresql+"):
        raise ChatBIError(
            ChatBIErrorCategory.CREDENTIAL_RESOLUTION_FAILED,
            "the configured database credential must use PostgreSQL",
        )
    # asyncpg consumes the plain PostgreSQL URL.  Rendering happens only in
    # memory and is never placed in a model, log message, or exception.
    return ResolvedDatabaseCredentials(
        parsed.set(drivername="postgresql").render_as_string(hide_password=False)
    )


class EnvironmentCredentialResolver:
    """Resolve ``env:NAME`` and optionally delegate ``secret:NAME`` to a backend."""

    def __init__(self, *, secret_resolver: SecretReferenceResolver | None = None) -> None:
        self._secret_resolver = secret_resolver

    def resolve(self, connection_ref: str) -> ResolvedDatabaseCredentials:
        try:
            normalized = validate_connection_ref(connection_ref)
        except ValueError:
            raise ChatBIError(
                ChatBIErrorCategory.CREDENTIAL_RESOLUTION_FAILED,
                "the datasource connection reference is invalid",
            ) from None

        kind, name = normalized.split(":", maxsplit=1)
        if kind == "env":
            raw_url = os.environ.get(name)
        elif self._secret_resolver is not None:
            try:
                raw_url = self._secret_resolver.resolve(name)
            except Exception:
                raise ChatBIError(
                    ChatBIErrorCategory.CREDENTIAL_RESOLUTION_FAILED,
                    "the configured secret credential could not be resolved",
                ) from None
        else:
            raise ChatBIError(
                ChatBIErrorCategory.CREDENTIAL_RESOLUTION_FAILED,
                "secret credential references require a configured secret backend",
            )
        return _validate_postgresql_url(raw_url)


__all__ = [
    "CredentialResolver",
    "EnvironmentCredentialResolver",
    "ResolvedDatabaseCredentials",
    "SecretReferenceResolver",
]
