"""Conservative PostgreSQL AST validation for generated read-only SQL."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final

from pydantic import ValidationError
from sqlglot import exp, parse
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.qualify import qualify
from sqlglot.schema import MappingSchema

from .errors import ChatBIError, ChatBIErrorCategory
from .nl2sql_models import SQL_VALIDATION_VERSION, SQLCandidate, ValidatedSQL
from .policy import QueryPolicy
from .schema_models import (
    SchemaContextBudgetError,
    SchemaObjectKind,
    SchemaSnapshot,
    SemanticSchemaContext,
    build_semantic_schema_context,
)

MAX_SET_OPERATION_DEPTH: Final = 8
MAX_SQL_CHARS: Final = 100_000
MAX_AST_NODES: Final = 4_096
_POSTGRES_SYSTEM_SCHEMA_NAMES: Final = frozenset(
    {
        "information_schema",
        "pg_catalog",
        "pg_temp",
        "pg_toast",
    }
)

# Only the operator forms needed by ordinary filtering and arithmetic are
# admitted.  In particular, PostgreSQL's explicit/custom operator syntax and
# distance operators are never treated as harmless binary expressions.
_SAFE_BINARY_OPERATOR_NAMES: Final = frozenset(
    {
        "Add",
        "And",
        "Between",
        "DPipe",
        "EQ",
        "GT",
        "GTE",
        "ILike",
        "In",
        "Is",
        "Like",
        "LT",
        "LTE",
        "Mod",
        "Mul",
        "NEQ",
        "NullSafeEQ",
        "NullSafeNEQ",
        "Or",
        "Sub",
    }
)
_SAFE_UNARY_OPERATOR_NAMES: Final = frozenset({"Neg", "Not"})
_UNSUPPORTED_AST_NODE_NAMES: Final = frozenset(
    {
        "Pivot",
        "Qualify",
        "TableSample",
        "WithTableHint",
    }
)
_SAFE_CAST_TYPES: Final = frozenset(
    {
        "bigint",
        "boolean",
        "bytea",
        "char",
        "date",
        "decimal",
        "double precision",
        "int",
        "interval",
        "json",
        "jsonb",
        "real",
        "smallint",
        "text",
        "time",
        "timestamp",
        "timestamptz",
        "uuid",
        "varchar",
    }
)

# This list is intentionally an allow-list.  Unknown or qualified functions
# are rejected so a user-defined/system function cannot become an execution
# capability by appearing in generated SQL.
SAFE_SQL_FUNCTIONS: Final = frozenset(
    {
        "abs",
        "array_agg",
        "array_length",
        "avg",
        "bool_and",
        "bool_or",
        "btrim",
        "cast",
        "ceil",
        "ceiling",
        "char_length",
        "coalesce",
        "concat",
        "concat_ws",
        "count",
        "current_date",
        "current_timestamp",
        "date_part",
        "date_trunc",
        "dense_rank",
        "every",
        "extract",
        "floor",
        "greatest",
        "json_agg",
        "json_build_object",
        "jsonb_agg",
        "jsonb_build_object",
        "least",
        "length",
        "localtime",
        "localtimestamp",
        "lower",
        "ltrim",
        "max",
        "min",
        "mod",
        "nullif",
        "now",
        "position",
        "power",
        "regexp_replace",
        "replace",
        "right",
        "round",
        "row_number",
        "rtrim",
        "split_part",
        "sqrt",
        "string_agg",
        "sum",
        "substring",
        "to_char",
        "to_date",
        "to_timestamp",
        "trim",
        "upper",
        "first_value",
        "last_value",
        "lag",
        "lead",
        "nth_value",
        "rank",
    }
)

_FORBIDDEN_NODE_NAMES: Final = frozenset(
    {
        "Alter",
        "Analyze",
        "Attach",
        "Call",
        "Cache",
        "Command",
        "Commit",
        "Copy",
        "Create",
        "Deallocate",
        "Declare",
        "Delete",
        "Describe",
        "Detach",
        "Do",
        "Drop",
        "Execute",
        "Grant",
        "Insert",
        "Listen",
        "Load",
        "Merge",
        "Notify",
        "Prepare",
        "Pragma",
        "Refresh",
        "Release",
        "Reset",
        "Revoke",
        "Rollback",
        "Savepoint",
        "Set",
        "Show",
        "Transaction",
        "TruncateTable",
        "Update",
        "Use",
        "Vacuum",
    }
)


@dataclass(frozen=True)
class _Identifier:
    value: str
    quoted: bool

    def matches(self, actual: str) -> bool:
        # PostgreSQL folds an unquoted identifier to lower case; a quoted
        # identifier must match the catalog spelling exactly.
        return self.value == actual if self.quoted else self.value.lower() == actual

    def equals(self, other: _Identifier) -> bool:
        left = self.value if self.quoted else self.value.lower()
        right = other.value if other.quoted else other.value.lower()
        return left == right


@dataclass(frozen=True)
class _ResolvedRelation:
    schema_name: str
    relation_name: str
    kind: SchemaObjectKind

    @property
    def label(self) -> str:
        return f"{self.schema_name}.{self.relation_name}"


def policy_fingerprint(policy: QueryPolicy) -> str:
    """Hash the complete normalized policy with deterministic JSON settings."""
    serialized = json.dumps(
        policy.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _name_part(value: object) -> _Identifier:
    if isinstance(value, exp.Identifier):
        raw = value.this
        return _Identifier(str(raw), bool(value.args.get("quoted")))
    if isinstance(value, str):
        return _Identifier(value, False)
    return _Identifier(str(value), False)


def _safe_error(category: ChatBIErrorCategory, message: str) -> ChatBIError:
    return ChatBIError(category, message)


def _parse_statement(sql: str) -> exp.Expression:
    if len(sql) > MAX_SQL_CHARS:
        raise _safe_error(
            ChatBIErrorCategory.UNSAFE_QUERY,
            "generated SQL exceeds the maximum statement length",
        )
    try:
        statements = parse(sql, read="postgres")
    except (RecursionError, SqlglotError, ValueError):
        raise _safe_error(
            ChatBIErrorCategory.SQL_PARSE_ERROR,
            "generated SQL could not be parsed as PostgreSQL",
        ) from None
    if len(statements) != 1 or statements[0] is None:
        raise _safe_error(
            ChatBIErrorCategory.UNSAFE_QUERY,
            "exactly one SQL statement is required",
        )
    return statements[0]


def _validate_ast_complexity(statement: exp.Expression) -> None:
    """Bound AST traversal so pathological nesting fails as a safe policy error."""
    try:
        for count, _node in enumerate(statement.walk(), start=1):
            if count > MAX_AST_NODES:
                raise _safe_error(
                    ChatBIErrorCategory.UNSAFE_QUERY,
                    "query AST exceeds the safety complexity bound",
                )
    except RecursionError:
        raise _safe_error(
            ChatBIErrorCategory.UNSAFE_QUERY,
            "query AST exceeds the safety complexity bound",
        ) from None


def _validate_read_only_shape(statement: exp.Expression) -> None:
    if not isinstance(statement, (exp.Select, exp.SetOperation)):
        raise _safe_error(
            ChatBIErrorCategory.UNSAFE_QUERY,
            "only read-only SELECT statements are allowed",
        )

    def validate_set_operation(node: exp.Expression, depth: int) -> None:
        if not isinstance(node, exp.SetOperation):
            if not isinstance(node, exp.Select):
                raise _safe_error(
                    ChatBIErrorCategory.UNSAFE_QUERY,
                    "set operations may contain only SELECT statements",
                )
            return
        if depth > MAX_SET_OPERATION_DEPTH:
            raise _safe_error(
                ChatBIErrorCategory.UNSAFE_QUERY,
                "set-operation nesting exceeds the safety bound",
            )
        validate_set_operation(node.left, depth + 1)
        validate_set_operation(node.right, depth + 1)

    validate_set_operation(statement, 0)
    for node in statement.walk():
        if type(node).__name__ in _FORBIDDEN_NODE_NAMES:
            raise _safe_error(
                ChatBIErrorCategory.UNSAFE_QUERY,
                "the generated SQL contains a forbidden statement or command",
            )
        if isinstance(node, exp.Select):
            with_clause = node.args.get("with_")
            if isinstance(with_clause, exp.With) and with_clause.args.get("recursive"):
                raise _safe_error(
                    ChatBIErrorCategory.UNSAFE_QUERY,
                    "recursive CTEs are not allowed",
                )
            if node.args.get("into") is not None:
                raise _safe_error(
                    ChatBIErrorCategory.UNSAFE_QUERY,
                    "SELECT INTO is not allowed",
                )
            if node.args.get("locks"):
                raise _safe_error(
                    ChatBIErrorCategory.UNSAFE_QUERY,
                    "row-locking clauses are not allowed",
                )
        if type(node).__name__ in _UNSUPPORTED_AST_NODE_NAMES:
            raise _safe_error(
                ChatBIErrorCategory.UNSAFE_QUERY,
                "the query contains an unsupported PostgreSQL construct",
            )


def _cte_names_in_scope(table: exp.Table) -> tuple[_Identifier, ...]:
    """Return CTE names visible at ``table`` under PostgreSQL lexical rules."""
    names: list[_Identifier] = []

    def add_visible(with_clause: exp.With) -> None:
        containing_cte: exp.CTE | None = None
        cursor: exp.Expression | None = table
        while cursor is not None and cursor is not with_clause:
            if isinstance(cursor, exp.CTE) and cursor.parent is with_clause:
                containing_cte = cursor
                break
            cursor = cursor.parent
        expressions = list(with_clause.expressions)
        visible = expressions
        if containing_cte is not None:
            containing_index = next(
                index
                for index, expression in enumerate(expressions)
                if expression is containing_cte
            )
            visible = expressions[:containing_index]
        for cte in visible:
            alias = cte.args.get("alias")
            if isinstance(alias, exp.TableAlias) and alias.this is not None:
                names.append(_name_part(alias.this))

    node: exp.Expression | None = table
    while node is not None:
        if isinstance(node, exp.With):
            add_visible(node)
        elif isinstance(node, (exp.Select, exp.SetOperation)):
            with_clause = node.args.get("with_")
            if isinstance(with_clause, exp.With):
                add_visible(with_clause)
        node = node.parent
    return tuple(names)


def _allowed_schema(name: _Identifier, policy: QueryPolicy) -> bool:
    normalized = name.value.lower()
    if normalized in _POSTGRES_SYSTEM_SCHEMA_NAMES or normalized.startswith(
        ("pg_temp_", "pg_toast_")
    ):
        return False
    if any(name.matches(schema) for schema in policy.denied_schemas):
        return False
    return any(name.matches(schema) for schema in policy.allowed_schemas)


def _resolve_table(
    table: exp.Table,
    snapshot: SchemaSnapshot,
    context: SemanticSchemaContext,
    policy: QueryPolicy,
) -> _ResolvedRelation | None:
    table_name = _name_part(table.this)
    schema_part = table.args.get("db")
    catalog_part = table.args.get("catalog")
    if catalog_part:
        raise _safe_error(
            ChatBIErrorCategory.POLICY_VIOLATION,
            "cross-database object references are not allowed",
        )
    if schema_part is None and any(name.equals(table_name) for name in _cte_names_in_scope(table)):
        return None

    schema_name = _name_part(schema_part) if schema_part is not None else None
    if schema_name is not None and not _allowed_schema(schema_name, policy):
        raise _safe_error(
            ChatBIErrorCategory.POLICY_VIOLATION,
            "the query references a schema outside the query policy",
        )

    candidates = [
        relation
        for relation in snapshot.relations
        if (schema_name is None or schema_name.matches(relation.schema_name))
        and table_name.matches(relation.name)
        and _allowed_schema(_Identifier(relation.schema_name, False), policy)
    ]
    if not candidates:
        raise _safe_error(
            ChatBIErrorCategory.UNKNOWN_TABLE,
            "the query references an unknown or unavailable table",
        )
    if len(candidates) > 1:
        raise _safe_error(
            ChatBIErrorCategory.UNKNOWN_TABLE,
            "an unqualified table name is ambiguous under the query policy",
        )
    relation = candidates[0]
    label = f"{relation.schema_name}.{relation.name}"
    if label not in context.included_relations:
        raise _safe_error(
            ChatBIErrorCategory.POLICY_VIOLATION,
            "the query references a relation omitted from the approved context",
        )
    if relation.kind is SchemaObjectKind.VIEW:
        raise _safe_error(
            ChatBIErrorCategory.POLICY_VIOLATION,
            "views are not allowed before dependency analysis is implemented",
        )
    return _ResolvedRelation(
        schema_name=relation.schema_name,
        relation_name=relation.name,
        kind=relation.kind,
    )


def _resolve_tables(
    statement: exp.Expression,
    snapshot: SchemaSnapshot,
    context: SemanticSchemaContext,
    policy: QueryPolicy,
) -> tuple[_ResolvedRelation, ...]:
    resolved: dict[str, _ResolvedRelation] = {}
    for table in statement.find_all(exp.Table):
        relation = _resolve_table(table, snapshot, context, policy)
        if relation is not None:
            resolved[relation.label] = relation
    return tuple(sorted(resolved.values(), key=lambda item: item.label))


def _qualify_physical_tables(
    statement: exp.Expression,
    snapshot: SchemaSnapshot,
    context: SemanticSchemaContext,
    policy: QueryPolicy,
) -> None:
    """Rewrite approved physical tables to quoted schema-qualified references."""
    for table in statement.find_all(exp.Table):
        relation = _resolve_table(table, snapshot, context, policy)
        if relation is None:
            continue
        table.set("this", exp.to_identifier(relation.relation_name, quoted=True))
        table.set("db", exp.to_identifier(relation.schema_name, quoted=True))


def _source_expressions(statement: exp.Expression) -> tuple[exp.Expression, ...]:
    sources: list[exp.Expression] = []
    for select in statement.find_all(exp.Select):
        from_clause = select.args.get("from_")
        if from_clause is not None and from_clause.this is not None:
            sources.append(from_clause.this)
        if from_clause is not None:
            sources.extend(from_clause.expressions)
        for join in select.args.get("joins") or ():
            if join.this is not None:
                sources.append(join.this)
    return tuple(sources)


def _validate_sources(statement: exp.Expression) -> None:
    for source in _source_expressions(statement):
        if isinstance(source, (exp.Table, exp.Subquery, exp.Values)):
            continue
        if isinstance(source, exp.Lateral) and isinstance(source.this, exp.Subquery):
            continue
        raise _safe_error(
            ChatBIErrorCategory.UNSAFE_QUERY,
            "the query uses an unsupported table source",
        )


def _function_name(function: exp.Func) -> str:
    if isinstance(function, exp.Anonymous):
        return function.name.lower()
    return function.sql_name().lower()


def _validate_cast(node: exp.Cast) -> None:
    target = node.args.get("to")
    if not isinstance(target, exp.DataType):
        raise _safe_error(
            ChatBIErrorCategory.UNSAFE_QUERY,
            "only built-in scalar casts are allowed",
        )
    if target.args.get("kind") is not None or target.args.get("expressions"):
        raise _safe_error(
            ChatBIErrorCategory.UNSAFE_QUERY,
            "only unqualified built-in scalar casts are allowed",
        )
    type_name = " ".join(target.sql(dialect="postgres").lower().split())
    if type_name not in _SAFE_CAST_TYPES:
        raise _safe_error(
            ChatBIErrorCategory.UNSAFE_QUERY,
            "the cast target is outside the safe built-in scalar allow-list",
        )


def _validate_functions(statement: exp.Expression) -> None:
    for node in statement.walk():
        node_name = type(node).__name__
        if node_name == "Operator" or isinstance(node, exp.Operator):
            raise _safe_error(
                ChatBIErrorCategory.UNSAFE_QUERY,
                "explicit PostgreSQL operators are not allowed",
            )
        if isinstance(node, exp.Binary) and node_name not in _SAFE_BINARY_OPERATOR_NAMES:
            raise _safe_error(
                ChatBIErrorCategory.UNSAFE_QUERY,
                "the query contains an unsupported or custom operator",
            )
        if isinstance(node, exp.Unary) and node_name not in _SAFE_UNARY_OPERATOR_NAMES:
            raise _safe_error(
                ChatBIErrorCategory.UNSAFE_QUERY,
                "the query contains an unsupported unary operator",
            )
        if isinstance(node, exp.Dot) and isinstance(node.expression, exp.Func):
            raise _safe_error(
                ChatBIErrorCategory.UNSAFE_QUERY,
                "qualified function calls are not allowed",
            )
        if isinstance(node, exp.Cast):
            _validate_cast(node)
        if (
            isinstance(node, exp.Func)
            and not isinstance(node, (exp.And, exp.Or))
            and (_function_name(node) not in SAFE_SQL_FUNCTIONS)
        ):
            raise _safe_error(
                ChatBIErrorCategory.UNSAFE_QUERY,
                "the query contains a function outside the safe allow-list",
            )


def _schema_for_qualification(
    snapshot: SchemaSnapshot,
    context: SemanticSchemaContext,
    policy: QueryPolicy,
) -> MappingSchema:
    included = set(context.included_relations)
    schema: dict[str, dict[str, dict[str, str]]] = {}
    for relation in snapshot.relations:
        label = f"{relation.schema_name}.{relation.name}"
        if label not in included or not _allowed_schema(
            _Identifier(relation.schema_name, False), policy
        ):
            continue
        schema.setdefault(relation.schema_name, {})[relation.name] = {
            column.name: column.normalized_type for column in relation.columns
        }
    # Keep identifier spelling and quoting intact.  MappingSchema's default
    # normalization lower-cases unquoted mapping keys and cannot represent a
    # PostgreSQL quoted identifier such as "Order ID" faithfully.
    return MappingSchema(schema, dialect="postgres", normalize=False)


def _validate_columns(
    statement: exp.Expression,
    snapshot: SchemaSnapshot,
    context: SemanticSchemaContext,
    policy: QueryPolicy,
) -> None:
    qualified_statement = statement.copy()
    # SQLGlot cannot resolve an implicit table-name qualifier reliably when a
    # PostgreSQL table uses a quoted mixed-case or space-containing name.  An
    # alias with the exact same identifier preserves PostgreSQL semantics and
    # gives the qualifier an explicit, lossless lookup key.
    for table in qualified_statement.find_all(exp.Table):
        if table.args.get("alias") is None and isinstance(table.this, exp.Identifier):
            table.set("alias", exp.TableAlias(this=table.this.copy()))
    try:
        qualify(
            qualified_statement,
            schema=_schema_for_qualification(snapshot, context, policy),
            dialect="postgres",
            validate_qualify_columns=True,
            identify=False,
        )
    except (IndexError, KeyError, SqlglotError, TypeError, ValueError):
        raise _safe_error(
            ChatBIErrorCategory.UNKNOWN_COLUMN,
            "the query references an unknown or ambiguous column",
        ) from None


def _limit_value(statement: exp.Expression) -> tuple[int | None, str]:
    limit = statement.args.get("limit")
    if limit is None:
        return None, "injected"
    expression = limit.args.get("expression")
    if not isinstance(expression, exp.Literal) or not expression.is_int:
        raise _safe_error(
            ChatBIErrorCategory.POLICY_VIOLATION,
            "LIMIT must be a non-negative integer literal",
        )
    try:
        value = int(expression.this)
    except (TypeError, ValueError):
        raise _safe_error(
            ChatBIErrorCategory.POLICY_VIOLATION,
            "LIMIT must be a non-negative integer literal",
        ) from None
    if value < 0:
        raise _safe_error(
            ChatBIErrorCategory.POLICY_VIOLATION,
            "LIMIT must be non-negative",
        )
    return value, "explicit"


def _validate_resource_clauses(statement: exp.Expression) -> None:
    """Keep row/resource controls explicit and bounded for the whole AST."""
    top_level_limit = statement.args.get("limit")
    if top_level_limit is not None:
        _limit_value(statement)
    for node in statement.walk():
        node_name = type(node).__name__
        if isinstance(node, exp.Offset) or node_name == "Offset":
            raise _safe_error(
                ChatBIErrorCategory.POLICY_VIOLATION,
                "OFFSET is not supported by the current execution policy",
            )
        if isinstance(node, exp.Fetch) or node_name == "Fetch":
            raise _safe_error(
                ChatBIErrorCategory.POLICY_VIOLATION,
                "FETCH is not supported by the current execution policy",
            )
        if isinstance(node, exp.Limit) and node is not top_level_limit:
            raise _safe_error(
                ChatBIErrorCategory.POLICY_VIOLATION,
                "nested LIMIT clauses are not supported",
            )


class _SQLSafetyValidator:
    """Validate generated SQL against the authoritative snapshot and policy."""

    def validate(
        self,
        candidate: object,
        *,
        schema_snapshot: SchemaSnapshot,
        semantic_context: SemanticSchemaContext,
        policy: QueryPolicy,
    ) -> ValidatedSQL:
        if not isinstance(candidate, SQLCandidate):
            raise TypeError("candidate must be an SQLCandidate")
        try:
            candidate = SQLCandidate.model_validate(candidate.model_dump())
            schema_snapshot = SchemaSnapshot.model_validate(schema_snapshot.model_dump())
            semantic_context = SemanticSchemaContext.model_validate(semantic_context.model_dump())
            policy = QueryPolicy.model_validate(policy.model_dump())
        except ValidationError:
            raise _safe_error(
                ChatBIErrorCategory.MALFORMED_MODEL_OUTPUT,
                "SQL candidate contract is invalid",
            ) from None
        try:
            expected_context = build_semantic_schema_context(
                schema_snapshot,
                max_chars=semantic_context.max_chars,
            )
        except SchemaContextBudgetError:
            raise _safe_error(
                ChatBIErrorCategory.POLICY_VIOLATION,
                "semantic context cannot be rebuilt from the schema snapshot",
            ) from None
        if expected_context != semantic_context:
            raise _safe_error(
                ChatBIErrorCategory.POLICY_VIOLATION,
                "semantic context is not the canonical snapshot context",
            )
        if candidate.datasource_id != schema_snapshot.datasource_id:
            raise _safe_error(
                ChatBIErrorCategory.POLICY_VIOLATION,
                "SQL candidate datasource does not match the schema snapshot",
            )
        if (
            candidate.dialect is not schema_snapshot.dialect
            or candidate.dialect is not policy.dialect
        ):
            raise _safe_error(
                ChatBIErrorCategory.POLICY_VIOLATION,
                "SQL candidate dialect does not match the configured policy",
            )
        if candidate.context_fingerprint != schema_snapshot.fingerprint:
            raise _safe_error(
                ChatBIErrorCategory.POLICY_VIOLATION,
                "SQL candidate context fingerprint is stale",
            )
        if semantic_context.snapshot_fingerprint != schema_snapshot.fingerprint:
            raise _safe_error(
                ChatBIErrorCategory.POLICY_VIOLATION,
                "semantic context fingerprint is stale",
            )
        known_labels = {
            f"{relation.schema_name}.{relation.name}" for relation in schema_snapshot.relations
        }
        if any(label not in known_labels for label in semantic_context.included_relations):
            raise _safe_error(
                ChatBIErrorCategory.POLICY_VIOLATION,
                "semantic context contains an unknown relation",
            )

        try:
            statement = _parse_statement(candidate.sql)
            _validate_ast_complexity(statement)
            _validate_read_only_shape(statement)
            _validate_sources(statement)
            _validate_functions(statement)
            _validate_resource_clauses(statement)
            relations = _resolve_tables(statement, schema_snapshot, semantic_context, policy)
            _validate_columns(statement, schema_snapshot, semantic_context, policy)

            explicit_limit, limit_source = _limit_value(statement)
            if explicit_limit is None:
                effective_limit = policy.max_rows
                normalized_statement = statement.limit(exp.Literal.number(effective_limit))
            elif explicit_limit > policy.max_rows:
                effective_limit = policy.max_rows
                limit_source = "clamped"
                normalized_statement = statement.limit(exp.Literal.number(effective_limit))
            else:
                effective_limit = explicit_limit
                normalized_statement = statement.copy()

            _qualify_physical_tables(
                normalized_statement,
                schema_snapshot,
                semantic_context,
                policy,
            )
            normalized_sql = normalized_statement.sql(dialect="postgres")
            if len(normalized_sql) > MAX_SQL_CHARS:
                raise _safe_error(
                    ChatBIErrorCategory.POLICY_VIOLATION,
                    "validated SQL exceeds the maximum statement length",
                )

            # The serialized form is a second trust boundary.  Reparse and
            # reapply the policy so normalization cannot silently introduce a
            # construct that was absent from the original AST.
            normalized_parsed = _parse_statement(normalized_sql)
            _validate_ast_complexity(normalized_parsed)
            _validate_read_only_shape(normalized_parsed)
            _validate_sources(normalized_parsed)
            _validate_functions(normalized_parsed)
            _validate_resource_clauses(normalized_parsed)
            normalized_relations = _resolve_tables(
                normalized_parsed,
                schema_snapshot,
                semantic_context,
                policy,
            )
            _validate_columns(normalized_parsed, schema_snapshot, semantic_context, policy)
            if tuple(relation.label for relation in normalized_relations) != tuple(
                relation.label for relation in relations
            ):
                raise _safe_error(
                    ChatBIErrorCategory.UNSAFE_QUERY,
                    "normalized SQL changed the approved relation set",
                )
        except RecursionError:
            raise _safe_error(
                ChatBIErrorCategory.UNSAFE_QUERY,
                "query complexity exceeds the safety bound",
            ) from None

        return ValidatedSQL._from_validator(
            validation_version=SQL_VALIDATION_VERSION,
            datasource_id=candidate.datasource_id,
            dialect=candidate.dialect,
            provider=candidate.provider,
            model=candidate.model,
            prompt_version=candidate.prompt_version,
            original_sql=candidate.sql,
            normalized_sql=normalized_sql,
            referenced_schemas=tuple(sorted({relation.schema_name for relation in relations})),
            referenced_relations=tuple(relation.label for relation in relations),
            schema_fingerprint=schema_snapshot.fingerprint,
            policy_fingerprint=policy_fingerprint(policy),
            limit_bounded=True,
            effective_limit=effective_limit,
            limit_source=limit_source,
        )


def _validate_sql_candidate(
    candidate: object,
    *,
    schema_snapshot: SchemaSnapshot,
    semantic_context: SemanticSchemaContext,
    policy: QueryPolicy,
) -> ValidatedSQL:
    """Functional entry point for tests and small integrations."""
    return _SQLSafetyValidator().validate(
        candidate,
        schema_snapshot=schema_snapshot,
        semantic_context=semantic_context,
        policy=policy,
    )


__all__ = [
    "MAX_SET_OPERATION_DEPTH",
    "SAFE_SQL_FUNCTIONS",
    "policy_fingerprint",
]
