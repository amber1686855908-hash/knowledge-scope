"""Small, reproducible A3.3 review sample built from A3.2 runtime records."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from time import perf_counter

from knowledge_scope.graph.models import GraphEntity, evidence_id_for
from knowledge_scope.graph.neo4j import Neo4jGraphStore
from knowledge_scope.linking.service import (
    LinkingGateway,
    LinkingRun,
    LinkingValidationError,
    link_entities,
)
from knowledge_scope.linking.service_types import LocalEntityContext
from knowledge_scope.shared.config import Settings

DEFAULT_INPUT: tuple[Path, ...] = (
    Path("data/evaluation/a3-2/debug-9/sample.jsonl"),
    Path("data/evaluation/a3-2/holdout-18/sample.jsonl"),
    Path("data/evaluation/a3-2/fresh-18/sample.jsonl"),
)
DEFAULT_OUTPUT = Path("data/evaluation/a3-3")
SAMPLE_SCHEMA_VERSION = "1.0"


def _bounded_excerpt(value: str, limit: int = 280) -> str:
    normalized = " ".join(value.split())
    return normalized[:limit] + ("…" if len(normalized) > limit else "")


def _lineage(entity: GraphEntity) -> dict[str, object]:
    provenance = sorted(entity.provenance, key=evidence_id_for)
    return {
        "document_id": str(entity.document_id),
        "pages": sorted({(item.page_start, item.page_end) for item in provenance}),
        "chunk_ids": sorted({item.chunk_id for item in provenance}),
        "source_block_ids": sorted(
            {block_id for item in provenance for block_id in item.source_block_ids}
        ),
        "section_paths": sorted({tuple(item.section_path) for item in provenance}),
    }


def _review_entity(context: LocalEntityContext) -> dict[str, object]:
    return {
        "entity_id": context.entity.entity_id,
        "knowledge_base_id": str(context.entity.knowledge_base_id),
        "document_id": str(context.entity.document_id),
        "canonical_name": context.entity.canonical_name,
        "entity_type": context.entity.entity_type,
        "aliases": list(context.entity.aliases),
        "lineage": _lineage(context.entity),
        "source_excerpt": _bounded_excerpt(context.source_excerpt),
        "subject": context.subject,
    }


def _json_default(value: object) -> str:
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _atomic_write_text(path: Path, content: str) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_sample_entity_contexts(paths: Sequence[Path]) -> tuple[LocalEntityContext, ...]:
    """Load accepted A3.2 entities and merge repeated local IDs safely.

    Only bounded ``text_excerpt`` values already present in the A3.2 review
    records are carried into the A3.3 pack.  No absolute input path is emitted.
    """

    grouped: dict[str, list[LocalEntityContext]] = {}
    for path in sorted(paths, key=lambda value: str(value)):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise LinkingValidationError(
                        f"A3.2 sample line {line_number} is not valid JSON"
                    ) from error
                if not isinstance(record, dict) or record.get("status") != "accepted":
                    continue
                raw_entities = record.get("entities", [])
                if not isinstance(raw_entities, list):
                    raise LinkingValidationError("A3.2 entity list has an invalid shape")
                subject = record.get("subject")
                excerpt = record.get("text_excerpt")
                if not isinstance(subject, str) or not isinstance(excerpt, str):
                    raise LinkingValidationError("A3.2 sample is missing bounded review context")
                for raw_entity in raw_entities:
                    try:
                        entity = GraphEntity.model_validate(raw_entity)
                    except (TypeError, ValueError) as error:
                        raise LinkingValidationError(
                            "A3.2 sample contains an invalid GraphEntity"
                        ) from error
                    grouped.setdefault(entity.entity_id, []).append(
                        LocalEntityContext(
                            entity=entity,
                            source_excerpt=excerpt,
                            subject=subject,
                        )
                    )

    contexts: list[LocalEntityContext] = []
    for entity_id, values in sorted(grouped.items()):
        ordered = sorted(
            values,
            key=lambda value: (
                value.subject or "",
                str(value.entity.document_id),
                value.source_excerpt,
            ),
        )
        first = ordered[0].entity
        if any(
            value.entity.knowledge_base_id != first.knowledge_base_id
            or value.entity.document_id != first.document_id
            or value.entity.canonical_name != first.canonical_name
            or value.entity.entity_type != first.entity_type
            for value in ordered[1:]
        ):
            raise LinkingValidationError(
                f"local entity {entity_id} has conflicting identity fields"
            )
        aliases = [alias for value in ordered for alias in value.entity.aliases]
        provenance_by_id = {
            evidence_id_for(provenance): provenance
            for value in ordered
            for provenance in value.entity.provenance
        }
        merged = GraphEntity(
            entity_id=first.entity_id,
            knowledge_base_id=first.knowledge_base_id,
            document_id=first.document_id,
            canonical_name=first.canonical_name,
            entity_type=first.entity_type,
            aliases=sorted(set(aliases)),
            provenance=list(provenance_by_id.values()),
        )
        excerpts = sorted({value.source_excerpt for value in ordered})
        contexts.append(
            LocalEntityContext(
                entity=merged,
                source_excerpt="\n".join(excerpts),
                subject=ordered[0].subject,
            )
        )
    return tuple(
        sorted(
            contexts,
            key=lambda value: (
                value.subject or "",
                str(value.entity.document_id),
                value.entity.entity_id,
            ),
        )
    )


def _review_record(
    run: LinkingRun,
    decision_index: int,
    *,
    context_by_id: dict[str, LocalEntityContext],
    mapping_by_local_id: dict[str, str],
) -> dict[str, object]:
    decision = run.decisions[decision_index]
    candidate = next(item for item in run.candidates if item.link_pair_id == decision.link_pair_id)
    return {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "knowledge_base_id": str(run.knowledge_base_id),
        "link_pair_id": decision.link_pair_id,
        "decision_id": decision.decision_id,
        "run_id": str(decision.run_id),
        "local_entity_a": _review_entity(context_by_id[candidate.local_entity_a_id]),
        "local_entity_b": _review_entity(context_by_id[candidate.local_entity_b_id]),
        "candidate_signals": candidate.signals.model_dump(mode="json"),
        "decision": decision.decision,
        "method": decision.method,
        "confidence": decision.confidence,
        "reason": decision.reason,
        "created_at": decision.created_at.isoformat(),
        "provider": decision.provider,
        "model": decision.model,
        "prompt_version": decision.prompt_version,
        "canonical_entity_id_a": mapping_by_local_id.get(candidate.local_entity_a_id),
        "canonical_entity_id_b": mapping_by_local_id.get(candidate.local_entity_b_id),
    }


async def run_linking_review_sample(
    paths: Sequence[Path],
    *,
    settings: Settings,
    output_dir: Path = DEFAULT_OUTPUT,
    gateway: LinkingGateway | None = None,
    store: Neo4jGraphStore | None = None,
    max_candidates: int = 200,
    max_block_size: int = 64,
) -> dict[str, object]:
    """Run a bounded review sample and write only ignored runtime artifacts."""

    contexts = load_sample_entity_contexts(paths)
    if not contexts:
        raise LinkingValidationError("sample contains no accepted extracted entities")
    started = perf_counter()
    run = await link_entities(
        [context.entity for context in contexts],
        gateway=gateway,
        settings=settings if gateway is not None else None,
        contexts=contexts,
        max_candidates=max_candidates,
        max_block_size=max_block_size,
    )
    persisted = False
    if store is not None:
        await asyncio.to_thread(
            store.upsert_linking_result,
            run.canonical_entities,
            run.decisions,
            run.mappings,
        )
        persisted = True

    context_by_id = {context.entity.entity_id: context for context in contexts}
    mapping_by_local_id = {
        mapping.local_entity_id: mapping.canonical_entity_id for mapping in run.mappings
    }
    records = [
        _review_record(
            run,
            index,
            context_by_id=context_by_id,
            mapping_by_local_id=mapping_by_local_id,
        )
        for index in range(len(run.decisions))
    ]
    sample_content = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, default=_json_default) + "\n"
        for record in records
    )
    summary: dict[str, object] = {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "sample_files": [
            str(Path(path.parent.name) / path.name)
            for path in sorted(paths, key=lambda value: str(value))
        ],
        "sample_entity_count": len(contexts),
        "candidate_count": run.stats.candidate_count,
        "canonical_entity_count": len(run.canonical_entities),
        "mapped_local_entity_count": len(run.mappings),
        "deterministic_decisions": run.stats.deterministic_decisions,
        "manual_review_fallbacks": run.stats.manual_review_fallbacks,
        "provider_failure_fallbacks": run.stats.provider_failure_fallbacks,
        "skipped_candidate_blocks": run.stats.skipped_candidate_blocks,
        "skipped_candidate_pairs": run.stats.skipped_candidate_pairs,
        "candidate_budget_exhausted": run.stats.candidate_budget_exhausted,
        "llm_adjudications": run.stats.llm_adjudications,
        "llm_failures": run.stats.llm_failures,
        "decision_counts": {
            "LINK": run.stats.link_count,
            "NO_LINK": run.stats.no_link_count,
            "UNCERTAIN": run.stats.uncertain_count,
        },
        "input_tokens": run.stats.input_tokens,
        "output_tokens": run.stats.output_tokens,
        "latency_ms": round(run.stats.latency_ms, 3),
        "estimated_cost": run.stats.estimated_cost,
        "elapsed_ms": round((perf_counter() - started) * 1000, 3),
        "persisted": persisted,
        "note": (
            "This is a deterministic review sample from A3.2 accepted extractions; "
            "it does not measure linking precision, recall, or accuracy."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary["review_record_count"] = len(records)
    summary["review_sha256"] = hashlib.sha256(sample_content.encode("utf-8")).hexdigest()
    _atomic_write_text(output_dir / "review.jsonl", sample_content)
    _atomic_write_text(
        output_dir / "summary.json",
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2, default=_json_default)
        + "\n",
    )
    return summary


__all__ = [
    "DEFAULT_INPUT",
    "DEFAULT_OUTPUT",
    "SAMPLE_SCHEMA_VERSION",
    "load_sample_entity_contexts",
    "run_linking_review_sample",
]
