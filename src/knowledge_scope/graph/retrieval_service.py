"""Conservative, bounded graph retrieval over the A3.1/A3.3 Neo4j graph."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Protocol
from uuid import UUID

from knowledge_scope.linking.models import normalize_linking_label

from .retrieval import (
    GraphEntitySnapshot,
    GraphEvidence,
    GraphEvidenceResult,
    GraphNeighbor,
    GraphPath,
    GraphPathKind,
    GraphRetrievalConfig,
    GraphRetrievalResult,
    GraphSeedCandidate,
    SeedResolutionMethod,
    graph_score,
    seed_method_priority,
)


class GraphRetrievalError(RuntimeError):
    """Raised when graph retrieval cannot safely use the stored graph state."""


class GraphRetrievalStore(Protocol):
    """The small typed adapter surface required by the retrieval service."""

    def list_retrieval_entities(
        self,
        knowledge_base_id: UUID,
        *,
        max_entities: int,
    ) -> tuple[GraphEntitySnapshot, ...]:
        """Return bounded local entities and their current canonical bridges."""

    def get_retrieval_neighbors(
        self,
        knowledge_base_id: UUID,
        entity_id: str,
        *,
        max_neighbors: int,
        max_relations: int,
    ) -> tuple[GraphNeighbor, ...]:
        """Return one bounded local-entity edge expansion."""


@dataclass(frozen=True, slots=True)
class _Match:
    score: float
    method: SeedResolutionMethod
    term: str
    canonical_entity_id: str | None


@dataclass(frozen=True, slots=True)
class _TraversalState:
    seed_entity_id: str
    current_entity_id: str
    current_document_id: UUID
    seed_score: float
    path: GraphPath
    evidence: tuple[GraphEvidence, ...]


@dataclass(slots=True)
class _EvidenceAggregate:
    evidence: GraphEvidence
    score: float
    best_hop_distance: int
    seed_entity_id: str
    retrieval_reason: str
    paths: dict[str, GraphPath]


def _is_ascii_identifier(value: str) -> bool:
    return all(
        character.isascii() and (character.isalnum() or character == "_") for character in value
    )


def _contains_term(query: str, term: str) -> bool:
    if term not in query:
        return False
    if not _is_ascii_identifier(term):
        return True
    start = 0
    while (match_start := query.find(term, start)) >= 0:
        end = match_start + len(term)
        before = query[match_start - 1] if match_start else ""
        after = query[end] if end < len(query) else ""
        if not (before.isascii() and (before.isalnum() or before == "_")) and not (
            after.isascii() and (after.isalnum() or after == "_")
        ):
            return True
        start = match_start + 1
    return False


def _path_key(path: GraphPath) -> str:
    return path.model_dump_json()


def _path_kind(path: GraphPath, direction: str) -> GraphPathKind:
    next_hop = path.hop_distance + 1
    if next_hop == 1:
        return "canonical_bridge" if direction == "canonical_bridge" else "relation"
    return "mixed"


def _path_reason(path: GraphPath) -> str:
    if path.kind == "seed":
        return "seed_entity_evidence"
    if path.kind == "relation":
        return "direct_relation_evidence"
    if path.kind == "canonical_bridge":
        return "canonical_membership_evidence"
    return "bounded_multi_hop_evidence"


def _relation_ids(path: GraphPath, neighbor: GraphNeighbor) -> tuple[str, ...]:
    if neighbor.relation is None:
        return tuple(path.relation_ids)
    return (*path.relation_ids, neighbor.relation.relation_id)


def _canonical_ids(path: GraphPath, neighbor: GraphNeighbor) -> tuple[str, ...]:
    if neighbor.canonical is None:
        return tuple(path.canonical_entity_ids)
    return (*path.canonical_entity_ids, neighbor.canonical.canonical_entity_id)


def _dedupe_evidence(values: Sequence[GraphEvidence]) -> tuple[GraphEvidence, ...]:
    by_id = {value.evidence_id: value for value in values}
    return tuple(by_id[key] for key in sorted(by_id))


def _unique_snapshots(
    snapshots: Sequence[GraphEntitySnapshot],
) -> tuple[GraphEntitySnapshot, ...]:
    """Reject conflicting adapter rows and remove identical duplicate rows."""

    by_id: dict[str, GraphEntitySnapshot] = {}
    for snapshot in snapshots:
        entity_id = snapshot.entity.entity_id
        previous = by_id.get(entity_id)
        if previous is not None and previous != snapshot:
            raise GraphRetrievalError("graph adapter returned conflicting entity snapshots")
        by_id[entity_id] = snapshot
    return tuple(by_id.values())


class GraphRetrievalService:
    """Resolve query seeds and perform a bounded, evidence-first graph walk."""

    def __init__(
        self,
        store: GraphRetrievalStore,
        *,
        config: GraphRetrievalConfig | None = None,
    ) -> None:
        self.store = store
        self.config = config or GraphRetrievalConfig()

    def _matches_for_snapshot(
        self,
        query_normalized: str,
        snapshot: GraphEntitySnapshot,
    ) -> list[_Match]:
        values: list[tuple[str, str, str | None]] = [
            (snapshot.entity.canonical_name, "local_name", None),
            *((alias, "local_alias", None) for alias in snapshot.entity.aliases),
        ]
        for canonical in snapshot.canonical_entities:
            values.append(
                (canonical.canonical_name, "canonical_name", canonical.canonical_entity_id)
            )
            values.extend(
                (alias, "canonical_alias", canonical.canonical_entity_id)
                for alias in canonical.aliases
            )

        matches: list[_Match] = []
        for display_term, term_kind, canonical_entity_id in values:
            term = normalize_linking_label(display_term)
            if len(term) < 2:
                continue
            method_prefix = term_kind
            if query_normalized == term:
                method = f"exact_{method_prefix}"  # type: ignore[assignment]
                matches.append(
                    _Match(1.0, method, display_term, canonical_entity_id)  # type: ignore[arg-type]
                )
                continue
            if len(term) >= 2 and _contains_term(query_normalized, term):
                method = f"contains_{method_prefix}"  # type: ignore[assignment]
                matches.append(
                    _Match(0.93, method, display_term, canonical_entity_id)  # type: ignore[arg-type]
                )
                continue
            if len(term) < self.config.min_lexical_term_length:
                continue
            similarity = SequenceMatcher(None, query_normalized, term).ratio()
            if similarity >= self.config.lexical_threshold:
                method = f"lexical_{method_prefix}"  # type: ignore[assignment]
                matches.append(
                    _Match(round(similarity * 0.88, 6), method, display_term, canonical_entity_id)  # type: ignore[arg-type]
                )
        return matches

    def _resolve_seed_candidates_from_snapshots(
        self,
        query: str,
        knowledge_base_id: UUID,
        snapshots: Sequence[GraphEntitySnapshot],
    ) -> tuple[GraphSeedCandidate, ...]:
        """Resolve only bounded, explainable local seeds for one KB."""

        query_normalized = normalize_linking_label(query)
        if not query_normalized:
            raise ValueError("query must not be blank")
        term_entities: dict[str, set[str]] = defaultdict(set)
        for snapshot in snapshots:
            if snapshot.entity.knowledge_base_id != knowledge_base_id:
                raise GraphRetrievalError("graph adapter returned a cross-KB entity")
            if not snapshot.evidence:
                continue
            terms = [snapshot.entity.canonical_name, *snapshot.entity.aliases]
            terms.extend(
                term
                for canonical in snapshot.canonical_entities
                for term in (canonical.canonical_name, *canonical.aliases)
            )
            for value in terms:
                normalized = normalize_linking_label(value)
                if len(normalized) >= 2:
                    term_entities[normalized].add(snapshot.entity.entity_id)

        candidates: list[GraphSeedCandidate] = []
        for snapshot in snapshots:
            matches = self._matches_for_snapshot(query_normalized, snapshot)
            if not snapshot.evidence or not matches:
                continue
            valid_matches = [
                match
                for match in matches
                if len(normalize_linking_label(match.term)) >= 2
                and len(term_entities[normalize_linking_label(match.term)])
                <= self.config.max_seed_entities * 4
            ]
            if not valid_matches:
                continue
            best = min(
                valid_matches,
                key=lambda match: (
                    -match.score,
                    seed_method_priority(match.method),
                    normalize_linking_label(match.term),
                    match.term,
                    match.canonical_entity_id or "",
                ),
            )
            candidates.append(
                GraphSeedCandidate(
                    seed_entity_id=snapshot.entity.entity_id,
                    knowledge_base_id=snapshot.entity.knowledge_base_id,
                    document_id=snapshot.entity.document_id,
                    canonical_name=snapshot.entity.canonical_name,
                    entity_type=snapshot.entity.entity_type,
                    matched_term=best.term,
                    method=best.method,
                    score=best.score,
                    matched_canonical_entity_id=best.canonical_entity_id,
                )
            )

        candidates.sort(
            key=lambda candidate: (
                -candidate.score,
                seed_method_priority(candidate.method),
                candidate.seed_entity_id,
            )
        )
        return tuple(candidates[: self.config.max_seed_entities])

    def resolve_seed_candidates(
        self,
        query: str,
        knowledge_base_id: UUID,
    ) -> tuple[GraphSeedCandidate, ...]:
        """Resolve only bounded, explainable local seeds for one KB."""

        snapshots = _unique_snapshots(
            self.store.list_retrieval_entities(
                knowledge_base_id,
                max_entities=self.config.max_entity_scan,
            )
        )
        return self._resolve_seed_candidates_from_snapshots(query, knowledge_base_id, snapshots)

    def search(self, query: str, knowledge_base_id: UUID) -> GraphRetrievalResult:
        """Return deterministic, bounded source evidence for one KB-scoped query."""

        if not query.strip():
            raise ValueError("query must not be blank")
        snapshots = _unique_snapshots(
            self.store.list_retrieval_entities(
                knowledge_base_id,
                max_entities=self.config.max_entity_scan,
            )
        )
        snapshot_by_id = {snapshot.entity.entity_id: snapshot for snapshot in snapshots}
        seeds = self._resolve_seed_candidates_from_snapshots(query, knowledge_base_id, snapshots)
        aggregates: dict[str, _EvidenceAggregate] = {}
        frontier: list[_TraversalState] = []

        def add_evidence(
            evidence: GraphEvidence,
            *,
            seed_entity_id: str,
            seed_score: float,
            path: GraphPath,
        ) -> None:
            try:
                evidence = GraphEvidence.model_validate(evidence.model_dump(mode="json"))
            except (TypeError, ValueError) as error:
                raise GraphRetrievalError(
                    "graph adapter returned invalid evidence lineage"
                ) from error
            if evidence.knowledge_base_id != knowledge_base_id:
                raise GraphRetrievalError("graph adapter returned cross-KB evidence")
            score = graph_score(seed_score, path.hop_distance, path.kind)
            reason = _path_reason(path)
            key = _path_key(path)
            existing = aggregates.get(evidence.evidence_id)
            if existing is None:
                aggregates[evidence.evidence_id] = _EvidenceAggregate(
                    evidence=evidence,
                    score=score,
                    best_hop_distance=path.hop_distance,
                    seed_entity_id=seed_entity_id,
                    retrieval_reason=reason,
                    paths={key: path},
                )
                return
            existing.paths.setdefault(key, path)
            if (score, -path.hop_distance, seed_entity_id) > (
                existing.score,
                -existing.best_hop_distance,
                existing.seed_entity_id,
            ):
                existing.score = score
                existing.best_hop_distance = path.hop_distance
                existing.seed_entity_id = seed_entity_id
                existing.retrieval_reason = reason

        for seed in seeds:
            snapshot = snapshot_by_id.get(seed.seed_entity_id)
            if snapshot is None:
                raise GraphRetrievalError("resolved seed is missing from the graph snapshot")
            path = GraphPath(
                seed_entity_id=seed.seed_entity_id,
                entity_ids=[seed.seed_entity_id],
                hop_distance=0,
                kind="seed",
            )
            for evidence in snapshot.evidence:
                add_evidence(
                    evidence,
                    seed_entity_id=seed.seed_entity_id,
                    seed_score=seed.score,
                    path=path,
                )
            frontier.append(
                _TraversalState(
                    seed_entity_id=seed.seed_entity_id,
                    current_entity_id=seed.seed_entity_id,
                    current_document_id=snapshot.entity.document_id,
                    seed_score=seed.score,
                    path=path,
                    evidence=tuple(snapshot.evidence),
                )
            )

        relation_ids_seen: set[str] = set()
        for _depth in range(1, self.config.max_hops + 1):
            next_frontier: list[_TraversalState] = []
            seen_next_states: set[tuple[str, str, tuple[str, ...], tuple[str, ...]]] = set()
            for state in frontier:
                neighbors = self.store.get_retrieval_neighbors(
                    knowledge_base_id,
                    state.current_entity_id,
                    max_neighbors=self.config.max_neighbors,
                    max_relations=self.config.max_relations,
                )
                # The adapter returns deterministic rows grouped by neighbor.  Select a
                # bounded set of neighbors first, visit one edge per selected neighbor,
                # and only then consume additional edges in round-robin order.  This
                # preserves a fair opportunity for later neighbors instead of letting
                # one high-degree neighbor consume the request-wide relation budget.
                ordered_neighbors = sorted(
                    neighbors,
                    key=lambda value: (
                        value.neighbor.entity_id,
                        value.relation.relation_id if value.relation is not None else "",
                        value.canonical.canonical_entity_id if value.canonical is not None else "",
                        value.direction,
                    ),
                )
                neighbor_groups: dict[str, list[GraphNeighbor]] = defaultdict(list)
                for neighbor in ordered_neighbors:
                    neighbor_groups[neighbor.neighbor.entity_id].append(neighbor)
                selected_neighbor_ids = list(neighbor_groups)[: self.config.max_neighbors]
                selected_groups = [
                    neighbor_groups[entity_id] for entity_id in selected_neighbor_ids
                ]
                first_pass = [group[0] for group in selected_groups]
                remainder: list[GraphNeighbor] = []
                for offset in range(1, max((len(group) for group in selected_groups), default=1)):
                    remainder.extend(
                        group[offset] for group in selected_groups if offset < len(group)
                    )
                for neighbor in (*first_pass, *remainder):
                    if neighbor.seed_entity_id != state.current_entity_id:
                        raise GraphRetrievalError(
                            "graph adapter returned a mismatched traversal seed"
                        )
                    if neighbor.neighbor.knowledge_base_id != knowledge_base_id:
                        raise GraphRetrievalError("graph adapter returned a cross-KB neighbor")
                    if neighbor.neighbor.entity_id in state.path.entity_ids:
                        continue
                    if not neighbor.neighbor_evidence:
                        raise GraphRetrievalError(
                            "graph adapter returned an unsupported neighbor entity"
                        )
                    if (
                        neighbor.canonical is not None
                        and neighbor.canonical.canonical_entity_id
                        in state.path.canonical_entity_ids
                    ):
                        continue
                    if neighbor.relation is not None and (
                        neighbor.relation.document_id != state.current_document_id
                        or neighbor.neighbor.document_id != state.current_document_id
                    ):
                        raise GraphRetrievalError(
                            "graph adapter returned a cross-document relation"
                        )
                    relation = neighbor.relation
                    if relation is not None and not neighbor.relation_evidence:
                        raise GraphRetrievalError("graph adapter returned an unsupported relation")
                    relation_ids = _relation_ids(state.path, neighbor)
                    if relation is not None and relation.relation_id not in relation_ids_seen:
                        if len(relation_ids_seen) >= self.config.max_relations:
                            continue
                        relation_ids_seen.add(relation.relation_id)
                    canonical_ids = _canonical_ids(state.path, neighbor)
                    path = GraphPath(
                        seed_entity_id=state.seed_entity_id,
                        entity_ids=[*state.path.entity_ids, neighbor.neighbor.entity_id],
                        relation_ids=list(relation_ids),
                        canonical_entity_ids=list(canonical_ids),
                        directions=[*state.path.directions, neighbor.direction],
                        hop_distance=state.path.hop_distance + 1,
                        kind=_path_kind(state.path, neighbor.direction),
                    )
                    # ``evidence`` is an adapter convenience aggregate.  Only the
                    # explicitly validated support lists may add new evidence; the
                    # current state already carries the evidence for the current node.
                    evidence = _dedupe_evidence(
                        [
                            *state.evidence,
                            *neighbor.neighbor_evidence,
                            *neighbor.relation_evidence,
                        ]
                    )
                    for evidence_item in evidence:
                        add_evidence(
                            evidence_item,
                            seed_entity_id=state.seed_entity_id,
                            seed_score=state.seed_score,
                            path=path,
                        )
                    if path.hop_distance >= self.config.max_hops:
                        continue
                    state_key = (
                        state.seed_entity_id,
                        neighbor.neighbor.entity_id,
                        relation_ids,
                        canonical_ids,
                    )
                    if state_key in seen_next_states:
                        continue
                    seen_next_states.add(state_key)
                    next_frontier.append(
                        _TraversalState(
                            seed_entity_id=state.seed_entity_id,
                            current_entity_id=neighbor.neighbor.entity_id,
                            current_document_id=neighbor.neighbor.document_id,
                            seed_score=state.seed_score,
                            path=path,
                            evidence=evidence,
                        )
                    )
            frontier = next_frontier
            if not frontier:
                break

        items = [
            GraphEvidenceResult(
                evidence=aggregate.evidence,
                score=aggregate.score,
                seed_entity_id=aggregate.seed_entity_id,
                retrieval_reason=aggregate.retrieval_reason,
                paths=list(aggregate.paths.values()),
            )
            for aggregate in aggregates.values()
        ]
        items.sort(
            key=lambda item: (
                -item.score,
                min(path.hop_distance for path in item.paths),
                str(item.evidence.document_id),
                item.evidence.page_start,
                item.evidence.page_end,
                item.evidence.chunk_id,
                item.evidence.evidence_id,
            )
        )
        return GraphRetrievalResult(
            query=query,
            knowledge_base_id=knowledge_base_id,
            seeds=list(seeds),
            items=items[: self.config.max_evidence],
        )


__all__ = ["GraphRetrievalError", "GraphRetrievalService", "GraphRetrievalStore"]
