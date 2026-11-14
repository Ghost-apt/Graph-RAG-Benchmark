"""
Entity Linking Module.
Maps extracted query mentions to verified GraphEntity nodes in the Knowledge Graph.
Enforces type-safe matching, alias lookup, and conservative ambiguity detection.
Does not mutate or create graph nodes.
"""

import re
import unicodedata

from graphrag_bench.graph_retrieval_schemas import (
    EntityLinkingResult,
    LinkedEntity,
    MatchType,
    QueryEntity,
)
from graphrag_bench.graph_schemas import GraphEntity
from graphrag_bench.graph_store import GraphStore
from graphrag_bench.logger import get_logger

logger = get_logger("entity_linker")


def normalize_string(text: str) -> str:
    """Standardize string for matching via NFKC, lowercase folding, and whitespace collapsing."""
    if not text:
        return ""
    folded = unicodedata.normalize("NFKC", text).lower().strip()
    return re.sub(r"\s+", " ", folded)


class EntityLinker:
    """Links query entity mentions to existing Knowledge Graph nodes."""

    def __init__(self, store: GraphStore) -> None:
        """Initialize entity linker with active GraphStore.

        Args:
            store: GraphStore instance containing populated entities and relationships.
        """
        self.store = store
        self._build_indexes()

    def _build_indexes(self) -> None:
        """Construct multi-tiered in-memory lookup indexes from GraphStore entities."""
        # canonical_norm -> list[GraphEntity] (supports same name in different types)
        self._canonical_index: dict[str, list[GraphEntity]] = {}
        # alias_norm -> list[GraphEntity]
        self._alias_index: dict[str, list[GraphEntity]] = {}
        # (canonical_norm, entity_type) -> GraphEntity
        self._type_index: dict[tuple[str, str], GraphEntity] = {}

        all_entities = self.store.get_entities()
        for ent in all_entities:
            c_norm = normalize_string(ent.canonical_name)
            if c_norm:
                self._canonical_index.setdefault(c_norm, []).append(ent)
                self._type_index[(c_norm, ent.entity_type.upper())] = ent

            for alias in ent.aliases:
                a_norm = normalize_string(alias)
                if a_norm and a_norm != c_norm:
                    self._alias_index.setdefault(a_norm, []).append(ent)

    def refresh_indexes(self) -> None:
        """Rebuild indexes if graph contents have been updated."""
        self._build_indexes()

    def link_entities(
        self,
        query_entities: list[QueryEntity],
        query: str = "",
    ) -> EntityLinkingResult:
        """Map extracted query entities to graph nodes with ambiguity handling.

        Args:
            query_entities: List of QueryEntity items identified from user query.
            query: Original user query string.

        Returns:
            EntityLinkingResult with linked entities, unlinked mentions, and ambiguity flags.
        """
        linked: list[LinkedEntity] = []
        unlinked: list[str] = []
        has_ambiguity = False
        seen_entity_ids: set[str] = set()

        for q_ent in query_entities:
            target_norm = normalize_string(q_ent.canonical_name) or normalize_string(
                q_ent.mention_text
            )
            if not target_norm:
                continue

            # 1. Exact Canonical Name Match
            if target_norm in self._canonical_index:
                candidates = self._canonical_index[target_norm]
                if len(candidates) == 1:
                    matched_node = candidates[0]
                    if matched_node.entity_id not in seen_entity_ids:
                        linked.append(
                            LinkedEntity(
                                query_mention=q_ent.mention_text,
                                entity_id=matched_node.entity_id,
                                canonical_name=matched_node.canonical_name,
                                entity_type=matched_node.entity_type,
                                match_type=MatchType.EXACT,
                                is_ambiguous=False,
                                match_explanation=f"Exact match on canonical name '{matched_node.canonical_name}'",
                            )
                        )
                        seen_entity_ids.add(matched_node.entity_id)
                else:
                    # Multiple candidates with same canonical name across distinct types
                    # Check if query entity type disambiguates
                    q_type = q_ent.entity_type.upper()
                    type_matched = [c for c in candidates if c.entity_type.upper() == q_type]
                    if len(type_matched) == 1:
                        matched_node = type_matched[0]
                        if matched_node.entity_id not in seen_entity_ids:
                            linked.append(
                                LinkedEntity(
                                    query_mention=q_ent.mention_text,
                                    entity_id=matched_node.entity_id,
                                    canonical_name=matched_node.canonical_name,
                                    entity_type=matched_node.entity_type,
                                    match_type=MatchType.EXACT,
                                    is_ambiguous=False,
                                    match_explanation=f"Exact match disambiguated by type '{q_type}'",
                                )
                            )
                            seen_entity_ids.add(matched_node.entity_id)
                    else:
                        has_ambiguity = True
                        cand_dicts = [
                            {
                                "entity_id": c.entity_id,
                                "canonical_name": c.canonical_name,
                                "entity_type": c.entity_type,
                            }
                            for c in candidates
                        ]
                        linked.append(
                            LinkedEntity(
                                query_mention=q_ent.mention_text,
                                entity_id=candidates[0].entity_id,
                                canonical_name=candidates[0].canonical_name,
                                entity_type=candidates[0].entity_type,
                                match_type=MatchType.EXACT,
                                is_ambiguous=True,
                                candidates=cand_dicts,
                                match_explanation=f"Ambiguous match: {len(candidates)} entities share name '{target_norm}' across different types",
                            )
                        )
                continue

            # 2. Registered Alias Match
            if target_norm in self._alias_index:
                candidates = self._alias_index[target_norm]
                if len(candidates) == 1:
                    matched_node = candidates[0]
                    if matched_node.entity_id not in seen_entity_ids:
                        linked.append(
                            LinkedEntity(
                                query_mention=q_ent.mention_text,
                                entity_id=matched_node.entity_id,
                                canonical_name=matched_node.canonical_name,
                                entity_type=matched_node.entity_type,
                                match_type=MatchType.ALIAS,
                                is_ambiguous=False,
                                match_explanation=f"Matched registered alias of canonical entity '{matched_node.canonical_name}'",
                            )
                        )
                        seen_entity_ids.add(matched_node.entity_id)
                else:
                    has_ambiguity = True
                    cand_dicts = [
                        {
                            "entity_id": c.entity_id,
                            "canonical_name": c.canonical_name,
                            "entity_type": c.entity_type,
                        }
                        for c in candidates
                    ]
                    linked.append(
                        LinkedEntity(
                            query_mention=q_ent.mention_text,
                            entity_id=candidates[0].entity_id,
                            canonical_name=candidates[0].canonical_name,
                            entity_type=candidates[0].entity_type,
                            match_type=MatchType.ALIAS,
                            is_ambiguous=True,
                            candidates=cand_dicts,
                            match_explanation=f"Ambiguous alias: {len(candidates)} entities match alias '{target_norm}'",
                        )
                    )
                continue

            # 3. Normalized Substring Match (Conservative)
            partial_matches: list[GraphEntity] = []
            for c_norm, ents in self._canonical_index.items():
                if len(target_norm) >= 4 and (target_norm in c_norm or c_norm in target_norm):
                    partial_matches.extend(ents)

            if len(partial_matches) == 1:
                matched_node = partial_matches[0]
                if matched_node.entity_id not in seen_entity_ids:
                    linked.append(
                        LinkedEntity(
                            query_mention=q_ent.mention_text,
                            entity_id=matched_node.entity_id,
                            canonical_name=matched_node.canonical_name,
                            entity_type=matched_node.entity_type,
                            match_type=MatchType.NORMALIZED,
                            is_ambiguous=False,
                            match_explanation=f"Normalized partial match to '{matched_node.canonical_name}'",
                        )
                    )
                    seen_entity_ids.add(matched_node.entity_id)
            elif len(partial_matches) > 1:
                has_ambiguity = True
                cand_dicts = [
                    {
                        "entity_id": c.entity_id,
                        "canonical_name": c.canonical_name,
                        "entity_type": c.entity_type,
                    }
                    for c in partial_matches[:5]
                ]
                linked.append(
                    LinkedEntity(
                        query_mention=q_ent.mention_text,
                        entity_id=partial_matches[0].entity_id,
                        canonical_name=partial_matches[0].canonical_name,
                        entity_type=partial_matches[0].entity_type,
                        match_type=MatchType.NORMALIZED,
                        is_ambiguous=True,
                        candidates=cand_dicts,
                        match_explanation=f"Ambiguous partial match: multiple entities matched query term '{target_norm}'",
                    )
                )
            else:
                unlinked.append(q_ent.mention_text)

        return EntityLinkingResult(
            query=query,
            linked_entities=linked,
            unlinked_mentions=unlinked,
            has_ambiguity=has_ambiguity,
        )
