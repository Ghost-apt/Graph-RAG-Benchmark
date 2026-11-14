"""Entity Resolution and Disambiguation Engine for Knowledge Graph Construction.

This module resolves and consolidates raw entity mentions across chunks and documents into
canonical GraphEntity nodes:
- Unicode NFKC normalization and case-folding.
- Type-compatibility validation: prevents merging entities of differing types.
- Deterministic entity ID generation.
- Alias indexing and multi-mention resolution.
- Provenance tracking (mention counts, source document IDs, source chunk IDs).
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from collections.abc import Sequence

from graphrag_bench.graph_schemas import GraphEntity, normalize_entity_type

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Normalize text using NFKC, lowercase folding, and collapsed whitespace."""
    if not text:
        return ""
    nfkc = unicodedata.normalize("NFKC", text)
    folded = nfkc.lower().strip()
    return re.sub(r"\s+", " ", folded)


def generate_entity_id(entity_type: str, normalized_name: str) -> str:
    """Generate deterministic entity ID based on type and normalized canonical name.

    Ensures that different entity types with the same name (e.g. Java [PROGRAMMING_LANGUAGE]
    vs Java [LOCATION]) generate distinct entity IDs and cannot accidentally collide.
    """
    clean_type = normalize_entity_type(entity_type).lower()
    clean_name = normalize_text(normalized_name)
    digest = hashlib.sha256(f"{clean_type}:{clean_name}".encode()).hexdigest()[:12]
    return f"ent_{clean_type}_{digest}"


class EntityResolver:
    """Resolves raw entity mentions to canonical GraphEntity nodes with provenance tracking."""

    def __init__(self) -> None:
        # Primary storage: entity_id -> GraphEntity
        self._entities: dict[str, GraphEntity] = {}

        # Lookup index: (normalized_string, normalized_entity_type) -> entity_id
        # Key can be normalized canonical name, mention text, or registered alias.
        self._lookup: dict[tuple[str, str], str] = {}

    def _index_keys_for_entity(self, entity: GraphEntity) -> None:
        """Register canonical name, aliases, and known keys in lookup index."""
        etype = entity.entity_type
        # Index canonical name
        canon_norm = normalize_text(entity.canonical_name)
        if canon_norm:
            self._lookup[(canon_norm, etype)] = entity.entity_id

        # Index aliases
        for alias in entity.aliases:
            alias_norm = normalize_text(alias)
            if alias_norm:
                self._lookup[(alias_norm, etype)] = entity.entity_id

    def resolve_entity(
        self,
        mention_text: str,
        canonical_name: str,
        entity_type: str,
        aliases: Sequence[str] | None = None,
        doc_id: str = "",
        chunk_id: str = "",
        description: str = "",
    ) -> GraphEntity:
        """Resolve a mention to an existing canonical entity or create a new GraphEntity.

        Args:
            mention_text: Verbatim mention in source text.
            canonical_name: Canonical name extracted or proposed.
            entity_type: Entity category (e.g. 'CONCEPT', 'TECHNOLOGY').
            aliases: Optional list of alternate names.
            doc_id: Source document ID for provenance.
            chunk_id: Source chunk ID for provenance.
            description: Grounded description of the entity.

        Returns:
            Resolved GraphEntity (updated if existing, newly created if not).
        """
        clean_type = normalize_entity_type(entity_type)
        norm_mention = normalize_text(mention_text)
        norm_canon = normalize_text(canonical_name) or norm_mention

        if not norm_canon:
            raise ValueError("Entity canonical_name and mention_text cannot both be empty.")

        # Prepare alias set
        raw_aliases = list(aliases) if aliases else []
        if norm_mention and norm_mention != norm_canon:
            raw_aliases.append(mention_text.strip())

        # Check existing match in lookup table with STRICT type matching
        matched_id: str | None = None

        # 1. Match by canonical name + type
        if (norm_canon, clean_type) in self._lookup:
            matched_id = self._lookup[(norm_canon, clean_type)]
        # 2. Match by mention text + type
        elif (norm_mention, clean_type) in self._lookup:
            matched_id = self._lookup[(norm_mention, clean_type)]
        else:
            # 3. Match by any alias + type
            for a in raw_aliases:
                a_norm = normalize_text(a)
                if (a_norm, clean_type) in self._lookup:
                    matched_id = self._lookup[(a_norm, clean_type)]
                    break

        if matched_id and matched_id in self._entities:
            existing = self._entities[matched_id]
            # Update existing entity
            existing.mention_count += 1

            # Merge source provenance
            if doc_id and doc_id not in existing.source_document_ids:
                existing.source_document_ids.append(doc_id)
            if chunk_id and chunk_id not in existing.source_chunk_ids:
                existing.source_chunk_ids.append(chunk_id)

            # Merge aliases
            existing_aliases_norm = {normalize_text(a) for a in existing.aliases}
            for a in raw_aliases:
                a_clean = a.strip()
                if a_clean and normalize_text(a_clean) not in existing_aliases_norm:
                    existing.aliases.append(a_clean)
                    existing_aliases_norm.add(normalize_text(a_clean))
                    self._lookup[(normalize_text(a_clean), clean_type)] = existing.entity_id

            # Merge description: preserve the richer/longer grounded description
            if description and len(description.strip()) > len(existing.description.strip()):
                existing.description = description.strip()

            return existing

        # Create new entity
        entity_id = generate_entity_id(clean_type, norm_canon)

        # Fallback check if ID already exists (e.g. loaded without full index)
        if entity_id in self._entities:
            existing = self._entities[entity_id]
            existing.mention_count += 1
            if doc_id and doc_id not in existing.source_document_ids:
                existing.source_document_ids.append(doc_id)
            if chunk_id and chunk_id not in existing.source_chunk_ids:
                existing.source_chunk_ids.append(chunk_id)
            self._index_keys_for_entity(existing)
            return existing

        clean_aliases: list[str] = []
        seen_aliases: set[str] = {norm_canon}
        for a in raw_aliases:
            a_clean = a.strip()
            a_norm = normalize_text(a_clean)
            if a_norm and a_norm not in seen_aliases:
                clean_aliases.append(a_clean)
                seen_aliases.add(a_norm)

        new_entity = GraphEntity(
            entity_id=entity_id,
            canonical_name=canonical_name.strip() or mention_text.strip(),
            entity_type=clean_type,
            description=description.strip(),
            aliases=clean_aliases,
            mention_count=1,
            source_document_ids=[doc_id] if doc_id else [],
            source_chunk_ids=[chunk_id] if chunk_id else [],
        )

        self._entities[entity_id] = new_entity
        self._index_keys_for_entity(new_entity)
        if norm_mention:
            self._lookup[(norm_mention, clean_type)] = entity_id

        return new_entity

    def find_entity(
        self,
        name_or_mention: str,
        entity_type: str | None = None,
    ) -> GraphEntity | None:
        """Find an entity by canonical name, mention, or alias.

        Args:
            name_or_mention: The name, mention, or alias to look up.
            entity_type: Optional entity type. If omitted, searches across all types,
                         disambiguating by mention count if multiple candidates match.

        Returns:
            Matching GraphEntity or None.
        """
        norm = normalize_text(name_or_mention)
        if not norm:
            return None

        if entity_type is not None:
            clean_type = normalize_entity_type(entity_type)
            eid = self._lookup.get((norm, clean_type))
            if eid and eid in self._entities:
                return self._entities[eid]
            return None

        # Search across types
        candidates: list[GraphEntity] = []
        for (k_norm, _k_type), eid in self._lookup.items():
            if k_norm == norm and eid in self._entities:
                candidates.append(self._entities[eid])

        if not candidates:
            return None

        # Disambiguate by highest mention count
        candidates.sort(key=lambda e: e.mention_count, reverse=True)
        return candidates[0]

    def get_entity(self, entity_id: str) -> GraphEntity | None:
        """Retrieve entity by its deterministic entity ID."""
        return self._entities.get(entity_id)

    def get_all_entities(self) -> list[GraphEntity]:
        """Return list of all registered GraphEntity objects."""
        return list(self._entities.values())

    def register_alias(self, entity_id: str, alias: str) -> bool:
        """Manually register an alias for an existing entity."""
        entity = self._entities.get(entity_id)
        if not entity:
            return False
        clean_alias = alias.strip()
        alias_norm = normalize_text(clean_alias)
        if not alias_norm:
            return False

        if clean_alias not in entity.aliases:
            entity.aliases.append(clean_alias)
        self._lookup[(alias_norm, entity.entity_type)] = entity.entity_id
        return True

    def remove_document(
        self,
        doc_id: str,
        chunk_ids: Sequence[str] | None = None,
    ) -> tuple[int, list[str]]:
        """Remove document and chunk references from entities.

        Returns:
            Tuple of (entities_modified_count, list_of_orphan_entity_ids).
            Orphan entities have 0 remaining source documents and 0 chunk references.
        """
        target_chunks = set(chunk_ids) if chunk_ids else set()
        modified_count = 0
        orphan_ids: list[str] = []

        for entity in list(self._entities.values()):
            changed = False
            if doc_id in entity.source_document_ids:
                entity.source_document_ids.remove(doc_id)
                changed = True

            if target_chunks:
                before_len = len(entity.source_chunk_ids)
                entity.source_chunk_ids = [
                    cid for cid in entity.source_chunk_ids if cid not in target_chunks
                ]
                if len(entity.source_chunk_ids) < before_len:
                    changed = True

            if changed:
                modified_count += 1
                if not entity.source_document_ids and not entity.source_chunk_ids:
                    orphan_ids.append(entity.entity_id)

        # Remove orphan entities
        for oid in orphan_ids:
            self._remove_entity(oid)

        return modified_count, orphan_ids

    def _remove_entity(self, entity_id: str) -> None:
        """Internal helper to safely remove an entity and all its lookup references."""
        entity = self._entities.pop(entity_id, None)
        if not entity:
            return
        keys_to_remove = [k for k, v in self._lookup.items() if v == entity_id]
        for k in keys_to_remove:
            self._lookup.pop(k, None)

    def load_entities(self, entities: Sequence[GraphEntity]) -> None:
        """Load an existing list of GraphEntities into memory and rebuild lookup indices."""
        self.clear()
        for ent in entities:
            self._entities[ent.entity_id] = ent
            self._index_keys_for_entity(ent)

    def clear(self) -> None:
        """Clear all entities and lookup tables."""
        self._entities.clear()
        self._lookup.clear()

    def __len__(self) -> int:
        return len(self._entities)
