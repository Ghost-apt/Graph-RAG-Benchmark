"""Unit tests for EntityResolver in Knowledge Graph Construction Engine."""

from graphrag_bench.entity_resolver import (
    EntityResolver,
    generate_entity_id,
    normalize_text,
)
from graphrag_bench.graph_schemas import GraphEntity


def test_normalize_text():
    assert normalize_text("  Graph   RAG  ") == "graph rag"
    assert normalize_text("Retrieval-Augmented Generation") == "retrieval-augmented generation"
    assert normalize_text("") == ""
    # Unicode NFKC normalization
    assert normalize_text("Ｋｎｏｗｌｅｄｇｅ") == "knowledge"


def test_generate_entity_id():
    id1 = generate_entity_id("CONCEPT", "Knowledge Graph")
    id2 = generate_entity_id("concept", "  knowledge graph  ")
    assert id1 == id2
    assert id1.startswith("ent_concept_")

    # Different type must produce different ID even with same name
    id_lang = generate_entity_id("PROGRAMMING_LANGUAGE", "Python")
    id_tech = generate_entity_id("TECHNOLOGY", "Python")
    assert id_lang != id_tech
    assert id_lang.startswith("ent_programming_language_")
    assert id_tech.startswith("ent_technology_")


def test_type_compatibility_prevents_incorrect_merge():
    """Verify entities with identical names but differing types remain distinct."""
    resolver = EntityResolver()

    ent_lang = resolver.resolve_entity(
        mention_text="Java",
        canonical_name="Java",
        entity_type="PROGRAMMING_LANGUAGE",
        doc_id="doc_1",
        chunk_id="chunk_1",
        description="Object-oriented programming language",
    )

    ent_loc = resolver.resolve_entity(
        mention_text="Java",
        canonical_name="Java",
        entity_type="LOCATION",
        doc_id="doc_2",
        chunk_id="chunk_2",
        description="An island of Indonesia",
    )

    assert ent_lang.entity_id != ent_loc.entity_id
    assert ent_lang.entity_type == "PROGRAMMING_LANGUAGE"
    assert ent_loc.entity_type == "LOCATION"
    assert len(resolver) == 2

    # Querying with type retrieves exact entity
    found_lang = resolver.find_entity("Java", "PROGRAMMING_LANGUAGE")
    found_loc = resolver.find_entity("Java", "LOCATION")
    assert found_lang.entity_id == ent_lang.entity_id
    assert found_loc.entity_id == ent_loc.entity_id


def test_multi_mention_consolidation():
    """Verify that multiple mentions of the same entity consolidate provenance and mention count."""
    resolver = EntityResolver()

    # First mention
    e1 = resolver.resolve_entity(
        mention_text="ChromaDB",
        canonical_name="ChromaDB",
        entity_type="SOFTWARE",
        doc_id="doc_a",
        chunk_id="chunk_1",
        description="Vector database",
    )
    assert e1.mention_count == 1
    assert e1.source_document_ids == ["doc_a"]
    assert e1.source_chunk_ids == ["chunk_1"]

    # Second mention from another chunk in the same document
    e2 = resolver.resolve_entity(
        mention_text="chromadb",
        canonical_name="ChromaDB",
        entity_type="SOFTWARE",
        doc_id="doc_a",
        chunk_id="chunk_2",
        description="Open-source embedding database for AI",
    )
    assert e2.entity_id == e1.entity_id
    assert e2.mention_count == 2
    assert e2.source_document_ids == ["doc_a"]
    assert e2.source_chunk_ids == ["chunk_1", "chunk_2"]
    # Richer description preserved
    assert "Open-source embedding database for AI" in e2.description

    # Third mention from another document
    e3 = resolver.resolve_entity(
        mention_text="Chroma",
        canonical_name="ChromaDB",
        entity_type="SOFTWARE",
        aliases=["Chroma"],
        doc_id="doc_b",
        chunk_id="chunk_3",
    )
    assert e3.entity_id == e1.entity_id
    assert e3.mention_count == 3
    assert set(e3.source_document_ids) == {"doc_a", "doc_b"}
    assert set(e3.source_chunk_ids) == {"chunk_1", "chunk_2", "chunk_3"}
    assert "Chroma" in e3.aliases
    assert len(resolver) == 1


def test_alias_resolution():
    """Verify resolving through known alias maps back to canonical entity."""
    resolver = EntityResolver()

    ent = resolver.resolve_entity(
        mention_text="Retrieval-Augmented Generation",
        canonical_name="Retrieval-Augmented Generation",
        entity_type="CONCEPT",
        aliases=["RAG"],
        doc_id="doc_1",
        chunk_id="chunk_1",
        description="Architecture combining retrieval and generation",
    )

    # Resolve using alias
    resolved = resolver.resolve_entity(
        mention_text="RAG",
        canonical_name="RAG",
        entity_type="CONCEPT",
        doc_id="doc_2",
        chunk_id="chunk_2",
    )

    assert resolved.entity_id == ent.entity_id
    assert resolved.mention_count == 2
    assert "doc_2" in resolved.source_document_ids


def test_find_entity_and_disambiguation():
    resolver = EntityResolver()

    e1 = resolver.resolve_entity(
        mention_text="BERT",
        canonical_name="BERT",
        entity_type="TECHNOLOGY",
        doc_id="doc_1",
        chunk_id="chunk_1",
    )
    # Mention it again to boost mention count
    resolver.resolve_entity(
        mention_text="BERT",
        canonical_name="BERT",
        entity_type="TECHNOLOGY",
        doc_id="doc_2",
        chunk_id="chunk_2",
    )

    assert resolver.find_entity("bert") == e1
    assert resolver.find_entity("BERT", "TECHNOLOGY") == e1
    assert resolver.find_entity("BERT", "LOCATION") is None
    assert resolver.find_entity("nonexistent") is None


def test_register_alias():
    resolver = EntityResolver()
    e = resolver.resolve_entity(
        mention_text="NetworkX",
        canonical_name="NetworkX",
        entity_type="FRAMEWORK",
    )

    assert resolver.register_alias(e.entity_id, "nx") is True
    assert "nx" in e.aliases
    assert resolver.find_entity("nx", "FRAMEWORK") == e
    assert resolver.register_alias("invalid_id", "alias") is False


def test_remove_document_and_orphan_cleanup():
    resolver = EntityResolver()

    # Entity 1 only in doc_1
    e1 = resolver.resolve_entity(
        mention_text="Sole Entity",
        canonical_name="Sole Entity",
        entity_type="CONCEPT",
        doc_id="doc_1",
        chunk_id="chunk_1",
    )

    # Entity 2 in doc_1 and doc_2
    e2 = resolver.resolve_entity(
        mention_text="Shared Entity",
        canonical_name="Shared Entity",
        entity_type="CONCEPT",
        doc_id="doc_1",
        chunk_id="chunk_1",
    )
    resolver.resolve_entity(
        mention_text="Shared Entity",
        canonical_name="Shared Entity",
        entity_type="CONCEPT",
        doc_id="doc_2",
        chunk_id="chunk_2",
    )

    assert len(resolver) == 2

    # Remove doc_1
    modified, orphans = resolver.remove_document("doc_1", chunk_ids=["chunk_1"])
    assert modified == 2
    assert e1.entity_id in orphans
    # e1 should have been purged because it became orphan
    assert resolver.get_entity(e1.entity_id) is None
    # e2 remains because it still has doc_2
    assert resolver.get_entity(e2.entity_id) is not None
    assert len(resolver) == 1


def test_load_entities_rebuilds_indices():
    resolver = EntityResolver()
    entities = [
        GraphEntity(
            entity_id="ent_concept_1234",
            canonical_name="Hybrid Search",
            entity_type="CONCEPT",
            description="Combining dense and sparse search",
            aliases=["Hybrid Retrieval"],
            mention_count=3,
            source_document_ids=["doc_1"],
            source_chunk_ids=["chunk_1"],
        )
    ]
    resolver.load_entities(entities)
    assert len(resolver) == 1
    assert resolver.find_entity("hybrid search") is not None
    assert resolver.find_entity("hybrid retrieval") is not None
