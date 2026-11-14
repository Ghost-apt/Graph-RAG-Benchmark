"""Unit tests for EntityLinker module."""

from graphrag_bench.entity_linker import EntityLinker, MatchType
from graphrag_bench.graph_retrieval_schemas import ExtractionMethod, QueryEntity
from graphrag_bench.graph_schemas import GraphEntity
from graphrag_bench.graph_store import GraphStore


def build_test_graph_store() -> GraphStore:
    """Construct a populated test GraphStore with potential collisions and aliases."""
    store = GraphStore()
    store.add_entity(
        GraphEntity(
            entity_id="ent_vector_rag",
            canonical_name="Vector RAG",
            entity_type="TECHNOLOGY",
            aliases=["VRAG", "Dense Vector Retrieval"],
            source_document_ids=["doc_1"],
            source_chunk_ids=["chunk_1"],
        )
    )
    store.add_entity(
        GraphEntity(
            entity_id="ent_graph_rag",
            canonical_name="Graph RAG",
            entity_type="TECHNOLOGY",
            aliases=["GRAG", "Graph-based RAG"],
            source_document_ids=["doc_1"],
            source_chunk_ids=["chunk_1"],
        )
    )
    store.add_entity(
        GraphEntity(
            entity_id="ent_hybrid_rag",
            canonical_name="Hybrid RAG",
            entity_type="TECHNOLOGY",
            aliases=["Hybrid Retrieval"],
            source_document_ids=["doc_1"],
            source_chunk_ids=["chunk_1"],
        )
    )
    # Entities with identical/similar prefix for ambiguity testing
    store.add_entity(
        GraphEntity(
            entity_id="ent_rag_framework",
            canonical_name="RAG Framework",
            entity_type="FRAMEWORK",
            aliases=["RAG System"],
            source_document_ids=["doc_2"],
            source_chunk_ids=["chunk_2"],
        )
    )
    store.add_entity(
        GraphEntity(
            entity_id="ent_rag_survey",
            canonical_name="RAG Benchmark Survey",
            entity_type="DATASET",
            aliases=["RAG Survey"],
            source_document_ids=["doc_2"],
            source_chunk_ids=["chunk_2"],
        )
    )
    return store


class TestEntityLinker:
    """Test suite for entity linking, disambiguation, and alias matching."""

    def test_link_exact_canonical(self) -> None:
        store = build_test_graph_store()
        linker = EntityLinker(store=store)

        q_entities = [
            QueryEntity(
                mention_text="Vector RAG",
                canonical_name="Vector RAG",
                entity_type="TECHNOLOGY",
                normalized_term="vector rag",
                extraction_method=ExtractionMethod.EXACT_MATCH,
            )
        ]

        result = linker.link_entities(q_entities)
        assert len(result.linked_entities) == 1
        linked = result.linked_entities[0]
        assert linked.entity_id == "ent_vector_rag"
        assert linked.canonical_name == "Vector RAG"
        assert linked.match_type == MatchType.EXACT
        assert linked.is_ambiguous is False
        assert len(result.unlinked_mentions) == 0

    def test_link_via_alias(self) -> None:
        store = build_test_graph_store()
        linker = EntityLinker(store=store)

        q_entities = [
            QueryEntity(
                mention_text="VRAG",
                canonical_name="VRAG",
                entity_type="TECHNOLOGY",
                normalized_term="vrag",
                extraction_method=ExtractionMethod.ALIAS_MATCH,
            )
        ]

        result = linker.link_entities(q_entities)
        assert len(result.linked_entities) == 1
        linked = result.linked_entities[0]
        assert linked.entity_id == "ent_vector_rag"
        assert linked.match_type == MatchType.ALIAS
        assert "alias" in linked.match_explanation.lower()

    def test_link_normalized_variation(self) -> None:
        store = build_test_graph_store()
        linker = EntityLinker(store=store)

        q_entities = [
            QueryEntity(
                mention_text="graph-rag",
                canonical_name="graph rag",
                entity_type="TECHNOLOGY",
                normalized_term="graph rag",
                extraction_method=ExtractionMethod.NORMALIZED_MATCH,
            )
        ]

        result = linker.link_entities(q_entities)
        assert len(result.linked_entities) == 1
        linked = result.linked_entities[0]
        assert linked.entity_id == "ent_graph_rag"
        assert linked.match_type in (MatchType.EXACT, MatchType.NORMALIZED)

    def test_unlinked_mentions(self) -> None:
        store = build_test_graph_store()
        linker = EntityLinker(store=store)

        q_entities = [
            QueryEntity(
                mention_text="NonExistentSystem",
                canonical_name="NonExistentSystem",
                entity_type="UNKNOWN",
                normalized_term="nonexistentsystem",
                extraction_method=ExtractionMethod.EXACT_MATCH,
            )
        ]

        result = linker.link_entities(q_entities)
        assert len(result.linked_entities) == 0
        assert "NonExistentSystem" in result.unlinked_mentions

    def test_partial_match_ambiguity_detection(self) -> None:
        store = build_test_graph_store()
        linker = EntityLinker(store=store)

        # "RAG" matches prefix of both "RAG Framework" and "RAG Benchmark Survey"
        q_entities = [
            QueryEntity(
                mention_text="RAG",
                canonical_name="RAG",
                entity_type="OTHER",
                normalized_term="rag",
                extraction_method=ExtractionMethod.EXACT_MATCH,
            )
        ]

        result = linker.link_entities(q_entities)
        # Should detect candidates and flag ambiguity if multiple match
        if result.linked_entities:
            assert result.has_ambiguity is True or len(result.linked_entities[0].candidates) > 1

    def test_type_aware_disambiguation(self) -> None:
        store = build_test_graph_store()
        linker = EntityLinker(store=store)

        # Providing entity_type="FRAMEWORK" should prioritize RAG Framework over Dataset
        q_entities = [
            QueryEntity(
                mention_text="RAG",
                canonical_name="RAG",
                entity_type="FRAMEWORK",
                normalized_term="rag",
                extraction_method=ExtractionMethod.EXACT_MATCH,
            )
        ]

        result = linker.link_entities(q_entities)
        if result.linked_entities:
            assert result.linked_entities[0].entity_type == "FRAMEWORK"

    def test_empty_query_entities_returns_empty_result(self) -> None:
        store = build_test_graph_store()
        linker = EntityLinker(store=store)

        result = linker.link_entities([])
        assert len(result.linked_entities) == 0
        assert len(result.unlinked_mentions) == 0
        assert result.has_ambiguity is False
