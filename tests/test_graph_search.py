"""Unit tests for GraphSearchEngine and search_graph entry points."""

from unittest.mock import MagicMock

from graphrag_bench.graph_retrieval_schemas import (
    GraphSearchStatus,
    QueryEntityExtractionResult,
)
from graphrag_bench.graph_schemas import GraphEntity, GraphRelationship
from graphrag_bench.graph_search import GraphSearchEngine, search_graph
from graphrag_bench.graph_store import GraphStore


def build_search_test_store() -> GraphStore:
    """Build a test GraphStore for end-to-end search orchestration tests."""
    store = GraphStore()
    store.add_entity(
        GraphEntity(
            entity_id="ent_vrag",
            canonical_name="Vector RAG",
            entity_type="TECHNOLOGY",
            aliases=["VRAG"],
            source_document_ids=["doc_1"],
            source_chunk_ids=["chunk_1"],
        )
    )
    store.add_entity(
        GraphEntity(
            entity_id="ent_chroma",
            canonical_name="ChromaDB",
            entity_type="SOFTWARE",
            aliases=["Chroma"],
            source_document_ids=["doc_1"],
            source_chunk_ids=["chunk_1"],
        )
    )
    store.add_relationship(
        GraphRelationship(
            relationship_id="rel_1",
            subject_entity_id="ent_vrag",
            predicate="USES",
            object_entity_id="ent_chroma",
            evidence_quote="Vector RAG uses ChromaDB for high-throughput semantic similarity search.",
            confidence=0.95,
            source_document_id="doc_1",
            source_chunk_id="chunk_1",
            source_filename="guide.md",
        )
    )
    return store


class TestGraphSearch:
    """Test suite for GraphSearchEngine end-to-end querying and error resilience."""

    def test_search_graph_success(self) -> None:
        store = build_search_test_store()
        engine = GraphSearchEngine(store=store)

        resp = engine.search_graph("How does Vector RAG store embeddings?")

        assert resp.status == GraphSearchStatus.SUCCESS
        assert len(resp.linked_entities) >= 1
        assert len(resp.results) >= 1
        assert resp.results[0].seed_entity_name == "Vector RAG"
        assert len(resp.structured_context) > 0
        assert "ChromaDB" in resp.structured_context
        assert resp.latency_ms > 0
        assert resp.statistics["paths_retrieved"] == len(resp.results)

    def test_search_graph_empty_query_returns_invalid_query_status(self) -> None:
        store = build_search_test_store()
        engine = GraphSearchEngine(store=store)

        resp = engine.search_graph("    ")
        assert resp.status == GraphSearchStatus.INVALID_QUERY
        assert len(resp.results) == 0

    def test_search_graph_empty_store_returns_unavailable(self) -> None:
        empty_store = GraphStore()
        engine = GraphSearchEngine(store=empty_store)

        resp = engine.search_graph("Vector RAG")
        assert resp.status == GraphSearchStatus.GRAPH_UNAVAILABLE
        assert len(resp.results) == 0

    def test_search_graph_no_matching_entities(self) -> None:
        store = build_search_test_store()
        engine = GraphSearchEngine(store=store)
        engine.extractor.extract_query_entities = MagicMock(
            return_value=QueryEntityExtractionResult(
                query="What is Quantum Entanglement?", entities=[]
            )
        )

        resp = engine.search_graph("What is Quantum Entanglement?")
        assert resp.status == GraphSearchStatus.NO_MATCHING_ENTITIES
        assert len(resp.results) == 0

    def test_search_graph_convenience_function(self) -> None:
        store = build_search_test_store()
        resp = search_graph(
            query="Tell me about ChromaDB",
            direction="both",
            store=store,
        )

        assert resp.status == GraphSearchStatus.SUCCESS
        assert any(e.canonical_name == "ChromaDB" for e in resp.linked_entities)
        assert len(resp.results) >= 1

    def test_search_response_to_export_dict(self) -> None:
        store = build_search_test_store()
        engine = GraphSearchEngine(store=store)

        resp = engine.search_graph("VRAG and ChromaDB")
        export_dict = resp.to_export_dict()

        assert isinstance(export_dict, dict)
        assert "query" in export_dict
        assert "status" in export_dict
        assert "results" in export_dict
        assert "structured_context" in export_dict
        assert export_dict["status"] == "SUCCESS"
