"""Unit tests for GraphRetriever module."""

from graphrag_bench.entity_linker import MatchType
from graphrag_bench.graph_ranker import GraphRanker
from graphrag_bench.graph_retrieval_schemas import LinkedEntity
from graphrag_bench.graph_retriever import GraphRetriever
from graphrag_bench.graph_schemas import GraphEntity, GraphRelationship
from graphrag_bench.graph_store import GraphStore
from graphrag_bench.graph_traversal import GraphTraversalEngine


def build_retriever_test_store() -> GraphStore:
    """Build a multi-entity test graph."""
    store = GraphStore()
    entities = [
        ("node_vrag", "Vector RAG", "TECHNOLOGY"),
        ("node_grag", "Graph RAG", "TECHNOLOGY"),
        ("node_hrag", "Hybrid RAG", "TECHNOLOGY"),
        ("node_chroma", "ChromaDB", "SOFTWARE"),
        ("node_nx", "NetworkX", "SOFTWARE"),
    ]
    for e_id, name, t in entities:
        store.add_entity(
            GraphEntity(
                entity_id=e_id,
                canonical_name=name,
                entity_type=t,
                source_document_ids=["doc_1"],
                source_chunk_ids=["chunk_1"],
            )
        )

    relationships = [
        ("r1", "node_vrag", "USES", "node_chroma", "doc_1", "c1", 0.95),
        ("r2", "node_grag", "USES", "node_nx", "doc_2", "c2", 0.90),
        ("r3", "node_hrag", "COMBINES", "node_vrag", "doc_3", "c3", 0.85),
        ("r4", "node_hrag", "COMBINES", "node_grag", "doc_3", "c4", 0.88),
    ]
    for r_id, s, p, o, d, c, conf in relationships:
        store.add_relationship(
            GraphRelationship(
                relationship_id=r_id,
                subject_entity_id=s,
                predicate=p,
                object_entity_id=o,
                evidence_quote=f"{s} {p} {o} quotation",
                confidence=conf,
                source_document_id=d,
                source_chunk_id=c,
                source_filename=f"{d}.md",
            )
        )
    return store


class TestGraphRetriever:
    """Test suite for high-level GraphRetriever orchestrator."""

    def test_retrieve_single_seed(self) -> None:
        store = build_retriever_test_store()
        traversal = GraphTraversalEngine(store=store)
        ranker = GraphRanker()
        retriever = GraphRetriever(store=store, traversal=traversal, ranker=ranker)

        seeds = [
            LinkedEntity(
                query_mention="Vector RAG",
                entity_id="node_vrag",
                canonical_name="Vector RAG",
                entity_type="TECHNOLOGY",
                match_type=MatchType.EXACT,
            )
        ]

        results = retriever.retrieve(
            query="Vector RAG usage",
            linked_entities=seeds,
            max_hops=2,
            top_k=5,
        )

        assert len(results) >= 1
        assert results[0].seed_entity_name == "Vector RAG"
        assert results[0].rank == 1

    def test_retrieve_multiple_seeds(self) -> None:
        store = build_retriever_test_store()
        traversal = GraphTraversalEngine(store=store)
        ranker = GraphRanker()
        retriever = GraphRetriever(store=store, traversal=traversal, ranker=ranker)

        seeds = [
            LinkedEntity(
                query_mention="Vector RAG",
                entity_id="node_vrag",
                canonical_name="Vector RAG",
                entity_type="TECHNOLOGY",
                match_type=MatchType.EXACT,
            ),
            LinkedEntity(
                query_mention="Graph RAG",
                entity_id="node_grag",
                canonical_name="Graph RAG",
                entity_type="TECHNOLOGY",
                match_type=MatchType.EXACT,
            ),
        ]

        results = retriever.retrieve(
            query="Compare Vector RAG and Graph RAG",
            linked_entities=seeds,
            max_hops=1,
            top_k=10,
        )

        seed_names = {r.seed_entity_name for r in results}
        assert "Vector RAG" in seed_names
        assert "Graph RAG" in seed_names

    def test_retrieve_with_predicate_filter(self) -> None:
        store = build_retriever_test_store()
        traversal = GraphTraversalEngine(store=store)
        ranker = GraphRanker()
        retriever = GraphRetriever(store=store, traversal=traversal, ranker=ranker)

        seeds = [
            LinkedEntity(
                query_mention="Hybrid RAG",
                entity_id="node_hrag",
                canonical_name="Hybrid RAG",
                entity_type="TECHNOLOGY",
                match_type=MatchType.EXACT,
            )
        ]

        # Only COMBINES predicate
        results = retriever.retrieve(
            query="Hybrid RAG composition",
            linked_entities=seeds,
            max_hops=2,
            predicates=["COMBINES"],
            top_k=10,
        )

        assert len(results) >= 1
        for res in results:
            assert all(e.predicate == "COMBINES" for e in res.path.edges)

    def test_retrieve_with_document_filter(self) -> None:
        store = build_retriever_test_store()
        traversal = GraphTraversalEngine(store=store)
        ranker = GraphRanker()
        retriever = GraphRetriever(store=store, traversal=traversal, ranker=ranker)

        seeds = [
            LinkedEntity(
                query_mention="Hybrid RAG",
                entity_id="node_hrag",
                canonical_name="Hybrid RAG",
                entity_type="TECHNOLOGY",
                match_type=MatchType.EXACT,
            )
        ]

        # Restrict to doc_3
        results = retriever.retrieve(
            query="Hybrid RAG",
            linked_entities=seeds,
            max_hops=2,
            doc_id_filter="doc_3",
            top_k=10,
        )

        for res in results:
            assert all(e.source_document_id == "doc_3" for e in res.path.edges)

    def test_retrieve_empty_seeds_returns_empty_list(self) -> None:
        store = build_retriever_test_store()
        traversal = GraphTraversalEngine(store=store)
        ranker = GraphRanker()
        retriever = GraphRetriever(store=store, traversal=traversal, ranker=ranker)

        results = retriever.retrieve(
            query="Empty seeds query",
            linked_entities=[],
            max_hops=2,
        )
        assert len(results) == 0

    def test_retrieve_respects_top_k_limit(self) -> None:
        store = build_retriever_test_store()
        traversal = GraphTraversalEngine(store=store)
        ranker = GraphRanker()
        retriever = GraphRetriever(store=store, traversal=traversal, ranker=ranker)

        seeds = [
            LinkedEntity(
                query_mention="Hybrid RAG",
                entity_id="node_hrag",
                canonical_name="Hybrid RAG",
                entity_type="TECHNOLOGY",
                match_type=MatchType.EXACT,
            )
        ]

        results = retriever.retrieve(
            query="Hybrid RAG",
            linked_entities=seeds,
            max_hops=2,
            direction="both",
            top_k=1,
        )
        assert len(results) <= 1
