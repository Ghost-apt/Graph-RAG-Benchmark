"""Unit tests for HybridRetriever coordinator and orchestrator."""

from unittest.mock import MagicMock

from graphrag_bench.config import Settings
from graphrag_bench.document_repository import DocumentRepository
from graphrag_bench.graph_retrieval_schemas import (
    GraphPath,
    GraphRetrievalResult,
    GraphSearchResponse,
    GraphSearchStatus,
    PathEdge,
    PathNode,
)
from graphrag_bench.graph_search import GraphSearchEngine
from graphrag_bench.hybrid_retriever import HybridRetriever, search_hybrid
from graphrag_bench.hybrid_schemas import HybridSearchStatus, RetrievalSource
from graphrag_bench.schemas import DocumentChunk
from graphrag_bench.semantic_search import SearchResponse, SemanticSearch
from graphrag_bench.vector_store import SearchResultItem


def build_mock_services() -> tuple[MagicMock, MagicMock, MagicMock, Settings]:
    """Build mock vector search, graph search, repository, and settings."""
    settings = Settings(
        hybrid_top_k=5,
        hybrid_vector_top_k=5,
        hybrid_graph_top_k=5,
        hybrid_rrf_k=60,
        hybrid_vector_weight=1.0,
        hybrid_graph_weight=1.0,
        hybrid_parallel_retrieval=False,  # Sequential for deterministic test execution
    )

    repo = MagicMock(spec=DocumentRepository)
    chunk1 = DocumentChunk(
        chunk_id="chunk_1",
        document_id="doc_1",
        chunk_index=0,
        start_char=0,
        end_char=100,
        text="Content of canonical chunk 1.",
        source_filename="doc1.md",
        file_type="md",
        page_number=1,
    )
    chunk2 = DocumentChunk(
        chunk_id="chunk_2",
        document_id="doc_2",
        chunk_index=0,
        start_char=0,
        end_char=100,
        text="Content of canonical chunk 2.",
        source_filename="doc2.txt",
        file_type="txt",
        page_number=2,
    )
    repo._get_all_chunks.return_value = [chunk1, chunk2]

    vector_search = MagicMock(spec=SemanticSearch)
    vector_search.search.return_value = SearchResponse(
        query="test query",
        top_k=5,
        total_found=1,
        duration_ms=5.0,
        items=[
            SearchResultItem(
                rank=1,
                chunk_id="chunk_1",
                document_id="doc_1",
                text="Content of canonical chunk 1.",
                distance=0.1,
                cosine_similarity=0.95,
                normalized_score=0.95,
                metadata={"source_filename": "doc1.md", "file_type": "md", "page_number": 1},
            )
        ],
        model_used="nomic-embed-text",
    )

    graph_search = MagicMock(spec=GraphSearchEngine)
    edge = PathEdge(
        relationship_id="rel_1",
        subject_entity_id="node_a",
        predicate="LINKS",
        object_entity_id="node_b",
        direction="outgoing",
        confidence=0.9,
        evidence_quote="Node A links Node B",
        source_document_id="doc_2",
        source_chunk_id="chunk_2",
        source_filename="doc2.txt",
    )
    path = GraphPath(
        path_id="p1",
        seed_entity_id="node_a",
        seed_entity_name="Node A",
        hop_count=1,
        nodes=[
            PathNode(
                entity_id="node_a", canonical_name="Node A", entity_type="CONCEPT", hop_index=0
            ),
            PathNode(
                entity_id="node_b", canonical_name="Node B", entity_type="CONCEPT", hop_index=1
            ),
        ],
        edges=[edge],
        path_score=0.9,
    )
    graph_search.search_graph.return_value = GraphSearchResponse(
        query="test query",
        status=GraphSearchStatus.SUCCESS,
        message="Success",
        linked_entities=[],
        results=[
            GraphRetrievalResult(
                result_id="res_1",
                rank=1,
                seed_entity_name="Node A",
                path=path,
                score=0.9,
            )
        ],
        execution_time_ms=8.0,
    )

    return vector_search, graph_search, repo, settings


class TestHybridRetriever:
    """Test suite for HybridRetriever coordination, error isolation, and filtering."""

    def test_search_empty_query_returns_invalid_query_status(self) -> None:
        """Verify empty query returns INVALID_QUERY without executing retrieval."""
        v_search, g_search, repo, settings = build_mock_services()
        retriever = HybridRetriever(
            vector_search=v_search,
            graph_search=g_search,
            repository=repo,
            settings=settings,
        )

        resp = retriever.search(query="   ")
        assert resp.status == HybridSearchStatus.INVALID_QUERY
        assert resp.fused_candidates == []
        assert not v_search.search.called
        assert not g_search.search_graph.called

    def test_search_both_engines_succeed(self) -> None:
        """Verify normal hybrid search fusing results from both engines."""
        v_search, g_search, repo, settings = build_mock_services()
        retriever = HybridRetriever(
            vector_search=v_search,
            graph_search=g_search,
            repository=repo,
            settings=settings,
        )

        resp = retriever.search(query="What is Graph RAG?")
        assert resp.status == HybridSearchStatus.SUCCESS
        assert resp.vector_status == "SUCCESS"
        assert resp.graph_status == "SUCCESS"
        assert resp.total_candidates == 2
        # chunk_1 is VECTOR, chunk_2 is GRAPH
        sources = {c.source for c in resp.fused_candidates}
        assert RetrievalSource.VECTOR in sources
        assert RetrievalSource.GRAPH in sources
        assert resp.structured_context != ""

    def test_search_vector_fails_graph_succeeds_partial_success(self) -> None:
        """Verify error isolation: vector failure results in PARTIAL_SUCCESS with graph candidates."""
        v_search, g_search, repo, settings = build_mock_services()
        v_search.search.side_effect = RuntimeError("Vector DB connection timeout")

        retriever = HybridRetriever(
            vector_search=v_search,
            graph_search=g_search,
            repository=repo,
            settings=settings,
        )

        resp = retriever.search(query="What is Graph RAG?")
        assert resp.status == HybridSearchStatus.PARTIAL_SUCCESS
        assert resp.vector_status == "FAILED"
        assert resp.graph_status == "SUCCESS"
        assert len(resp.fused_candidates) == 1
        assert resp.fused_candidates[0].source == RetrievalSource.GRAPH
        assert any("Vector search failed" in w for w in resp.warnings)

    def test_search_graph_fails_vector_succeeds_partial_success(self) -> None:
        """Verify error isolation: graph failure results in PARTIAL_SUCCESS with vector candidates."""
        v_search, g_search, repo, settings = build_mock_services()
        g_search.search_graph.side_effect = ValueError("Graph traversal error")

        retriever = HybridRetriever(
            vector_search=v_search,
            graph_search=g_search,
            repository=repo,
            settings=settings,
        )

        resp = retriever.search(query="What is Graph RAG?")
        assert resp.status == HybridSearchStatus.PARTIAL_SUCCESS
        assert resp.vector_status == "SUCCESS"
        assert resp.graph_status == "FAILED"
        assert len(resp.fused_candidates) == 1
        assert resp.fused_candidates[0].source == RetrievalSource.VECTOR
        assert any("Graph search failed" in w for w in resp.warnings)

    def test_search_both_engines_fail(self) -> None:
        """Verify that when both engines fail, ALL_ENGINES_FAILED is returned gracefully."""
        v_search, g_search, repo, settings = build_mock_services()
        v_search.search.side_effect = RuntimeError("Vector DB dead")
        g_search.search_graph.side_effect = RuntimeError("Graph dead")

        retriever = HybridRetriever(
            vector_search=v_search,
            graph_search=g_search,
            repository=repo,
            settings=settings,
        )

        resp = retriever.search(query="What is Graph RAG?")
        assert resp.status == HybridSearchStatus.ALL_ENGINES_FAILED
        assert resp.vector_status == "FAILED"
        assert resp.graph_status == "FAILED"
        assert resp.fused_candidates == []
        assert len(resp.warnings) >= 2

    def test_search_filtering_by_document_id(self) -> None:
        """Verify document_ids filtering retains only candidates from specified documents."""
        v_search, g_search, repo, settings = build_mock_services()
        retriever = HybridRetriever(
            vector_search=v_search,
            graph_search=g_search,
            repository=repo,
            settings=settings,
        )

        resp = retriever.search(query="What is Graph RAG?", document_ids=["doc_1"])
        assert len(resp.fused_candidates) == 1
        assert resp.fused_candidates[0].document_id == "doc_1"

    def test_search_filtering_by_file_type(self) -> None:
        """Verify file_types filtering retains only candidates with matching extensions."""
        v_search, g_search, repo, settings = build_mock_services()
        retriever = HybridRetriever(
            vector_search=v_search,
            graph_search=g_search,
            repository=repo,
            settings=settings,
        )

        resp = retriever.search(query="What is Graph RAG?", file_types=["md"])
        assert len(resp.fused_candidates) == 1
        assert resp.fused_candidates[0].source_filename.endswith(".md")

    def test_search_final_top_k_bounding(self) -> None:
        """Verify final_top_k limits the number of fused candidates returned."""
        v_search, g_search, repo, settings = build_mock_services()
        retriever = HybridRetriever(
            vector_search=v_search,
            graph_search=g_search,
            repository=repo,
            settings=settings,
        )

        resp = retriever.search(query="What is Graph RAG?", final_top_k=1)
        assert len(resp.fused_candidates) == 1

    def test_search_hybrid_functional_api(self) -> None:
        """Verify standalone search_hybrid() functional wrapper works identically."""
        v_search, g_search, repo, settings = build_mock_services()
        resp = search_hybrid(
            query="Functional API query",
            vector_search=v_search,
            graph_search=g_search,
            repository=repo,
            settings=settings,
        )
        assert resp.status == HybridSearchStatus.SUCCESS
        assert len(resp.fused_candidates) == 2
