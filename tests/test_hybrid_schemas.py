"""Unit tests for Phase 6 Hybrid Retrieval Schemas and Data Models."""

import pytest
from pydantic import ValidationError

from graphrag_bench.hybrid_schemas import (
    FusionCandidate,
    GraphCandidate,
    HybridSearchResponse,
    HybridSearchStatus,
    RetrievalSource,
    VectorCandidate,
)


class TestHybridSchemas:
    """Test suite for Hybrid Schemas validation, constraints, and serialization."""

    def test_retrieval_source_enum(self) -> None:
        """Verify enum string representations."""
        assert RetrievalSource.VECTOR.value == "VECTOR"
        assert RetrievalSource.GRAPH.value == "GRAPH"
        assert RetrievalSource.VECTOR_AND_GRAPH.value == "VECTOR_AND_GRAPH"

    def test_hybrid_search_status_enum(self) -> None:
        """Verify hybrid status values."""
        assert HybridSearchStatus.SUCCESS.value == "SUCCESS"
        assert HybridSearchStatus.PARTIAL_SUCCESS.value == "PARTIAL_SUCCESS"
        assert HybridSearchStatus.NO_RESULTS.value == "NO_RESULTS"
        assert HybridSearchStatus.INVALID_QUERY.value == "INVALID_QUERY"
        assert HybridSearchStatus.ALL_ENGINES_FAILED.value == "ALL_ENGINES_FAILED"

    def test_vector_candidate_valid(self) -> None:
        """Verify valid VectorCandidate construction."""
        cand = VectorCandidate(
            chunk_id="chunk_001",
            document_id="doc_abc",
            source_filename="test.md",
            file_type="md",
            chunk_text="This is a test chunk.",
            vector_rank=1,
            cosine_distance=0.2,
            cosine_similarity=0.8,
            normalized_score=0.9,
            source_page=1,
        )
        assert cand.chunk_id == "chunk_001"
        assert cand.vector_rank == 1
        assert cand.cosine_similarity == 0.8
        assert cand.metadata == {}

    def test_vector_candidate_empty_field_raises(self) -> None:
        """Verify empty required strings fail validation."""
        with pytest.raises(ValidationError):
            VectorCandidate(
                chunk_id="   ",
                document_id="doc_abc",
                source_filename="test.md",
                file_type="md",
                chunk_text="Valid text",
                vector_rank=1,
                cosine_distance=0.2,
                cosine_similarity=0.8,
                normalized_score=0.9,
            )

    def test_graph_candidate_valid(self) -> None:
        """Verify valid GraphCandidate construction."""
        cand = GraphCandidate(
            chunk_id="chunk_002",
            document_id="doc_xyz",
            source_filename="graph_doc.md",
            chunk_text="Graph grounded content.",
            graph_rank=2,
            matched_entities=["Vector RAG", "ChromaDB"],
            supporting_paths=[{"path_id": "p1", "edges": []}],
            evidence_quotes=["Quote 1", "Quote 2"],
            best_path_score=0.85,
            hop_count=2,
        )
        assert cand.chunk_id == "chunk_002"
        assert cand.graph_rank == 2
        assert len(cand.matched_entities) == 2
        assert cand.hop_count == 2

    def test_fusion_candidate_dual_modality(self) -> None:
        """Verify FusionCandidate representing dual-modality match."""
        cand = FusionCandidate(
            chunk_id="chunk_003",
            document_id="doc_xyz",
            source_filename="hybrid.md",
            chunk_text="Hybrid content.",
            source=RetrievalSource.VECTOR_AND_GRAPH,
            vector_rank=1,
            graph_rank=2,
            vector_rrf_contribution=0.016393,
            graph_rrf_contribution=0.016129,
            rrf_score=0.032522,
            final_rank=1,
            cosine_similarity=0.85,
            matched_entities=["Vector RAG"],
            evidence_quotes=["Quote A"],
        )
        assert cand.source == RetrievalSource.VECTOR_AND_GRAPH
        assert cand.vector_rank == 1
        assert cand.graph_rank == 2
        assert cand.final_rank == 1
        assert cand.rrf_score == pytest.approx(0.032522, abs=1e-5)

    def test_hybrid_search_response_serialization(self) -> None:
        """Verify HybridSearchResponse serialization to dict and JSON."""
        cand = FusionCandidate(
            chunk_id="chunk_001",
            document_id="doc_1",
            source_filename="doc1.md",
            chunk_text="Sample text",
            source=RetrievalSource.VECTOR,
            vector_rank=1,
            vector_rrf_contribution=0.016393,
            rrf_score=0.016393,
            final_rank=1,
        )
        resp = HybridSearchResponse(
            query="test query",
            status=HybridSearchStatus.SUCCESS,
            fused_candidates=[cand],
            vector_candidates_count=1,
            graph_candidates_count=0,
            total_candidates=1,
            fusion_params={"rrf_k": 60, "vector_weight": 1.0, "graph_weight": 1.0},
            vector_latency_ms=12.5,
            graph_latency_ms=0.0,
            fusion_latency_ms=1.2,
            total_latency_ms=13.7,
            structured_context="### Retrieved Evidence\nSample",
        )
        data = resp.model_dump(mode="json")
        assert data["query"] == "test query"
        assert data["status"] == "SUCCESS"
        assert len(data["results"]) == 1
        assert data["results"][0]["chunk_id"] == "chunk_001"
