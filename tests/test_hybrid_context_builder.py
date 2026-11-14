"""Unit tests for HybridContextBuilder module."""

from graphrag_bench.hybrid_context_builder import HybridContextBuilder
from graphrag_bench.hybrid_schemas import FusionCandidate, RetrievalSource


class TestHybridContextBuilder:
    """Test suite for structured hybrid prompt context generation."""

    def test_build_context_empty_candidates(self) -> None:
        """Verify fallback text when no candidates are provided."""
        builder = HybridContextBuilder()
        text = builder.build_context([])
        assert "No relevant hybrid retrieval context found" in text

    def test_build_context_vector_only_candidate(self) -> None:
        """Verify formatting of vector-only candidate."""
        builder = HybridContextBuilder()
        cand = FusionCandidate(
            chunk_id="chunk_vec_1",
            document_id="doc_1",
            source_filename="doc1.md",
            chunk_text="Content from vector search.",
            source=RetrievalSource.VECTOR,
            vector_rank=1,
            vector_rrf_contribution=0.016393,
            rrf_score=0.016393,
            final_rank=1,
            cosine_similarity=0.88,
        )
        ctx = builder.build_context([cand])
        assert "[Evidence #1] Chunk ID: `chunk_vec_1` | Source: **VECTOR**" in ctx
        assert "Document: `doc1.md`" in ctx
        assert "Vector Rank #1" in ctx
        assert "Cosine Sim: 0.8800" in ctx
        assert "Content from vector search." in ctx

    def test_build_context_dual_modality_candidate(self) -> None:
        """Verify formatting of dual-modality candidate with graph provenance."""
        builder = HybridContextBuilder()
        cand = FusionCandidate(
            chunk_id="chunk_dual_1",
            document_id="doc_2",
            source_filename="dual.md",
            chunk_text="Content grounded in both vector and graph.",
            source=RetrievalSource.VECTOR_AND_GRAPH,
            vector_rank=1,
            graph_rank=2,
            vector_rrf_contribution=0.016393,
            graph_rrf_contribution=0.016129,
            rrf_score=0.032522,
            final_rank=1,
            cosine_similarity=0.92,
            min_hops=1,
            matched_entities=["Vector RAG", "Graph RAG"],
            evidence_quotes=[
                "Quote 1: Vector RAG uses ChromaDB",
                "Quote 2: Graph RAG uses NetworkX",
            ],
        )
        ctx = builder.build_context([cand])
        assert "[Evidence #1] Chunk ID: `chunk_dual_1` | Source: **VECTOR_AND_GRAPH**" in ctx
        assert "Vector Rank #1" in ctx
        assert "Graph Rank #2" in ctx
        assert "**Grounded Knowledge Graph Quotes:**" in ctx
        assert '"Quote 1: Vector RAG uses ChromaDB"' in ctx
        assert '"Quote 2: Graph RAG uses NetworkX"' in ctx

    def test_build_context_max_context_chunks(self) -> None:
        """Verify max_context_chunks limits the formatted evidence blocks."""
        builder = HybridContextBuilder()
        cand1 = FusionCandidate(
            chunk_id="chunk_1",
            document_id="doc_1",
            source_filename="doc1.md",
            chunk_text="Content 1",
            source=RetrievalSource.VECTOR,
            vector_rank=1,
            vector_rrf_contribution=0.016393,
            rrf_score=0.016393,
            final_rank=1,
        )
        cand2 = FusionCandidate(
            chunk_id="chunk_2",
            document_id="doc_2",
            source_filename="doc2.md",
            chunk_text="Content 2",
            source=RetrievalSource.VECTOR,
            vector_rank=2,
            vector_rrf_contribution=0.016129,
            rrf_score=0.016129,
            final_rank=2,
        )
        ctx = builder.build_context([cand1, cand2], max_context_chunks=1)
        assert "Chunk ID: `chunk_1`" in ctx
        assert "Chunk ID: `chunk_2`" not in ctx
