"""Unit tests for GraphContextBuilder module."""

from graphrag_bench.graph_context_builder import GraphContextBuilder
from graphrag_bench.graph_retrieval_schemas import (
    GraphPath,
    GraphRetrievalResult,
    PathEdge,
    PathNode,
)


def make_sample_result() -> GraphRetrievalResult:
    """Helper to synthesize a GraphRetrievalResult for context construction."""
    path = GraphPath(
        path_id="path_test_1",
        seed_entity_id="node_a",
        seed_entity_name="Vector RAG",
        nodes=[
            PathNode(
                entity_id="node_a",
                canonical_name="Vector RAG",
                entity_type="TECHNOLOGY",
                hop_index=0,
            ),
            PathNode(
                entity_id="node_b",
                canonical_name="ChromaDB",
                entity_type="SOFTWARE",
                hop_index=1,
            ),
        ],
        edges=[
            PathEdge(
                relationship_id="rel_ab",
                subject_entity_id="node_a",
                predicate="USES",
                object_entity_id="node_b",
                direction="outgoing",
                evidence_quote="Vector RAG uses ChromaDB for persistent vector storage.",
                confidence=0.95,
                source_document_id="doc_1",
                source_chunk_id="chunk_1",
                source_filename="bench.md",
            )
        ],
        hop_count=1,
        source_document_ids=["doc_1"],
        source_chunk_ids=["chunk_1"],
    )

    return GraphRetrievalResult(
        result_id="res_1",
        rank=1,
        seed_entity_name="Vector RAG",
        path=path,
        score=0.92,
        score_explanation="Strong seed match with grounded evidence.",
        source_document_ids=["doc_1"],
        source_chunk_ids=["chunk_1"],
        source_filenames=["bench.md"],
        evidence_quotes=["Vector RAG uses ChromaDB for persistent vector storage."],
    )


class TestGraphContextBuilder:
    """Test suite for structured graph context packaging and provenance formatting."""

    def test_build_context_with_results(self) -> None:
        builder = GraphContextBuilder()
        res = make_sample_result()

        ctx = builder.build_context([res])

        assert len(ctx.triples) >= 1
        assert any("USES" in t for t in ctx.triples)
        assert len(ctx.evidence_quotes) == 1
        assert "ChromaDB for persistent vector storage" in ctx.evidence_quotes[0]
        assert len(ctx.source_citations) >= 1
        assert ctx.source_citations[0]["source_filename"] == "bench.md"

        # Check formatted markdown string structure
        text = ctx.formatted_context
        assert "### Knowledge Graph Relational Context" in text
        assert "#### Asserted Relational Triples:" in text
        assert "#### Grounded Evidence Quotations:" in text
        assert "bench.md" in text

    def test_build_context_empty_results(self) -> None:
        builder = GraphContextBuilder()
        ctx = builder.build_context([])

        assert len(ctx.triples) == 0
        assert len(ctx.evidence_quotes) == 0
        assert len(ctx.source_citations) == 0
        assert "No relevant knowledge graph context retrieved" in ctx.formatted_context

    def test_deduplication_of_parallel_evidence(self) -> None:
        builder = GraphContextBuilder()
        res1 = make_sample_result()
        # Duplicate result sharing exact same evidence quote
        res2 = make_sample_result()

        ctx = builder.build_context([res1, res2])
        # Quotes should be deduplicated
        assert len(ctx.evidence_quotes) == 1
