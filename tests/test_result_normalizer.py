"""Unit tests for ResultNormalizer module."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from graphrag_bench.document_repository import DocumentRepository
from graphrag_bench.graph_retrieval_schemas import (
    GraphPath,
    GraphRetrievalResult,
    GraphSearchResponse,
    GraphSearchStatus,
    PathEdge,
    PathNode,
)
from graphrag_bench.result_normalizer import ResultNormalizer
from graphrag_bench.schemas import DocumentChunk
from graphrag_bench.semantic_search import SearchResponse
from graphrag_bench.vector_store import SearchResultItem


def create_mock_doc_repo() -> DocumentRepository:
    """Create a mock repository pre-loaded with canonical chunks."""
    repo = MagicMock(spec=DocumentRepository)
    chunk1 = DocumentChunk(
        chunk_id="chunk_canon_1",
        document_id="doc_canon_1",
        chunk_index=0,
        start_char=0,
        end_char=100,
        text="Canonical content for chunk 1.",
        source_filename="canon_guide.md",
        file_type="md",
        page_number=1,
    )
    chunk2 = DocumentChunk(
        chunk_id="chunk_canon_2",
        document_id="doc_canon_2",
        chunk_index=0,
        start_char=0,
        end_char=120,
        text="Canonical content for chunk 2.",
        source_filename="canon_guide.md",
        file_type="md",
        page_number=2,
    )
    repo._get_all_chunks.return_value = [chunk1, chunk2]
    return repo


class TestResultNormalizer:
    """Test suite for ResultNormalizer vector and graph normalization."""

    def test_normalize_vector_results_success(self) -> None:
        """Verify standard vector search response normalization."""
        repo = create_mock_doc_repo()
        normalizer = ResultNormalizer(repository=repo)

        search_resp = SearchResponse(
            query="test query",
            top_k=2,
            total_found=2,
            duration_ms=5.0,
            items=[
                SearchResultItem(
                    rank=1,
                    chunk_id="chunk_canon_1",
                    document_id="doc_canon_1",
                    text="Canonical content for chunk 1.",
                    distance=0.2,
                    cosine_similarity=0.8,
                    normalized_score=0.9,
                    metadata={
                        "source_filename": "canon_guide.md",
                        "file_type": "md",
                        "page_number": 1,
                    },
                ),
                SearchResultItem(
                    rank=2,
                    chunk_id="chunk_canon_2",
                    document_id="doc_canon_2",
                    text="Canonical content for chunk 2.",
                    distance=0.4,
                    cosine_similarity=0.6,
                    normalized_score=0.8,
                    metadata={
                        "source_filename": "canon_guide.md",
                        "file_type": "md",
                        "page_number": 2,
                    },
                ),
            ],
            model_used="nomic-embed-text",
        )

        candidates, _ = normalizer.normalize_vector_results(search_resp)
        assert len(candidates) == 2
        assert candidates[0].chunk_id == "chunk_canon_1"
        assert candidates[0].vector_rank == 1
        assert candidates[0].cosine_distance == 0.2
        assert candidates[0].cosine_similarity == pytest_approx(0.8)
        assert candidates[0].normalized_score == pytest_approx(0.9)
        assert candidates[1].vector_rank == 2

    def test_normalize_vector_results_missing_chunk_id(self) -> None:
        """Verify vector item with empty or missing chunk_id is skipped."""
        repo = create_mock_doc_repo()
        normalizer = ResultNormalizer(repository=repo)

        search_resp = SearchResponse(
            query="test query",
            top_k=1,
            total_found=1,
            duration_ms=5.0,
            items=[
                SearchResultItem(
                    rank=1,
                    chunk_id="",
                    document_id="doc_canon_1",
                    text="Invalid item without chunk ID",
                    distance=0.2,
                    similarity_score=0.8,
                    metadata={},
                ),
            ],
            model_used="nomic-embed-text",
        )

        candidates, _ = normalizer.normalize_vector_results(search_resp)
        assert len(candidates) == 0

    def test_normalize_graph_results_deduplication_and_aggregation(self) -> None:
        """Verify multiple graph paths pointing to the same chunk are deduplicated."""
        repo = create_mock_doc_repo()
        normalizer = ResultNormalizer(repository=repo)

        # Path 1: references chunk_canon_1, rank 1
        edge1 = PathEdge(
            relationship_id="rel_1",
            subject_entity_id="node_a",
            predicate="USES",
            object_entity_id="node_b",
            direction="outgoing",
            confidence=0.9,
            evidence_quote="Node A uses Node B",
            source_document_id="doc_canon_1",
            source_chunk_id="chunk_canon_1",
            source_filename="canon_guide.md",
        )
        path1 = GraphPath(
            path_id="p1",
            seed_entity_id="node_a",
            seed_entity_name="Node A",
            hop_count=1,
            nodes=[
                PathNode(
                    entity_id="node_a",
                    canonical_name="Node A",
                    entity_type="TECHNOLOGY",
                    hop_index=0,
                ),
                PathNode(
                    entity_id="node_b", canonical_name="Node B", entity_type="SOFTWARE", hop_index=1
                ),
            ],
            edges=[edge1],
            path_score=0.9,
        )

        # Path 2: also references chunk_canon_1 via second edge, rank 2
        edge2 = PathEdge(
            relationship_id="rel_2",
            subject_entity_id="node_b",
            predicate="CONNECTS",
            object_entity_id="node_c",
            direction="outgoing",
            confidence=0.85,
            evidence_quote="Node B connects Node C",
            source_document_id="doc_canon_1",
            source_chunk_id="chunk_canon_1",
            source_filename="canon_guide.md",
        )
        path2 = GraphPath(
            path_id="p2",
            seed_entity_id="node_b",
            seed_entity_name="Node B",
            hop_count=1,
            nodes=[
                PathNode(
                    entity_id="node_b", canonical_name="Node B", entity_type="SOFTWARE", hop_index=0
                ),
                PathNode(
                    entity_id="node_c", canonical_name="Node C", entity_type="CONCEPT", hop_index=1
                ),
            ],
            edges=[edge2],
            path_score=0.85,
        )

        graph_resp = GraphSearchResponse(
            query="graph query",
            status=GraphSearchStatus.SUCCESS,
            message="Success",
            linked_entities=[],
            results=[
                GraphRetrievalResult(
                    result_id="res_1",
                    rank=1,
                    seed_entity_name="Node A",
                    path=path1,
                    score=0.9,
                ),
                GraphRetrievalResult(
                    result_id="res_2",
                    rank=2,
                    seed_entity_name="Node B",
                    path=path2,
                    score=0.85,
                ),
            ],
            execution_time_ms=10.0,
        )

        candidates, _ = normalizer.normalize_graph_results(graph_resp)
        # Should be deduplicated into 1 candidate with aggregated paths and evidence quotes
        assert len(candidates) == 1
        c = candidates[0]
        assert c.chunk_id == "chunk_canon_1"
        assert c.graph_rank == 1
        assert c.best_path_score == 0.9
        assert len(c.supporting_paths) == 2
        assert "Node A uses Node B" in c.evidence_quotes
        assert "Node B connects Node C" in c.evidence_quotes


def pytest_approx(val: float, abs_val: float = 1e-4) -> Any:
    return pytest.approx(val, abs=abs_val)
