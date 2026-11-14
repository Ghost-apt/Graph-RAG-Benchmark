"""Unit tests for Phase 8 Benchmark Runner Engine."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from graphrag_bench.benchmark_runner import BenchmarkRunner
from graphrag_bench.benchmark_schemas import (
    BenchmarkQuery,
    BenchmarkQueryCategory,
    BenchmarkSplit,
    RelevanceJudgment,
    RetrievalStrategy,
)
from graphrag_bench.benchmark_storage import BenchmarkStorage
from graphrag_bench.semantic_search import SearchResponse
from graphrag_bench.vector_store import SearchResultItem


@pytest.fixture
def mock_runner(tmp_path: Path) -> BenchmarkRunner:
    storage = BenchmarkStorage(root_dir=tmp_path / "phase8_benchmark")

    # Mock Vector Search
    mock_vec_search = MagicMock()
    mock_vec_search.search.return_value = SearchResponse(
        query="What is Graph RAG?",
        top_k=5,
        total_found=2,
        duration_ms=12.5,
        items=[
            SearchResultItem(
                rank=1,
                chunk_id="chunk_test_1",
                document_id="doc_1",
                text="Graph RAG combines knowledge graphs and language models.",
                distance=0.15,
                cosine_similarity=0.85,
                normalized_score=0.925,
                metadata={"document_title": "Doc 1"},
            ),
            SearchResultItem(
                rank=2,
                chunk_id="chunk_test_2",
                document_id="doc_1",
                text="Another chunk.",
                distance=0.35,
                cosine_similarity=0.65,
                normalized_score=0.825,
                metadata={"document_title": "Doc 1"},
            ),
        ],
        model_used="nomic-embed-text",
    )

    # Mock Graph Search & Normalizer
    mock_graph_search = MagicMock()
    mock_normalizer = MagicMock()
    mock_cand1 = MagicMock()
    mock_cand1.chunk_id = "chunk_test_1"
    mock_cand1.document_id = "doc_1"
    mock_cand1.source_filename = "doc_1.md"
    mock_cand1.graph_score = 0.95
    mock_cand1.chunk_text = "Graph chunk text"

    mock_normalizer.normalize_graph_results.return_value = ([mock_cand1], [])

    # Mock Hybrid Retriever
    mock_hybrid = MagicMock()
    mock_fused1 = MagicMock()
    mock_fused1.fused_rank = 1
    mock_fused1.chunk_id = "chunk_test_1"
    mock_fused1.document_id = "doc_1"
    mock_fused1.source_filename = "doc_1.md"
    mock_fused1.rrf_score = 0.033
    mock_fused1.retrieval_sources = MagicMock(value="hybrid")
    mock_fused1.chunk_text = "Fused chunk text"

    mock_hybrid_resp = MagicMock()
    mock_hybrid_resp.candidates = [mock_fused1]
    mock_hybrid_resp.vector_latency_ms = 10.0
    mock_hybrid_resp.graph_latency_ms = 5.0
    mock_hybrid_resp.fusion_latency_ms = 1.0
    mock_hybrid_resp.llm_invoked = False
    mock_hybrid.search_hybrid.return_value = mock_hybrid_resp

    runner = BenchmarkRunner(
        storage=storage,
        vector_search=mock_vec_search,
        graph_search=mock_graph_search,
        normalizer=mock_normalizer,
        hybrid_retriever=mock_hybrid,
    )
    return runner


def test_runner_run_query_retrieval(mock_runner: BenchmarkRunner):
    """Verify single query retrieval across all 3 strategies."""
    query = BenchmarkQuery(
        query_id="BM-001",
        question="What is Graph RAG?",
        category=BenchmarkQueryCategory.SINGLE_HOP,
        split=BenchmarkSplit.DEV,
        relevant_chunk_ids=["chunk_test_1"],
        relevance_judgments=[
            RelevanceJudgment(chunk_id="chunk_test_1", document_id="doc_1", grade=3),
        ],
    )

    # Vector
    res_vec = mock_runner.run_query_retrieval(query, RetrievalStrategy.VECTOR, k_values=[1, 5])
    assert res_vec.strategy == RetrievalStrategy.VECTOR
    assert len(res_vec.retrieved_items) == 2
    assert res_vec.retrieved_items[0].is_relevant is True
    assert res_vec.metrics_by_k[1].precision == 1.0
    assert res_vec.metrics_by_k[5].precision == 0.2

    # Graph
    res_graph = mock_runner.run_query_retrieval(query, RetrievalStrategy.GRAPH, k_values=[1, 5])
    assert res_graph.strategy == RetrievalStrategy.GRAPH
    assert len(res_graph.retrieved_items) == 1
    assert res_graph.retrieved_items[0].chunk_id == "chunk_test_1"
    assert res_graph.metrics_by_k[1].precision == 1.0

    # Hybrid
    res_hybrid = mock_runner.run_query_retrieval(query, RetrievalStrategy.HYBRID, k_values=[1, 5])
    assert res_hybrid.strategy == RetrievalStrategy.HYBRID
    assert len(res_hybrid.retrieved_items) == 1
    assert res_hybrid.retrieved_items[0].chunk_id == "chunk_test_1"
    assert res_hybrid.metrics_by_k[1].hit_rate == 1.0


def test_runner_experiment_a_batch_and_compile(mock_runner: BenchmarkRunner):
    """Verify Experiment A execution and compile_run_manifest."""
    queries = [
        BenchmarkQuery(
            query_id="BM-001",
            question="What is Graph RAG?",
            category=BenchmarkQueryCategory.SINGLE_HOP,
            split=BenchmarkSplit.DEV,
            relevant_chunk_ids=["chunk_test_1"],
            relevance_judgments=[
                RelevanceJudgment(chunk_id="chunk_test_1", document_id="doc_1", grade=3)
            ],
        )
    ]

    run_id, results = mock_runner.run_experiment_a(
        queries=queries,
        strategies=[RetrievalStrategy.VECTOR, RetrievalStrategy.HYBRID],
        k_values=[1, 5],
        run_id="test_exp_a_run",
    )

    assert run_id == "test_exp_a_run"
    assert len(results) == 2  # 1 query * 2 strategies

    # Compile manifest
    manifest = mock_runner.compile_run_manifest(
        run_id=run_id,
        retrieval_results=results,
        k_values=[1, 5],
    )

    assert manifest.run_id == run_id
    assert manifest.status.lower() == "completed"
    assert "vector" in manifest.strategy_summaries
    assert "hybrid" in manifest.strategy_summaries
    assert len(manifest.paired_comparisons) >= 1
