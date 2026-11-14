"""Unit tests for Phase 8 Benchmark Pydantic Schemas."""

import pytest
from pydantic import ValidationError

from graphrag_bench.benchmark_schemas import (
    AnswerQualityMetrics,
    BenchmarkQuery,
    BenchmarkQueryCategory,
    BenchmarkQueryResult,
    BenchmarkSplit,
    LatencyTelemetry,
    RelevanceJudgment,
    RetrievalMetrics,
    RetrievalRankedItem,
    RetrievalStrategy,
)


def test_relevance_judgment_validation():
    """Verify RelevanceJudgment grades and validation."""
    j = RelevanceJudgment(
        chunk_id="chunk_1", document_id="doc_1", grade=3, rationale="Core definition"
    )
    assert j.grade == 3
    assert j.is_relevant is True

    j2 = RelevanceJudgment(chunk_id="chunk_2", document_id="doc_1", grade=1)
    assert j2.grade == 1
    assert j2.is_relevant is False

    with pytest.raises(ValidationError):
        RelevanceJudgment(chunk_id="chunk_3", document_id="doc_1", grade=4)  # type: ignore

    with pytest.raises(ValidationError):
        RelevanceJudgment(chunk_id="", document_id="doc_1", grade=2)


def test_benchmark_query_validation():
    """Verify BenchmarkQuery schema and defaults."""
    q = BenchmarkQuery(
        query_id="BM-001",
        question="What is Graph RAG?",
        category=BenchmarkQueryCategory.SINGLE_HOP,
        split=BenchmarkSplit.DEV,
        relevant_chunk_ids=["chunk_1"],
        relevance_judgments=[
            RelevanceJudgment(chunk_id="chunk_1", document_id="doc_1", grade=3),
        ],
    )
    assert q.query_id == "BM-001"
    assert q.answerable is True
    assert len(q.relevant_chunk_ids) == 1

    with pytest.raises(ValidationError):
        BenchmarkQuery(
            query_id="BM-002",
            question="   ",
            category=BenchmarkQueryCategory.MULTI_HOP,
            split=BenchmarkSplit.TEST,
        )


def test_retrieval_ranked_item():
    """Verify RetrievalRankedItem schema validation."""
    item = RetrievalRankedItem(
        rank=1,
        chunk_id="chunk_1",
        document_id="doc_1",
        modality="vector",
        score=0.89,
        gold_grade=3,
        is_relevant=True,
    )
    assert item.rank == 1
    assert item.gold_grade == 3

    with pytest.raises(ValidationError):
        RetrievalRankedItem(
            rank=0,  # Rank must be >= 1
            chunk_id="c1",
            document_id="d1",
            modality="vector",
        )


def test_retrieval_metrics():
    """Verify RetrievalMetrics bounds and validation."""
    m = RetrievalMetrics(
        k=5,
        precision=0.4,
        recall=0.8,
        hit_rate=1.0,
        mrr=0.5,
        ndcg=0.82,
        map_score=0.65,
    )
    assert m.k == 5
    assert m.precision == 0.4

    with pytest.raises(ValidationError):
        RetrievalMetrics(
            k=0,  # k >= 1
            precision=0.5,
            recall=0.5,
            hit_rate=1.0,
            mrr=0.5,
            ndcg=0.5,
        )

    with pytest.raises(ValidationError):
        RetrievalMetrics(
            k=5,
            precision=1.5,  # precision <= 1.0
            recall=0.5,
            hit_rate=1.0,
            mrr=0.5,
            ndcg=0.5,
        )


def test_answer_quality_metrics():
    """Verify AnswerQualityMetrics bounds."""
    aq = AnswerQualityMetrics(
        answer_correctness=0.9,
        answer_relevance=1.0,
        faithfulness_score=0.95,
        citation_reference_precision=1.0,
        citation_coverage=0.8,
        refusal_accuracy=None,
    )
    assert aq.faithfulness_score == 0.95

    with pytest.raises(ValidationError):
        AnswerQualityMetrics(faithfulness_score=1.2)


def test_benchmark_query_result():
    """Verify BenchmarkQueryResult encapsulation."""
    res = BenchmarkQueryResult(
        query_id="BM-001",
        strategy=RetrievalStrategy.HYBRID,
        category=BenchmarkQueryCategory.COMPARISON,
        split=BenchmarkSplit.TEST,
        answerable=True,
        metrics_by_k={
            5: RetrievalMetrics(
                k=5,
                precision=0.4,
                recall=0.8,
                hit_rate=1.0,
                mrr=0.5,
                ndcg=0.75,
                map_score=0.6,
            )
        },
        latency=LatencyTelemetry(total_latency_ms=45.2),
    )
    assert res.strategy == RetrievalStrategy.HYBRID
    assert 5 in res.metrics_by_k
    assert res.latency.total_latency_ms == 45.2
