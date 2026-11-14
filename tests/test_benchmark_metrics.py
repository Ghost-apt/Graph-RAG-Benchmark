"""Unit tests for Phase 8 Benchmark Information Retrieval Metrics Engine."""

import pytest

from graphrag_bench.benchmark_metrics import (
    aggregate_strategy_metrics,
    calculate_dcg_at_k,
    calculate_hit_rate_at_k,
    calculate_map_at_k,
    calculate_ndcg_at_k,
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_reciprocal_rank,
    evaluate_query_retrieval,
)
from graphrag_bench.benchmark_schemas import (
    BenchmarkQuery,
    BenchmarkQueryCategory,
    BenchmarkQueryResult,
    BenchmarkSplit,
    LatencyTelemetry,
    RelevanceJudgment,
    RetrievalRankedItem,
    RetrievalStrategy,
)


@pytest.fixture
def sample_retrieved_items() -> list[RetrievalRankedItem]:
    """Sample candidate ranking:
    Rank 1: chunk_A (grade 3, relevant)
    Rank 2: chunk_B (grade 0, non-relevant)
    Rank 3: chunk_C (grade 2, relevant)
    Rank 4: chunk_D (grade 1, non-relevant)
    """
    return [
        RetrievalRankedItem(
            rank=1,
            chunk_id="chunk_A",
            document_id="doc_1",
            modality="vector",
            gold_grade=3,
            is_relevant=True,
        ),
        RetrievalRankedItem(
            rank=2,
            chunk_id="chunk_B",
            document_id="doc_1",
            modality="vector",
            gold_grade=0,
            is_relevant=False,
        ),
        RetrievalRankedItem(
            rank=3,
            chunk_id="chunk_C",
            document_id="doc_2",
            modality="vector",
            gold_grade=2,
            is_relevant=True,
        ),
        RetrievalRankedItem(
            rank=4,
            chunk_id="chunk_D",
            document_id="doc_2",
            modality="vector",
            gold_grade=1,
            is_relevant=False,
        ),
    ]


@pytest.fixture
def gold_relevant_chunk_ids() -> set[str]:
    # 3 total gold relevant chunks in corpus: chunk_A, chunk_C, and chunk_E (which was unretrieved)
    return {"chunk_A", "chunk_C", "chunk_E"}


@pytest.fixture
def query_judgments_map() -> dict[str, int]:
    return {
        "chunk_A": 3,
        "chunk_B": 0,
        "chunk_C": 2,
        "chunk_D": 1,
        "chunk_E": 3,
    }


def test_precision_at_k_hand_calculated(sample_retrieved_items, gold_relevant_chunk_ids):
    """Verify Precision@K at K=1, 2, 3, 5 with strict denominator K."""
    # K = 1: 1 relevant out of 1 -> 1.0
    p1 = calculate_precision_at_k(sample_retrieved_items, gold_relevant_chunk_ids, k=1)
    assert p1 == 1.0

    # K = 2: 1 relevant (chunk_A) out of 2 -> 0.5
    p2 = calculate_precision_at_k(sample_retrieved_items, gold_relevant_chunk_ids, k=2)
    assert p2 == 0.5

    # K = 3: 2 relevant (chunk_A, chunk_C) out of 3 -> 2/3
    p3 = calculate_precision_at_k(sample_retrieved_items, gold_relevant_chunk_ids, k=3)
    assert pytest.approx(p3, 0.0001) == 2.0 / 3.0

    # K = 5: Only 4 items retrieved, 2 relevant out of K=5 -> 2/5 = 0.4
    p5 = calculate_precision_at_k(sample_retrieved_items, gold_relevant_chunk_ids, k=5)
    assert p5 == 0.4


def test_recall_at_k_hand_calculated(sample_retrieved_items, gold_relevant_chunk_ids):
    """Verify Recall@K at K=1, 2, 3, 5 against total 3 relevant chunks."""
    # K = 1: 1 / 3
    r1 = calculate_recall_at_k(sample_retrieved_items, gold_relevant_chunk_ids, k=1)
    assert pytest.approx(r1, 0.0001) == 1.0 / 3.0

    # K = 2: 1 / 3
    r2 = calculate_recall_at_k(sample_retrieved_items, gold_relevant_chunk_ids, k=2)
    assert pytest.approx(r2, 0.0001) == 1.0 / 3.0

    # K = 3: 2 / 3
    r3 = calculate_recall_at_k(sample_retrieved_items, gold_relevant_chunk_ids, k=3)
    assert pytest.approx(r3, 0.0001) == 2.0 / 3.0

    # K = 5: 2 / 3 (since chunk_E was never retrieved)
    r5 = calculate_recall_at_k(sample_retrieved_items, gold_relevant_chunk_ids, k=5)
    assert pytest.approx(r5, 0.0001) == 2.0 / 3.0


def test_hit_rate_and_mrr(sample_retrieved_items, gold_relevant_chunk_ids):
    """Verify Hit Rate and Reciprocal Rank."""
    # First relevant chunk is at rank 1
    assert calculate_hit_rate_at_k(sample_retrieved_items, gold_relevant_chunk_ids, k=1) == 1.0
    assert (
        calculate_reciprocal_rank(sample_retrieved_items, gold_relevant_chunk_ids, max_k=5) == 1.0
    )

    # If first relevant chunk was at rank 3:
    offset_items = [
        sample_retrieved_items[1],  # rank 1: chunk_B (irrelevant)
        sample_retrieved_items[3],  # rank 2: chunk_D (irrelevant)
        sample_retrieved_items[0],  # rank 3: chunk_A (relevant)
    ]
    assert calculate_hit_rate_at_k(offset_items, gold_relevant_chunk_ids, k=2) == 0.0
    assert calculate_hit_rate_at_k(offset_items, gold_relevant_chunk_ids, k=3) == 1.0
    assert calculate_reciprocal_rank(offset_items, gold_relevant_chunk_ids, max_k=2) == 0.0
    assert (
        pytest.approx(
            calculate_reciprocal_rank(offset_items, gold_relevant_chunk_ids, max_k=5), 0.0001
        )
        == 1.0 / 3.0
    )


def test_dcg_and_ndcg_graded(sample_retrieved_items, query_judgments_map):
    """Verify graded DCG and nDCG calculation:
    Retrieved:
    Rank 1: grade 3 -> gain = 2^3 - 1 = 7. discount = log2(2) = 1.0. val = 7.0
    Rank 2: grade 0 -> gain = 2^0 - 1 = 0. val = 0.0
    Rank 3: grade 2 -> gain = 2^2 - 1 = 3. discount = log2(4) = 2.0. val = 1.5
    Rank 4: grade 1 -> gain = 2^1 - 1 = 1. discount = log2(5) = 2.3219. val = 1 / 2.3219 = 0.4307
    """
    # DCG@1 = 7.0
    assert calculate_dcg_at_k([3, 0, 2, 1], k=1) == 7.0
    # DCG@3 = 7.0 + 0 + 1.5 = 8.5
    assert pytest.approx(calculate_dcg_at_k([3, 0, 2, 1], k=3), 0.0001) == 8.5

    # Ideal ranking for K=1: [3] -> IDCG@1 = 7.0. nDCG@1 = 7.0 / 7.0 = 1.0
    ndcg1 = calculate_ndcg_at_k(sample_retrieved_items, query_judgments_map, k=1)
    assert ndcg1 == 1.0

    # Ideal ranking for K=3: [3, 3, 2] (chunk_A=3, chunk_E=3, chunk_C=2)
    # IDCG@3 = 7/1 + 7/log2(3) + 3/2 = 7 + 7/1.58496 + 1.5 = 7 + 4.4170 + 1.5 = 12.9170
    # Actual DCG@3 = 8.5
    # nDCG@3 = 8.5 / 12.9170 = 0.6580
    ndcg3 = calculate_ndcg_at_k(sample_retrieved_items, query_judgments_map, k=3)
    assert 0.65 <= ndcg3 <= 0.67


def test_map_at_k_hand_calculated(sample_retrieved_items, gold_relevant_chunk_ids):
    """Verify MAP@K formula:
    AP@1 = (1/min(1, 3)) * (P@1 * 1) = 1.0 * 1.0 = 1.0
    AP@3 = (1/min(3, 3)) * (P@1 * 1 + P@2 * 0 + P@3 * 1)
         = (1/3) * (1.0 + 0.0 + 2/3) = (1/3) * (5/3) = 5/9 = 0.5556
    """
    ap1 = calculate_map_at_k(sample_retrieved_items, gold_relevant_chunk_ids, k=1)
    assert ap1 == 1.0

    ap3 = calculate_map_at_k(sample_retrieved_items, gold_relevant_chunk_ids, k=3)
    assert pytest.approx(ap3, 0.001) == 5.0 / 9.0


def test_edge_cases_empty_and_unanswerable():
    """Verify edge cases: 0 items, empty gold set, unanswerable queries."""
    # Empty items
    assert calculate_precision_at_k([], {"c1"}, k=5) == 0.0
    assert calculate_recall_at_k([], {"c1"}, k=5) == 0.0
    assert calculate_hit_rate_at_k([], {"c1"}, k=5) == 0.0
    assert calculate_reciprocal_rank([], {"c1"}, max_k=5) == 0.0
    assert calculate_ndcg_at_k([], {"c1": 3}, k=5) == 0.0
    assert calculate_map_at_k([], {"c1"}, k=5) == 0.0

    # Unanswerable query with 0 relevant chunks and no retrieved items
    assert calculate_ndcg_at_k([], {}, k=5) == 1.0


def test_evaluate_query_retrieval_and_aggregation():
    """Verify evaluate_query_retrieval and macro-aggregation across queries."""
    q1 = BenchmarkQuery(
        query_id="Q1",
        question="What is RAG?",
        category=BenchmarkQueryCategory.SINGLE_HOP,
        split=BenchmarkSplit.DEV,
        relevant_chunk_ids=["chunk_A"],
        relevance_judgments=[RelevanceJudgment(chunk_id="chunk_A", document_id="doc_1", grade=3)],
    )
    items1 = [
        RetrievalRankedItem(
            rank=1,
            chunk_id="chunk_A",
            document_id="doc_1",
            modality="vector",
            gold_grade=3,
            is_relevant=True,
        ),
        RetrievalRankedItem(
            rank=2,
            chunk_id="chunk_X",
            document_id="doc_1",
            modality="vector",
            gold_grade=0,
            is_relevant=False,
        ),
    ]

    metrics1 = evaluate_query_retrieval(items1, q1, k_values=[1, 3])
    assert metrics1[1].precision == 1.0
    assert metrics1[1].recall == 1.0
    assert metrics1[3].precision == round(1.0 / 3.0, 4)

    r1 = BenchmarkQueryResult(
        query_id="Q1",
        strategy=RetrievalStrategy.VECTOR,
        category=BenchmarkQueryCategory.SINGLE_HOP,
        split=BenchmarkSplit.DEV,
        answerable=True,
        metrics_by_k=metrics1,
        latency=LatencyTelemetry(total_latency_ms=20.0),
    )

    q2 = BenchmarkQuery(
        query_id="Q2",
        question="What is KG?",
        category=BenchmarkQueryCategory.SINGLE_HOP,
        split=BenchmarkSplit.DEV,
        relevant_chunk_ids=["chunk_B"],
        relevance_judgments=[RelevanceJudgment(chunk_id="chunk_B", document_id="doc_1", grade=3)],
    )
    # System failed to retrieve chunk_B
    items2 = [
        RetrievalRankedItem(
            rank=1,
            chunk_id="chunk_Z",
            document_id="doc_1",
            modality="vector",
            gold_grade=0,
            is_relevant=False,
        ),
    ]
    metrics2 = evaluate_query_retrieval(items2, q2, k_values=[1, 3])
    r2 = BenchmarkQueryResult(
        query_id="Q2",
        strategy=RetrievalStrategy.VECTOR,
        category=BenchmarkQueryCategory.SINGLE_HOP,
        split=BenchmarkSplit.DEV,
        answerable=True,
        metrics_by_k=metrics2,
        latency=LatencyTelemetry(total_latency_ms=40.0),
    )

    summary = aggregate_strategy_metrics([r1, r2], k_values=[1, 3])
    assert summary.strategy == RetrievalStrategy.VECTOR
    assert summary.total_queries == 2
    # P@1 macro average = (1.0 + 0.0) / 2 = 0.5
    assert summary.mean_precision_at_k[1] == 0.5
    # R@1 macro average = (1.0 + 0.0) / 2 = 0.5
    assert summary.mean_recall_at_k[1] == 0.5
    assert summary.mean_latency_ms == 30.0
