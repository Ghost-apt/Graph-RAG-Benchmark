"""Information Retrieval & Evaluation Metrics Engine for Phase 8 Benchmark.

Implements mathematically rigorous IR metrics:
- Precision@K
- Recall@K
- Hit Rate@K
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (nDCG@K) with graded relevance (0-3)
- Mean Average Precision (MAP@K)
- Category-level and strategy-level macro/micro aggregations
"""

import math
from collections.abc import Sequence

from graphrag_bench.benchmark_schemas import (
    BenchmarkQuery,
    BenchmarkQueryResult,
    RetrievalMetrics,
    RetrievalRankedItem,
    StrategyAggregateSummary,
)


def calculate_precision_at_k(
    retrieved_items: Sequence[RetrievalRankedItem],
    gold_relevant_chunk_ids: set[str],
    k: int,
) -> float:
    """Calculate Precision@K: Fraction of top-K retrieved items that are relevant.

    Formula:
        Precision@K = |{item in top_K : item.chunk_id in gold_relevant}| / K

    Convention:
        Denominator is strictly K (not min(K, len(retrieved))). If a system retrieves
        fewer than K items, unretrieved slots are treated as non-relevant.

    Args:
        retrieved_items: Ranked candidate list.
        gold_relevant_chunk_ids: Set of canonical chunk IDs with relevance grade >= 2.
        k: Rank cutoff (must be >= 1).

    Returns:
        Float in [0.0, 1.0].
    """
    if k < 1:
        raise ValueError(f"Cutoff k must be >= 1, got {k}")
    if not retrieved_items or not gold_relevant_chunk_ids:
        return 0.0

    top_items = retrieved_items[:k]
    relevant_retrieved = sum(1 for item in top_items if item.chunk_id in gold_relevant_chunk_ids)
    return relevant_retrieved / float(k)


def calculate_recall_at_k(
    retrieved_items: Sequence[RetrievalRankedItem],
    gold_relevant_chunk_ids: set[str],
    k: int,
) -> float:
    """Calculate Recall@K: Fraction of gold relevant items present in top-K retrieved.

    Formula:
        Recall@K = |{item in top_K : item.chunk_id in gold_relevant}| / |gold_relevant|

    Convention:
        If gold_relevant is empty (e.g. unanswerable query with 0 relevant chunks),
        recall is 1.0 if top_K contains 0 relevant items, or 0.0 if unanswerable query
        is not applicable. We return 0.0 when gold_relevant is empty by standard IR convention.

    Args:
        retrieved_items: Ranked candidate list.
        gold_relevant_chunk_ids: Set of canonical chunk IDs with relevance grade >= 2.
        k: Rank cutoff (must be >= 1).

    Returns:
        Float in [0.0, 1.0].
    """
    if k < 1:
        raise ValueError(f"Cutoff k must be >= 1, got {k}")
    total_gold = len(gold_relevant_chunk_ids)
    if total_gold == 0:
        return 0.0
    if not retrieved_items:
        return 0.0

    top_items = retrieved_items[:k]
    # Use set to count unique gold items retrieved in top-K
    retrieved_chunk_ids = {item.chunk_id for item in top_items}
    intersection_count = len(retrieved_chunk_ids.intersection(gold_relevant_chunk_ids))
    return min(1.0, intersection_count / float(total_gold))


def calculate_hit_rate_at_k(
    retrieved_items: Sequence[RetrievalRankedItem],
    gold_relevant_chunk_ids: set[str],
    k: int,
) -> float:
    """Calculate Hit Rate@K: 1.0 if at least one relevant item is in top-K, else 0.0.

    Args:
        retrieved_items: Ranked candidate list.
        gold_relevant_chunk_ids: Set of canonical chunk IDs with relevance grade >= 2.
        k: Rank cutoff (must be >= 1).

    Returns:
        1.0 or 0.0.
    """
    if k < 1:
        raise ValueError(f"Cutoff k must be >= 1, got {k}")
    if not retrieved_items or not gold_relevant_chunk_ids:
        return 0.0

    top_items = retrieved_items[:k]
    return 1.0 if any(item.chunk_id in gold_relevant_chunk_ids for item in top_items) else 0.0


def calculate_reciprocal_rank(
    retrieved_items: Sequence[RetrievalRankedItem],
    gold_relevant_chunk_ids: set[str],
    max_k: int | None = None,
) -> float:
    """Calculate Reciprocal Rank (RR): 1 / rank of first relevant item in candidates.

    Formula:
        RR = 1.0 / rank_first_relevant (where rank is 1-based)
        RR = 0.0 if no relevant items retrieved within max_k.

    Args:
        retrieved_items: Ranked candidate list.
        gold_relevant_chunk_ids: Set of canonical chunk IDs with relevance grade >= 2.
        max_k: Maximum rank cutoff considered (or None for all items).

    Returns:
        Float in [0.0, 1.0].
    """
    if not retrieved_items or not gold_relevant_chunk_ids:
        return 0.0

    items_to_check = retrieved_items[:max_k] if max_k is not None else retrieved_items
    for idx, item in enumerate(items_to_check, start=1):
        if item.chunk_id in gold_relevant_chunk_ids:
            return 1.0 / float(idx)

    return 0.0


def calculate_dcg_at_k(
    grades: Sequence[int],
    k: int,
) -> float:
    """Calculate Discounted Cumulative Gain at rank K using standard exponential gain.

    Formula:
        DCG@K = sum_{i=1}^K (2^{rel_i} - 1) / log2(i + 1)

    Args:
        grades: List of integer relevance grades (0 to 3) in retrieved rank order.
        k: Cutoff rank.

    Returns:
        Non-negative float DCG score.
    """
    dcg = 0.0
    for idx, grade in enumerate(grades[:k], start=1):
        gain = float((1 << grade) - 1)  # 2^grade - 1
        discount = math.log2(idx + 1)
        dcg += gain / discount
    return dcg


def calculate_ndcg_at_k(
    retrieved_items: Sequence[RetrievalRankedItem],
    query_judgments_map: dict[str, int],
    k: int,
) -> float:
    """Calculate Normalized Discounted Cumulative Gain (nDCG@K) with graded relevance.

    Formula:
        nDCG@K = DCG@K / IDCG@K
        where IDCG@K is the ideal DCG obtained by sorting all available gold grades descending.

    Special Conventions:
        - If IDCG@K == 0.0 and DCG@K == 0.0 (e.g. unanswerable query or zero positive gold grades):
          return 1.0 if the query had zero relevant chunks (perfect rejection), else 0.0.
        - If IDCG@K == 0.0: return 0.0.

    Args:
        retrieved_items: Ranked candidate items.
        query_judgments_map: Mapping of chunk_id -> integer grade (0 to 3).
        k: Rank cutoff.

    Returns:
        Float in [0.0, 1.0].
    """
    if k < 1:
        raise ValueError(f"Cutoff k must be >= 1, got {k}")

    # Extract retrieved grades
    retrieved_grades = [query_judgments_map.get(item.chunk_id, 0) for item in retrieved_items[:k]]
    actual_dcg = calculate_dcg_at_k(retrieved_grades, k)

    # Calculate Ideal DCG from all gold judgments sorted descending
    all_positive_grades = sorted([g for g in query_judgments_map.values() if g > 0], reverse=True)
    ideal_dcg = calculate_dcg_at_k(all_positive_grades, k)

    if ideal_dcg <= 0.0:
        # If there were no positive grades possible in the corpus for this query
        return 1.0 if actual_dcg == 0.0 else 0.0

    return min(1.0, max(0.0, actual_dcg / ideal_dcg))


def calculate_map_at_k(
    retrieved_items: Sequence[RetrievalRankedItem],
    gold_relevant_chunk_ids: set[str],
    k: int,
) -> float:
    """Calculate Average Precision at K (AP@K) for a single query.

    Formula:
        AP@K = (1 / min(K, |gold_relevant|)) * sum_{i=1}^K (P@i * I(item_i is relevant))

    Args:
        retrieved_items: Ranked candidate items.
        gold_relevant_chunk_ids: Set of relevant chunk IDs.
        k: Rank cutoff.

    Returns:
        Float in [0.0, 1.0].
    """
    if k < 1:
        raise ValueError(f"Cutoff k must be >= 1, got {k}")
    total_gold = len(gold_relevant_chunk_ids)
    if total_gold == 0 or not retrieved_items:
        return 0.0

    norm_factor = min(k, total_gold)
    cumulative_hits = 0
    running_precision_sum = 0.0

    for idx, item in enumerate(retrieved_items[:k], start=1):
        if item.chunk_id in gold_relevant_chunk_ids:
            cumulative_hits += 1
            running_precision_sum += float(cumulative_hits) / float(idx)

    return min(1.0, running_precision_sum / float(norm_factor))


def evaluate_query_retrieval(
    retrieved_items: Sequence[RetrievalRankedItem],
    query: BenchmarkQuery,
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> dict[int, RetrievalMetrics]:
    """Compute complete retrieval metric suite across all evaluation cutoffs K.

    Args:
        retrieved_items: Ranked candidate items.
        query: BenchmarkQuery with validated relevant_chunk_ids and relevance_judgments.
        k_values: Sequence of integer K thresholds.

    Returns:
        Dictionary mapping K -> RetrievalMetrics.
    """
    gold_relevant_set = set(query.relevant_chunk_ids)
    judgments_map = {j.chunk_id: j.grade for j in query.relevance_judgments}

    metrics_map: dict[int, RetrievalMetrics] = {}

    for k in sorted(k_values):
        prec = calculate_precision_at_k(retrieved_items, gold_relevant_set, k)
        rec = calculate_recall_at_k(retrieved_items, gold_relevant_set, k)
        hit = calculate_hit_rate_at_k(retrieved_items, gold_relevant_set, k)
        mrr = calculate_reciprocal_rank(retrieved_items, gold_relevant_set, max_k=k)
        ndcg = calculate_ndcg_at_k(retrieved_items, judgments_map, k)
        map_sc = calculate_map_at_k(retrieved_items, gold_relevant_set, k)

        metrics_map[k] = RetrievalMetrics(
            k=k,
            precision=round(prec, 4),
            recall=round(rec, 4),
            hit_rate=round(hit, 4),
            mrr=round(mrr, 4),
            ndcg=round(ndcg, 4),
            map_score=round(map_sc, 4),
        )

    return metrics_map


def aggregate_strategy_metrics(
    query_results: Sequence[BenchmarkQueryResult],
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> StrategyAggregateSummary:
    """Aggregate per-query results into a macro-averaged StrategyAggregateSummary.

    Args:
        query_results: List of BenchmarkQueryResult for a single strategy.
        k_values: Cutoff K values evaluated.

    Returns:
        StrategyAggregateSummary with macro averages and latency statistics.
    """
    if not query_results:
        raise ValueError("query_results cannot be empty for aggregation.")

    strategy = query_results[0].strategy
    n = len(query_results)

    mean_prec: dict[int, float] = {}
    mean_rec: dict[int, float] = {}
    mean_hit: dict[int, float] = {}
    mean_ndcg: dict[int, float] = {}
    mean_map: dict[int, float] = {}

    for k in k_values:
        mean_prec[k] = round(
            sum(r.metrics_by_k[k].precision for r in query_results if k in r.metrics_by_k) / n, 4
        )
        mean_rec[k] = round(
            sum(r.metrics_by_k[k].recall for r in query_results if k in r.metrics_by_k) / n, 4
        )
        mean_hit[k] = round(
            sum(r.metrics_by_k[k].hit_rate for r in query_results if k in r.metrics_by_k) / n, 4
        )
        mean_ndcg[k] = round(
            sum(r.metrics_by_k[k].ndcg for r in query_results if k in r.metrics_by_k) / n, 4
        )
        mean_map[k] = round(
            sum(r.metrics_by_k[k].map_score for r in query_results if k in r.metrics_by_k) / n, 4
        )

    # Global MRR at max K
    max_k = max(k_values) if k_values else 10
    mean_mrr = round(
        sum(r.metrics_by_k.get(max_k, r.metrics_by_k[k_values[0]]).mrr for r in query_results) / n,
        4,
    )

    # Generation metrics if available
    faith_scores = [
        r.answer_quality.faithfulness_score
        for r in query_results
        if r.answer_quality and r.answer_quality.faithfulness_score is not None
    ]
    mean_faith = round(sum(faith_scores) / len(faith_scores), 4) if faith_scores else None

    cit_prec_scores = [
        r.answer_quality.citation_reference_precision
        for r in query_results
        if r.answer_quality and r.answer_quality.citation_reference_precision is not None
    ]
    mean_cit_prec = (
        round(sum(cit_prec_scores) / len(cit_prec_scores), 4) if cit_prec_scores else None
    )

    refusal_scores = [
        r.answer_quality.refusal_accuracy
        for r in query_results
        if r.answer_quality and r.answer_quality.refusal_accuracy is not None
    ]
    mean_refusal = round(sum(refusal_scores) / len(refusal_scores), 4) if refusal_scores else None

    # Latencies
    ret_latencies = sorted(r.latency.retrieval_latency_ms for r in query_results)
    tot_latencies = [r.latency.total_latency_ms for r in query_results]
    mean_ret_lat = round(sum(ret_latencies) / n, 2)
    median_ret_lat = round(ret_latencies[n // 2], 2)
    p95_idx = min(n - 1, int(0.95 * n))
    p95_ret_lat = round(ret_latencies[p95_idx], 2)
    mean_tot_lat = round(sum(tot_latencies) / n, 2)

    total_model_calls = sum(r.latency.total_model_invocations for r in query_results)

    return StrategyAggregateSummary(
        strategy=strategy,
        query_count=n,
        mean_precision_at_k=mean_prec,
        mean_recall_at_k=mean_rec,
        mean_hit_rate_at_k=mean_hit,
        mean_mrr=mean_mrr,
        mean_ndcg_at_k=mean_ndcg,
        mean_map_at_k=mean_map,
        mean_faithfulness=mean_faith,
        mean_citation_precision=mean_cit_prec,
        refusal_accuracy=mean_refusal,
        mean_retrieval_latency_ms=mean_ret_lat,
        median_retrieval_latency_ms=median_ret_lat,
        p95_retrieval_latency_ms=p95_ret_lat,
        mean_total_latency_ms=mean_tot_lat,
        total_model_invocations=total_model_calls,
    )
