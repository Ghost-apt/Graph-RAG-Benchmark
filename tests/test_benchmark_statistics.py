"""Unit tests for Phase 8 Benchmark Statistical Analysis & Hypothesis Testing."""

from graphrag_bench.benchmark_schemas import (
    BenchmarkQueryCategory,
    BenchmarkQueryResult,
    BenchmarkSplit,
    LatencyTelemetry,
    RetrievalMetrics,
    RetrievalStrategy,
)
from graphrag_bench.benchmark_statistics import (
    bootstrap_confidence_interval,
    calculate_paired_wilcoxon,
    compare_strategies_paired,
    compute_latency_summary,
)


def test_latency_summary():
    """Verify latency summary statistics with known sequence."""
    latencies = [10.0, 20.0, 30.0, 40.0, 50.0]
    summary = compute_latency_summary(latencies)

    assert summary["mean_ms"] == 30.0
    assert summary["median_ms"] == 30.0
    assert summary["min_ms"] == 10.0
    assert summary["max_ms"] == 50.0
    assert summary["p95_ms"] == 50.0
    assert summary["std_dev_ms"] > 0.0

    # Empty latency sequence
    empty_summary = compute_latency_summary([])
    assert empty_summary["mean_ms"] == 0.0
    assert empty_summary["p95_ms"] == 0.0


def test_bootstrap_confidence_interval():
    """Verify bootstrap 95% CI generates valid bounds."""
    data = [0.7, 0.75, 0.8, 0.82, 0.85, 0.9, 0.92]
    ci_lower, ci_upper = bootstrap_confidence_interval(
        data, num_bootstraps=500, ci_level=0.95, seed=42
    )

    assert ci_lower <= ci_upper
    assert min(data) <= ci_lower
    assert ci_upper <= max(data)

    # Constant sequence
    c_lower, c_upper = bootstrap_confidence_interval([0.5, 0.5, 0.5], num_bootstraps=100)
    assert c_lower == 0.5
    assert c_upper == 0.5


def test_wilcoxon_paired_test():
    """Verify paired Wilcoxon signed-rank test."""
    # Identical arrays -> zero difference
    stat, p_val = calculate_paired_wilcoxon([0.5, 0.6, 0.7], [0.5, 0.6, 0.7])
    assert stat == 0.0
    assert p_val == 1.0

    # Strictly superior treatment
    stat, p_val = calculate_paired_wilcoxon(
        [0.2, 0.3, 0.4, 0.3, 0.2, 0.4, 0.5, 0.3, 0.2, 0.3],
        [0.8, 0.9, 0.85, 0.9, 0.8, 0.95, 0.9, 0.85, 0.9, 0.8],
    )
    assert stat is not None and stat >= 0.0
    assert p_val is not None and p_val < 0.05


def test_compare_strategies_paired():
    """Verify compare_strategies_paired win/loss/tie accounting."""
    # Create 3 baseline queries
    b_results = [
        BenchmarkQueryResult(
            query_id=f"Q{i}",
            strategy=RetrievalStrategy.VECTOR,
            category=BenchmarkQueryCategory.SINGLE_HOP,
            split=BenchmarkSplit.DEV,
            answerable=True,
            metrics_by_k={
                10: RetrievalMetrics(
                    k=10, precision=0.2, recall=0.4, hit_rate=1.0, mrr=0.5, ndcg=0.4, map_score=0.3
                )
            },
            latency=LatencyTelemetry(total_latency_ms=15.0),
        )
        for i in range(1, 4)
    ]

    # Create 3 treatment queries (Q1 improves, Q2 ties, Q3 degrades)
    t_ndcgs = [0.8, 0.4, 0.1]
    t_results = [
        BenchmarkQueryResult(
            query_id=f"Q{i}",
            strategy=RetrievalStrategy.HYBRID,
            category=BenchmarkQueryCategory.SINGLE_HOP,
            split=BenchmarkSplit.DEV,
            answerable=True,
            metrics_by_k={
                10: RetrievalMetrics(
                    k=10,
                    precision=0.4,
                    recall=0.8,
                    hit_rate=1.0,
                    mrr=1.0,
                    ndcg=t_ndcgs[i - 1],
                    map_score=0.6,
                )
            },
            latency=LatencyTelemetry(total_latency_ms=25.0),
        )
        for i in range(1, 4)
    ]

    comp = compare_strategies_paired(
        baseline_results=b_results,
        treatment_results=t_results,
        baseline_strategy=RetrievalStrategy.VECTOR,
        treatment_strategy=RetrievalStrategy.HYBRID,
        k=10,
    )

    assert comp.baseline_strategy == RetrievalStrategy.VECTOR
    assert comp.treatment_strategy == RetrievalStrategy.HYBRID
    assert comp.wins == 1  # Q1 improved
    assert comp.ties == 1  # Q2 tied (0.4 vs 0.4)
    assert comp.losses == 1  # Q3 degraded (0.1 vs 0.4)
    assert comp.paired_query_count == 3
