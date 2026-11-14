"""Statistical Analysis and Comparative Testing Engine for Phase 8 Benchmark.

Implements:
- Latency statistics (mean, median/p50, p95, min, max, std dev)
- Paired differences across identical queries
- 95% bootstrap confidence intervals
- Non-parametric paired hypothesis testing (Wilcoxon signed-rank test)
- Win/Loss/Tie accounting
- Statistical caveats and sample-size limitation disclosures
"""

import math
import random
from collections.abc import Callable, Sequence

from graphrag_bench.benchmark_schemas import (
    BenchmarkQueryResult,
    PairedComparisonResult,
    RetrievalStrategy,
)


def compute_latency_summary(latencies_ms: Sequence[float]) -> dict[str, float]:
    """Compute summary statistics over a sequence of latency observations in ms.

    Args:
        latencies_ms: List of non-negative latency measurements in milliseconds.

    Returns:
        Dictionary with mean, median, p95, min, max, and std_dev in milliseconds.
    """
    if not latencies_ms:
        return {
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "std_dev_ms": 0.0,
        }

    sorted_vals = sorted(latencies_ms)
    n = len(sorted_vals)
    mean_val = sum(sorted_vals) / float(n)

    # Median (p50)
    if n % 2 == 1:
        median_val = sorted_vals[n // 2]
    else:
        median_val = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0

    # p95 (nearest rank / linear interpolation)
    p95_idx = min(n - 1, math.ceil(0.95 * n) - 1)
    p95_val = sorted_vals[max(0, p95_idx)]

    # Variance and standard deviation
    if n > 1:
        variance = sum((x - mean_val) ** 2 for x in sorted_vals) / float(n - 1)
        std_val = math.sqrt(variance)
    else:
        std_val = 0.0

    return {
        "mean_ms": round(mean_val, 2),
        "median_ms": round(median_val, 2),
        "p95_ms": round(p95_val, 2),
        "min_ms": round(sorted_vals[0], 2),
        "max_ms": round(sorted_vals[-1], 2),
        "std_dev_ms": round(std_val, 2),
    }


def compute_bootstrap_confidence_interval(
    paired_differences: Sequence[float],
    num_bootstrap_samples: int = 2000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
    num_bootstraps: int | None = None,
    ci_level: float | None = None,
    seed: int | None = None,
) -> tuple[float, float]:
    """Compute non-parametric bootstrap confidence interval for the mean paired difference.

    Args:
        paired_differences: List of diffs d_i = score_A(q_i) - score_B(q_i).
        num_bootstrap_samples: Number of resamples (default: 2000).
        confidence_level: Desired coverage (default: 0.95).
        random_seed: Deterministic seed for reproducibility.
        num_bootstraps: Alias for num_bootstrap_samples.
        ci_level: Alias for confidence_level.
        seed: Alias for random_seed.

    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    samples_count = num_bootstraps if num_bootstraps is not None else num_bootstrap_samples
    cov_level = ci_level if ci_level is not None else confidence_level
    actual_seed = seed if seed is not None else random_seed

    if not paired_differences:
        return (0.0, 0.0)

    n = len(paired_differences)
    if n == 1 or all(abs(x - paired_differences[0]) < 1e-9 for x in paired_differences):
        val = round(paired_differences[0], 4)
        return (val, val)

    rng = random.Random(actual_seed)
    boot_means: list[float] = []

    for _ in range(samples_count):
        sample = [rng.choice(paired_differences) for _ in range(n)]
        boot_means.append(sum(sample) / float(n))

    boot_means.sort()
    alpha = (1.0 - cov_level) / 2.0
    low_idx = int(alpha * samples_count)
    high_idx = int((1.0 - alpha) * samples_count) - 1

    low = round(boot_means[max(0, low_idx)], 4)
    high = round(boot_means[min(samples_count - 1, high_idx)], 4)
    return (low, high)


def compute_wilcoxon_signed_rank(
    paired_differences: Sequence[float],
    treatment_values: Sequence[float] | None = None,
) -> tuple[float | None, float | None]:
    """Compute non-parametric Wilcoxon signed-rank test on paired differences.

    Args:
        paired_differences: List of d_i = score_A(q_i) - score_B(q_i), or baseline sequence.
        treatment_values: Optional treatment sequence if baseline and treatment passed separately.

    Returns:
        Tuple of (statistic, p_value), or (None, None) if sample size too small.
    """
    if treatment_values is not None:
        diffs = [float(t) - float(b) for b, t in zip(paired_differences, treatment_values)]
    else:
        diffs = list(paired_differences)

    # Filter out zero differences (ties)
    non_zero_diffs = [d for d in diffs if abs(d) > 1e-9]
    n = len(non_zero_diffs)
    if n == 0:
        # All differences are identical ties
        return 0.0, 1.0

    if n < 5:
        # Too few non-zero differences for asymptotic or ranked Wilcoxon
        return 0.0, 1.0

    try:
        from scipy import stats  # type: ignore

        res = stats.wilcoxon(non_zero_diffs, alternative="two-sided")
        return round(float(res.statistic), 4), round(float(res.pvalue), 6)
    except (ImportError, Exception):
        # Deterministic ranking and normal approximation fallback
        abs_diffs = [(abs(d), 1 if d > 0 else -1) for d in non_zero_diffs]
        abs_diffs.sort(key=lambda x: x[0])

        ranks = []
        i = 0
        while i < n:
            j = i
            while j < n and abs(abs_diffs[j][0] - abs_diffs[i][0]) < 1e-9:
                j += 1
            avg_rank = (i + 1 + j) / 2.0
            for k in range(i, j):
                ranks.append((avg_rank, abs_diffs[k][1]))
            i = j

        # Sum of positive ranks
        w_plus = sum(r for r, sign in ranks if sign > 0)
        w_minus = sum(r for r, sign in ranks if sign < 0)
        w_stat = min(w_plus, w_minus)

        # Normal approximation for n >= 5
        e_w = n * (n + 1) / 4.0
        var_w = n * (n + 1) * (2 * n + 1) / 24.0
        if var_w <= 0.0:
            return float(w_stat), 1.0

        z = (w_stat - e_w) / math.sqrt(var_w)
        p_val = 2.0 * 0.5 * math.erfc(abs(z) / math.sqrt(2.0))
        return round(float(w_stat), 4), round(float(p_val), 6)


# Aliases for convenience and backward compatibility
bootstrap_confidence_interval = compute_bootstrap_confidence_interval
calculate_paired_wilcoxon = compute_wilcoxon_signed_rank


def compare_strategies_paired(
    results_a: Sequence[BenchmarkQueryResult] | None = None,
    results_b: Sequence[BenchmarkQueryResult] | None = None,
    metric_extractor: Callable[[BenchmarkQueryResult], float] | None = None,
    metric_name: str | None = None,
    strategy_a: RetrievalStrategy | None = None,
    strategy_b: RetrievalStrategy | None = None,
    baseline_results: Sequence[BenchmarkQueryResult] | None = None,
    treatment_results: Sequence[BenchmarkQueryResult] | None = None,
    baseline_strategy: RetrievalStrategy | None = None,
    treatment_strategy: RetrievalStrategy | None = None,
    k: int = 10,
) -> PairedComparisonResult:
    """Execute paired statistical comparison of Strategy A (baseline) vs Strategy B (treatment) across identical queries.

    Args:
        results_a: List of query results for Strategy A (or baseline_results).
        results_b: List of query results for Strategy B (or treatment_results).
        metric_extractor: Optional callable (BenchmarkQueryResult -> float) extracting the target metric.
        metric_name: Optional human-readable metric label (e.g. 'nDCG@10').
        strategy_a: Strategy A enum (or baseline_strategy).
        strategy_b: Strategy B enum (or treatment_strategy).
        baseline_results: Alias for results_a.
        treatment_results: Alias for results_b.
        baseline_strategy: Alias for strategy_a.
        treatment_strategy: Alias for strategy_b.
        k: Default cutoff K if metric_extractor is None.

    Returns:
        PairedComparisonResult with mean diff, median diff, win/loss counts, CIs, and p-values.
    """
    actual_results_a = results_a if results_a is not None else baseline_results
    actual_results_b = results_b if results_b is not None else treatment_results
    actual_strat_a = strategy_a if strategy_a is not None else baseline_strategy
    actual_strat_b = strategy_b if strategy_b is not None else treatment_strategy

    if actual_results_a is None or actual_results_b is None:
        raise ValueError("Both results_a and results_b must be provided.")

    if actual_strat_a is None:
        actual_strat_a = (
            actual_results_a[0].strategy if actual_results_a else RetrievalStrategy.VECTOR
        )
    if actual_strat_b is None:
        actual_strat_b = (
            actual_results_b[0].strategy if actual_results_b else RetrievalStrategy.HYBRID
        )

    if metric_name is None:
        metric_name = f"nDCG@{k}"

    if metric_extractor is None:

        def default_extractor(r: BenchmarkQueryResult) -> float:
            if k in r.metrics_by_k:
                return r.metrics_by_k[k].ndcg
            # Fallback to first available K or 0.0
            for m in r.metrics_by_k.values():
                return m.ndcg
            return 0.0

        extractor = default_extractor
    else:
        extractor = metric_extractor

    map_a = {r.query_id: r for r in actual_results_a}
    map_b = {r.query_id: r for r in actual_results_b}

    common_qids = sorted(set(map_a.keys()).intersection(map_b.keys()))
    if not common_qids:
        raise ValueError("No common query IDs between Strategy A and Strategy B results.")

    diffs: list[float] = []
    wins_a = 0
    wins_b = 0
    ties = 0

    for qid in common_qids:
        score_a = float(extractor(map_a[qid]))
        score_b = float(extractor(map_b[qid]))
        diff = score_a - score_b
        diffs.append(diff)

        if diff > 1e-4:
            wins_a += 1
        elif diff < -1e-4:
            wins_b += 1
        else:
            ties += 1

    n = len(diffs)
    mean_diff = round(sum(diffs) / float(n), 4)

    sorted_diffs = sorted(diffs)
    if n % 2 == 1:
        median_diff = round(sorted_diffs[n // 2], 4)
    else:
        median_diff = round((sorted_diffs[n // 2 - 1] + sorted_diffs[n // 2]) / 2.0, 4)

    ci_95 = compute_bootstrap_confidence_interval(diffs)
    w_stat, p_val = compute_wilcoxon_signed_rank(diffs)

    sig = (p_val is not None) and (p_val < 0.05) and (ci_95[0] > 0.0 or ci_95[1] < 0.0)

    notes = ""
    if n < 30:
        notes = f"Exploratory sample size (N={n}). Statistical power is limited; interpret p-values cautiously."
    elif sig:
        favored = actual_strat_a.value.upper() if mean_diff > 0 else actual_strat_b.value.upper()
        notes = f"Statistically significant difference favoring {favored} (p={p_val:.4f}, 95% CI [{ci_95[0]:.4f}, {ci_95[1]:.4f}])."
    else:
        notes = f"No statistically significant difference detected at alpha=0.05 (p={p_val if p_val is not None else 'N/A'})."

    return PairedComparisonResult(
        strategy_a=actual_strat_a,
        strategy_b=actual_strat_b,
        metric_name=metric_name,
        mean_diff=mean_diff,
        median_diff=median_diff,
        win_count_a=wins_a,
        win_count_b=wins_b,
        tie_count=ties,
        bootstrap_ci_95=ci_95,
        wilcoxon_stat=w_stat,
        wilcoxon_p_value=p_val,
        is_statistically_significant=sig,
        sample_size=n,
        notes=notes,
    )
