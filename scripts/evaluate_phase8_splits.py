"""Script to evaluate Phase 8 Benchmark results across individual dataset splits.

Produces separate evaluations for:
1. Development Split (15 queries)
2. Validation Split (15 queries)
3. Held-Out Test Split (30 queries)

Calculates:
- Macro-averaged retrieval metrics across K in {1, 3, 5, 10}
- Answerable-only metrics (excluding unanswerable queries from recall/precision denominators)
- Unanswerable query metrics (empty result rate, false positive retrieval rate, refusal rate)
- Paired statistical contrasts on held-out test split (95% bootstrap CIs, Wilcoxon signed-rank tests)
- Holm-Bonferroni multiple testing corrections
- Dedicated run directory: data/phase8_benchmark/results/run_20261114_heldout_eval/
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphrag_bench.benchmark_schemas import (
    BenchmarkQueryResult,
    BenchmarkRunManifest,
)
from graphrag_bench.benchmark_statistics import (
    compute_bootstrap_confidence_interval,
    compute_wilcoxon_signed_rank,
)
from graphrag_bench.benchmark_storage import BenchmarkStorage


def holm_bonferroni_correction(p_values: list[float]) -> list[float]:
    """Apply Holm-Bonferroni step-down correction to a list of p-values.

    Args:
        p_values: List of raw unadjusted p-values.

    Returns:
        List of adjusted p-values in original input order.
    """
    m = len(p_values)
    if m == 0:
        return []

    # Pair with original index and sort ascending by p-value
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])

    adjusted = [0.0] * m
    running_max = 0.0

    for rank, (orig_idx, p) in enumerate(indexed_p):
        multiplier = m - rank
        adj = min(1.0, p * multiplier)
        adj = max(adj, running_max)  # Enforce monotonicity
        running_max = adj
        adjusted[orig_idx] = round(adj, 6)

    return adjusted


def compute_metrics_for_subset(
    query_results: list[BenchmarkQueryResult],
    k_values: list[int] = (1, 3, 5, 10),
    answerable_only: bool = False,
) -> dict[str, dict[str, float]]:
    """Compute macro-averaged retrieval metrics for a subset of results."""
    # Filter if requested
    filtered = [r for r in query_results if not answerable_only or r.answerable]
    n = len(filtered)
    if n == 0:
        return {}

    by_strategy: dict[str, list[BenchmarkQueryResult]] = {}
    for r in filtered:
        s_val = r.strategy.value if hasattr(r.strategy, "value") else str(r.strategy)
        by_strategy.setdefault(s_val, []).append(r)

    strat_metrics: dict[str, dict[str, float]] = {}

    for strat, res_list in by_strategy.items():
        n_strat = len(res_list)
        metrics: dict[str, float] = {"query_count": float(n_strat)}

        for k in k_values:
            p_vals = [r.metrics_by_k[k].precision for r in res_list if k in r.metrics_by_k]
            r_vals = [r.metrics_by_k[k].recall for r in res_list if k in r.metrics_by_k]
            h_vals = [r.metrics_by_k[k].hit_rate for r in res_list if k in r.metrics_by_k]
            ndcg_vals = [r.metrics_by_k[k].ndcg for r in res_list if k in r.metrics_by_k]
            map_vals = [r.metrics_by_k[k].map_score for r in res_list if k in r.metrics_by_k]
            mrr_vals = [r.metrics_by_k[k].mrr for r in res_list if k in r.metrics_by_k]

            metrics[f"P@{k}"] = round(sum(p_vals) / n_strat, 4) if p_vals else 0.0
            metrics[f"R@{k}"] = round(sum(r_vals) / n_strat, 4) if r_vals else 0.0
            metrics[f"Hit@{k}"] = round(sum(h_vals) / n_strat, 4) if h_vals else 0.0
            metrics[f"nDCG@{k}"] = round(sum(ndcg_vals) / n_strat, 4) if ndcg_vals else 0.0
            metrics[f"MAP@{k}"] = round(sum(map_vals) / n_strat, 4) if map_vals else 0.0
            metrics[f"MRR@{k}"] = round(sum(mrr_vals) / n_strat, 4) if mrr_vals else 0.0

        latencies = [r.latency.total_latency_ms for r in res_list]
        metrics["Mean_Latency_ms"] = round(sum(latencies) / n_strat, 2) if latencies else 0.0
        strat_metrics[strat] = metrics

    return strat_metrics


def run_split_analysis():
    print("=" * 80)
    print("   GRAPH RAG BENCHMARK — DATA SPLIT & HELD-OUT TEST RESEARCH AUDIT")
    print("=" * 80)

    storage = BenchmarkStorage()
    base_run_id = "run_20261114_060224"
    heldout_run_id = "run_20261114_heldout_eval"

    print(f"\n[+] Loading baseline execution results from: {base_run_id}")
    all_retrieval_results = storage.load_run_results(base_run_id, "retrieval_results.jsonl")
    all_answer_results = storage.load_run_results(base_run_id, "answer_results.jsonl")

    print(f"    - Loaded {len(all_retrieval_results)} retrieval results")
    print(f"    - Loaded {len(all_answer_results)} answer generation results")

    # Group by split
    by_split: dict[str, list[BenchmarkQueryResult]] = {}
    for r in all_retrieval_results:
        s_name = r.split.value if hasattr(r.split, "value") else str(r.split)
        by_split.setdefault(s_name.lower(), []).append(r)

    print("\n[+] Results partition by split:")
    for sp_name, r_list in by_split.items():
        unique_q = len({r.query_id for r in r_list})
        print(f"    * Split '{sp_name}': {len(r_list)} results ({unique_q} unique queries)")

    # -------------------------------------------------------------------------
    # PART A: SEPARATE RESULTS PER SPLIT (All queries in split)
    # -------------------------------------------------------------------------
    k_cutoffs = [1, 3, 5, 10]
    split_tables: dict[str, dict[str, dict[str, float]]] = {}

    for sp_name in ["dev", "validation", "test"]:
        res_list = by_split.get(sp_name, [])
        split_tables[sp_name] = compute_metrics_for_subset(
            res_list, k_cutoffs, answerable_only=False
        )

    print("\n" + "=" * 80)
    print("                     EVALUATION RESULTS BY DATASET SPLIT")
    print("=" * 80)

    for sp_name in ["dev", "validation", "test"]:
        tbl = split_tables[sp_name]
        q_count = int(next(iter(tbl.values()))["query_count"]) if tbl else 0
        print(f"\n--- SPLIT: {sp_name.upper()} ({q_count} queries per strategy) ---")
        print(
            f"{'Strategy':<10} | {'P@5':<8} | {'R@5':<8} | {'Hit@5':<8} | {'nDCG@5':<8} | {'nDCG@10':<8} | {'MAP@10':<8} | {'Mean Latency':<12}"
        )
        print("-" * 80)
        for st in ["vector", "graph", "hybrid"]:
            m = tbl.get(st, {})
            print(
                f"{st.upper():<10} | {m.get('P@5', 0):<8.4f} | {m.get('R@5', 0):<8.4f} | {m.get('Hit@5', 0):<8.4f} | {m.get('nDCG@5', 0):<8.4f} | {m.get('nDCG@10', 0):<8.4f} | {m.get('MAP@10', 0):<8.4f} | {m.get('Mean_Latency_ms', 0):<10.2f} ms"
            )

    # -------------------------------------------------------------------------
    # PART B: ANSWERABLE-ONLY VS UNANSWERABLE (Held-out Test Set)
    # -------------------------------------------------------------------------
    test_results = by_split.get("test", [])
    test_answerable_table = compute_metrics_for_subset(
        test_results, k_cutoffs, answerable_only=True
    )
    test_ans_count = (
        int(next(iter(test_answerable_table.values()))["query_count"])
        if test_answerable_table
        else 0
    )

    print("\n" + "=" * 80)
    print(f"      HELD-OUT TEST SET: ANSWERABLE-ONLY QUERIES ({test_ans_count} queries)")
    print("=" * 80)
    print(
        f"{'Strategy':<10} | {'P@5':<8} | {'R@5':<8} | {'Hit@5':<8} | {'nDCG@5':<8} | {'nDCG@10':<8} | {'MAP@10':<8} | {'Mean Latency':<12}"
    )
    print("-" * 80)
    for st in ["vector", "graph", "hybrid"]:
        m = test_answerable_table.get(st, {})
        print(
            f"{st.upper():<10} | {m.get('P@5', 0):<8.4f} | {m.get('R@5', 0):<8.4f} | {m.get('Hit@5', 0):<8.4f} | {m.get('nDCG@5', 0):<8.4f} | {m.get('nDCG@10', 0):<8.4f} | {m.get('MAP@10', 0):<8.4f} | {m.get('Mean_Latency_ms', 0):<10.2f} ms"
        )

    # Unanswerable query behavior
    test_unans = [r for r in test_results if not r.answerable]
    print("\n" + "-" * 80)
    print(
        f"      HELD-OUT TEST SET: UNANSWERABLE QUERIES ({len({r.query_id for r in test_unans})} queries)"
    )
    print("-" * 80)
    by_strat_unans: dict[str, list[BenchmarkQueryResult]] = {}
    for r in test_unans:
        s_val = r.strategy.value if hasattr(r.strategy, "value") else str(r.strategy)
        by_strat_unans.setdefault(s_val, []).append(r)

    print(
        f"{'Strategy':<10} | {'Total':<6} | {'Empty Results':<14} | {'False Pos Retr Rate':<20} | {'Mean Retr Items':<15}"
    )
    print("-" * 80)
    for st in ["vector", "graph", "hybrid"]:
        u_list = by_strat_unans.get(st, [])
        n_u = len(u_list)
        empty_cnt = sum(1 for r in u_list if len(r.retrieved_items) == 0)
        fp_rate = (n_u - empty_cnt) / n_u if n_u > 0 else 0.0
        avg_items = sum(len(r.retrieved_items) for r in u_list) / n_u if n_u > 0 else 0.0
        print(
            f"{st.upper():<10} | {n_u:<6} | {empty_cnt:<14} | {fp_rate * 100:<18.1f}% | {avg_items:<15.1f}"
        )

    # -------------------------------------------------------------------------
    # PART C: PAIRED STATISTICAL COMPARISONS ON HELD-OUT TEST (nDCG@10)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("      HELD-OUT TEST SET: PAIRED STATISTICAL HYPOTHESIS TESTING (nDCG@10)")
    print("=" * 80)

    # Group test results by strategy and query_id
    test_by_strat_qid: dict[str, dict[str, BenchmarkQueryResult]] = {}
    for r in test_results:
        s_val = r.strategy.value if hasattr(r.strategy, "value") else str(r.strategy)
        test_by_strat_qid.setdefault(s_val, {})[r.query_id] = r

    vec_test = test_by_strat_qid.get("vector", {})
    graph_test = test_by_strat_qid.get("graph", {})
    hybrid_test = test_by_strat_qid.get("hybrid", {})

    contrasts_data = [
        ("HYBRID vs VECTOR", "vector", "hybrid", vec_test, hybrid_test),
        ("HYBRID vs GRAPH", "graph", "hybrid", graph_test, hybrid_test),
        ("VECTOR vs GRAPH", "graph", "vector", graph_test, vec_test),
    ]

    raw_p_values = []
    contrast_results = []

    for name, base_strat, treat_strat, base_map, treat_map in contrasts_data:
        common_qids = sorted(set(base_map.keys()) & set(treat_map.keys()))
        diffs = []
        wins_treat = 0
        wins_base = 0
        ties = 0

        for qid in common_qids:
            b_val = base_map[qid].metrics_by_k[10].ndcg
            t_val = treat_map[qid].metrics_by_k[10].ndcg
            d = t_val - b_val
            diffs.append(d)
            if abs(d) < 1e-6:
                ties += 1
            elif d > 0:
                wins_treat += 1
            else:
                wins_base += 1

        mean_diff = round(sum(diffs) / len(diffs), 4)
        median_diff = round(sorted(diffs)[len(diffs) // 2], 4)
        ci_low, ci_high = compute_bootstrap_confidence_interval(
            diffs, num_bootstrap_samples=2000, confidence_level=0.95, random_seed=42
        )
        w_stat, w_pval = compute_wilcoxon_signed_rank(diffs)
        raw_p_values.append(w_pval)

        contrast_results.append(
            {
                "name": name,
                "base_strat": base_strat,
                "treat_strat": treat_strat,
                "n_pairs": len(diffs),
                "mean_diff": mean_diff,
                "median_diff": median_diff,
                "ci_95": (ci_low, ci_high),
                "w_stat": w_stat,
                "raw_pval": w_pval,
                "wins_treat": wins_treat,
                "wins_base": wins_base,
                "ties": ties,
            }
        )

    # Apply Holm-Bonferroni correction
    adjusted_p_values = holm_bonferroni_correction(raw_p_values)

    for idx, c in enumerate(contrast_results):
        c["adj_pval"] = adjusted_p_values[idx]
        c["significant"] = c["adj_pval"] < 0.05

        print(f"\nContrast: {c['name']} (N = {c['n_pairs']} held-out test queries)")
        print(f"  - Mean Difference (Delta) : {c['mean_diff']:+.4f}")
        print(f"  - Median Difference       : {c['median_diff']:+.4f}")
        print(f"  - 95% Bootstrap CI        : [{c['ci_95'][0]:.4f}, {c['ci_95'][1]:.4f}]")
        print(f"  - Wilcoxon W-statistic    : {c['w_stat']}")
        print(f"  - Unadjusted p-value      : {c['raw_pval']:.6f}")
        print(f"  - Holm-Adjusted p-value   : {c['adj_pval']:.6f}")
        print(
            f"  - Win / Loss / Tie        : {c['wins_treat']} Wins / {c['wins_base']} Losses / {c['ties']} Ties"
        )
        sig_str = "YES (p < 0.05)" if c["significant"] else "NO (p >= 0.05)"
        print(f"  - Statistically Significant: {sig_str}")

    # -------------------------------------------------------------------------
    # PART D: PERSIST DEDICATED HELD-OUT RUN
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"[+] Persisting dedicated Held-Out Test Run: {heldout_run_id}")
    print("=" * 80)

    heldout_dir = storage.results_dir / heldout_run_id
    heldout_dir.mkdir(parents=True, exist_ok=True)

    # Save retrieval_results.jsonl for heldout test set
    with open(heldout_dir / "retrieval_results.jsonl", "w", encoding="utf-8") as f:
        for r in test_results:
            f.write(r.model_dump_json() + "\n")

    # Save answer_results.jsonl for test set if any exist
    test_qids = {r.query_id for r in test_results}
    test_answers = [a for a in all_answer_results if a.query_id in test_qids]
    with open(heldout_dir / "answer_results.jsonl", "w", encoding="utf-8") as f:
        f.writelines(a.model_dump_json() + "\n" for a in test_answers)

    # Save metrics.json for heldout test set
    heldout_metrics_export = {
        "all_test_queries": split_tables["test"],
        "answerable_only_test_queries": test_answerable_table,
    }
    with open(heldout_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(heldout_metrics_export, f, indent=2)

    # Save comparison.json for heldout test set
    with open(heldout_dir / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(contrast_results, f, indent=2)

    # Export CSV for heldout test set
    storage.export_results_csv(heldout_run_id)

    # Compile and save run_manifest.json
    manifest = BenchmarkRunManifest(
        run_id=heldout_run_id,
        timestamp_utc=datetime.now(UTC).isoformat(),
        system_version="0.8.0",
        dataset_name="Phase 8 Held-Out Test Evaluation Partition (30 Queries)",
        total_queries=len(test_qids),
        strategies_evaluated=["vector", "graph", "hybrid"],
        k_values=k_cutoffs,
        strategy_summaries={},
        paired_comparisons=[],
        status="COMPLETED",
    )
    storage.save_run_manifest(heldout_run_id, manifest)

    # Save summary json across all splits
    all_splits_summary = {
        "dev": split_tables["dev"],
        "validation": split_tables["validation"],
        "test_all": split_tables["test"],
        "test_answerable_only": test_answerable_table,
        "paired_comparisons_test": contrast_results,
    }
    with open(heldout_dir / "splits_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_splits_summary, f, indent=2)

    print(f"[+] Held-out run artifacts persisted successfully to: {heldout_dir}")
    print("=" * 80)


if __name__ == "__main__":
    run_split_analysis()
