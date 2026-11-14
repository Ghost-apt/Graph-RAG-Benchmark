"""Phase 8 Experiment Runner Script.

Executes:
1. Experiment A: Retrieval-only comparison across all 60 benchmark queries
   for Vector RAG, Graph RAG, and Hybrid RAG (180 query executions).
2. Experiment B: Grounded answer generation evaluation across representative queries.
3. Computes paired statistical contrasts (Wilcoxon signed-rank tests, 95% bootstrap CIs).
4. Persists complete run manifest, JSONL records, aggregate metrics, and CSV exports to
   data/phase8_benchmark/results/<run_id>/.
"""

import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphrag_bench.benchmark_runner import BenchmarkRunner
from graphrag_bench.benchmark_schemas import (
    BenchmarkSplit,
    RetrievalStrategy,
)
from graphrag_bench.logger import get_logger

logger = get_logger("run_phase8_experiments")


def run_experiments():
    print("=" * 80)
    print("       GRAPH RAG BENCHMARK — PHASE 8 FORMAL COMPARATIVE EVALUATION")
    print("=" * 80)

    runner = BenchmarkRunner()
    print(f"[+] Loaded {len(runner.queries)} benchmark queries.")

    if len(sys.argv) > 1 and sys.argv[1].startswith("run_"):
        run_id = sys.argv[1]
    else:
        run_id = f"run_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    print(f"[+] Initiating Benchmark Run: {run_id}")

    # -------------------------------------------------------------------------
    # Experiment A: Retrieval-Only Comparative Evaluation (All 60 Queries)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("STEP 1: Executing Experiment A (Retrieval Comparison across 60 queries * 3 strategies)")
    print("-" * 80)

    def progress_callback(cur, total, msg):
        pct = (cur / total) * 100.0
        sys.stdout.write(f"\r[{cur:3d}/{total:3d}] ({pct:5.1f}%) {msg[:60]:<60}")
        sys.stdout.flush()

    t_start_a = time.perf_counter()
    _, retrieval_results = runner.run_experiment_a(
        queries=runner.queries,
        strategies=[RetrievalStrategy.VECTOR, RetrievalStrategy.GRAPH, RetrievalStrategy.HYBRID],
        k_values=[1, 3, 5, 10],
        run_id=run_id,
        resume=True,
        progress_callback=progress_callback,
    )
    t_elapsed_a = time.perf_counter() - t_start_a
    print(
        f"\n[+] Experiment A completed in {t_elapsed_a:.2f}s ({len(retrieval_results)} queries evaluated)."
    )

    # -------------------------------------------------------------------------
    # Experiment B: Grounded Answer Generation (Representative Evaluation Sample)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("STEP 2: Executing Experiment B (Grounded Answer Generation on Representative Queries)")
    print("-" * 80)

    # Select 5 representative queries (1 per category: single-hop, multi-hop, entity-rel, comparison, unanswerable)
    selected_queries = []
    seen_cats = set()
    for q in runner.queries:
        if q.category not in seen_cats and q.split in (
            BenchmarkSplit.TEST,
            BenchmarkSplit.VALIDATION,
        ):
            selected_queries.append(q)
            seen_cats.add(q.category)
        if len(selected_queries) == 5:
            break

    print(
        f"[+] Selected {len(selected_queries)} representative queries across {len(seen_cats)} categories:"
    )
    for sq in selected_queries:
        print(f"    - [{sq.query_id}] ({sq.category.value}): {sq.question}")

    t_start_b = time.perf_counter()
    _, answer_results = runner.run_experiment_b(
        queries=selected_queries,
        strategies=[RetrievalStrategy.VECTOR, RetrievalStrategy.GRAPH, RetrievalStrategy.HYBRID],
        top_k=5,
        run_id=run_id,
        resume=True,
        progress_callback=progress_callback,
    )
    t_elapsed_b = time.perf_counter() - t_start_b
    print(
        f"\n[+] Experiment B completed in {t_elapsed_b:.2f}s ({len(answer_results)} answers evaluated)."
    )

    # -------------------------------------------------------------------------
    # STEP 3: Compile Manifest & Persist Statistical Aggregates
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("STEP 3: Compiling Run Manifest, Paired Statistics & Visual Summary")
    print("-" * 80)

    manifest = runner.compile_run_manifest(
        run_id=run_id,
        retrieval_results=retrieval_results,
        answer_results=answer_results,
        k_values=[1, 3, 5, 10],
    )

    print("\n================================================================================")
    print("                     PHASE 8 RETRIEVAL EVALUATION SUMMARY")
    print("================================================================================")
    print(
        f"{'Strategy':<10} | {'P@5':<8} | {'R@5':<8} | {'Hit@5':<8} | {'nDCG@5':<8} | {'nDCG@10':<8} | {'MAP@10':<8} | {'Mean Latency':<12}"
    )
    print("-" * 80)

    for strat_key, summary in manifest.strategy_summaries.items():
        p5 = summary.mean_precision_at_k.get(5, 0.0)
        r5 = summary.mean_recall_at_k.get(5, 0.0)
        h5 = summary.mean_hit_rate_at_k.get(5, 0.0)
        ndcg5 = summary.mean_ndcg_at_k.get(5, 0.0)
        ndcg10 = summary.mean_ndcg_at_k.get(10, 0.0)
        map10 = summary.mean_map_at_k.get(10, 0.0)
        lat = summary.mean_total_latency_ms
        print(
            f"{strat_key.upper():<10} | {p5:<8.4f} | {r5:<8.4f} | {h5:<8.4f} | {ndcg5:<8.4f} | {ndcg10:<8.4f} | {map10:<8.4f} | {lat:<10.2f} ms"
        )

    print("\n================================================================================")
    print("                      PAIRED STATISTICAL CONTRASTS (nDCG@10)")
    print("================================================================================")
    for comp in manifest.paired_comparisons:
        strat_a = comp.strategy_a.value.upper()
        strat_b = comp.strategy_b.value.upper()
        print(f"\nContrast: {strat_b} vs {strat_a}")
        print(f"  - Mean Difference  : {comp.mean_diff:+.4f}")
        print(
            f"  - 95% Bootstrap CI : [{comp.bootstrap_ci_95[0]:.4f}, {comp.bootstrap_ci_95[1]:.4f}]"
        )
        print(f"  - Wilcoxon p-value : {comp.wilcoxon_p_value}")
        print(
            f"  - Win/Loss/Tie     : {comp.win_count_b} Wins / {comp.win_count_a} Losses / {comp.tie_count} Ties"
        )
        print(
            f"  - Significant      : {'YES (p < 0.05)' if comp.is_statistically_significant else 'NO (p >= 0.05)'}"
        )
        print(f"  - Notes            : {comp.notes}")

    csv_path = runner.storage.results_dir / run_id / "results.csv"
    print("\n================================================================================")
    print(f"[+] All artifacts successfully persisted to: {runner.storage.results_dir / run_id}")
    print(f"[+] CSV exported to: {csv_path}")
    print("================================================================================")


if __name__ == "__main__":
    run_experiments()
