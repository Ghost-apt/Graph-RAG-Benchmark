"""Persistence and Artifact Management Engine for Phase 8 Benchmark.

Handles structured storage for:
- Benchmark evaluation dataset (JSON / JSONL / CSV)
- Corpus metadata and document manifests
- Experiment run directories:
  data/phase8_benchmark/results/<run_id>/
    ├── run_manifest.json
    ├── retrieval_results.jsonl
    ├── answer_results.jsonl
    ├── metrics.json
    ├── latency.json
    ├── comparison.json
    ├── results.csv
    └── run.log
- Resume capabilities for interrupted experiments
"""

import csv
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from graphrag_bench.benchmark_schemas import (
    BenchmarkDocument,
    BenchmarkQuery,
    BenchmarkQueryResult,
    BenchmarkRunManifest,
    PairedComparisonResult,
    StrategyAggregateSummary,
)
from graphrag_bench.logger import get_logger

logger = get_logger("benchmark_storage")

DEFAULT_BENCHMARK_ROOT = Path("data/phase8_benchmark")


class BenchmarkStorage:
    """Manages reading, writing, and resuming Phase 8 benchmark datasets and results."""

    def __init__(self, root_dir: Path | str | None = None) -> None:
        """Initialize storage manager with base directory.

        Args:
            root_dir: Base directory for Phase 8. Defaults to data/phase8_benchmark.
        """
        self.root_dir = Path(root_dir) if root_dir else DEFAULT_BENCHMARK_ROOT
        self.raw_docs_dir = self.root_dir / "raw_documents"
        self.processed_dir = self.root_dir / "processed"
        self.chroma_dir = self.root_dir / "chroma"
        self.kg_dir = self.root_dir / "knowledge_graph"
        self.ground_truth_dir = self.root_dir / "ground_truth"
        self.results_dir = self.root_dir / "results"
        self.snapshots_dir = self.root_dir / "snapshots"
        self.logs_dir = self.root_dir / "logs"

        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensure all required subdirectories exist."""
        for d in (
            self.root_dir,
            self.raw_docs_dir,
            self.processed_dir,
            self.chroma_dir,
            self.kg_dir,
            self.ground_truth_dir,
            self.results_dir,
            self.snapshots_dir,
            self.logs_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Benchmark Dataset Persistence
    # -------------------------------------------------------------------------

    def save_benchmark_dataset(
        self,
        queries: Sequence[BenchmarkQuery],
        documents: Sequence[BenchmarkDocument] | None = None,
        filename: str = "benchmark_dataset.json",
    ) -> Path:
        """Persist benchmark queries and documents to ground_truth directory.

        Args:
            queries: List of BenchmarkQuery instances.
            documents: Optional list of BenchmarkDocument instances.
            filename: Target JSON filename.

        Returns:
            Path to saved JSON file.
        """
        out_path = self.ground_truth_dir / filename
        data: dict[str, Any] = {
            "version": "1.0.0",
            "updated_at": datetime.now(UTC).isoformat(),
            "query_count": len(queries),
            "document_count": len(documents) if documents else 0,
            "queries": [q.model_dump(mode="json") for q in queries],
            "documents": [d.model_dump(mode="json") for d in (documents or [])],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info("Persisted %d benchmark queries to %s", len(queries), out_path)
        return out_path

    def load_dataset(
        self, filename: str = "benchmark_dataset.json"
    ) -> tuple[list[BenchmarkQuery], list[BenchmarkDocument]]:
        """Load benchmark queries and documents from ground_truth directory."""
        file_path = self.ground_truth_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Benchmark dataset not found at {file_path}")

        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        raw_queries = data.get("queries", [])
        queries = [BenchmarkQuery.model_validate(q) for q in raw_queries]

        raw_docs = data.get("documents", [])
        documents = [BenchmarkDocument.model_validate(d) for d in raw_docs]

        logger.info(
            "Loaded %d queries and %d documents from %s", len(queries), len(documents), file_path
        )
        return queries, documents

    def load_benchmark_dataset(
        self, filename: str = "benchmark_dataset.json"
    ) -> list[BenchmarkQuery]:
        """Convenience loader returning only BenchmarkQuery instances."""
        queries, _ = self.load_dataset(filename)
        return queries

    # -------------------------------------------------------------------------
    # Experiment Run Persistence & Resuming
    # -------------------------------------------------------------------------

    def create_run_directory(
        self,
        run_id: str | None = None,
        config_metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Create a new isolated run directory and optionally save run configuration.

        Args:
            run_id: Optional explicit run ID. Defaults to timestamp-based ID.
            config_metadata: Optional dictionary with configuration parameters.

        Returns:
            Path to created run directory.
        """
        if not run_id:
            ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            run_id = f"run_{ts}"

        run_path = self.results_dir / run_id
        run_path.mkdir(parents=True, exist_ok=True)

        if config_metadata:
            with open(run_path / "config.json", "w", encoding="utf-8") as f:
                json.dump(config_metadata, f, indent=2)

        return run_path

    def append_query_result(
        self,
        arg1: Any,
        arg2: Any,
        filename: str = "retrieval_results.jsonl",
    ) -> None:
        """Append a single query result to the run's JSONL results file.

        Accepts either (run_id, result, filename) or (result, run_id).
        """
        if isinstance(arg1, str):
            run_id = arg1
            result = arg2
        else:
            result = arg1
            run_id = arg2

        run_dir = self.results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        out_path = run_dir / filename
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(result.model_dump_json() + "\n")

    def load_completed_query_keys(
        self,
        run_id: str,
        filename: str = "retrieval_results.jsonl",
    ) -> set[tuple[str, str]]:
        """Load set of completed (query_id, strategy) pairs for resuming.

        Args:
            run_id: Experiment run ID.
            filename: Target JSONL filename.

        Returns:
            Set of (query_id, strategy_value) tuples already processed.
        """
        results_file = self.results_dir / run_id / filename
        if not results_file.exists():
            return set()

        completed: set[tuple[str, str]] = set()
        with open(results_file, encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    try:
                        data = json.loads(line_str)
                        strat = data.get("strategy")
                        strat_val = (
                            strat.get("value", strat) if isinstance(strat, dict) else str(strat)
                        )
                        completed.add((data["query_id"], strat_val))
                    except Exception:  # noqa: S112
                        continue
        return completed

    def load_completed_query_ids(self, run_id: str) -> set[tuple[str, str]]:
        """Alias for load_completed_query_keys."""
        return self.load_completed_query_keys(run_id, "retrieval_results.jsonl")

    def save_run_manifest(self, arg1: Any, arg2: Any = None) -> Path:
        """Save run manifest JSON. Accepts (manifest, run_id) or (run_id, manifest)."""
        if isinstance(arg1, BenchmarkRunManifest):
            manifest = arg1
            run_id = arg2 or manifest.run_id
        else:
            run_id = arg1
            manifest = arg2

        run_dir = self.results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        out_path = run_dir / "run_manifest.json"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(manifest.model_dump_json(indent=2))
        return out_path

    def load_run_manifest(self, run_id: str) -> BenchmarkRunManifest | None:
        """Load run manifest for a specific run ID."""
        run_dir = self.results_dir / run_id
        man_path = run_dir / "run_manifest.json"
        if not man_path.exists():
            return None
        with open(man_path, encoding="utf-8") as f:
            return BenchmarkRunManifest.model_validate_json(f.read())

    def save_run_aggregates(
        self,
        run_id: str,
        strategy_summaries: dict[str, StrategyAggregateSummary],
        paired_comparisons: list[PairedComparisonResult],
    ) -> None:
        """Save final aggregate metrics and pairwise comparisons for a completed run."""
        run_dir = self.results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        metrics_data = {
            strat: agg.model_dump(mode="json") for strat, agg in strategy_summaries.items()
        }
        with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=2)

        comp_data = [c.model_dump(mode="json") for c in paired_comparisons]
        with open(run_dir / "comparison.json", "w", encoding="utf-8") as f:
            json.dump(comp_data, f, indent=2)

    def save_aggregates_and_comparisons(
        self,
        aggregates: dict[str, StrategyAggregateSummary],
        comparisons: list[PairedComparisonResult],
        run_id: str,
    ) -> None:
        """Alias for save_run_aggregates."""
        self.save_run_aggregates(run_id, aggregates, comparisons)

    def load_run_results(
        self,
        run_id: str,
        filename: str = "retrieval_results.jsonl",
    ) -> list[BenchmarkQueryResult]:
        """Load list of BenchmarkQueryResult objects from a run's JSONL file."""
        run_dir = self.results_dir / run_id
        res_path = run_dir / filename
        results: list[BenchmarkQueryResult] = []
        if res_path.exists():
            with open(res_path, encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if line_str:
                        try:
                            results.append(BenchmarkQueryResult.model_validate_json(line_str))
                        except Exception as exc:
                            logger.warning("Failed to parse result line: %s", exc)
        return results

    def list_runs(self) -> list[str]:
        """List all run IDs in results directory."""
        if not self.results_dir.exists():
            return []
        return [d.name for d in sorted(self.results_dir.iterdir(), reverse=True) if d.is_dir()]

    def list_historical_runs(self) -> list[dict[str, Any]]:
        """List detailed metadata for all benchmark runs in results_dir."""
        runs = []
        if not self.results_dir.exists():
            return runs

        for d in sorted(self.results_dir.iterdir(), reverse=True):
            if d.is_dir():
                manifest_file = d / "run_manifest.json"
                manifest_data: dict[str, Any] = {}
                if manifest_file.exists():
                    try:
                        with open(manifest_file, encoding="utf-8") as f:
                            manifest_data = json.load(f)
                    except Exception:  # noqa: S110
                        pass
                runs.append(
                    {
                        "run_id": d.name,
                        "path": str(d),
                        "timestamp": manifest_data.get(
                            "timestamp_utc", manifest_data.get("timestamp", "")
                        ),
                        "status": manifest_data.get("status", "unknown"),
                        "queries_evaluated": manifest_data.get(
                            "total_queries", manifest_data.get("queries_evaluated", 0)
                        ),
                        "strategies": manifest_data.get("strategies_evaluated", []),
                    }
                )
        return runs

    def export_results_csv(
        self,
        run_id: str,
        destination_csv: Path | str | None = None,
        filename: str = "retrieval_results.jsonl",
    ) -> Path:
        """Export run results into flat CSV format for tabular analysis."""
        run_dir = self.results_dir / run_id
        results = self.load_run_results(run_id, filename)

        out_path = Path(destination_csv) if destination_csv else (run_dir / "results.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for r in results:
            strat_str = r.strategy.value if hasattr(r.strategy, "value") else str(r.strategy)
            cat_str = r.category.value if hasattr(r.category, "value") else str(r.category)
            split_str = r.split.value if hasattr(r.split, "value") else str(r.split)

            row: dict[str, Any] = {
                "query_id": r.query_id,
                "strategy": strat_str,
                "category": cat_str,
                "split": split_str,
                "answerable": r.answerable,
                "retrieved_count": len(r.retrieved_items),
                "total_latency_ms": r.latency.total_latency_ms,
                "status": r.status,
            }

            for k, m in r.metrics_by_k.items():
                row[f"P@{k}"] = m.precision
                row[f"R@{k}"] = m.recall
                row[f"HitRate@{k}"] = m.hit_rate
                row[f"MRR@{k}"] = m.mrr
                row[f"nDCG@{k}"] = m.ndcg
                row[f"MAP@{k}"] = m.map_score

            if r.answer_quality:
                row["faithfulness"] = r.answer_quality.faithfulness_score
                row["citation_precision"] = r.answer_quality.citation_reference_precision
                row["citation_coverage"] = r.answer_quality.citation_coverage
                row["refusal_accuracy"] = r.answer_quality.refusal_accuracy

            rows.append(row)

        if rows:
            fieldnames = list(rows[0].keys())
            # Ensure any extra columns are added
            for r_dict in rows:
                for k in r_dict:
                    if k not in fieldnames:
                        fieldnames.append(k)

            with open(out_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        else:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("query_id,strategy,status\n")

        logger.info("Exported %d result rows to CSV: %s", len(rows), out_path)
        return out_path
