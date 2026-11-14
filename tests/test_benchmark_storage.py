"""Unit tests for Phase 8 Benchmark Storage and Persistence Engine."""

from pathlib import Path

import pytest

from graphrag_bench.benchmark_schemas import (
    BenchmarkQuery,
    BenchmarkQueryCategory,
    BenchmarkQueryResult,
    BenchmarkRunManifest,
    BenchmarkSplit,
    LatencyTelemetry,
    RelevanceJudgment,
    RetrievalMetrics,
    RetrievalStrategy,
)
from graphrag_bench.benchmark_storage import BenchmarkStorage


@pytest.fixture
def test_storage(tmp_path: Path) -> BenchmarkStorage:
    return BenchmarkStorage(root_dir=tmp_path / "phase8_benchmark")


def test_storage_directories(test_storage: BenchmarkStorage):
    """Verify storage initializes directory structure."""
    assert test_storage.root_dir.exists()
    assert test_storage.raw_docs_dir.exists()
    assert test_storage.processed_dir.exists()
    assert test_storage.chroma_dir.exists()
    assert test_storage.kg_dir.exists()
    assert test_storage.ground_truth_dir.exists()
    assert test_storage.results_dir.exists()


def test_dataset_save_and_load(test_storage: BenchmarkStorage):
    """Verify benchmark queries save and load round-trip."""
    queries = [
        BenchmarkQuery(
            query_id="BM-001",
            question="What is Graph RAG?",
            category=BenchmarkQueryCategory.SINGLE_HOP,
            split=BenchmarkSplit.DEV,
            relevant_chunk_ids=["chunk_1"],
            relevance_judgments=[
                RelevanceJudgment(chunk_id="chunk_1", document_id="doc_1", grade=3)
            ],
        ),
        BenchmarkQuery(
            query_id="BM-002",
            question="Explain hybrid reciprocal rank fusion.",
            category=BenchmarkQueryCategory.MULTI_HOP,
            split=BenchmarkSplit.TEST,
            relevant_chunk_ids=["chunk_2"],
            relevance_judgments=[
                RelevanceJudgment(chunk_id="chunk_2", document_id="doc_1", grade=2)
            ],
        ),
    ]

    saved_path = test_storage.save_benchmark_dataset(queries)
    assert saved_path.exists()

    loaded = test_storage.load_benchmark_dataset()
    assert len(loaded) == 2
    assert loaded[0].query_id == "BM-001"
    assert loaded[1].query_id == "BM-002"
    assert loaded[1].category == BenchmarkQueryCategory.MULTI_HOP


def test_run_directory_and_incremental_saving(test_storage: BenchmarkStorage):
    """Verify run directory creation, incremental JSONL appending, and resume capability."""
    run_id = "test_run_001"
    run_dir = test_storage.create_run_directory(run_id, config_metadata={"test": True})
    assert run_dir.exists()

    res1 = BenchmarkQueryResult(
        query_id="BM-001",
        strategy=RetrievalStrategy.VECTOR,
        category=BenchmarkQueryCategory.SINGLE_HOP,
        split=BenchmarkSplit.DEV,
        answerable=True,
        metrics_by_k={
            5: RetrievalMetrics(
                k=5, precision=0.2, recall=0.5, hit_rate=1.0, mrr=1.0, ndcg=0.6, map_score=0.4
            )
        },
        latency=LatencyTelemetry(total_latency_ms=12.5),
    )
    test_storage.append_query_result(run_id, res1, "retrieval_results.jsonl")

    res2 = BenchmarkQueryResult(
        query_id="BM-001",
        strategy=RetrievalStrategy.GRAPH,
        category=BenchmarkQueryCategory.SINGLE_HOP,
        split=BenchmarkSplit.DEV,
        answerable=True,
        metrics_by_k={
            5: RetrievalMetrics(
                k=5, precision=0.4, recall=0.8, hit_rate=1.0, mrr=1.0, ndcg=0.8, map_score=0.7
            )
        },
        latency=LatencyTelemetry(total_latency_ms=22.5),
    )
    test_storage.append_query_result(run_id, res2, "retrieval_results.jsonl")

    # Verify resume check finds both completed keys
    keys = test_storage.load_completed_query_keys(run_id, "retrieval_results.jsonl")
    assert ("BM-001", "vector") in keys
    assert ("BM-001", "graph") in keys
    assert len(keys) == 2

    # Load results
    results = test_storage.load_run_results(run_id, "retrieval_results.jsonl")
    assert len(results) == 2
    assert results[0].query_id == "BM-001"


def test_manifest_and_csv_export(test_storage: BenchmarkStorage):
    """Verify manifest persistence and CSV export."""
    run_id = "test_run_002"
    test_storage.create_run_directory(run_id)

    res1 = BenchmarkQueryResult(
        query_id="BM-001",
        strategy=RetrievalStrategy.HYBRID,
        category=BenchmarkQueryCategory.SINGLE_HOP,
        split=BenchmarkSplit.DEV,
        answerable=True,
        metrics_by_k={
            5: RetrievalMetrics(
                k=5, precision=0.6, recall=1.0, hit_rate=1.0, mrr=1.0, ndcg=0.9, map_score=0.85
            )
        },
        latency=LatencyTelemetry(total_latency_ms=18.0),
    )
    test_storage.append_query_result(run_id, res1, "retrieval_results.jsonl")

    manifest = BenchmarkRunManifest(
        run_id=run_id,
        timestamp_utc="2026-11-14T00:00:00Z",
        system_version="0.8.0",
        dataset_name="Test Dataset",
        total_queries=1,
        strategies_evaluated=["hybrid"],
        k_values=[5],
        status="COMPLETED",
    )
    test_storage.save_run_manifest(run_id, manifest)

    loaded_manifest = test_storage.load_run_manifest(run_id)
    assert loaded_manifest is not None
    assert loaded_manifest.run_id == run_id

    # Export CSV
    csv_path = test_storage.export_results_csv(run_id)
    assert csv_path.exists()
    assert csv_path.stat().st_size > 0

    # List runs
    runs = test_storage.list_runs()
    assert run_id in runs
