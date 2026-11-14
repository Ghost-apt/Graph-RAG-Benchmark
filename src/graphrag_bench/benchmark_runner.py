"""Benchmark Runner Engine for Phase 8 Formal Comparative Evaluation.

Orchestrates:
- Experiment A: Retrieval-Only Evaluation across Vector RAG, Graph RAG, and Hybrid RAG.
- Experiment B: End-to-End Grounded Generation & Provenance Verification.
- Seamless execution against isolated benchmark environment (data/phase8_benchmark).
- Incremental JSONL saving, resume-on-failure, and multi-metric aggregation.
"""

import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from graphrag_bench.answer_schemas import AnswerStatus, GroundedAnswer
from graphrag_bench.benchmark_metrics import (
    aggregate_strategy_metrics,
    evaluate_query_retrieval,
)
from graphrag_bench.benchmark_schemas import (
    AnswerQualityMetrics,
    BenchmarkQuery,
    BenchmarkQueryResult,
    BenchmarkRunManifest,
    LatencyTelemetry,
    PairedComparisonResult,
    RetrievalMetrics,
    RetrievalRankedItem,
    RetrievalStrategy,
    StrategyAggregateSummary,
)
from graphrag_bench.benchmark_statistics import (
    compare_strategies_paired,
)
from graphrag_bench.benchmark_storage import BenchmarkStorage
from graphrag_bench.config import Settings, get_settings
from graphrag_bench.document_repository import DocumentRepository
from graphrag_bench.embedding_service import EmbeddingService
from graphrag_bench.graph_search import GraphSearchEngine
from graphrag_bench.graph_store import GraphStore
from graphrag_bench.hybrid_retriever import HybridRetriever
from graphrag_bench.logger import get_logger
from graphrag_bench.ollama_client import OllamaClient
from graphrag_bench.rag_pipeline import RAGPipeline
from graphrag_bench.result_normalizer import ResultNormalizer
from graphrag_bench.semantic_search import SemanticSearch
from graphrag_bench.vector_store import VectorStore

logger = get_logger("benchmark_runner")


def _calculate_lexical_overlap(text_a: str, text_b: str) -> float:
    """Calculate token Jaccard similarity between two texts for lexical alignment."""
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / float(len(union))


class BenchmarkRunner:
    """Executes comparative evaluation runs across Vector, Graph, and Hybrid RAG."""

    def __init__(
        self,
        benchmark_root: Path | str | None = None,
        vector_store: VectorStore | None = None,
        graph_store: GraphStore | None = None,
        doc_repo: DocumentRepository | None = None,
        embedding_service: EmbeddingService | None = None,
        vector_search: SemanticSearch | None = None,
        graph_search: GraphSearchEngine | None = None,
        normalizer: ResultNormalizer | None = None,
        hybrid_retriever: HybridRetriever | None = None,
        rag_pipeline: RAGPipeline | None = None,
        storage: BenchmarkStorage | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize benchmark runner with isolated benchmark environment."""
        self.settings = settings or get_settings()
        self.storage = storage or BenchmarkStorage(benchmark_root)
        self.root_dir = self.storage.root_dir

        # Initialize DocumentRepository against benchmark processed dir
        self.doc_repo = doc_repo or DocumentRepository(
            processed_dir=self.storage.processed_dir,
            uploads_dir=self.storage.raw_docs_dir,
        )

        # Initialize isolated VectorStore
        self.vector_store = vector_store or VectorStore(
            persist_directory=self.storage.chroma_dir,
            collection_name="phase8_benchmark_documents",
            settings=self.settings,
        )

        # Initialize isolated GraphStore
        if graph_store is not None:
            self.graph_store = graph_store
        else:
            self.graph_store = GraphStore()
            kg_path = self.storage.kg_dir / "knowledge_graph.json"
            if kg_path.exists():
                try:
                    self.graph_store.load_json(kg_path)
                    logger.info(
                        "Loaded benchmark KG: %d nodes, %d edges",
                        self.graph_store.node_count,
                        self.graph_store.edge_count,
                    )
                except Exception as exc:
                    logger.error("Failed to load benchmark knowledge graph: %s", exc)

        # Initialize shared Ollama client & Embedding service
        self.ollama_client = OllamaClient(settings=self.settings)
        self.embedding_service = embedding_service or EmbeddingService(
            client=self.ollama_client,
            settings=self.settings,
        )

        # Initialize retrievers
        self.vector_search = vector_search or SemanticSearch(
            vector_store=self.vector_store,
            embedding_service=self.embedding_service,
            settings=self.settings,
        )
        self.graph_search = graph_search or GraphSearchEngine(
            store=self.graph_store,
            settings=self.settings,
        )
        self.normalizer = normalizer or ResultNormalizer(repository=self.doc_repo)
        self.hybrid_retriever = hybrid_retriever or HybridRetriever(
            vector_search=self.vector_search,
            graph_search=self.graph_search,
            repository=self.doc_repo,
            normalizer=self.normalizer,
            settings=self.settings,
        )

        # Initialize end-to-end RAG Pipeline
        self.rag_pipeline = rag_pipeline or RAGPipeline(
            hybrid_retriever=self.hybrid_retriever,
            vector_search=self.vector_search,
            graph_retriever=self.graph_search.retriever,
            ollama_client=self.ollama_client,
        )

        # Load benchmark queries
        self.queries: list[BenchmarkQuery] = []
        self.queries_by_id: dict[str, BenchmarkQuery] = {}
        self.reload_queries()

    def reload_queries(self) -> None:
        """Load or refresh benchmark queries from storage."""
        try:
            self.queries = self.storage.load_benchmark_dataset()
            self.queries_by_id = {q.query_id: q for q in self.queries}
            logger.info(
                "Loaded %d benchmark queries from %s",
                len(self.queries),
                self.storage.ground_truth_dir,
            )
        except Exception as exc:
            logger.warning("Could not load benchmark dataset: %s", exc)
            self.queries = []
            self.queries_by_id = {}

    def run_query_retrieval(
        self,
        query: BenchmarkQuery,
        strategy: RetrievalStrategy,
        k_values: Sequence[int] = (1, 3, 5, 10),
    ) -> BenchmarkQueryResult:
        """Execute single retrieval query under a specific strategy and evaluate metrics.

        Args:
            query: Validated BenchmarkQuery instance.
            strategy: Target retrieval strategy (VECTOR, GRAPH, or HYBRID).
            k_values: Cutoff ranks to evaluate.

        Returns:
            BenchmarkQueryResult with ranked items, IR metrics, and timing breakdown.
        """
        max_k = max(k_values) if k_values else 10
        gold_map = {j.chunk_id: j.grade for j in query.relevance_judgments}
        gold_relevant_set = set(query.relevant_chunk_ids)

        retrieved_items: list[RetrievalRankedItem] = []
        latency = LatencyTelemetry()
        status = "SUCCESS"
        error_msg: str | None = None

        t0 = time.perf_counter()

        try:
            if strategy == RetrievalStrategy.VECTOR:
                search_resp = self.vector_search.search(query=query.question, top_k=max_k)
                t_retrieval = (time.perf_counter() - t0) * 1000.0

                for item in search_resp.items:
                    grade = gold_map.get(item.chunk_id, 0)
                    retrieved_items.append(
                        RetrievalRankedItem(
                            rank=item.rank,
                            chunk_id=item.chunk_id,
                            document_id=item.document_id,
                            document_title=str(
                                item.metadata.get("document_title", item.document_id)
                            ),
                            modality="vector",
                            score=float(item.normalized_score),
                            raw_score=float(item.cosine_similarity),
                            gold_grade=grade,
                            is_relevant=(grade >= 2 or item.chunk_id in gold_relevant_set),
                            evidence_text=item.text,
                        )
                    )

                latency = LatencyTelemetry(
                    retrieval_latency_ms=round(t_retrieval, 2),
                    vector_search_ms=round(search_resp.duration_ms, 2),
                    total_latency_ms=round(t_retrieval, 2),
                    total_model_invocations=0,
                )

            elif strategy == RetrievalStrategy.GRAPH:
                g_resp = self.graph_search.search_graph(
                    query=query.question,
                    top_k=max_k,
                    max_hops=2,
                )
                t_retrieval = (time.perf_counter() - t0) * 1000.0

                candidates, _ = self.normalizer.normalize_graph_results(g_resp)
                for idx, cand in enumerate(candidates[:max_k], start=1):
                    grade = gold_map.get(cand.chunk_id, 0)
                    retrieved_items.append(
                        RetrievalRankedItem(
                            rank=idx,
                            chunk_id=cand.chunk_id,
                            document_id=cand.document_id,
                            document_title=cand.source_filename,
                            modality="graph",
                            score=float(cand.graph_score),
                            raw_score=float(cand.graph_score),
                            gold_grade=grade,
                            is_relevant=(grade >= 2 or cand.chunk_id in gold_relevant_set),
                            evidence_text=cand.chunk_text[:300] if cand.chunk_text else "",
                        )
                    )

                model_calls = 1 if getattr(g_resp, "llm_invoked", False) else 0
                latency = LatencyTelemetry(
                    retrieval_latency_ms=round(t_retrieval, 2),
                    query_entity_extraction_ms=round(
                        getattr(g_resp, "extraction_latency_ms", 0.0), 2
                    ),
                    graph_traversal_ms=round(getattr(g_resp, "traversal_latency_ms", 0.0), 2),
                    total_latency_ms=round(t_retrieval, 2),
                    query_extraction_model_calls=model_calls,
                    total_model_invocations=model_calls,
                )

            elif strategy == RetrievalStrategy.HYBRID:
                h_resp = self.hybrid_retriever.search_hybrid(
                    query=query.question,
                    final_top_k=max_k,
                )
                t_retrieval = (time.perf_counter() - t0) * 1000.0

                for cand in h_resp.candidates[:max_k]:
                    grade = gold_map.get(cand.chunk_id, 0)
                    retrieved_items.append(
                        RetrievalRankedItem(
                            rank=cand.fused_rank,
                            chunk_id=cand.chunk_id,
                            document_id=cand.document_id,
                            document_title=cand.source_filename,
                            modality="hybrid",
                            score=float(cand.rrf_score),
                            raw_score=float(cand.rrf_score),
                            gold_grade=grade,
                            is_relevant=(grade >= 2 or cand.chunk_id in gold_relevant_set),
                            evidence_text=cand.chunk_text[:300] if cand.chunk_text else "",
                        )
                    )

                model_calls = 1 if getattr(h_resp, "llm_invoked", False) else 0
                latency = LatencyTelemetry(
                    retrieval_latency_ms=round(t_retrieval, 2),
                    vector_search_ms=round(getattr(h_resp, "vector_latency_ms", 0.0), 2),
                    graph_traversal_ms=round(getattr(h_resp, "graph_latency_ms", 0.0), 2),
                    fusion_ms=round(getattr(h_resp, "fusion_latency_ms", 0.0), 2),
                    total_latency_ms=round(t_retrieval, 2),
                    query_extraction_model_calls=model_calls,
                    total_model_invocations=model_calls,
                )

        except Exception as exc:
            logger.error(
                "Error executing %s retrieval on %s: %s", strategy.value, query.query_id, exc
            )
            status = "ERROR"
            error_msg = str(exc)
            latency = LatencyTelemetry(
                total_latency_ms=round((time.perf_counter() - t0) * 1000.0, 2)
            )

        # Evaluate metrics across all cutoffs
        metrics_by_k = evaluate_query_retrieval(retrieved_items, query, k_values=k_values)

        return BenchmarkQueryResult(
            query_id=query.query_id,
            strategy=strategy,
            category=query.category,
            split=query.split,
            answerable=query.answerable,
            retrieved_items=retrieved_items,
            metrics_by_k=metrics_by_k,
            latency=latency,
            status=status,
            error_message=error_msg,
        )

    def run_query_generation(
        self,
        query: BenchmarkQuery,
        strategy: RetrievalStrategy,
        top_k: int = 5,
    ) -> BenchmarkQueryResult:
        """Execute single end-to-end question answering query (Experiment B).

        Args:
            query: BenchmarkQuery instance.
            strategy: Target retrieval strategy.
            top_k: Evidence items to retrieve.

        Returns:
            BenchmarkQueryResult with generated answer, quality metrics, and latency breakdown.
        """
        gold_relevant_set = set(query.relevant_chunk_ids)
        t0 = time.perf_counter()
        status = "SUCCESS"
        error_msg: str | None = None
        ans_obj: GroundedAnswer | None = None
        retrieved_items: list[RetrievalRankedItem] = []
        metrics_by_k: dict[int, RetrievalMetrics] = {}
        answer_quality: AnswerQualityMetrics | None = None
        latency = LatencyTelemetry()

        try:
            mode_str = "hybrid"
            if strategy == RetrievalStrategy.VECTOR:
                mode_str = "vector"
            elif strategy == RetrievalStrategy.GRAPH:
                mode_str = "graph"

            ans_obj = self.rag_pipeline.answer_question(
                request=query.question,
                retrieval_mode=mode_str,
                top_k=top_k,
                require_citations=True,
                enable_faithfulness_check=True,
            )

            # Map evidence items to RetrievalRankedItem
            gold_map = {j.chunk_id: j.grade for j in query.relevance_judgments}
            for idx, ev in enumerate(ans_obj.evidence_items, start=1):
                grade = gold_map.get(ev.chunk_id, 0)
                ev_score = float(getattr(ev, "retrieval_score", getattr(ev, "score", 0.0)))
                ev_text = getattr(ev, "content", getattr(ev, "text", "")) or ""
                retrieved_items.append(
                    RetrievalRankedItem(
                        rank=idx,
                        chunk_id=ev.chunk_id,
                        document_id=ev.document_id,
                        document_title=ev.document_title,
                        modality=strategy.value,
                        score=ev_score,
                        raw_score=ev_score,
                        gold_grade=grade,
                        is_relevant=(grade >= 2 or ev.chunk_id in gold_relevant_set),
                        evidence_text=ev_text[:300],
                    )
                )

            metrics_by_k = evaluate_query_retrieval(retrieved_items, query, k_values=[top_k])

            # Evaluate answer quality metrics
            cited_chunk_ids = {c.chunk_id for c in ans_obj.citations if c.chunk_id}
            if gold_relevant_set:
                citation_coverage = len(cited_chunk_ids & gold_relevant_set) / float(
                    len(gold_relevant_set)
                )
            else:
                citation_coverage = 1.0 if not cited_chunk_ids else 0.0

            refusal_acc: float | None = None
            if not query.answerable:
                # Correctly refused if pipeline rejected or status is REFUSED / INSUFFICIENT
                refused_keywords = [
                    "insufficient",
                    "not enough",
                    "cannot answer",
                    "cannot find",
                    "no information",
                ]
                ans_lower = ans_obj.answer.lower()
                is_refused = ans_obj.status in (
                    AnswerStatus.REFUSED,
                    AnswerStatus.INSUFFICIENT_EVIDENCE,
                    AnswerStatus.GUARDRAIL_BLOCKED,
                ) or any(kw in ans_lower for kw in refused_keywords)
                refusal_acc = 1.0 if is_refused else 0.0

            lexical_correctness: float | None = None
            if query.answerable and query.expected_answer:
                lexical_correctness = round(
                    _calculate_lexical_overlap(ans_obj.answer, query.expected_answer), 4
                )

            answer_quality = AnswerQualityMetrics(
                answer_correctness=lexical_correctness,
                answer_relevance=1.0 if ans_obj.status == AnswerStatus.SUCCESS else 0.5,
                faithfulness_score=ans_obj.faithfulness_score,
                citation_reference_precision=ans_obj.citation_reference_precision,
                citation_coverage=round(citation_coverage, 4),
                refusal_accuracy=refusal_acc,
            )

            # Map telemetry
            t_obj = getattr(ans_obj, "telemetry", None)
            if t_obj:
                latency = LatencyTelemetry(
                    retrieval_latency_ms=round(getattr(t_obj, "retrieval_ms", 0.0), 2),
                    query_entity_extraction_ms=round(
                        getattr(t_obj, "query_entity_extraction_ms", 0.0), 2
                    ),
                    graph_traversal_ms=round(getattr(t_obj, "graph_traversal_ms", 0.0), 2),
                    vector_search_ms=round(getattr(t_obj, "vector_retrieval_ms", 0.0), 2),
                    fusion_ms=round(getattr(t_obj, "fusion_ms", 0.0), 2),
                    generation_latency_ms=round(getattr(t_obj, "generation_ms", 0.0), 2),
                    total_latency_ms=round(
                        getattr(t_obj, "total_duration_ms", (time.perf_counter() - t0) * 1000.0), 2
                    ),
                    query_extraction_model_calls=getattr(t_obj, "query_extraction_model_calls", 0),
                    answer_generation_model_calls=getattr(
                        t_obj, "answer_generation_model_calls", 1
                    ),
                    total_model_invocations=getattr(t_obj, "total_model_invocations", 1),
                )
            else:
                latency = LatencyTelemetry(
                    total_latency_ms=round((time.perf_counter() - t0) * 1000.0, 2)
                )

        except Exception as exc:
            logger.error(
                "Error executing %s generation on %s: %s", strategy.value, query.query_id, exc
            )
            status = "ERROR"
            error_msg = str(exc)
            latency = LatencyTelemetry(
                total_latency_ms=round((time.perf_counter() - t0) * 1000.0, 2)
            )

        return BenchmarkQueryResult(
            query_id=query.query_id,
            strategy=strategy,
            category=query.category,
            split=query.split,
            answerable=query.answerable,
            retrieved_items=retrieved_items,
            metrics_by_k=metrics_by_k,
            answer_quality=answer_quality,
            latency=latency,
            generated_answer=ans_obj.answer if ans_obj else "",
            status=status,
            error_message=error_msg,
        )

    def run_experiment_a(
        self,
        queries: Sequence[BenchmarkQuery] | None = None,
        strategies: Sequence[RetrievalStrategy] | None = None,
        k_values: Sequence[int] = (1, 3, 5, 10),
        run_id: str | None = None,
        resume: bool = True,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> tuple[str, list[BenchmarkQueryResult]]:
        """Execute Experiment A: Retrieval-Only Comparative Evaluation.

        Args:
            queries: Benchmark queries to evaluate (defaults to all queries).
            strategies: Strategies to evaluate (defaults to VECTOR, GRAPH, HYBRID).
            k_values: Evaluation cutoff thresholds.
            run_id: Optional existing or new run identifier.
            resume: Whether to resume and skip completed query/strategy pairs.
            progress_callback: Optional UI progress callback (current, total, message).

        Returns:
            Tuple of (run_id, list of BenchmarkQueryResult).
        """
        target_queries = list(queries) if queries is not None else self.queries
        target_strategies = (
            list(strategies)
            if strategies is not None
            else [
                RetrievalStrategy.VECTOR,
                RetrievalStrategy.GRAPH,
                RetrievalStrategy.HYBRID,
            ]
        )

        if not target_queries:
            raise ValueError("No benchmark queries available to execute Experiment A.")

        if not run_id:
            run_id = f"run_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"

        self.storage.create_run_directory(
            run_id=run_id,
            config_metadata={
                "experiment_type": "retrieval_only",
                "k_values": list(k_values),
                "strategies": [s.value for s in target_strategies],
                "query_count": len(target_queries),
            },
        )

        completed_keys: set[tuple[str, str]] = set()
        existing_results: list[BenchmarkQueryResult] = []
        if resume:
            completed_keys = self.storage.load_completed_query_keys(
                run_id, "retrieval_results.jsonl"
            )
            existing_results = self.storage.load_run_results(run_id, "retrieval_results.jsonl")
            logger.info(
                "Resuming run %s: found %d completed retrieval queries", run_id, len(completed_keys)
            )

        total_work = len(target_queries) * len(target_strategies)
        current_work = len(completed_keys)

        results_map: dict[tuple[str, str], BenchmarkQueryResult] = {
            (r.query_id, r.strategy.value): r for r in existing_results
        }

        for q_idx, q in enumerate(target_queries):
            for s in target_strategies:
                key = (q.query_id, s.value)
                if resume and key in completed_keys:
                    continue

                current_work += 1
                if progress_callback:
                    progress_callback(
                        current_work,
                        total_work,
                        f"Running {s.value.upper()} on {q.query_id} ({q.category.value})...",
                    )

                res = self.run_query_retrieval(query=q, strategy=s, k_values=k_values)
                self.storage.append_query_result(run_id, res, "retrieval_results.jsonl")
                results_map[key] = res

        all_results = list(results_map.values())
        return run_id, all_results

    def run_experiment_b(
        self,
        queries: Sequence[BenchmarkQuery] | None = None,
        strategies: Sequence[RetrievalStrategy] | None = None,
        top_k: int = 5,
        run_id: str | None = None,
        resume: bool = True,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> tuple[str, list[BenchmarkQueryResult]]:
        """Execute Experiment B: End-to-End Grounded Generation Evaluation.

        Args:
            queries: Target benchmark queries.
            strategies: Target strategies.
            top_k: Evidence context size.
            run_id: Run identifier.
            resume: Whether to skip existing answers.
            progress_callback: UI progress callback.

        Returns:
            Tuple of (run_id, list of BenchmarkQueryResult).
        """
        target_queries = list(queries) if queries is not None else self.queries
        target_strategies = (
            list(strategies)
            if strategies is not None
            else [
                RetrievalStrategy.VECTOR,
                RetrievalStrategy.GRAPH,
                RetrievalStrategy.HYBRID,
            ]
        )

        if not target_queries:
            raise ValueError("No benchmark queries available for Experiment B.")

        if not run_id:
            run_id = f"run_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"

        self.storage.create_run_directory(
            run_id=run_id,
            config_metadata={
                "experiment_type": "end_to_end_rag",
                "top_k": top_k,
                "strategies": [s.value for s in target_strategies],
                "query_count": len(target_queries),
            },
        )

        completed_keys: set[tuple[str, str]] = set()
        existing_results: list[BenchmarkQueryResult] = []
        if resume:
            completed_keys = self.storage.load_completed_query_keys(run_id, "answer_results.jsonl")
            existing_results = self.storage.load_run_results(run_id, "answer_results.jsonl")
            logger.info(
                "Resuming run %s: found %d completed generation queries",
                run_id,
                len(completed_keys),
            )

        total_work = len(target_queries) * len(target_strategies)
        current_work = len(completed_keys)

        results_map: dict[tuple[str, str], BenchmarkQueryResult] = {
            (r.query_id, r.strategy.value): r for r in existing_results
        }

        for q in target_queries:
            for s in target_strategies:
                key = (q.query_id, s.value)
                if resume and key in completed_keys:
                    continue

                current_work += 1
                if progress_callback:
                    progress_callback(
                        current_work,
                        total_work,
                        f"Generating answer via {s.value.upper()} for {q.query_id}...",
                    )

                res = self.run_query_generation(query=q, strategy=s, top_k=top_k)
                self.storage.append_query_result(run_id, res, "answer_results.jsonl")
                results_map[key] = res

        all_results = list(results_map.values())
        return run_id, all_results

    def compile_run_manifest(
        self,
        run_id: str,
        retrieval_results: Sequence[BenchmarkQueryResult],
        answer_results: Sequence[BenchmarkQueryResult] | None = None,
        k_values: Sequence[int] = (1, 3, 5, 10),
    ) -> BenchmarkRunManifest:
        """Aggregate run results, execute paired statistical tests, and persist artifacts.

        Args:
            run_id: Unique run identifier.
            retrieval_results: List of retrieval query results.
            answer_results: Optional list of generation query results.
            k_values: Cutoff K values evaluated.

        Returns:
            BenchmarkRunManifest summarizing complete evaluation.
        """
        # Group retrieval results by strategy
        by_strategy: dict[RetrievalStrategy, list[BenchmarkQueryResult]] = {}
        for r in retrieval_results:
            by_strategy.setdefault(r.strategy, []).append(r)

        strategy_summaries: dict[str, StrategyAggregateSummary] = {}
        ans_by_strat: dict[str, list[BenchmarkQueryResult]] = {}
        if answer_results:
            for ar in answer_results:
                s_key = ar.strategy.value if hasattr(ar.strategy, "value") else str(ar.strategy)
                ans_by_strat.setdefault(s_key, []).append(ar)

        for strat, r_list in by_strategy.items():
            strat_key = strat.value if hasattr(strat, "value") else str(strat)
            summary = aggregate_strategy_metrics(r_list, k_values=k_values)
            strat_answers = ans_by_strat.get(strat_key, [])
            if strat_answers:
                faith_scores = [
                    a.answer_quality.faithfulness_score
                    for a in strat_answers
                    if a.answer_quality and a.answer_quality.faithfulness_score is not None
                ]
                if faith_scores:
                    summary.mean_faithfulness = round(sum(faith_scores) / len(faith_scores), 4)

                cit_prec_scores = [
                    a.answer_quality.citation_reference_precision
                    for a in strat_answers
                    if a.answer_quality
                    and a.answer_quality.citation_reference_precision is not None
                ]
                if cit_prec_scores:
                    summary.mean_citation_precision = round(
                        sum(cit_prec_scores) / len(cit_prec_scores), 4
                    )

                refusal_scores = [
                    a.answer_quality.refusal_accuracy
                    for a in strat_answers
                    if a.answer_quality and a.answer_quality.refusal_accuracy is not None
                ]
                if refusal_scores:
                    summary.refusal_accuracy = round(sum(refusal_scores) / len(refusal_scores), 4)

            strategy_summaries[strat_key] = summary

        # Paired comparisons across primary contrasts
        paired_comparisons: list[PairedComparisonResult] = []
        vec_results = by_strategy.get(RetrievalStrategy.VECTOR, [])
        graph_results = by_strategy.get(RetrievalStrategy.GRAPH, [])
        hybrid_results = by_strategy.get(RetrievalStrategy.HYBRID, [])

        if hybrid_results and vec_results:
            comp = compare_strategies_paired(
                baseline_results=vec_results,
                treatment_results=hybrid_results,
                baseline_strategy=RetrievalStrategy.VECTOR,
                treatment_strategy=RetrievalStrategy.HYBRID,
                k=10,
            )
            paired_comparisons.append(comp)

        if hybrid_results and graph_results:
            comp = compare_strategies_paired(
                baseline_results=graph_results,
                treatment_results=hybrid_results,
                baseline_strategy=RetrievalStrategy.GRAPH,
                treatment_strategy=RetrievalStrategy.HYBRID,
                k=10,
            )
            paired_comparisons.append(comp)

        if graph_results and vec_results:
            comp = compare_strategies_paired(
                baseline_results=vec_results,
                treatment_results=graph_results,
                baseline_strategy=RetrievalStrategy.VECTOR,
                treatment_strategy=RetrievalStrategy.GRAPH,
                k=10,
            )
            paired_comparisons.append(comp)

        # Build manifest
        manifest = BenchmarkRunManifest(
            run_id=run_id,
            timestamp_utc=datetime.now(UTC).isoformat(),
            system_version=self.settings.app_version,
            dataset_name="Phase 8 Canonical Benchmark Dataset",
            total_queries=len({r.query_id for r in retrieval_results}),
            strategies_evaluated=[s for s in strategy_summaries],
            k_values=list(k_values),
            strategy_summaries=strategy_summaries,
            paired_comparisons=paired_comparisons,
            status="COMPLETED",
        )

        # Save all artifacts via storage engine
        self.storage.save_run_manifest(run_id, manifest)
        self.storage.save_run_aggregates(
            run_id=run_id,
            strategy_summaries=strategy_summaries,
            paired_comparisons=paired_comparisons,
        )
        self.storage.export_results_csv(run_id)

        logger.info("Compiled and persisted run manifest for %s successfully", run_id)
        return manifest
