"""Hybrid Retriever Coordinator for Phase 6.

Integrates SemanticSearch (Phase 3) and GraphSearchEngine (Phase 5), normalizes their outputs
to canonical chunk identifiers, applies source filters, executes Reciprocal Rank Fusion,
and packages unified, grounded evidence contexts.
"""

import time

from graphrag_bench.config import Settings, get_settings
from graphrag_bench.document_repository import DocumentRepository
from graphrag_bench.graph_search import GraphSearchEngine, get_graph_search_engine
from graphrag_bench.hybrid_context_builder import HybridContextBuilder
from graphrag_bench.hybrid_schemas import (
    GraphCandidate,
    HybridSearchResponse,
    HybridSearchStatus,
    VectorCandidate,
)
from graphrag_bench.logger import get_logger
from graphrag_bench.reciprocal_rank_fusion import ReciprocalRankFusion
from graphrag_bench.result_normalizer import ResultNormalizer
from graphrag_bench.semantic_search import SearchResponse, SemanticSearch

logger = get_logger("hybrid_retriever")


class HybridRetriever:
    """Orchestrates multi-modal Hybrid Retrieval fusing Vector and Knowledge Graph evidence."""

    def __init__(
        self,
        vector_search: SemanticSearch | None = None,
        graph_search: GraphSearchEngine | None = None,
        repository: DocumentRepository | None = None,
        normalizer: ResultNormalizer | None = None,
        fusion_engine: ReciprocalRankFusion | None = None,
        context_builder: HybridContextBuilder | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize Hybrid Retriever with injected dependencies or defaults.

        Args:
            vector_search: SemanticSearch instance for dense retrieval.
            graph_search: GraphSearchEngine instance for graph traversal.
            repository: DocumentRepository instance for canonical chunk validation.
            normalizer: ResultNormalizer instance.
            fusion_engine: ReciprocalRankFusion instance.
            context_builder: HybridContextBuilder instance.
            settings: Settings configuration instance.
        """
        self.settings = settings or get_settings()
        self.repository = repository or DocumentRepository()
        self.vector_search = vector_search or SemanticSearch(settings=self.settings)
        self.graph_search = graph_search or get_graph_search_engine()
        self.normalizer = normalizer or ResultNormalizer(repository=self.repository)
        self.fusion_engine = fusion_engine or ReciprocalRankFusion(
            k=self.settings.hybrid_rrf_k,
            vector_weight=self.settings.hybrid_vector_weight,
            graph_weight=self.settings.hybrid_graph_weight,
            min_rrf_score=self.settings.hybrid_min_rrf_score,
        )
        self.context_builder = context_builder or HybridContextBuilder()

    def search_hybrid(
        self,
        query: str,
        vector_top_k: int | None = None,
        graph_top_k: int | None = None,
        final_top_k: int | None = None,
        rrf_k: int | None = None,
        vector_weight: float | None = None,
        graph_weight: float | None = None,
        max_hops: int | None = None,
        direction: str = "both",
        predicates: list[str] | set[str] | None = None,
        document_ids: list[str] | set[str] | str | None = None,
        file_types: list[str] | set[str] | str | None = None,
    ) -> HybridSearchResponse:
        """Execute unified hybrid retrieval across vector and graph systems.

        Args:
            query: Natural language user question.
            vector_top_k: Top-K items to fetch from vector search.
            graph_top_k: Top-K paths to fetch from graph search.
            final_top_k: Final top-K fused chunks to return.
            rrf_k: Optional override for RRF smoothing constant.
            vector_weight: Optional override for vector modality weight.
            graph_weight: Optional override for graph modality weight.
            max_hops: Maximum relationship traversal hops (1 to 3).
            direction: Edge traversal direction ('both', 'out', 'in', etc.).
            predicates: Optional relationship predicate filter.
            document_ids: Optional document ID or list of document IDs to filter.
            file_types: Optional file format filter (e.g. 'pdf', 'docx', 'txt', 'md').

        Returns:
            HybridSearchResponse containing fused candidates, latency, and context.
        """
        t0 = time.perf_counter()
        clean_query = query.strip() if query else ""
        if not clean_query:
            return HybridSearchResponse(
                query=query,
                status=HybridSearchStatus.INVALID_QUERY,
                vector_search_status="SKIPPED",
                graph_search_status="SKIPPED",
                errors=["Query cannot be empty or whitespace-only."],
                total_latency_ms=0.0,
            )

        v_top_k = (
            vector_top_k
            if vector_top_k is not None
            else getattr(self.settings, "hybrid_vector_top_k", 10)
        )
        g_top_k = (
            graph_top_k
            if graph_top_k is not None
            else getattr(self.settings, "hybrid_graph_top_k", 10)
        )
        f_top_k = (
            final_top_k if final_top_k is not None else getattr(self.settings, "hybrid_top_k", 10)
        )
        hops = max_hops if max_hops is not None else getattr(self.settings, "graph_max_hops", 2)

        # Normalize document filter
        doc_filter_set: set[str] | None = None
        single_doc_filter: str | None = None
        if document_ids:
            if isinstance(document_ids, str):
                single_doc_filter = document_ids.strip()
                doc_filter_set = {single_doc_filter}
            else:
                doc_filter_set = {d.strip() for d in document_ids if d.strip()}
                if len(doc_filter_set) == 1:
                    single_doc_filter = next(iter(doc_filter_set))

        # Normalize file type filter
        type_filter_set: set[str] | None = None
        single_type_filter: str | None = None
        if file_types:
            if isinstance(file_types, str):
                single_type_filter = file_types.strip().lower()
                type_filter_set = {single_type_filter}
            else:
                type_filter_set = {t.strip().lower() for t in file_types if t.strip()}
                if len(type_filter_set) == 1:
                    single_type_filter = next(iter(type_filter_set))

        all_warnings: list[str] = []
        all_errors: list[str] = []

        # Normalize edge direction for graph search
        dir_clean = direction.lower() if isinstance(direction, str) else "both"
        if "out" in dir_clean:
            graph_dir = "out"
        elif "in" in dir_clean:
            graph_dir = "in"
        else:
            graph_dir = "both"

        # ---------------------------------------------------------------------
        # 1. Execute Vector Retrieval
        # ---------------------------------------------------------------------
        t_v0 = time.perf_counter()
        vector_candidates: list[VectorCandidate] = []
        v_status = "NOT_RUN"
        try:
            vector_resp: SearchResponse = self.vector_search.search(
                query=clean_query,
                top_k=v_top_k,
                document_id=single_doc_filter,
                file_type=single_type_filter,
            )
            v_norm_cands, v_warns = self.normalizer.normalize_vector_results(vector_resp)
            vector_candidates = v_norm_cands
            all_warnings.extend(v_warns)
            v_status = "SUCCESS"
        except Exception as exc:
            v_status = f"FAILED: {exc}"
            err_msg = f"Vector search failed: {exc}"
            all_errors.append(err_msg)
            all_warnings.append(err_msg)
            logger.error("Vector retrieval error in hybrid search: %s", exc)
        v_latency_ms = round((time.perf_counter() - t_v0) * 1000.0, 2)

        # ---------------------------------------------------------------------
        # 2. Execute Knowledge Graph Retrieval
        # ---------------------------------------------------------------------
        t_g0 = time.perf_counter()
        graph_candidates: list[GraphCandidate] = []
        g_status = "NOT_RUN"
        g_extract_ms = 0.0
        g_llm_invoked = False
        try:
            graph_resp = self.graph_search.search_graph(
                query=clean_query,
                max_hops=hops,
                direction=graph_dir,
                top_k=g_top_k,
                predicates=predicates,
                doc_id_filter=single_doc_filter,
            )
            if graph_resp and graph_resp.extraction_result:
                g_extract_ms = graph_resp.extraction_result.duration_ms
                g_llm_invoked = graph_resp.extraction_result.llm_invoked
            g_norm_cands, g_warns = self.normalizer.normalize_graph_results(graph_resp)
            graph_candidates = g_norm_cands
            all_warnings.extend(g_warns)
            g_status = "SUCCESS"
        except Exception as exc:
            g_status = f"FAILED: {exc}"
            err_msg = f"Graph search failed: {exc}"
            all_errors.append(err_msg)
            all_warnings.append(err_msg)
            logger.error("Graph retrieval error in hybrid search: %s", exc)
        g_latency_ms = round((time.perf_counter() - t_g0) * 1000.0, 2)
        g_traversal_ms = max(0.0, round(g_latency_ms - g_extract_ms, 2))

        # ---------------------------------------------------------------------
        # 3. Apply Multi-Document & Multi-Type Filters Consistently
        # ---------------------------------------------------------------------
        if doc_filter_set:
            vector_candidates = [vc for vc in vector_candidates if vc.document_id in doc_filter_set]
            graph_candidates = [gc for gc in graph_candidates if gc.document_id in doc_filter_set]

        if type_filter_set:
            vector_candidates = [
                vc for vc in vector_candidates if vc.file_type.lower() in type_filter_set
            ]
            # Graph candidates check canonical file extension if available
            filtered_gc = []
            for gc in graph_candidates:
                ext = (
                    gc.source_filename.rsplit(".", 1)[-1].lower()
                    if "." in gc.source_filename
                    else "txt"
                )
                if ext in type_filter_set:
                    filtered_gc.append(gc)
            graph_candidates = filtered_gc

        # ---------------------------------------------------------------------
        # 4. Execute Reciprocal Rank Fusion
        # ---------------------------------------------------------------------
        t_f0 = time.perf_counter()
        fused_candidates = self.fusion_engine.fuse(
            vector_candidates=vector_candidates,
            graph_candidates=graph_candidates,
            top_k=f_top_k,
            k=rrf_k,
            vector_weight=vector_weight,
            graph_weight=graph_weight,
        )
        f_latency_ms = round((time.perf_counter() - t_f0) * 1000.0, 2)

        # ---------------------------------------------------------------------
        # 5. Build Unified Context Block
        # ---------------------------------------------------------------------
        structured_context = self.context_builder.build_context(
            fused_candidates=fused_candidates,
            max_context_chunks=f_top_k,
        )

        total_latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        # Determine overall hybrid status
        v_failed = v_status.startswith("FAILED")
        g_failed = g_status.startswith("FAILED")

        if v_failed and g_failed:
            overall_status = HybridSearchStatus.ALL_ENGINES_FAILED
        elif v_failed or g_failed:
            overall_status = (
                HybridSearchStatus.PARTIAL_SUCCESS
                if fused_candidates
                else HybridSearchStatus.NO_RESULTS
            )
        elif not fused_candidates:
            overall_status = HybridSearchStatus.NO_RESULTS
        else:
            overall_status = HybridSearchStatus.SUCCESS

        eff_k = rrf_k if rrf_k is not None else self.fusion_engine.k
        eff_vw = vector_weight if vector_weight is not None else self.fusion_engine.vector_weight
        eff_gw = graph_weight if graph_weight is not None else self.fusion_engine.graph_weight

        return HybridSearchResponse(
            query=clean_query,
            status=overall_status,
            vector_search_status="FAILED" if v_failed else "SUCCESS",
            graph_search_status="FAILED" if g_failed else "SUCCESS",
            vector_results_count=len(vector_candidates),
            graph_results_count=len(graph_candidates),
            fused_results_count=len(fused_candidates),
            results=fused_candidates,
            fusion_method="Reciprocal Rank Fusion (RRF)",
            rrf_constant=eff_k,
            vector_weight=eff_vw,
            graph_weight=eff_gw,
            vector_latency_ms=v_latency_ms,
            graph_latency_ms=g_latency_ms,
            query_entity_extraction_ms=g_extract_ms,
            graph_traversal_ms=g_traversal_ms,
            llm_extraction_invoked=g_llm_invoked,
            fusion_latency_ms=f_latency_ms,
            total_latency_ms=total_latency_ms,
            warnings=all_warnings,
            errors=all_errors,
            structured_context=structured_context,
        )

    # Class-level alias
    search = search_hybrid


# Global singleton helper
_hybrid_retriever_instance: HybridRetriever | None = None


def get_hybrid_retriever() -> HybridRetriever:
    """Return singleton HybridRetriever instance."""
    global _hybrid_retriever_instance
    if _hybrid_retriever_instance is None:
        _hybrid_retriever_instance = HybridRetriever()
    return _hybrid_retriever_instance


def search_hybrid(
    query: str,
    vector_top_k: int | None = None,
    graph_top_k: int | None = None,
    final_top_k: int | None = None,
    rrf_k: int | None = None,
    vector_weight: float | None = None,
    graph_weight: float | None = None,
    max_hops: int | None = None,
    direction: str = "both",
    predicates: list[str] | set[str] | None = None,
    document_ids: list[str] | set[str] | str | None = None,
    file_types: list[str] | set[str] | str | None = None,
    retriever: HybridRetriever | None = None,
    vector_search: SemanticSearch | None = None,
    graph_search: GraphSearchEngine | None = None,
    repository: DocumentRepository | None = None,
    settings: Settings | None = None,
) -> HybridSearchResponse:
    """Convenience functional interface for hybrid retrieval."""
    if retriever is None and (
        vector_search is not None
        or graph_search is not None
        or repository is not None
        or settings is not None
    ):
        retriever = HybridRetriever(
            vector_search=vector_search,
            graph_search=graph_search,
            repository=repository,
            settings=settings,
        )
    active_retriever = retriever or get_hybrid_retriever()
    return active_retriever.search_hybrid(
        query=query,
        vector_top_k=vector_top_k,
        graph_top_k=graph_top_k,
        final_top_k=final_top_k,
        rrf_k=rrf_k,
        vector_weight=vector_weight,
        graph_weight=graph_weight,
        max_hops=max_hops,
        direction=direction,
        predicates=predicates,
        document_ids=document_ids,
        file_types=file_types,
    )
