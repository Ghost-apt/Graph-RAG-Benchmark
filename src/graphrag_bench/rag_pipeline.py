"""End-to-End Grounded Question Answering Pipeline (Phase 7).

Orchestrates multi-modal retrieval (Hybrid, Vector, or Graph), evidence context
preparation, grounded LLM generation, citation provenance validation, guardrails,
and faithfulness scoring into a unified QA pipeline.
"""

import time
from typing import Literal

from graphrag_bench.answer_generator import AnswerGenerator
from graphrag_bench.answer_guardrails import (
    STANDARD_INSUFFICIENT_EVIDENCE_MESSAGE,
    AnswerGuardrails,
)
from graphrag_bench.answer_schemas import (
    AnswerGenerationRequest,
    AnswerStatus,
    EvidenceItem,
    GroundedAnswer,
)
from graphrag_bench.citation_validator import CitationValidator
from graphrag_bench.config import get_settings
from graphrag_bench.embedding_service import EmbeddingService
from graphrag_bench.evidence_preparer import EvidencePreparer
from graphrag_bench.faithfulness_checker import FaithfulnessChecker
from graphrag_bench.graph_retriever import GraphRetriever
from graphrag_bench.graph_search import GraphSearchEngine
from graphrag_bench.hybrid_retriever import HybridRetriever
from graphrag_bench.logger import get_logger
from graphrag_bench.ollama_client import OllamaClient, OllamaError
from graphrag_bench.semantic_search import SemanticSearch

logger = get_logger(__name__)


class RAGPipeline:
    """Master orchestrator for grounded question answering over vector, graph, and hybrid indexes."""

    def __init__(
        self,
        hybrid_retriever: HybridRetriever | None = None,
        vector_search: SemanticSearch | None = None,
        graph_retriever: GraphRetriever | None = None,
        evidence_preparer: EvidencePreparer | None = None,
        answer_generator: AnswerGenerator | None = None,
        citation_validator: CitationValidator | None = None,
        guardrails: AnswerGuardrails | None = None,
        faithfulness_checker: FaithfulnessChecker | None = None,
        ollama_client: OllamaClient | None = None,
    ) -> None:
        """Initialize the RAG pipeline with components or lazily construct defaults.

        Args:
            hybrid_retriever: Preconfigured HybridRetriever or None.
            vector_search: Preconfigured SemanticSearch or None.
            graph_retriever: Preconfigured GraphRetriever or None.
            evidence_preparer: Preconfigured EvidencePreparer or None.
            answer_generator: Preconfigured AnswerGenerator or None.
            citation_validator: Preconfigured CitationValidator or None.
            guardrails: Preconfigured AnswerGuardrails or None.
            faithfulness_checker: Preconfigured FaithfulnessChecker or None.
            ollama_client: Shared OllamaClient or None.
        """
        self.cfg = get_settings()
        self.ollama_client = ollama_client or OllamaClient()

        self.hybrid_retriever = hybrid_retriever
        self.vector_search = vector_search
        self.graph_retriever = graph_retriever

        self.preparer = evidence_preparer or EvidencePreparer(
            default_max_context_tokens=self.cfg.rag_max_context_tokens
        )
        self.generator = answer_generator or AnswerGenerator(
            ollama_client=self.ollama_client,
            model=self.cfg.rag_generation_model,
            default_temperature=self.cfg.rag_temperature,
            default_timeout=self.cfg.rag_generation_timeout,
        )
        self.validator = citation_validator or CitationValidator()
        self.guardrails = guardrails or AnswerGuardrails(
            min_evidence_items=self.cfg.rag_min_evidence_items
        )
        self.faithfulness = faithfulness_checker or FaithfulnessChecker(
            ollama_client=self.ollama_client,
            model=self.cfg.rag_generation_model,
        )

    def _ensure_retrievers(self, mode: str) -> None:
        """Lazily instantiate retrievers if not injected."""
        if mode == "hybrid" and self.hybrid_retriever is None:
            self.hybrid_retriever = HybridRetriever(settings=self.cfg)
        elif mode == "vector" and self.vector_search is None:
            embed_svc = EmbeddingService(ollama_client=self.ollama_client, settings=self.cfg)
            self.vector_search = SemanticSearch(embedding_service=embed_svc, settings=self.cfg)
        elif mode == "graph" and self.graph_retriever is None:
            g_search = GraphSearchEngine(settings=self.cfg)
            self.graph_retriever = g_search.retriever

    def answer_question(
        self,
        request: AnswerGenerationRequest | str,
        retrieval_mode: Literal["hybrid", "vector", "graph"] = "hybrid",
        top_k: int | None = None,
        temperature: float | None = None,
        max_context_tokens: int | None = None,
        max_generation_tokens: int | None = None,
        require_citations: bool | None = None,
        enable_faithfulness_check: bool | None = None,
        min_relevance_score: float | None = None,
    ) -> GroundedAnswer:
        """Execute the end-to-end question answering pipeline.

        Args:
            request: Either a validated AnswerGenerationRequest or a plain query string.
            retrieval_mode: Fallback retrieval mode if request is a string.
            top_k: Number of candidates to retrieve.
            temperature: LLM temperature.
            max_context_tokens: Upper bound on context tokens.
            max_generation_tokens: Maximum tokens for LLM response.
            require_citations: Whether to enforce [S1], [S2] markers.
            enable_faithfulness_check: Whether to verify claims against evidence.
            min_relevance_score: Minimum retrieval relevance score threshold to proceed to generation.

        Returns:
            GroundedAnswer containing answer text, citations, claims, scores, and latency.
        """
        t_pipeline_start = time.perf_counter()

        # Normalize request input
        if isinstance(request, str):
            req = AnswerGenerationRequest(
                query=request,
                retrieval_mode=retrieval_mode,
                top_k=top_k if top_k is not None else self.cfg.rag_top_k,
                temperature=temperature if temperature is not None else self.cfg.rag_temperature,
                max_context_tokens=max_context_tokens
                if max_context_tokens is not None
                else self.cfg.rag_max_context_tokens,
                max_generation_tokens=max_generation_tokens
                if max_generation_tokens is not None
                else self.cfg.rag_max_generation_tokens,
                require_citations=require_citations
                if require_citations is not None
                else self.cfg.rag_require_citations,
                enable_faithfulness_check=enable_faithfulness_check
                if enable_faithfulness_check is not None
                else self.cfg.rag_enable_faithfulness_check,
                min_relevance_score=min_relevance_score,
            )
        else:
            req = request

        query = req.query
        mode = req.retrieval_mode
        all_warnings: list[str] = []
        all_errors: list[str] = []

        # ---------------------------------------------------------
        # Step 0: Query Preprocessing
        # ---------------------------------------------------------
        t_prep_start = time.perf_counter()
        query_clean = query.strip()
        query_prep_latency_ms = (time.perf_counter() - t_prep_start) * 1000.0

        # ---------------------------------------------------------
        # Step 1: Retrieval
        # ---------------------------------------------------------
        self._ensure_retrievers(mode)
        evidence_items: list[EvidenceItem] = []
        retrieval_latency_ms = 0.0
        vec_lat_ms = 0.0
        graph_lat_ms = 0.0
        fusion_lat_ms = 0.0
        query_extraction_ms = 0.0
        graph_traversal_ms = 0.0
        query_extraction_calls = 0
        evidence_prep_latency_ms = 0.0

        t_ret_start = time.perf_counter()
        try:
            if mode == "hybrid":
                assert self.hybrid_retriever is not None
                hybrid_resp = self.hybrid_retriever.search_hybrid(
                    query=query_clean,
                    final_top_k=req.top_k,
                )
                all_warnings.extend(hybrid_resp.warnings)
                all_errors.extend(hybrid_resp.errors)
                vec_lat_ms = hybrid_resp.vector_latency_ms
                graph_lat_ms = hybrid_resp.graph_latency_ms
                fusion_lat_ms = hybrid_resp.fusion_latency_ms
                query_extraction_ms = hybrid_resp.query_entity_extraction_ms
                graph_traversal_ms = hybrid_resp.graph_traversal_ms
                if hybrid_resp.llm_extraction_invoked:
                    query_extraction_calls = 1

                t_ev_start = time.perf_counter()
                evidence_items = self.preparer.prepare_from_hybrid(
                    hybrid_response=hybrid_resp,
                    max_tokens=req.max_context_tokens,
                    top_k=req.top_k,
                )
                evidence_prep_latency_ms = (time.perf_counter() - t_ev_start) * 1000.0

            elif mode == "vector":
                assert self.vector_search is not None
                vec_resp = self.vector_search.search(
                    query=query_clean,
                    top_k=req.top_k,
                )
                vec_lat_ms = vec_resp.duration_ms
                t_ev_start = time.perf_counter()
                evidence_items = self.preparer.prepare_from_vector(
                    search_items=vec_resp.items,
                    max_tokens=req.max_context_tokens,
                    top_k=req.top_k,
                )
                evidence_prep_latency_ms = (time.perf_counter() - t_ev_start) * 1000.0

            elif mode == "graph":
                assert self.graph_retriever is not None
                t_g_start = time.perf_counter()
                graph_results = self.graph_retriever.retrieve(
                    query=query_clean,
                    top_k=req.top_k,
                )
                graph_lat_ms = (time.perf_counter() - t_g_start) * 1000.0
                if (
                    self.graph_retriever.extractor
                    and self.graph_retriever.extractor.last_extraction_result
                ):
                    ext = self.graph_retriever.extractor.last_extraction_result
                    query_extraction_ms = ext.duration_ms
                    if ext.llm_invoked:
                        query_extraction_calls = 1
                    graph_traversal_ms = max(0.0, round(graph_lat_ms - query_extraction_ms, 2))
                else:
                    graph_traversal_ms = graph_lat_ms

                t_ev_start = time.perf_counter()
                evidence_items = self.preparer.prepare_from_graph(
                    graph_results=graph_results,
                    max_tokens=req.max_context_tokens,
                    top_k=req.top_k,
                )
                evidence_prep_latency_ms = (time.perf_counter() - t_ev_start) * 1000.0

        except Exception as exc:
            msg = f"Evidence retrieval failed in mode '{mode}': {exc}"
            logger.error(msg)
            all_errors.append(msg)

        retrieval_latency_ms = (time.perf_counter() - t_ret_start) * 1000.0

        # ---------------------------------------------------------
        # Step 2: Pre-generation Guardrails
        # ---------------------------------------------------------
        effective_min_relevance = (
            req.min_relevance_score
            if req.min_relevance_score is not None
            else self.cfg.rag_min_relevance_score
        )
        pre_check = self.guardrails.evaluate_pre_generation(
            query=query_clean,
            evidence_items=evidence_items,
            min_relevance_threshold=effective_min_relevance,
        )
        all_warnings.extend(pre_check.warnings)

        if not pre_check.can_proceed:
            total_latency_ms = (time.perf_counter() - t_pipeline_start) * 1000.0
            breakdown = {
                "query_preprocessing_ms": round(query_prep_latency_ms, 2),
                "query_entity_extraction_ms": round(query_extraction_ms, 2),
                "graph_traversal_ms": round(graph_traversal_ms, 2),
                "vector_retrieval_ms": round(vec_lat_ms, 2),
                "graph_retrieval_ms": round(graph_lat_ms, 2),
                "hybrid_fusion_ms": round(fusion_lat_ms, 2),
                "retrieval_total_ms": round(retrieval_latency_ms, 2),
                "evidence_preparation_ms": round(evidence_prep_latency_ms, 2),
                "generation_ms": 0.0,
                "citation_validation_ms": 0.0,
                "faithfulness_checking_ms": 0.0,
                "total_pipeline_ms": round(total_latency_ms, 2),
            }
            return GroundedAnswer(
                query=query_clean,
                answer=pre_check.early_answer or STANDARD_INSUFFICIENT_EVIDENCE_MESSAGE,
                status=pre_check.status,
                evidence_items=evidence_items,
                citations=[],
                claims=[],
                faithfulness_score=None,
                citation_reference_precision=None,
                generation_invoked=False,
                total_model_invocations=query_extraction_calls,
                query_extraction_model_calls=query_extraction_calls,
                answer_generation_model_calls=0,
                query_entity_extraction_ms=query_extraction_ms,
                graph_traversal_ms=graph_traversal_ms,
                retrieval_mode=mode,
                model_used=self.generator.model,
                prompt_tokens_estimated=0,
                answer_tokens_estimated=0,
                latency_retrieval_ms=retrieval_latency_ms,
                latency_generation_ms=0.0,
                latency_total_ms=total_latency_ms,
                latency_breakdown=breakdown,
                warnings=all_warnings,
                errors=all_errors,
            )

        # ---------------------------------------------------------
        # Step 3: LLM Generation
        # ---------------------------------------------------------
        clean_answer = ""
        generation_stats: dict = {}
        generation_latency_ms = 0.0

        try:
            clean_answer, _, generation_stats = self.generator.generate(
                query=query_clean,
                evidence_items=evidence_items,
                temperature=req.temperature,
                max_tokens=req.max_generation_tokens,
                require_citations=req.require_citations,
            )
            generation_latency_ms = generation_stats.get("latency_ms", 0.0)
        except OllamaError as exc:
            msg = f"LLM generation failed: {exc}"
            logger.error(msg)
            all_errors.append(msg)
            total_latency_ms = (time.perf_counter() - t_pipeline_start) * 1000.0
            breakdown = {
                "query_preprocessing_ms": round(query_prep_latency_ms, 2),
                "query_entity_extraction_ms": round(query_extraction_ms, 2),
                "graph_traversal_ms": round(graph_traversal_ms, 2),
                "vector_retrieval_ms": round(vec_lat_ms, 2),
                "graph_retrieval_ms": round(graph_lat_ms, 2),
                "hybrid_fusion_ms": round(fusion_lat_ms, 2),
                "retrieval_total_ms": round(retrieval_latency_ms, 2),
                "evidence_preparation_ms": round(evidence_prep_latency_ms, 2),
                "generation_ms": 0.0,
                "citation_validation_ms": 0.0,
                "faithfulness_checking_ms": 0.0,
                "total_pipeline_ms": round(total_latency_ms, 2),
            }
            return GroundedAnswer(
                query=query_clean,
                answer="Generation error occurred while contacting local model.",
                status=AnswerStatus.FAILED,
                evidence_items=evidence_items,
                citations=[],
                claims=[],
                faithfulness_score=None,
                citation_reference_precision=None,
                generation_invoked=True,
                total_model_invocations=query_extraction_calls + 1,
                query_extraction_model_calls=query_extraction_calls,
                answer_generation_model_calls=1,
                query_entity_extraction_ms=query_extraction_ms,
                graph_traversal_ms=graph_traversal_ms,
                retrieval_mode=mode,
                model_used=self.generator.model,
                prompt_tokens_estimated=0,
                answer_tokens_estimated=0,
                latency_retrieval_ms=retrieval_latency_ms,
                latency_generation_ms=0.0,
                latency_total_ms=total_latency_ms,
                latency_breakdown=breakdown,
                warnings=all_warnings,
                errors=all_errors,
            )

        # ---------------------------------------------------------
        # Step 4: Citation Validation
        # ---------------------------------------------------------
        t_cit_start = time.perf_counter()
        cit_result = self.validator.validate_citations(
            text=clean_answer,
            evidence_items=evidence_items,
        )
        cit_latency_ms = (time.perf_counter() - t_cit_start) * 1000.0
        all_warnings.extend(cit_result.warnings)

        # ---------------------------------------------------------
        # Step 5: Post-generation Guardrails
        # ---------------------------------------------------------
        post_check = self.guardrails.evaluate_post_generation(
            answer=clean_answer,
            citations=cit_result.citations,
            require_citations=req.require_citations,
        )
        all_warnings.extend(post_check.warnings)
        final_answer_text = post_check.sanitized_answer
        final_status = post_check.status

        # ---------------------------------------------------------
        # Step 6: Faithfulness Verification (if enabled)
        # ---------------------------------------------------------
        claims = []
        faithfulness_score: float | None = None
        faith_latency_ms = 0.0

        if req.enable_faithfulness_check and final_status == AnswerStatus.SUCCESS:
            t_faith_start = time.perf_counter()
            try:
                # Use heuristic fallback internally to avoid latency spikes if LLM check is slow
                claims, faithfulness_score = self.faithfulness.check_faithfulness(
                    answer=final_answer_text,
                    evidence_items=evidence_items,
                    use_llm=False,  # Fast heuristic verification by default for pipeline
                )
            except Exception as exc:
                logger.warning("Faithfulness verification failed: %s", exc)
                all_warnings.append(f"Faithfulness check error: {exc}")
            faith_latency_ms = (time.perf_counter() - t_faith_start) * 1000.0

        total_latency_ms = (time.perf_counter() - t_pipeline_start) * 1000.0
        breakdown = {
            "query_preprocessing_ms": round(query_prep_latency_ms, 2),
            "query_entity_extraction_ms": round(query_extraction_ms, 2),
            "graph_traversal_ms": round(graph_traversal_ms, 2),
            "vector_retrieval_ms": round(vec_lat_ms, 2),
            "graph_retrieval_ms": round(graph_lat_ms, 2),
            "hybrid_fusion_ms": round(fusion_lat_ms, 2),
            "retrieval_total_ms": round(retrieval_latency_ms, 2),
            "evidence_preparation_ms": round(evidence_prep_latency_ms, 2),
            "generation_ms": round(generation_latency_ms, 2),
            "citation_validation_ms": round(cit_latency_ms, 2),
            "faithfulness_checking_ms": round(faith_latency_ms, 2),
            "total_pipeline_ms": round(total_latency_ms, 2),
        }

        return GroundedAnswer(
            query=query_clean,
            answer=final_answer_text,
            status=final_status,
            evidence_items=evidence_items,
            citations=cit_result.citations,
            claims=claims,
            faithfulness_score=faithfulness_score,
            citation_reference_precision=cit_result.citation_reference_precision,
            generation_invoked=True,
            total_model_invocations=query_extraction_calls + 1,
            query_extraction_model_calls=query_extraction_calls,
            answer_generation_model_calls=1,
            query_entity_extraction_ms=query_extraction_ms,
            graph_traversal_ms=graph_traversal_ms,
            retrieval_mode=mode,
            model_used=self.generator.model,
            prompt_tokens_estimated=generation_stats.get("prompt_tokens_estimated", 0),
            answer_tokens_estimated=generation_stats.get("answer_tokens_estimated", 0),
            latency_retrieval_ms=retrieval_latency_ms,
            latency_generation_ms=generation_latency_ms,
            latency_total_ms=total_latency_ms,
            latency_breakdown=breakdown,
            warnings=all_warnings,
            errors=all_errors,
        )


def answer_question(
    query: str,
    retrieval_mode: Literal["hybrid", "vector", "graph"] = "hybrid",
    top_k: int | None = None,
    temperature: float | None = None,
    require_citations: bool = True,
    enable_faithfulness_check: bool = True,
    min_relevance_score: float | None = None,
) -> GroundedAnswer:
    """Convenience functional API for answering a question with grounded citations.

    Args:
        query: User natural language question.
        retrieval_mode: Retrieval strategy ('hybrid', 'vector', or 'graph').
        top_k: Number of evidence candidates.
        temperature: Model sampling temperature.
        require_citations: Whether to enforce citations.
        enable_faithfulness_check: Whether to compute claim faithfulness.
        min_relevance_score: Minimum retrieval relevance score threshold to proceed to generation.

    Returns:
        GroundedAnswer instance with complete evidence provenance.
    """
    pipeline = RAGPipeline()
    return pipeline.answer_question(
        request=query,
        retrieval_mode=retrieval_mode,
        top_k=top_k,
        temperature=temperature,
        require_citations=require_citations,
        enable_faithfulness_check=enable_faithfulness_check,
        min_relevance_score=min_relevance_score,
    )
