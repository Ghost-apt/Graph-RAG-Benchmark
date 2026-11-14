"""Unit tests for RAG Pipeline orchestrator (graphrag_bench.rag_pipeline)."""

from unittest.mock import MagicMock

from graphrag_bench.answer_schemas import AnswerStatus, GroundedAnswer
from graphrag_bench.hybrid_schemas import (
    FusionCandidate,
    HybridSearchResponse,
    HybridSearchStatus,
    RetrievalSource,
)
from graphrag_bench.ollama_client import GenerateResponse, OllamaClient
from graphrag_bench.rag_pipeline import RAGPipeline, answer_question


class TestRAGPipeline:
    """Test suite for master Grounded RAG Pipeline."""

    def test_pipeline_hybrid_success(self):
        """Test full pipeline execution in hybrid retrieval mode."""
        # Mock OllamaClient
        mock_ollama = MagicMock(spec=OllamaClient)
        mock_ollama.default_llm = "qwen3:8b"
        mock_ollama.generate.return_value = GenerateResponse(
            model="qwen3:8b",
            response="Graph RAG evaluates multi-hop retrieval benchmarks [S1].",
            done=True,
            eval_count=25,
            prompt_eval_count=180,
        )

        # Mock HybridRetriever
        mock_hybrid = MagicMock()
        cand = FusionCandidate(
            chunk_id="chunk_8125_0000",
            document_id="doc_8125",
            source_filename="guide.md",
            chunk_text="Graph RAG evaluates multi-hop retrieval benchmarks across diverse datasets.",
            fused_rank=1,
            retrieval_sources=RetrievalSource.VECTOR_AND_GRAPH,
            rrf_score=0.88,
        )
        mock_hybrid.search_hybrid.return_value = HybridSearchResponse(
            query="What does Graph RAG evaluate?",
            status=HybridSearchStatus.SUCCESS,
            results=[cand],
            fused_results_count=1,
        )

        pipeline = RAGPipeline(
            hybrid_retriever=mock_hybrid,
            ollama_client=mock_ollama,
        )

        answer: GroundedAnswer = pipeline.answer_question(
            request="What does Graph RAG evaluate?",
            retrieval_mode="hybrid",
            top_k=5,
            temperature=0.1,
            require_citations=True,
            enable_faithfulness_check=True,
        )

        assert answer.status == AnswerStatus.SUCCESS
        assert answer.is_grounded is True
        assert len(answer.citations) == 1
        assert answer.citations[0].evidence_id == "S1"
        assert answer.citations[0].is_valid is True
        assert answer.latency_retrieval_ms > 0
        assert answer.latency_generation_ms > 0
        assert answer.latency_total_ms > 0
        assert answer.citation_precision == 1.0

    def test_pipeline_zero_evidence_short_circuit(self):
        """Test that zero retrieved candidates returns INSUFFICIENT_EVIDENCE immediately without calling LLM."""
        mock_ollama = MagicMock(spec=OllamaClient)
        mock_hybrid = MagicMock()
        mock_hybrid.search_hybrid.return_value = HybridSearchResponse(
            query="Unanswerable query",
            status=HybridSearchStatus.NO_RESULTS,
            results=[],
            fused_results_count=0,
        )

        pipeline = RAGPipeline(
            hybrid_retriever=mock_hybrid,
            ollama_client=mock_ollama,
        )

        answer: GroundedAnswer = pipeline.answer_question(
            request="Unanswerable query",
            retrieval_mode="hybrid",
        )

        assert answer.status == AnswerStatus.INSUFFICIENT_EVIDENCE
        assert "insufficient information" in answer.answer.lower()
        assert len(answer.evidence_items) == 0
        # Verify LLM was NOT called
        mock_ollama.generate.assert_not_called()

    def test_pipeline_vector_mode(self):
        """Test pipeline execution in vector retrieval mode."""
        mock_ollama = MagicMock(spec=OllamaClient)
        mock_ollama.default_llm = "qwen3:8b"
        mock_ollama.generate.return_value = GenerateResponse(
            model="qwen3:8b",
            response="Vector embeddings capture semantic similarity [S1].",
            done=True,
        )

        mock_vector = MagicMock()
        item = MagicMock()
        item.chunk_id = "chunk_vec_01"
        item.document_id = "doc_vec_01"
        item.text = "Vector embeddings capture semantic similarity across high-dimensional spaces."
        item.normalized_score = 0.92
        item.metadata = {"filename": "embeddings.txt"}
        mock_resp = MagicMock()
        mock_resp.items = [item]
        mock_vector.search.return_value = mock_resp

        pipeline = RAGPipeline(
            vector_search=mock_vector,
            ollama_client=mock_ollama,
        )

        answer = pipeline.answer_question(
            request="How do vector embeddings work?",
            retrieval_mode="vector",
        )

        assert answer.status == AnswerStatus.SUCCESS
        assert answer.retrieval_mode == "vector"
        assert len(answer.evidence_items) == 1
        assert answer.evidence_items[0].source_modality == "vector"
        assert answer.citations[0].is_valid is True

    def test_answer_question_functional_helper(self, monkeypatch):
        """Test the top-level answer_question convenience function."""
        # Patch RAGPipeline inside answer_question
        mock_pipeline_inst = MagicMock()
        mock_pipeline_inst.answer_question.return_value = GroundedAnswer(
            query="Test",
            answer="Functional answer [S1].",
            status=AnswerStatus.SUCCESS,
        )
        monkeypatch.setattr("graphrag_bench.rag_pipeline.RAGPipeline", lambda: mock_pipeline_inst)

        res = answer_question("Test", retrieval_mode="hybrid")
        assert res.answer == "Functional answer [S1]."

    def test_pipeline_low_relevance_short_circuit_q6(self):
        """Test that unrelated queries returning low-relevance candidates short-circuit with generation_invoked=False."""
        mock_ollama = MagicMock(spec=OllamaClient)
        mock_hybrid = MagicMock()
        # Candidate with low relevance score (similar to Q6 Zurich snowfall)
        cand = FusionCandidate(
            chunk_id="doc_8125aaecfd279966_c0000_afd4f832",
            document_id="doc_8125aaecfd279966",
            source_filename="graph_rag_benchmarking_guide.md",
            chunk_text="A comprehensive guide to Graph RAG benchmarking...",
            fused_rank=1,
            retrieval_sources=RetrievalSource.VECTOR,
            rrf_score=0.0164,
            source_metadata={"cosine_similarity": 0.3845, "normalized_score": 0.3845},
        )
        mock_hybrid.search_hybrid.return_value = HybridSearchResponse(
            query="What was the average snowfall in Zurich during January 1985?",
            status=HybridSearchStatus.SUCCESS,
            results=[cand],
            fused_results_count=1,
            vector_latency_ms=125.0,
            graph_latency_ms=45.0,
            fusion_latency_ms=1.2,
        )

        pipeline = RAGPipeline(
            hybrid_retriever=mock_hybrid,
            ollama_client=mock_ollama,
        )

        answer = pipeline.answer_question(
            request="What was the average snowfall in Zurich during January 1985?",
            retrieval_mode="hybrid",
            min_relevance_score=0.50,
        )

        assert answer.status == AnswerStatus.INSUFFICIENT_EVIDENCE
        assert answer.generation_invoked is False
        assert answer.answer_generation_model_calls == 0
        assert answer.latency_generation_ms == 0.0
        assert answer.faithfulness_score is None
        assert answer.citation_reference_precision is None
        assert answer.citation_precision is None
        assert answer.latency_breakdown["generation_ms"] == 0.0
        assert answer.latency_breakdown["vector_retrieval_ms"] == 125.0
        assert answer.latency_breakdown["graph_retrieval_ms"] == 45.0
        # Crucial check: verify LLM generate was never invoked
        mock_ollama.generate.assert_not_called()

    def test_pipeline_conflicting_sources_handling(self):
        """Test pipeline behavior when retrieved context contains genuine contradictory assertions."""
        mock_ollama = MagicMock(spec=OllamaClient)
        mock_ollama.default_llm = "qwen3:8b"
        mock_ollama.generate.return_value = GenerateResponse(
            model="qwen3:8b",
            response=(
                "The retrieved sources present conflicting information: [S1] states that the "
                "standard RRF smoothing constant k is 60, whereas [S2] asserts that k is 100. "
                "Due to this discrepancy, the exact benchmark constant cannot be definitively confirmed."
            ),
            done=True,
            eval_count=45,
            prompt_eval_count=220,
        )

        mock_hybrid = MagicMock()
        cand1 = FusionCandidate(
            chunk_id="chunk_conf_01",
            document_id="doc_conf_01",
            source_filename="guide.md",
            chunk_text="In our benchmark implementation, the standard RRF smoothing constant k is 60.",
            fused_rank=1,
            retrieval_sources=RetrievalSource.VECTOR,
            rrf_score=0.85,
            source_metadata={"normalized_score": 0.85},
        )
        cand2 = FusionCandidate(
            chunk_id="chunk_conf_02",
            document_id="doc_conf_02",
            source_filename="paper.docx",
            chunk_text="In our benchmark implementation, the standard RRF smoothing constant k is 100.",
            fused_rank=2,
            retrieval_sources=RetrievalSource.GRAPH,
            rrf_score=0.82,
            source_metadata={"graph_score": 0.82},
        )
        mock_hybrid.search_hybrid.return_value = HybridSearchResponse(
            query="What is the standard RRF constant k in the benchmark?",
            status=HybridSearchStatus.SUCCESS,
            results=[cand1, cand2],
            fused_results_count=2,
        )

        pipeline = RAGPipeline(
            hybrid_retriever=mock_hybrid,
            ollama_client=mock_ollama,
        )

        answer = pipeline.answer_question(
            request="What is the standard RRF constant k in the benchmark?",
            retrieval_mode="hybrid",
        )

        assert answer.status == AnswerStatus.SUCCESS
        assert answer.is_grounded is True
        assert len(answer.citations) == 2
        cited_ids = {c.evidence_id for c in answer.citations}
        assert cited_ids == {"S1", "S2"}
        assert all(c.is_valid for c in answer.citations)
        assert answer.citation_reference_precision == 1.0
        assert "conflicting" in answer.answer.lower()
        assert "[s1]" in answer.answer.lower()
        assert "[s2]" in answer.answer.lower()

    def test_pipeline_partial_evidence_grounding_distinction(self):
        """Test distinction between supported assertions and unsupported assertions in partial evidence answers."""
        mock_ollama = MagicMock(spec=OllamaClient)
        mock_ollama.default_llm = "qwen3:8b"
        mock_ollama.generate.return_value = GenerateResponse(
            model="qwen3:8b",
            response=(
                "Graph RAG utilizes ChromaDB as its persistent vector database [S1]. "
                "Furthermore, it incorporates quantum annealing for hyperdimensional graph routing."
            ),
            done=True,
            eval_count=35,
            prompt_eval_count=190,
        )

        mock_hybrid = MagicMock()
        cand = FusionCandidate(
            chunk_id="chunk_part_01",
            document_id="doc_part_01",
            source_filename="guide.md",
            chunk_text="Graph RAG utilizes ChromaDB as its persistent vector database for dense retrieval.",
            fused_rank=1,
            retrieval_sources=RetrievalSource.VECTOR,
            rrf_score=0.90,
            source_metadata={"normalized_score": 0.90},
        )
        mock_hybrid.search_hybrid.return_value = HybridSearchResponse(
            query="What vector database and routing does Graph RAG use?",
            status=HybridSearchStatus.SUCCESS,
            results=[cand],
            fused_results_count=1,
        )

        pipeline = RAGPipeline(
            hybrid_retriever=mock_hybrid,
            ollama_client=mock_ollama,
        )

        answer = pipeline.answer_question(
            request="What vector database and routing does Graph RAG use?",
            retrieval_mode="hybrid",
            enable_faithfulness_check=True,
        )

        assert answer.status == AnswerStatus.SUCCESS
        assert answer.is_grounded is True
        assert len(answer.claims) >= 2
        # Verify supported claim exists
        verdicts = [c.verdict for c in answer.claims]
        assert "SUPPORTED" in verdicts
        # Verify unsupported or uncertain claim exists for the quantum assertion
        assert any(v in {"UNSUPPORTED", "UNCERTAIN"} for v in verdicts)
        # Faithfulness score should reflect partial support (< 1.0)
        assert answer.faithfulness_score is not None
        assert 0.0 < answer.faithfulness_score < 1.0
