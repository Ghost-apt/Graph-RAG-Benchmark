"""Unit tests for Answer Guardrails (graphrag_bench.answer_guardrails)."""

from graphrag_bench.answer_guardrails import (
    STANDARD_INSUFFICIENT_EVIDENCE_MESSAGE,
    AnswerGuardrails,
)
from graphrag_bench.answer_schemas import AnswerCitation, AnswerStatus, EvidenceItem


class TestAnswerGuardrails:
    """Test suite for safety guardrails and refusal handling."""

    def test_pre_check_empty_query(self):
        """Test that empty queries fail pre-generation checks."""
        guard = AnswerGuardrails()
        res = guard.evaluate_pre_generation(query="   ", evidence_items=[])
        assert res.can_proceed is False
        assert res.status == AnswerStatus.FAILED

    def test_pre_check_zero_evidence(self):
        """Test short-circuiting when zero evidence chunks are retrieved."""
        guard = AnswerGuardrails(min_evidence_items=1)
        res = guard.evaluate_pre_generation(query="What is Graph RAG?", evidence_items=[])
        assert res.can_proceed is False
        assert res.status == AnswerStatus.INSUFFICIENT_EVIDENCE
        assert res.early_answer == STANDARD_INSUFFICIENT_EVIDENCE_MESSAGE

    def test_pre_check_valid_evidence(self):
        """Test passing pre-check with valid evidence."""
        item = EvidenceItem(
            evidence_id="S1",
            chunk_id="c1",
            document_id="d1",
            document_title="doc.md",
            content="Content",
            retrieval_score=0.8,
        )
        guard = AnswerGuardrails(min_evidence_items=1)
        res = guard.evaluate_pre_generation(query="Question?", evidence_items=[item])
        assert res.can_proceed is True
        assert res.status == AnswerStatus.SUCCESS

    def test_post_check_delimiter_leakage_sanitization(self):
        """Test sanitizing internal prompt delimiters from model answers."""
        guard = AnswerGuardrails()
        raw = "<<<EVIDENCE_BLOCK_START>>>\nGraph RAG combines graph traversal [S1].\n<<<EVIDENCE_BLOCK_END>>>"
        cit = AnswerCitation(
            citation_marker="[S1]",
            evidence_id="S1",
            chunk_id="c1",
            document_id="d1",
            document_title="d.md",
            is_valid=True,
        )
        res = guard.evaluate_post_generation(raw, [cit], require_citations=True)
        assert res.status == AnswerStatus.SUCCESS
        assert "<<<EVIDENCE_BLOCK_START>>>" not in res.sanitized_answer
        assert "<<<EVIDENCE_BLOCK_END>>>" not in res.sanitized_answer
        assert "Graph RAG combines graph traversal [S1]." in res.sanitized_answer
        assert len(res.warnings) >= 1

    def test_post_check_insufficient_evidence_phrase(self):
        """Test that model refusal phrases trigger INSUFFICIENT_EVIDENCE status."""
        guard = AnswerGuardrails()
        ans = "Based on the provided evidence, there is insufficient information to answer this question."
        res = guard.evaluate_post_generation(ans, [], require_citations=True)
        assert res.status == AnswerStatus.INSUFFICIENT_EVIDENCE

    def test_post_check_missing_required_citations(self):
        """Test warning when citations are required but none are valid."""
        guard = AnswerGuardrails()
        ans = "Graph RAG is a useful technology."
        res = guard.evaluate_post_generation(ans, [], require_citations=True)
        assert res.status == AnswerStatus.SUCCESS
        assert any("no valid citations" in w for w in res.warnings)

    def test_guardrail_case1_answerable_with_relevant_evidence(self):
        """Case 1: Answerable question with relevant evidence proceeds normally."""
        guard = AnswerGuardrails(min_relevance_threshold=0.50)
        item = EvidenceItem(
            evidence_id="S1",
            chunk_id="c1",
            document_id="d1",
            document_title="guide.md",
            content="Graph RAG integrates NetworkX and ChromaDB.",
            retrieval_score=0.82,
            metadata={"normalized_score": 0.82},
        )
        res = guard.evaluate_pre_generation(
            query="What is Graph RAG?",
            evidence_items=[item],
        )
        assert res.can_proceed is True
        assert res.status == AnswerStatus.SUCCESS

    def test_guardrail_case2_unrelated_query_no_evidence(self):
        """Case 2: Unrelated question with zero retrieved evidence short-circuits."""
        guard = AnswerGuardrails(min_evidence_items=1)
        res = guard.evaluate_pre_generation(
            query="What was the average snowfall in Zurich during January 1985?",
            evidence_items=[],
        )
        assert res.can_proceed is False
        assert res.status == AnswerStatus.INSUFFICIENT_EVIDENCE
        assert res.early_answer == STANDARD_INSUFFICIENT_EVIDENCE_MESSAGE

    def test_guardrail_case3_unrelated_query_irrelevant_vector_results(self):
        """Case 3: Unrelated question returning low-similarity vector chunks short-circuits before LLM call."""
        guard = AnswerGuardrails(min_relevance_threshold=0.50)
        # Vector search returned 3 chunks with low similarity (0.3845, 0.3359, 0.3312) like Q6
        item1 = EvidenceItem(
            evidence_id="S1",
            chunk_id="c1",
            document_id="d1",
            document_title="guide.md",
            content="Some benchmark architecture info.",
            retrieval_score=0.3845,
            metadata={"normalized_score": 0.3845},
        )
        item2 = EvidenceItem(
            evidence_id="S2",
            chunk_id="c2",
            document_id="d2",
            document_title="paper.md",
            content="Some vector search info.",
            retrieval_score=0.3359,
            metadata={"normalized_score": 0.3359},
        )
        res = guard.evaluate_pre_generation(
            query="What was the average snowfall in Zurich during January 1985?",
            evidence_items=[item1, item2],
            min_relevance_threshold=0.50,
        )
        assert res.can_proceed is False
        assert res.status == AnswerStatus.INSUFFICIENT_EVIDENCE
        assert res.early_answer == STANDARD_INSUFFICIENT_EVIDENCE_MESSAGE
        assert any("below required relevance threshold" in w for w in res.warnings)

    def test_guardrail_case4_unrelated_query_irrelevant_graph_results(self):
        """Case 4: Unrelated question returning weak graph paths short-circuits."""
        guard = AnswerGuardrails(min_relevance_threshold=0.50)
        item = EvidenceItem(
            evidence_id="S1",
            chunk_id="cg1",
            document_id="dg1",
            document_title="graph.json",
            content="Entity relation path.",
            retrieval_score=0.20,
            metadata={"normalized_score": 0.20},
        )
        res = guard.evaluate_pre_generation(
            query="Who invented the saxophone?",
            evidence_items=[item],
        )
        assert res.can_proceed is False
        assert res.status == AnswerStatus.INSUFFICIENT_EVIDENCE

    def test_guardrail_case5_partially_relevant_below_threshold(self):
        """Case 5: Partially relevant question whose evidence falls below threshold short-circuits."""
        guard = AnswerGuardrails(min_relevance_threshold=0.50)
        item = EvidenceItem(
            evidence_id="S1",
            chunk_id="c1",
            document_id="d1",
            document_title="guide.md",
            content="General database indexing concepts.",
            retrieval_score=0.45,
            metadata={"normalized_score": 0.45},
        )
        res = guard.evaluate_pre_generation(
            query="What specific neural architecture is used in Qwen3 MoE routing?",
            evidence_items=[item],
        )
        assert res.can_proceed is False
        assert res.status == AnswerStatus.INSUFFICIENT_EVIDENCE

    def test_guardrail_case6_conflicting_sources_refusal_capture(self):
        """Case 6: Model refusing or expressing contradictory uncertainty is flagged as INSUFFICIENT_EVIDENCE."""
        guard = AnswerGuardrails()
        refusal = (
            "The evidence does not provide consistent information to determine the exact date."
        )
        res = guard.evaluate_post_generation(refusal, [], require_citations=True)
        assert res.status == AnswerStatus.INSUFFICIENT_EVIDENCE

    def test_guardrail_case7_partial_answer_without_citations_warning(self):
        """Case 7: Answer addressing only partial aspect without citations generates warning."""
        guard = AnswerGuardrails()
        partial_answer = (
            "Graph RAG addresses retrieval, but the guide does not mention quantum encryption."
        )
        res = guard.evaluate_post_generation(partial_answer, [], require_citations=True)
        assert res.status == AnswerStatus.INSUFFICIENT_EVIDENCE or len(res.warnings) >= 1

    def test_guardrail_decoupled_modality_high_graph_low_vector(self):
        """Verify that high graph relevance allows generation even when vector relevance is low."""
        guard = AnswerGuardrails(min_relevance_threshold=0.50)
        item = EvidenceItem(
            evidence_id="S1",
            chunk_id="c_graph",
            document_id="doc_g",
            document_title="graph_doc.md",
            content="Graph relation path connecting concepts.",
            source_modality="hybrid",
            retrieval_score=0.032,  # RRF score
            metadata={
                "normalized_score": 0.35,  # Low vector score (< 0.50)
                "graph_score": 0.85,  # High graph score (>= 0.40)
            },
        )
        res = guard.evaluate_pre_generation(
            query="Explain relationship between X and Y.",
            evidence_items=[item],
            min_relevance_threshold=0.50,
        )
        assert res.can_proceed is True
        assert res.status == AnswerStatus.SUCCESS

    def test_guardrail_decoupled_modality_high_vector_low_graph(self):
        """Verify that high vector relevance allows generation even when graph relevance is low."""
        guard = AnswerGuardrails(min_relevance_threshold=0.50)
        item = EvidenceItem(
            evidence_id="S1",
            chunk_id="c_vec",
            document_id="doc_v",
            document_title="vector_doc.md",
            content="Detailed text description.",
            source_modality="hybrid",
            retrieval_score=0.030,
            metadata={
                "normalized_score": 0.78,  # High vector score (>= 0.50)
                "graph_score": 0.15,  # Low graph score (< 0.40)
            },
        )
        res = guard.evaluate_pre_generation(
            query="What is the concept definition?",
            evidence_items=[item],
            min_relevance_threshold=0.50,
        )
        assert res.can_proceed is True
        assert res.status == AnswerStatus.SUCCESS
