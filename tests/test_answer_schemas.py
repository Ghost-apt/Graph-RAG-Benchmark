"""Unit tests for Grounded QA answer schemas (graphrag_bench.answer_schemas)."""

import pytest
from pydantic import ValidationError

from graphrag_bench.answer_schemas import (
    AnswerCitation,
    AnswerClaim,
    AnswerGenerationRequest,
    AnswerStatus,
    EvidenceItem,
    GroundedAnswer,
)


class TestAnswerSchemas:
    """Test suite for Phase 7 domain schemas."""

    def test_evidence_item_valid(self):
        """Test valid EvidenceItem creation and property access."""
        item = EvidenceItem(
            evidence_id="S1",
            chunk_id="chunk_123_0000",
            document_id="doc_123",
            document_title="Sample Doc",
            content="This is sample content for testing.",
            token_count=10,
            source_modality="hybrid",
            retrieval_score=0.85,
            metadata={"rank": 1},
        )
        assert item.evidence_id == "S1"
        assert item.chunk_id == "chunk_123_0000"
        assert item.document_id == "doc_123"
        assert item.document_title == "Sample Doc"
        assert item.content == "This is sample content for testing."
        assert item.token_count == 10
        assert item.source_modality == "hybrid"
        assert item.retrieval_score == 0.85

    def test_evidence_item_validation(self):
        """Test that empty evidence_id raises ValidationError."""
        with pytest.raises(ValidationError):
            EvidenceItem(
                evidence_id="   ",
                chunk_id="chunk_1",
                document_id="doc_1",
                document_title="Doc",
                content="Text",
            )

    def test_answer_citation_valid(self):
        """Test valid AnswerCitation model."""
        cit = AnswerCitation(
            citation_marker="[S1]",
            evidence_id="S1",
            chunk_id="chunk_123_0000",
            document_id="doc_123",
            document_title="Doc Title",
            snippet="Snippet text",
            is_valid=True,
        )
        assert cit.citation_marker == "[S1]"
        assert cit.evidence_id == "S1"
        assert cit.is_valid is True
        assert cit.validation_error is None

    def test_answer_claim_model(self):
        """Test AnswerClaim creation and defaults."""
        claim = AnswerClaim(
            claim_id="C1",
            claim_text="Graph RAG improves multi-hop reasoning over vector RAG.",
            cited_sources=["S1", "S2"],
            verdict="SUPPORTED",
            reasoning="Explicitly stated in evidence chunk.",
            grounding_score=0.95,
        )
        assert claim.claim_id == "C1"
        assert claim.verdict == "SUPPORTED"
        assert claim.grounding_score == 0.95
        assert len(claim.cited_sources) == 2

    def test_answer_generation_request(self):
        """Test AnswerGenerationRequest validation."""
        req = AnswerGenerationRequest(query="What is Graph RAG?")
        assert req.query == "What is Graph RAG?"
        assert req.retrieval_mode == "hybrid"
        assert req.top_k == 5
        assert req.temperature == 0.1
        assert req.require_citations is True

        # Empty query rejection
        with pytest.raises(ValidationError):
            AnswerGenerationRequest(query="   ")

    def test_grounded_answer_model(self):
        """Test GroundedAnswer model with helper properties."""
        ev1 = EvidenceItem(
            evidence_id="S1",
            chunk_id="c1",
            document_id="d1",
            document_title="Title 1",
            content="Content 1",
            source_modality="hybrid",
        )
        ev2 = EvidenceItem(
            evidence_id="S2",
            chunk_id="c2",
            document_id="d2",
            document_title="Title 2",
            content="Content 2",
            source_modality="vector",
        )
        cit1 = AnswerCitation(
            citation_marker="[S1]",
            evidence_id="S1",
            chunk_id="c1",
            document_id="d1",
            document_title="Title 1",
            is_valid=True,
        )

        ans = GroundedAnswer(
            query="Explain Graph RAG.",
            answer="Graph RAG combines knowledge graphs with vectors [S1].",
            status=AnswerStatus.SUCCESS,
            evidence_items=[ev1, ev2],
            citations=[cit1],
            faithfulness_score=1.0,
            citation_precision=1.0,
        )

        assert ans.is_grounded is True
        assert len(ans.cited_evidence) == 1
        assert ans.cited_evidence[0].evidence_id == "S1"

    def test_grounded_answer_null_faithfulness_and_precision_aliases(self):
        """Test nullability of faithfulness_score and alias mapping for citation_reference_precision."""
        ans = GroundedAnswer(
            query="Unanswerable query",
            answer="Based on the provided evidence, there is insufficient information.",
            status=AnswerStatus.INSUFFICIENT_EVIDENCE,
            evidence_items=[],
            citations=[],
            faithfulness_score=None,  # Null when 0 claims evaluated
            citation_precision=None,  # Null when 0 citations exist
            generation_invoked=False,
            total_model_invocations=1,
            query_extraction_model_calls=1,
            answer_generation_model_calls=0,
            latency_breakdown={"retrieval_total_ms": 15.0, "generation_ms": 0.0},
        )
        assert ans.faithfulness_score is None
        assert ans.citation_reference_precision is None
        assert ans.citation_precision is None
        assert ans.generation_invoked is False
        assert ans.total_model_invocations == 1
        assert ans.query_extraction_model_calls == 1
        assert ans.answer_generation_model_calls == 0
        assert ans.latency_breakdown["generation_ms"] == 0.0

    def test_answer_citation_claim_supported_attribute(self):
        """Test that AnswerCitation supports claim_supported attribute (defaulting to None)."""
        cit = AnswerCitation(
            citation_marker="[S1]",
            evidence_id="S1",
            chunk_id="c1",
            document_id="d1",
            document_title="Doc",
            is_valid=True,
        )
        assert cit.claim_supported is None
        cit_verified = AnswerCitation(
            citation_marker="[S1]",
            evidence_id="S1",
            chunk_id="c1",
            document_id="d1",
            document_title="Doc",
            is_valid=True,
            claim_supported=True,
        )
        assert cit_verified.claim_supported is True
