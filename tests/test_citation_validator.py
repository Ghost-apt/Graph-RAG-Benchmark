"""Unit tests for Citation Validator (graphrag_bench.citation_validator)."""

from graphrag_bench.answer_schemas import EvidenceItem
from graphrag_bench.citation_validator import CitationValidator


class TestCitationValidator:
    """Test suite for citation regex extraction and provenance verification."""

    def test_extract_citation_ids(self):
        """Test regex extraction of diverse citation marker formats."""
        validator = CitationValidator()

        text = "Graph RAG uses vector search [S1] and knowledge graphs [S2]. Compound support is noted in [S1, S3] and [S4,S5]."
        ids = validator.extract_citation_ids(text)
        assert ids == ["S1", "S2", "S1", "S3", "S4", "S5"]

    def test_validate_citations_all_valid(self):
        """Test citation validation when all cited markers exist in evidence."""
        ev1 = EvidenceItem(
            evidence_id="S1",
            chunk_id="chunk_01",
            document_id="doc_01",
            document_title="guide.md",
            content="Evidence one content.",
        )
        ev2 = EvidenceItem(
            evidence_id="S2",
            chunk_id="chunk_02",
            document_id="doc_02",
            document_title="paper.md",
            content="Evidence two content.",
        )

        validator = CitationValidator()
        text = "This fact is from [S1], and another fact is from [S2]."
        result = validator.validate_citations(text, [ev1, ev2])

        assert len(result.citations) == 2
        assert len(result.valid_citations) == 2
        assert len(result.invalid_citations) == 0
        assert result.citation_precision == 1.0
        assert result.cited_evidence_ids == ["S1", "S2"]
        assert result.uncited_evidence_ids == []
        assert result.citations[0].chunk_id == "chunk_01"
        assert result.citations[1].chunk_id == "chunk_02"

    def test_validate_citations_with_hallucinations(self):
        """Test citation validation when model invents unknown markers."""
        ev1 = EvidenceItem(
            evidence_id="S1",
            chunk_id="chunk_01",
            document_id="doc_01",
            document_title="guide.md",
            content="Evidence one content.",
        )

        validator = CitationValidator()
        text = "Valid assertion [S1], but fabricated assertion [S9]."
        result = validator.validate_citations(text, [ev1])

        assert len(result.citations) == 2
        assert len(result.valid_citations) == 1
        assert len(result.invalid_citations) == 1
        assert result.citation_precision == 0.5  # 1 valid out of 2 total
        assert result.invalid_citations[0].evidence_id == "S9"
        assert result.invalid_citations[0].is_valid is False
        assert len(result.warnings) == 1

    def test_validate_citations_uncited_evidence(self):
        """Test reporting of uncited evidence items."""
        ev1 = EvidenceItem(
            evidence_id="S1",
            chunk_id="chunk_01",
            document_id="doc_01",
            document_title="guide.md",
            content="Evidence one.",
        )
        ev2 = EvidenceItem(
            evidence_id="S2",
            chunk_id="chunk_02",
            document_id="doc_02",
            document_title="paper.md",
            content="Evidence two.",
        )

        validator = CitationValidator()
        text = "Only source [S1] was used."
        result = validator.validate_citations(text, [ev1, ev2])

        assert result.cited_evidence_ids == ["S1"]
        assert result.uncited_evidence_ids == ["S2"]

    def test_compound_and_duplicate_markers(self):
        """Test compound markers [S1, S2] and deduplication of repeated [S1] occurrences."""
        ev1 = EvidenceItem(
            evidence_id="S1",
            chunk_id="chunk_01",
            document_id="doc_01",
            document_title="guide.md",
            content="Evidence one content.",
        )
        ev2 = EvidenceItem(
            evidence_id="S2",
            chunk_id="chunk_02",
            document_id="doc_02",
            document_title="paper.md",
            content="Evidence two content.",
        )
        validator = CitationValidator()
        text = "First assertion [S1]. Second assertion [S1, S2]. Re-citing [S1]."
        result = validator.validate_citations(text, [ev1, ev2])

        assert len(result.citations) == 2  # S1 and S2 deduplicated
        assert len(result.valid_citations) == 2
        assert result.citation_reference_precision == 1.0
        assert result.citation_precision == 1.0  # Alias property
        assert result.cited_evidence_ids == ["S1", "S2"]

    def test_missing_citations_empty(self):
        """Test validation when answer text contains no citation markers."""
        ev1 = EvidenceItem(
            evidence_id="S1",
            chunk_id="chunk_01",
            document_id="doc_01",
            document_title="guide.md",
            content="Evidence one content.",
        )
        validator = CitationValidator()
        text = "This response provides factual text without citing any source brackets."
        result = validator.validate_citations(text, [ev1])

        assert len(result.citations) == 0
        assert len(result.valid_citations) == 0
        assert result.citation_reference_precision is None
        assert result.citation_precision is None
        assert result.uncited_evidence_ids == ["S1"]

    def test_citations_referring_to_wrong_registry(self):
        """Test validation when citations refer to IDs not present in current active registry."""
        ev_active = EvidenceItem(
            evidence_id="S5",
            chunk_id="chunk_05",
            document_id="doc_05",
            document_title="guide5.md",
            content="Evidence five content.",
        )
        validator = CitationValidator()
        # Text cites S1 and S2 which belonged to another search registry
        text = "Assertion based on old search context [S1, S2]."
        result = validator.validate_citations(text, [ev_active])

        assert len(result.citations) == 2
        assert len(result.valid_citations) == 0
        assert len(result.invalid_citations) == 2
        assert result.citation_reference_precision == 0.0
        assert "S1" in result.invalid_citations[0].validation_error
        assert "S2" in result.invalid_citations[1].validation_error

    def test_reference_validity_vs_claim_support_semantics(self):
        """Verify that citation validation establishes reference validity and provenance without claiming semantic entailment."""
        ev1 = EvidenceItem(
            evidence_id="S1",
            chunk_id="chunk_01",
            document_id="doc_01",
            document_title="guide.md",
            content="ChromaDB uses HNSW indexing for vector search.",
        )
        validator = CitationValidator()
        # Falsely cites S1 for an ungrounded claim about astrophysics
        text = "Black holes emit Hawking radiation [S1]."
        result = validator.validate_citations(text, [ev1])

        assert len(result.valid_citations) == 1
        cit = result.valid_citations[0]
        # Reference validity and source provenance are true
        assert cit.is_valid is True
        assert cit.chunk_id == "chunk_01"
        assert cit.document_id == "doc_01"
        # Crucial distinction: claim_supported defaults to None because citation validation
        # strictly checks reference provenance, not semantic claim entailment
        assert cit.claim_supported is None
