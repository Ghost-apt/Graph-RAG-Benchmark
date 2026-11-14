"""Citation Extraction & Ground Truth Provenance Validator for Grounded QA (Phase 7).

Parses citation markers (e.g. [S1], [S2], [S1, S2]) from generated text,
validates them against the active evidence registry, and checks provenance.
"""

import re
from typing import NamedTuple

from graphrag_bench.answer_schemas import AnswerCitation, EvidenceItem
from graphrag_bench.logger import get_logger

logger = get_logger(__name__)


class CitationValidationResult(NamedTuple):
    """Container for citation validation output.

    NOTE: This result establishes CITATION REFERENCE VALIDITY and SOURCE PROVENANCE
    (i.e. whether [Sn] markers reference existing items in the active evidence registry
    and map to valid canonical chunks/documents). It does NOT establish semantic claim support
    (i.e. whether the cited evidence text semantically entails the asserted factual claim).
    Semantic grounding is evaluated separately by FaithfulnessChecker.
    """

    citations: list[AnswerCitation]
    valid_citations: list[AnswerCitation]
    invalid_citations: list[AnswerCitation]
    citation_reference_precision: float | None
    cited_evidence_ids: list[str]
    uncited_evidence_ids: list[str]
    warnings: list[str]

    @property
    def citation_precision(self) -> float | None:
        """Backward-compatible alias for citation_reference_precision."""
        return self.citation_reference_precision


class CitationValidator:
    """Extracts and verifies citation markers against provided evidence context."""

    # Matches bracketed citation patterns like [S1], [S2], [S1, S2], [S1,S3]
    CITATION_PATTERN = re.compile(r"\[(S\d+(?:[\s,]+S\d+)*)\]", re.IGNORECASE)
    INDIVIDUAL_TAG_PATTERN = re.compile(r"S\d+", re.IGNORECASE)

    def extract_citation_ids(self, text: str) -> list[str]:
        """Extract all raw evidence IDs (e.g. ['S1', 'S2']) from text in order of appearance.

        Args:
            text: Input generated text.

        Returns:
            List of normalized evidence ID strings (e.g. ['S1', 'S2']).
        """
        extracted_ids: list[str] = []
        if not text:
            return extracted_ids

        for match in self.CITATION_PATTERN.finditer(text):
            inner = match.group(1)
            for tag in self.INDIVIDUAL_TAG_PATTERN.finditer(inner):
                normalized = tag.group(0).upper()
                extracted_ids.append(normalized)

        return extracted_ids

    def validate_citations(
        self,
        text: str,
        evidence_items: list[EvidenceItem],
    ) -> CitationValidationResult:
        """Parse citations from text and validate against provided evidence items.

        Args:
            text: Generated answer text.
            evidence_items: List of EvidenceItem instances provided in prompt context.

        Returns:
            CitationValidationResult with validated citations and precision metrics.
        """
        evidence_map: dict[str, EvidenceItem] = {
            item.evidence_id.upper(): item for item in evidence_items
        }

        extracted_ids = self.extract_citation_ids(text)
        citations: list[AnswerCitation] = []
        valid_citations: list[AnswerCitation] = []
        invalid_citations: list[AnswerCitation] = []
        warnings: list[str] = []

        seen_markers: set[str] = set()

        for ev_id in extracted_ids:
            marker = f"[{ev_id}]"
            # Deduplicate multiple occurrences of same citation marker in citations list
            # while still recording that it is valid
            if ev_id in seen_markers:
                continue
            seen_markers.add(ev_id)

            if ev_id in evidence_map:
                item = evidence_map[ev_id]
                content = item.content.strip()
                snippet = content[:200] + ("..." if len(content) > 200 else "")
                cit = AnswerCitation(
                    citation_marker=marker,
                    evidence_id=ev_id,
                    chunk_id=item.chunk_id,
                    document_id=item.document_id,
                    document_title=item.document_title,
                    snippet=snippet,
                    is_valid=True,
                    validation_error=None,
                )
                citations.append(cit)
                valid_citations.append(cit)
            else:
                msg = f"Unknown citation marker {marker} is not present in retrieved context."
                logger.warning(msg)
                warnings.append(msg)
                cit = AnswerCitation(
                    citation_marker=marker,
                    evidence_id=ev_id,
                    chunk_id="",
                    document_id="",
                    document_title="",
                    snippet="",
                    is_valid=False,
                    validation_error=msg,
                )
                citations.append(cit)
                invalid_citations.append(cit)

        total_count = len(citations)
        valid_count = len(valid_citations)
        precision: float | None = (valid_count / total_count) if total_count > 0 else None

        cited_evidence_ids = sorted({c.evidence_id for c in valid_citations})
        all_evidence_ids = {item.evidence_id.upper() for item in evidence_items}
        uncited_evidence_ids = sorted(all_evidence_ids - set(cited_evidence_ids))

        return CitationValidationResult(
            citations=citations,
            valid_citations=valid_citations,
            invalid_citations=invalid_citations,
            citation_reference_precision=precision,
            cited_evidence_ids=cited_evidence_ids,
            uncited_evidence_ids=uncited_evidence_ids,
            warnings=warnings,
        )
