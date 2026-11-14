"""Domain schemas and data models for Grounded Question Answering (Phase 7).

Defines Pydantic v2 models for evidence items, citations, factual claims,
answer requests, and complete grounded answers with full provenance metadata.
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AnswerStatus(str, Enum):
    """Execution status for grounded answer generation."""

    SUCCESS = "success"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    FAILED = "failed"
    GUARDRAIL_BLOCKED = "guardrail_blocked"
    REFUSED = "refused"


class EvidenceItem(BaseModel):
    """Structured representation of a single retrieved evidence context chunk.

    Assigned a deterministic citation key (e.g., 'S1', 'S2') for LLM attribution.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    evidence_id: str = Field(
        ...,
        description="Deterministic source identifier for citation matching (e.g. 'S1', 'S2')",
    )
    chunk_id: str = Field(
        ...,
        description="Canonical chunk identifier (e.g. 'chunk_8125aaecfd279966_0000')",
    )
    document_id: str = Field(
        ...,
        description="Canonical document identifier (e.g. 'doc_8125aaecfd279966')",
    )
    document_title: str = Field(
        ...,
        description="Document title or source filename",
    )
    content: str = Field(
        ...,
        description="Cleaned text content or excerpt from the chunk",
    )
    token_count: int = Field(
        default=0,
        ge=0,
        description="Estimated token count for this evidence item",
    )
    source_modality: Literal["vector", "graph", "hybrid"] = Field(
        default="hybrid",
        description="Modality that retrieved this evidence item",
    )
    retrieval_score: float = Field(
        default=0.0,
        ge=0.0,
        description="Fused or raw retrieval relevance score",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional provenance metadata (e.g. entities, relations, rank)",
    )

    @field_validator("evidence_id")
    @classmethod
    def validate_evidence_id(cls, v: str) -> str:
        """Ensure evidence ID starts with 'S' followed by digits or is non-empty."""
        v = v.strip()
        if not v:
            raise ValueError("evidence_id cannot be empty.")
        return v

    @property
    def score(self) -> float:
        """Score alias for retrieval_score."""
        return self.retrieval_score

    @property
    def text(self) -> str:
        """Text alias for content."""
        return self.content


class AnswerCitation(BaseModel):
    """Citation extracted from generated answer attributing claims to evidence.

    Distinguishes citation reference validity (does marker [S1] exist in active context?)
    and source provenance mapping from semantic claim support.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    citation_marker: str = Field(
        ...,
        description="Citation marker string as written by LLM (e.g. '[S1]')",
    )
    evidence_id: str = Field(
        ...,
        description="Source identifier extracted from marker (e.g. 'S1')",
    )
    chunk_id: str = Field(
        default="",
        description="Canonical chunk identifier mapped from evidence registry",
    )
    document_id: str = Field(
        default="",
        description="Canonical document identifier mapped from evidence registry",
    )
    document_title: str = Field(
        default="",
        description="Document title or filename mapped from evidence registry",
    )
    snippet: str = Field(
        default="",
        description="Evidence text excerpt supporting this citation",
    )
    is_valid: bool = Field(
        default=True,
        description="Whether the citation corresponds to an actual retrieved evidence item in active registry",
    )
    validation_error: str | None = Field(
        default=None,
        description="Explanation if citation reference is invalid or hallucinated",
    )
    claim_supported: bool | None = Field(
        default=None,
        description="Whether cited evidence semantically supports the associated claim (evaluated by faithfulness checker, distinct from reference validity)",
    )


class AnswerClaim(BaseModel):
    """Atomic factual claim extracted from an answer for grounding verification."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    claim_id: str = Field(
        ...,
        description="Identifier for the claim (e.g. 'C1', 'C2')",
    )
    claim_text: str = Field(
        ...,
        description="Natural language statement of the extracted factual claim",
    )
    cited_sources: list[str] = Field(
        default_factory=list,
        description="List of evidence IDs cited for this claim (e.g. ['S1'])",
    )
    verdict: Literal["SUPPORTED", "UNSUPPORTED", "UNCERTAIN"] = Field(
        default="UNCERTAIN",
        description="Grounding verdict relative to retrieved evidence",
    )
    reasoning: str = Field(
        default="",
        description="Explanation supporting the verification verdict",
    )
    grounding_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Estimated confidence score for grounding verification",
    )


class AnswerGenerationRequest(BaseModel):
    """Parameters for executing a grounded QA query."""

    model_config = ConfigDict(extra="ignore")

    query: str = Field(
        ...,
        min_length=1,
        description="User natural-language query",
    )
    retrieval_mode: Literal["hybrid", "vector", "graph"] = Field(
        default="hybrid",
        description="Retrieval engine to use for evidence retrieval",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of evidence chunks to retrieve",
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="LLM generation temperature",
    )
    max_context_tokens: int = Field(
        default=3000,
        ge=200,
        le=16000,
        description="Maximum token budget for evidence packaging",
    )
    max_generation_tokens: int = Field(
        default=1024,
        ge=64,
        le=4096,
        description="Maximum generation token length",
    )
    require_citations: bool = Field(
        default=True,
        description="Whether to instruct and enforce citation markers",
    )
    min_relevance_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional minimum retrieval relevance score threshold required to proceed to generation",
    )
    enable_faithfulness_check: bool = Field(
        default=True,
        description="Whether to run preliminary claim faithfulness verification",
    )

    @field_validator("query")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        """Strip whitespace and reject blank queries."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Query string cannot be empty or whitespace.")
        return cleaned


class GroundedAnswer(BaseModel):
    """Comprehensive grounded answer response with full provenance and verification."""

    model_config = ConfigDict(extra="ignore")

    query: str = Field(
        ...,
        description="User natural-language query",
    )
    answer: str = Field(
        ...,
        description="Generated textual answer containing citations or status explanation",
    )
    status: AnswerStatus = Field(
        default=AnswerStatus.SUCCESS,
        description="Operational outcome of the QA pipeline",
    )
    evidence_items: list[EvidenceItem] = Field(
        default_factory=list,
        description="All evidence items packaged and presented to LLM context",
    )
    citations: list[AnswerCitation] = Field(
        default_factory=list,
        description="Extracted citations mapped to evidence registry",
    )
    claims: list[AnswerClaim] = Field(
        default_factory=list,
        description="Extracted factual claims and their grounding verdicts",
    )
    faithfulness_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Overall faithfulness metric (0.0 to 1.0), or None if zero claims were evaluated",
    )
    citation_reference_precision: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Reference precision of citations (valid active context references / total citations), or None if zero citations exist",
    )
    generation_invoked: bool = Field(
        default=True,
        description="Whether LLM answer generation was invoked (False if short-circuited by guardrails)",
    )
    total_model_invocations: int = Field(
        default=0,
        ge=0,
        description="Total count of LLM model invocations across all pipeline stages (query extraction + answer generation)",
    )
    query_extraction_model_calls: int = Field(
        default=0,
        ge=0,
        description="Count of LLM model invocations during query entity extraction",
    )
    answer_generation_model_calls: int = Field(
        default=0,
        ge=0,
        description="Count of LLM model invocations during final answer generation",
    )
    query_entity_extraction_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Latency for query entity extraction in milliseconds",
    )
    graph_traversal_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Latency for pure graph traversal/expansion in milliseconds",
    )
    retrieval_mode: str = Field(
        default="hybrid",
        description="Retrieval engine utilized ('hybrid', 'vector', 'graph')",
    )
    model_used: str = Field(
        default="qwen3:8b",
        description="LLM identifier used for generation",
    )
    prompt_tokens_estimated: int = Field(
        default=0,
        ge=0,
        description="Estimated token count of the prompt sent to LLM",
    )
    answer_tokens_estimated: int = Field(
        default=0,
        ge=0,
        description="Estimated token count of generated answer",
    )
    latency_retrieval_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Latency for evidence retrieval in milliseconds",
    )
    latency_generation_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Latency for answer generation in milliseconds",
    )
    latency_total_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Total end-to-end execution latency in milliseconds",
    )
    latency_breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Detailed execution latency breakdown by pipeline stage in milliseconds",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal warnings encountered during pipeline execution",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Fatal or degraded errors encountered during pipeline execution",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when answer was generated",
    )

    @model_validator(mode="before")
    @classmethod
    def handle_citation_precision_alias(cls, data: Any) -> Any:
        """Allow legacy citation_precision field to populate citation_reference_precision."""
        if isinstance(data, dict):
            d = dict(data)
            if "citation_precision" in d and "citation_reference_precision" not in d:
                d["citation_reference_precision"] = d["citation_precision"]
            return d
        return data

    @property
    def citation_precision(self) -> float | None:
        """Backward-compatible alias for citation_reference_precision."""
        return self.citation_reference_precision

    @property
    def is_grounded(self) -> bool:
        """Return True if answer is SUCCESS and has at least one valid citation.

        SEMANTIC DISTINCTION:
        This property certifies STRUCTURAL CITATION REFERENCE VALIDITY and SOURCE PROVENANCE
        (i.e. that the answer achieved SUCCESS status and contains at least one citation
        marker [Sn] that maps to an actual retrieved evidence chunk in the active context).
        It does NOT guarantee full semantic entailment or counterfactual truth of every claim;
        semantic claim factuality is separately evaluated by FaithfulnessChecker / faithfulness_score.
        """
        return (
            self.status == AnswerStatus.SUCCESS
            and len(self.citations) > 0
            and any(c.is_valid for c in self.citations)
        )

    @property
    def cited_evidence(self) -> list[EvidenceItem]:
        """Return evidence items that were cited at least once in valid citations."""
        valid_evidence_ids = {c.evidence_id for c in self.citations if c.is_valid}
        return [item for item in self.evidence_items if item.evidence_id in valid_evidence_ids]

    @property
    def telemetry(self) -> Any:
        """Provide structured telemetry object compatible with benchmark runner."""
        from types import SimpleNamespace

        b = self.latency_breakdown or {}
        return SimpleNamespace(
            retrieval_ms=self.latency_retrieval_ms or b.get("retrieval_total_ms", 0.0),
            query_entity_extraction_ms=self.query_entity_extraction_ms
            or b.get("query_entity_extraction_ms", 0.0),
            graph_traversal_ms=self.graph_traversal_ms or b.get("graph_traversal_ms", 0.0),
            vector_retrieval_ms=b.get("vector_retrieval_ms", 0.0),
            fusion_ms=b.get("hybrid_fusion_ms", 0.0),
            generation_ms=self.latency_generation_ms or b.get("generation_ms", 0.0),
            total_duration_ms=self.latency_total_ms or b.get("total_pipeline_ms", 0.0),
            query_extraction_model_calls=self.query_extraction_model_calls,
            answer_generation_model_calls=self.answer_generation_model_calls,
            total_model_invocations=self.total_model_invocations,
        )
