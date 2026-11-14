"""
Pydantic data models and schemas for Phase 5 Graph Retrieval & Multi-Hop Reasoning.
"""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ExtractionMethod(str, Enum):
    """Method utilized for discovering query entities."""

    EXACT_MATCH = "exact_match"
    ALIAS_MATCH = "alias_match"
    NORMALIZED_MATCH = "normalized_match"
    LLM_EXTRACTION = "llm_extraction"


class QueryEntity(BaseModel):
    """An entity mention extracted from a user question."""

    mention_text: str = Field(description="Exact substring mention from the user question")
    canonical_name: str = Field(description="Normalized or proposed canonical entity name")
    entity_type: str = Field(default="OTHER", description="Entity category taxonomy")
    normalized_term: str = Field(description="Search-normalized representation")
    extraction_method: ExtractionMethod = Field(
        default=ExtractionMethod.EXACT_MATCH,
        description="Method used to detect this entity",
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("mention_text", "canonical_name", "normalized_term")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Entity string fields cannot be empty.")
        return s


class QueryEntityExtractionResult(BaseModel):
    """Result of query entity identification."""

    query: str
    entities: list[QueryEntity] = Field(default_factory=list)
    duration_ms: float = Field(default=0.0, ge=0.0)
    llm_invoked: bool = Field(default=False)


class MatchType(str, Enum):
    """Type of match established during entity linking."""

    EXACT = "exact"
    ALIAS = "alias"
    NORMALIZED = "normalized"
    FUZZY = "fuzzy"


class LinkedEntity(BaseModel):
    """Mapping of a query mention to an existing GraphEntity in the Knowledge Graph."""

    query_mention: str = Field(description="Mention text in user query")
    entity_id: str = Field(description="Target GraphEntity deterministic identifier")
    canonical_name: str = Field(description="Canonical entity name in graph")
    entity_type: str = Field(description="Entity category in graph")
    match_type: MatchType = Field(description="How the match was resolved")
    is_ambiguous: bool = Field(
        default=False, description="Whether multiple candidate nodes matched"
    )
    candidates: list[dict[str, Any]] = Field(
        default_factory=list, description="Candidate matches if ambiguous"
    )
    match_explanation: str = Field(default="", description="Human-readable match explanation")


class EntityLinkingResult(BaseModel):
    """Result of entity linking step."""

    query: str
    linked_entities: list[LinkedEntity] = Field(default_factory=list)
    unlinked_mentions: list[str] = Field(default_factory=list)
    has_ambiguity: bool = Field(default=False)


class PathNode(BaseModel):
    """A single node along a graph retrieval path."""

    entity_id: str
    canonical_name: str
    entity_type: str
    hop_index: int = Field(default=0, ge=0)


class PathEdge(BaseModel):
    """A directed edge traversed along a graph retrieval path."""

    relationship_id: str
    subject_entity_id: str
    predicate: str
    object_entity_id: str
    direction: Literal["outgoing", "incoming"] = Field(
        description="Direction traversed relative to path sequence"
    )
    evidence_quote: str = Field(description="Verbatim supporting quote from source chunk")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_document_id: str
    source_chunk_id: str
    source_filename: str = Field(default="")


class GraphPath(BaseModel):
    """A validated sequential path through the knowledge graph from a seed entity."""

    path_id: str
    seed_entity_id: str
    seed_entity_name: str
    nodes: list[PathNode] = Field(default_factory=list)
    edges: list[PathEdge] = Field(default_factory=list)
    hop_count: int = Field(default=0, ge=0)
    source_document_ids: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    score: float = Field(default=0.0)
    score_breakdown: dict[str, float] = Field(default_factory=dict)


class GraphRetrievalResult(BaseModel):
    """Ranked retrieval result wrapping a validated graph path with evidence citations."""

    result_id: str
    rank: int = Field(ge=1)
    seed_entity_name: str
    path: GraphPath
    score: float = Field(ge=0.0)
    score_explanation: str = Field(default="")
    source_document_ids: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    source_filenames: list[str] = Field(default_factory=list)
    evidence_quotes: list[str] = Field(default_factory=list)


class GraphContextEvidence(BaseModel):
    """Structured context packaging graph paths for downstream synthesis."""

    triples: list[str] = Field(default_factory=list)
    entity_descriptions: dict[str, str] = Field(default_factory=dict)
    evidence_quotes: list[str] = Field(default_factory=list)
    source_citations: list[dict[str, Any]] = Field(default_factory=list)
    formatted_context: str = Field(default="")


class GraphSearchStatus(str, Enum):
    """Operational status of a graph search query."""

    SUCCESS = "SUCCESS"
    NO_MATCHING_ENTITIES = "NO_MATCHING_ENTITIES"
    NO_GRAPH_RESULTS = "NO_GRAPH_RESULTS"
    AMBIGUOUS_ENTITY = "AMBIGUOUS_ENTITY"
    INVALID_QUERY = "INVALID_QUERY"
    GRAPH_UNAVAILABLE = "GRAPH_UNAVAILABLE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class GraphSearchResponse(BaseModel):
    """Top-level response object returned by the Graph Search Orchestrator."""

    query: str
    status: GraphSearchStatus
    message: str
    extraction_result: QueryEntityExtractionResult | None = None
    linked_entities: list[LinkedEntity] = Field(default_factory=list)
    results: list[GraphRetrievalResult] = Field(default_factory=list)
    structured_context: str = Field(default="")
    context_evidence: GraphContextEvidence | None = None
    statistics: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = Field(default=0.0, ge=0.0)

    def to_export_dict(self) -> dict[str, Any]:
        """Convert response to a serializable dictionary for JSON export."""
        return self.model_dump(mode="json")
