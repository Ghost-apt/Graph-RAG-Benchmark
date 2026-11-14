"""Data schemas for Knowledge Graph Construction Engine.

Defines Pydantic models for:
- ExtractedEntity: Raw entity mention extracted from a chunk
- ExtractedRelationship: Raw relationship triple with evidence quote
- ChunkGraphExtraction: Container of extracted entities and relationships for a chunk
- GraphEntity: Canonical resolved knowledge graph node with aggregated provenance
- GraphRelationship: Directed knowledge graph edge with chunk provenance and evidence
"""

import re
import unicodedata
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

EntityType = Literal[
    "PERSON",
    "ORGANIZATION",
    "CONCEPT",
    "TECHNOLOGY",
    "PROGRAMMING_LANGUAGE",
    "FRAMEWORK",
    "SOFTWARE",
    "DATASET",
    "METHOD",
    "LOCATION",
    "OTHER",
]

ALLOWED_ENTITY_TYPES: set[str] = {
    "PERSON",
    "ORGANIZATION",
    "CONCEPT",
    "TECHNOLOGY",
    "PROGRAMMING_LANGUAGE",
    "FRAMEWORK",
    "SOFTWARE",
    "DATASET",
    "METHOD",
    "LOCATION",
    "OTHER",
}

# Mapping common synonym or granular categories to standard taxonomy
ENTITY_TYPE_SYNONYMS: dict[str, str] = {
    "ALGORITHM": "METHOD",
    "APPROACH": "METHOD",
    "TECHNIQUE": "METHOD",
    "MODEL": "TECHNOLOGY",
    "TOOL": "TECHNOLOGY",
    "SYSTEM": "TECHNOLOGY",
    "ARCHITECTURE": "CONCEPT",
    "PARADIGM": "CONCEPT",
    "METRIC": "CONCEPT",
    "BENCHMARK": "DATASET",
    "CORPUS": "DATASET",
    "LIBRARY": "FRAMEWORK",
    "PACKAGE": "FRAMEWORK",
    "LANGUAGE": "PROGRAMMING_LANGUAGE",
    "DATABASE": "SOFTWARE",
    "COMPANY": "ORGANIZATION",
    "INSTITUTION": "ORGANIZATION",
    "UNIVERSITY": "ORGANIZATION",
    "AUTHOR": "PERSON",
    "RESEARCHER": "PERSON",
    "CITY": "LOCATION",
    "COUNTRY": "LOCATION",
    "PLACE": "LOCATION",
}


def normalize_predicate(predicate: str) -> str:
    """Normalize relationship predicate to uppercase snake_case."""
    pred = predicate.strip().upper()
    pred = re.sub(r"[^\w\s-]", "", pred)
    pred = re.sub(r"[\s-]+", "_", pred)
    return pred or "RELATED_TO"


def normalize_entity_type(raw_type: str) -> str:
    """Normalize entity type string to standard EntityType taxonomy."""
    clean = raw_type.strip().upper()
    clean = re.sub(r"[\s-]+", "_", clean)
    if clean in ALLOWED_ENTITY_TYPES:
        return clean
    if clean in ENTITY_TYPE_SYNONYMS:
        return ENTITY_TYPE_SYNONYMS[clean]
    return "OTHER"


class ExtractedEntity(BaseModel):
    """Raw entity extracted by LLM from a document chunk."""

    mention_text: str = Field(description="Exact textual mention in the source chunk")
    canonical_name: str = Field(description="Standardized name of the entity")
    entity_type: str = Field(default="CONCEPT", description="Entity category taxonomy")
    description: str = Field(default="", description="Source-grounded description")
    aliases: list[str] = Field(default_factory=list, description="Alternative names or aliases")

    @field_validator("mention_text", "canonical_name")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Entity mention_text and canonical_name cannot be empty.")
        return unicodedata.normalize("NFKC", s)

    @field_validator("entity_type")
    @classmethod
    def normalize_type(cls, v: str) -> str:
        return normalize_entity_type(v)

    @field_validator("aliases")
    @classmethod
    def clean_aliases(cls, v: list[str]) -> list[str]:
        cleaned = []
        for a in v:
            if isinstance(a, str) and a.strip():
                cleaned.append(unicodedata.normalize("NFKC", a.strip()))
        return list(dict.fromkeys(cleaned))


class ExtractedRelationship(BaseModel):
    """Raw directional relationship extracted from a document chunk."""

    subject: str = Field(description="Mention or canonical name of the subject entity")
    predicate: str = Field(description="Relationship label connecting subject to object")
    object: str = Field(description="Mention or canonical name of the object entity")
    evidence_quote: str = Field(description="Exact verbatim supporting quote from the chunk text")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Model extraction confidence score between 0.0 and 1.0",
    )

    @field_validator("subject", "object", "evidence_quote")
    @classmethod
    def validate_non_empty_strings(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Relationship subject, object, and evidence_quote cannot be empty.")
        return unicodedata.normalize("NFKC", s)

    @field_validator("predicate")
    @classmethod
    def clean_predicate(cls, v: str) -> str:
        return normalize_predicate(v)


class ChunkGraphExtraction(BaseModel):
    """Structured extraction outcome for a single document chunk."""

    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)
    source_chunk_id: str | None = Field(default=None, description="Identifier of the origin chunk")
    raw_response: str | None = Field(
        default=None, description="Optional raw text response for debugging"
    )
    extraction_duration_ms: float = Field(default=0.0, ge=0.0)


class GraphEntity(BaseModel):
    """Resolved and consolidated knowledge graph node."""

    entity_id: str = Field(description="Deterministic unique entity identifier")
    canonical_name: str = Field(description="Primary canonical entity name")
    entity_type: str = Field(description="Validated entity category")
    description: str = Field(default="", description="Aggregated entity description")
    aliases: list[str] = Field(default_factory=list, description="Unique aliases across mentions")
    mention_count: int = Field(default=1, ge=1, description="Total mentions across all chunks")
    source_document_ids: list[str] = Field(
        default_factory=list, description="IDs of documents supporting this entity"
    )
    source_chunk_ids: list[str] = Field(
        default_factory=list, description="IDs of chunks supporting this entity"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("entity_id", "canonical_name", "entity_type")
    @classmethod
    def validate_node_strings(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("GraphEntity fields cannot be empty.")
        return s


class GraphRelationship(BaseModel):
    """Directed knowledge graph edge representing an asserted relationship."""

    relationship_id: str = Field(description="Deterministic unique edge identifier")
    subject_entity_id: str = Field(description="Source entity node ID")
    predicate: str = Field(description="Normalized relationship predicate")
    object_entity_id: str = Field(description="Target entity node ID")
    evidence_quote: str = Field(description="Verbatim supporting quote from source chunk")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_document_id: str = Field(description="Origin document identifier")
    source_chunk_id: str = Field(description="Origin chunk identifier")
    source_filename: str = Field(default="", description="Origin document filename")
    extraction_model: str = Field(default="qwen3:8b", description="Model used for extraction")

    @field_validator(
        "relationship_id",
        "subject_entity_id",
        "predicate",
        "object_entity_id",
        "source_document_id",
        "source_chunk_id",
    )
    @classmethod
    def validate_edge_strings(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("GraphRelationship identifiers and predicate cannot be empty.")
        return s
