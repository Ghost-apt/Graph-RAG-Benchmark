"""Pydantic schemas and data models for Phase 6 Hybrid Retrieval & Reciprocal Rank Fusion."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class RetrievalSource(str, Enum):
    """Origin of a retrieved evidence candidate."""

    VECTOR = "VECTOR"
    GRAPH = "GRAPH"
    VECTOR_AND_GRAPH = "VECTOR_AND_GRAPH"


class HybridSearchStatus(str, Enum):
    """Operational status of a hybrid search query."""

    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    NO_RESULTS = "NO_RESULTS"
    INVALID_QUERY = "INVALID_QUERY"
    ALL_ENGINES_FAILED = "ALL_ENGINES_FAILED"


class VectorCandidate(BaseModel):
    """Normalized candidate chunk retrieved by Vector Semantic Search."""

    chunk_id: str = Field(description="Canonical Phase 2 chunk identifier")
    document_id: str = Field(description="Canonical Phase 2 document identifier")
    source_filename: str = Field(description="Original source filename")
    file_type: str = Field(default="txt", description="Document file extension")
    chunk_text: str = Field(description="Verbatim canonical text content of the chunk")
    vector_rank: int = Field(ge=1, description="1-based ordinal rank in vector search results")
    cosine_distance: float = Field(description="ChromaDB cosine distance metric in [0.0, 2.0]")
    cosine_similarity: float = Field(description="Raw cosine similarity: 1.0 - cosine_distance")
    normalized_score: float = Field(
        description="Normalized cosine score: 1.0 - (cosine_distance / 2.0) in [0.0, 1.0]"
    )
    source_page: int | None = Field(default=None, description="Page number where chunk starts")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Diagnostic vector metadata")

    @field_validator("chunk_id", "document_id", "chunk_text")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Field cannot be empty or whitespace-only.")
        return s


class GraphCandidate(BaseModel):
    """Normalized candidate chunk supported by one or more Knowledge Graph paths."""

    chunk_id: str = Field(description="Canonical Phase 2 chunk identifier mapped via repository")
    document_id: str = Field(description="Canonical Phase 2 document identifier")
    source_filename: str = Field(description="Original source filename")
    chunk_text: str = Field(description="Verbatim canonical text content of the chunk")
    graph_rank: int = Field(ge=1, description="1-based ordinal rank based on highest-scoring path")
    matched_entities: list[str] = Field(
        default_factory=list, description="Entities participating in supporting paths"
    )
    supporting_paths: list[dict[str, Any]] = Field(
        default_factory=list, description="Serialized GraphPath objects supporting this chunk"
    )
    relationship_ids: list[str] = Field(
        default_factory=list, description="Unique relationship IDs in supporting paths"
    )
    hop_count: int = Field(default=1, ge=1, description="Minimum hop count among supporting paths")
    evidence_quotes: list[str] = Field(
        default_factory=list, description="Deduplicated verbatim relationship quotations"
    )
    graph_score: float = Field(
        default=0.0, description="Highest composite score among supporting paths"
    )

    @model_validator(mode="before")
    @classmethod
    def preprocess_graph_candidate_aliases(cls, data: Any) -> Any:
        """Allow convenience alias field names."""
        if isinstance(data, dict):
            d = dict(data)
            if "best_path_score" in d and "graph_score" not in d:
                d["graph_score"] = d["best_path_score"]
            return d
        return data

    @property
    def best_path_score(self) -> float:
        """Convenience alias for graph_score."""
        return self.graph_score

    @field_validator("chunk_id", "document_id", "chunk_text")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Field cannot be empty or whitespace-only.")
        return s


class FusionCandidate(BaseModel):
    """Consolidated chunk candidate resulting from Reciprocal Rank Fusion."""

    chunk_id: str = Field(description="Canonical Phase 2 chunk identifier")
    document_id: str = Field(description="Canonical Phase 2 document identifier")
    source_filename: str = Field(description="Original source filename")
    chunk_text: str = Field(description="Verbatim canonical chunk text")
    vector_rank: int | None = Field(
        default=None, ge=1, description="1-based rank in vector results, if retrieved"
    )
    graph_rank: int | None = Field(
        default=None, ge=1, description="1-based rank in graph results, if retrieved"
    )
    vector_contribution: float = Field(
        default=0.0, ge=0.0, description="Weighted reciprocal rank component from vector search"
    )
    graph_contribution: float = Field(
        default=0.0, ge=0.0, description="Weighted reciprocal rank component from graph search"
    )
    rrf_score: float = Field(
        default=0.0, ge=0.0, description="Composite RRF score: vector_contrib + graph_contrib"
    )
    fused_rank: int = Field(ge=1, description="1-based final fused rank in hybrid results")
    retrieval_sources: RetrievalSource = Field(description="Modality attribution of this candidate")
    supporting_graph_paths: list[dict[str, Any]] = Field(
        default_factory=list, description="Associated graph reasoning paths if retrieved via graph"
    )
    evidence_quotes: list[str] = Field(
        default_factory=list, description="Verbatim relationship evidence quotes"
    )
    source_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Metadata including page number, cosine scores, etc."
    )

    @model_validator(mode="before")
    @classmethod
    def preprocess_fusion_candidate_aliases(cls, data: Any) -> Any:
        """Allow convenience alias field names for compatibility."""
        if isinstance(data, dict):
            d = dict(data)
            if "source" in d and "retrieval_sources" not in d:
                d["retrieval_sources"] = d["source"]
            if "final_rank" in d and "fused_rank" not in d:
                d["fused_rank"] = d["final_rank"]
            if "vector_rrf_contribution" in d and "vector_contribution" not in d:
                d["vector_contribution"] = d["vector_rrf_contribution"]
            if "graph_rrf_contribution" in d and "graph_contribution" not in d:
                d["graph_contribution"] = d["graph_rrf_contribution"]
            if "supporting_paths" in d and "supporting_graph_paths" not in d:
                d["supporting_graph_paths"] = d["supporting_paths"]
            # Map extra metrics into source_metadata if provided at top-level
            meta = dict(d.get("source_metadata", {}))
            for k in (
                "cosine_distance",
                "cosine_similarity",
                "normalized_score",
                "source_page",
                "hop_count",
                "min_hops",
                "matched_entities",
            ):
                if k in d and k not in meta:
                    meta[k] = d[k]
            d["source_metadata"] = meta
            return d
        return data

    @property
    def final_rank(self) -> int:
        """Convenience alias for fused_rank."""
        return self.fused_rank

    @property
    def source(self) -> RetrievalSource:
        """Convenience alias for retrieval_sources."""
        return self.retrieval_sources

    @property
    def vector_rrf_contribution(self) -> float:
        """Convenience alias for vector_contribution."""
        return self.vector_contribution

    @property
    def graph_rrf_contribution(self) -> float:
        """Convenience alias for graph_contribution."""
        return self.graph_contribution

    @property
    def supporting_paths(self) -> list[dict[str, Any]]:
        """Convenience alias for supporting_graph_paths."""
        return self.supporting_graph_paths

    @property
    def cosine_distance(self) -> float | None:
        """Retrieve cosine distance from metadata if present."""
        return self.source_metadata.get("cosine_distance")

    @property
    def cosine_similarity(self) -> float | None:
        """Retrieve cosine similarity from metadata if present."""
        return self.source_metadata.get("cosine_similarity")

    @property
    def source_page(self) -> int | None:
        """Retrieve source page from metadata if present."""
        return self.source_metadata.get("source_page")

    @property
    def min_hops(self) -> int | None:
        """Retrieve min hops from metadata if present."""
        return self.source_metadata.get("hop_count") or self.source_metadata.get("min_hops")

    @property
    def matched_entities(self) -> list[str]:
        """Retrieve matched entities from metadata if present."""
        return self.source_metadata.get("matched_entities", [])


class HybridSearchResponse(BaseModel):
    """Complete container for hybrid retrieval query execution output."""

    query: str = Field(description="Search query string")
    status: HybridSearchStatus = Field(description="Overall hybrid retrieval operational status")
    vector_search_status: str = Field(
        default="SUCCESS", description="Status of vector search modality"
    )
    graph_search_status: str = Field(
        default="SUCCESS", description="Status of graph search modality"
    )
    vector_results_count: int = Field(default=0, ge=0)
    graph_results_count: int = Field(default=0, ge=0)
    fused_results_count: int = Field(default=0, ge=0)
    results: list[FusionCandidate] = Field(
        default_factory=list, description="Ranked fused chunk candidates"
    )
    fusion_method: str = Field(default="Reciprocal Rank Fusion (RRF)")
    rrf_constant: int = Field(default=60, ge=1)
    vector_weight: float = Field(default=1.0, ge=0.0)
    graph_weight: float = Field(default=1.0, ge=0.0)
    vector_latency_ms: float = Field(default=0.0, ge=0.0)
    graph_latency_ms: float = Field(default=0.0, ge=0.0)
    query_entity_extraction_ms: float = Field(default=0.0, ge=0.0)
    graph_traversal_ms: float = Field(default=0.0, ge=0.0)
    llm_extraction_invoked: bool = Field(default=False)
    fusion_latency_ms: float = Field(default=0.0, ge=0.0)
    total_latency_ms: float = Field(default=0.0, ge=0.0)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    structured_context: str = Field(
        default="", description="Unified formatted context for downstream RAG consumption"
    )

    @property
    def candidates(self) -> list[FusionCandidate]:
        """Convenience alias for results list."""
        return self.results

    def to_export_dict(self) -> dict[str, Any]:
        """Convert response to a serializable dictionary for JSON export."""
        return self.model_dump(mode="json")

    @model_validator(mode="before")
    @classmethod
    def preprocess_response_aliases(cls, data: Any) -> Any:
        """Allow convenience alias field names for compatibility."""
        if isinstance(data, dict):
            d = dict(data)
            if "vector_status" in d and "vector_search_status" not in d:
                d["vector_search_status"] = d["vector_status"]
            if "graph_status" in d and "graph_search_status" not in d:
                d["graph_search_status"] = d["graph_status"]
            if "vector_candidates_count" in d and "vector_results_count" not in d:
                d["vector_results_count"] = d["vector_candidates_count"]
            if "graph_candidates_count" in d and "graph_results_count" not in d:
                d["graph_results_count"] = d["graph_candidates_count"]
            if "total_candidates" in d and "fused_results_count" not in d:
                d["fused_results_count"] = d["total_candidates"]
            if "fused_candidates" in d and "results" not in d:
                d["results"] = d["fused_candidates"]
            if "fusion_params" in d:
                fp = d["fusion_params"]
                if "rrf_k" in fp and "rrf_constant" not in d:
                    d["rrf_constant"] = fp["rrf_k"]
                if "vector_weight" in fp and "vector_weight" not in d:
                    d["vector_weight"] = fp["vector_weight"]
                if "graph_weight" in fp and "graph_weight" not in d:
                    d["graph_weight"] = fp["graph_weight"]
            return d
        return data

    @property
    def fused_candidates(self) -> list[FusionCandidate]:
        """Convenience alias for results."""
        return self.results

    @property
    def vector_status(self) -> str:
        """Convenience alias for vector_search_status."""
        return self.vector_search_status

    @property
    def graph_status(self) -> str:
        """Convenience alias for graph_search_status."""
        return self.graph_search_status

    @property
    def vector_candidates_count(self) -> int:
        """Convenience alias for vector_results_count."""
        return self.vector_results_count

    @property
    def graph_candidates_count(self) -> int:
        """Convenience alias for graph_results_count."""
        return self.graph_results_count

    @property
    def total_candidates(self) -> int:
        """Convenience alias for fused_results_count."""
        return self.fused_results_count

    @property
    def fusion_params(self) -> dict[str, Any]:
        """Convenience dictionary representation of fusion hyperparameters."""
        return {
            "rrf_k": self.rrf_constant,
            "vector_weight": self.vector_weight,
            "graph_weight": self.graph_weight,
            "fusion_method": self.fusion_method,
        }
