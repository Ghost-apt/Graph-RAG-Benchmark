"""Configuration management for Graph RAG Benchmark.

Loads and validates environment variables using pydantic-settings and python-dotenv.
Ensures sensitive fields are masked when serialized or represented as strings.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Application and infrastructure configuration settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_version: str = Field(
        default="0.9.0",
        description="Application semantic version string",
    )

    # Ollama settings
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for the local Ollama HTTP REST server",
    )
    ollama_llm_model: str = Field(
        default="qwen3:8b",
        description="Ollama model identifier for text generation and QA reasoning",
    )
    ollama_embed_model: str = Field(
        default="nomic-embed-text",
        description="Ollama model identifier for vector embedding generation",
    )
    ollama_timeout: int = Field(
        default=120,
        description="Request timeout in seconds for Ollama API calls",
    )

    # Logging settings
    log_level: LogLevel = Field(
        default="INFO",
        description="Standard logging verbosity level",
    )
    log_file: str = Field(
        default="logs/graphrag.log",
        description="Path to persistent log file on disk",
    )

    # Document Ingestion & Storage Settings (Phase 2)
    data_dir: str = Field(
        default="data",
        description="Root data directory",
    )
    upload_dir: str = Field(
        default="data/uploads",
        description="Directory where original uploaded documents are preserved",
    )
    processed_dir: str = Field(
        default="data/processed",
        description="Directory where processed documents, chunks, and manifests are saved",
    )
    max_upload_size_mb: int = Field(
        default=25,
        description="Maximum allowed upload file size in megabytes",
    )
    default_chunk_size: int = Field(
        default=400,
        description="Default word chunk size for document splitting",
    )
    default_chunk_overlap: int = Field(
        default=60,
        description="Default word chunk overlap for document splitting",
    )

    # Vector Database & Retrieval Settings (Phase 3)
    chroma_path: str = Field(
        default="data/chroma",
        description="Path to persistent ChromaDB directory on disk",
    )
    chroma_collection_name: str = Field(
        default="graphrag_documents",
        description="ChromaDB collection name for indexed document chunk embeddings",
    )
    embedding_batch_size: int = Field(
        default=32,
        description="Maximum batch size when computing embeddings via Ollama",
    )
    retrieval_top_k: int = Field(
        default=5,
        description="Default number of top semantic neighbors to retrieve during search",
    )

    # Knowledge Graph Construction Settings (Phase 4)
    graph_dir: str = Field(
        default="data/knowledge_graph",
        description="Root knowledge graph persistence directory",
    )
    graph_path: str = Field(
        default="data/knowledge_graph/knowledge_graph.json",
        description="Canonical JSON knowledge graph file path",
    )
    graph_export_path: str = Field(
        default="data/knowledge_graph/knowledge_graph.graphml",
        description="Sanitized GraphML export file path",
    )
    graph_extractions_path: str = Field(
        default="data/knowledge_graph/extractions.jsonl",
        description="Chunk extraction history JSONL file path",
    )
    graph_manifest_path: str = Field(
        default="data/knowledge_graph/manifest.json",
        description="Knowledge graph construction manifest file path",
    )
    graph_extraction_model: str = Field(
        default="qwen3:8b",
        description="LLM model used for entity and relationship extraction",
    )
    graph_extraction_batch_size: int = Field(
        default=1,
        description="Batch size for chunk extraction processing",
    )
    graph_min_confidence: float = Field(
        default=0.50,
        description="Minimum confidence threshold for relationship extraction",
    )
    graph_auto_rebuild: bool = Field(
        default=False,
        description="Whether to automatically synchronize knowledge graph on repository changes",
    )
    graph_extraction_timeout: int = Field(
        default=600,
        ge=30,
        description="Request timeout in seconds for LLM knowledge graph extraction",
    )

    # Graph Retrieval & Multi-Hop Reasoning Settings (Phase 5)
    graph_max_hops: int = Field(
        default=2,
        ge=1,
        le=3,
        description="Maximum traversal hops for graph retrieval (1 to 3)",
    )
    graph_max_results: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of retrieved graph results",
    )
    graph_max_paths: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Maximum number of candidate graph traversal paths",
    )
    graph_traversal_direction: Literal["both", "out", "in"] = Field(
        default="both",
        description="Default edge traversal direction ('both', 'out', 'in')",
    )
    graph_max_degree: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum node degree threshold to prevent unbounded expansion on high-degree hubs",
    )

    # Hybrid Retrieval & Reciprocal Rank Fusion Settings (Phase 6)
    hybrid_top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Final number of hybrid fused results to return",
    )
    hybrid_vector_top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of vector candidates to retrieve before fusion",
    )
    hybrid_graph_top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of graph candidate paths to retrieve before fusion",
    )
    hybrid_rrf_k: int = Field(
        default=60,
        ge=1,
        le=1000,
        description="Reciprocal Rank Fusion smoothing constant (default: 60)",
    )
    hybrid_vector_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight multiplier for vector retrieval in weighted RRF",
    )
    hybrid_graph_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight multiplier for graph retrieval in weighted RRF",
    )
    hybrid_min_rrf_score: float = Field(
        default=0.0,
        ge=0.0,
        description="Minimum RRF score threshold for inclusion in final results",
    )
    hybrid_parallel_retrieval: bool = Field(
        default=False,
        description="Whether to execute vector and graph retrieval in parallel",
    )

    # Grounded Answer Generation & QA Settings (Phase 7)
    rag_generation_model: str = Field(
        default="qwen3:8b",
        description="Ollama LLM model identifier for grounded answer generation",
    )
    rag_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for grounded generation (low values enhance factuality)",
    )
    rag_top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of retrieved evidence candidates to pass to generation prompt",
    )
    rag_max_context_tokens: int = Field(
        default=3000,
        ge=500,
        le=16000,
        description="Strict context window token budget for retrieved evidence",
    )
    rag_max_generation_tokens: int = Field(
        default=1024,
        ge=64,
        le=4096,
        description="Maximum token length for generated answers",
    )
    rag_generation_timeout: int = Field(
        default=180,
        ge=15,
        description="Timeout in seconds for Ollama answer generation requests",
    )
    rag_min_evidence_items: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Minimum evidence items required before triggering generation",
    )
    rag_min_relevance_score: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Minimum retrieval relevance score required to proceed to generation (queries with lower top relevance short-circuit to INSUFFICIENT_EVIDENCE)",
    )
    rag_require_citations: bool = Field(
        default=True,
        description="Enforce citation markers [S1], [S2] in generated responses",
    )
    rag_enable_faithfulness_check: bool = Field(
        default=True,
        description="Enable automated claim-level faithfulness verification against retrieved context",
    )

    # Application settings
    app_env: str = Field(
        default="development",
        description="Runtime environment identifier (e.g., development, testing, production)",
    )

    # Phase 9: Production Readiness, Deployment & Monitoring
    app_host: str = Field(
        default="127.0.0.1",
        description="Network host interface to bind application services",
    )
    app_port: int = Field(
        default=8501,
        ge=1,
        le=65535,
        description="TCP port to bind application services",
    )
    benchmark_dir: str = Field(
        default="data/phase8_benchmark",
        description="Root directory for Phase 8 benchmark corpus and results",
    )
    backup_dir: str = Field(
        default="data/backups",
        description="Root directory for persistent backup archives and manifests",
    )
    log_format: Literal["text", "json"] = Field(
        default="text",
        description="Structured logging output format ('text' or 'json')",
    )
    log_max_bytes: int = Field(
        default=10485760,
        ge=1024,
        description="Maximum log file size in bytes before rotating (default: 10 MB)",
    )
    log_backup_count: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of rotated backup log files to retain on disk",
    )
    max_concurrent_generations: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Maximum concurrent in-flight LLM generation jobs",
    )
    max_concurrent_benchmarks: int = Field(
        default=1,
        ge=1,
        le=2,
        description="Maximum concurrent benchmark suite execution jobs",
    )
    request_timeout: int = Field(
        default=60,
        ge=5,
        le=600,
        description="Default HTTP request timeout in seconds",
    )
    auth_mode: Literal["none", "basic", "header", "dev", "oidc"] = Field(
        default="none",
        description="Deployment authentication mode ('none', 'basic', 'header', 'dev', 'oidc')",
    )
    enable_telemetry: bool = Field(
        default=True,
        description="Whether to track in-memory latency and error telemetry",
    )

    # Phase 10: Real-User Pilot Deployment, Identity Management & RBAC
    auth_redirect_uri: str = Field(
        default="http://localhost:8501/oauth2callback",
        description="OIDC redirect callback URI (must match identity provider registration)",
    )
    oidc_issuer: str | None = Field(
        default=None,
        description="Expected OpenID Connect token issuer URL for token validation",
    )
    admin_bootstrap_email: str | None = Field(
        default=None,
        description="Verified identity provider email automatically assigned ADMIN role upon initial login",
    )
    user_registry_path: str = Field(
        default="data/users/users.json",
        description="Path to persistent JSON user registry",
    )
    user_audit_log_path: str = Field(
        default="logs/user_audit.jsonl",
        description="Path to user audit log stream",
    )
    session_timeout_minutes: int = Field(
        default=120,
        ge=5,
        le=1440,
        description="User session inactivity timeout in minutes",
    )
    require_email_verified: bool = Field(
        default=True,
        description="Whether OIDC authentication requires email_verified claim to be True",
    )

    @field_validator("auth_mode", mode="before")
    @classmethod
    def normalize_auth_mode(cls, v: Any) -> Any:
        """Strip and lowercase auth_mode string."""
        if isinstance(v, str):
            return v.strip().lower()
        return v

    @field_validator("ollama_base_url")
    @classmethod
    def validate_ollama_base_url(cls, v: str) -> str:
        """Validate and normalize Ollama base URL."""
        v = v.strip()
        if not v:
            raise ValueError("OLLAMA_BASE_URL cannot be empty.")
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"OLLAMA_BASE_URL must start with http:// or https://, got: {v}")
        return v.rstrip("/")

    @field_validator(
        "ollama_timeout",
        "embedding_batch_size",
        "retrieval_top_k",
        "graph_extraction_batch_size",
        "graph_max_results",
        "graph_max_paths",
        "graph_max_degree",
        "rag_top_k",
        "rag_max_context_tokens",
        "rag_max_generation_tokens",
        "rag_generation_timeout",
        "rag_min_evidence_items",
    )
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        """Validate value is a positive integer > 0."""
        if v <= 0:
            raise ValueError(f"Value must be greater than 0, got: {v}")
        return v

    @field_validator("graph_max_hops")
    @classmethod
    def validate_max_hops(cls, v: int) -> int:
        """Validate max hops is within bounded range [1, 3]."""
        if not (1 <= v <= 3):
            raise ValueError(f"graph_max_hops must be between 1 and 3, got: {v}")
        return v

    @field_validator("graph_traversal_direction", mode="before")
    @classmethod
    def validate_traversal_direction(cls, v: Any) -> str:
        """Validate and normalize traversal direction."""
        if isinstance(v, str):
            v_clean = v.strip().lower()
            if v_clean not in {"both", "out", "in"}:
                raise ValueError(
                    f"graph_traversal_direction must be one of 'both', 'out', 'in', got: {v}"
                )
            return v_clean
        return v

    @field_validator("graph_min_confidence")
    @classmethod
    def validate_confidence_range(cls, v: float) -> float:
        """Validate confidence score is within [0.0, 1.0]."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"graph_min_confidence must be between 0.0 and 1.0, got: {v}")
        return v

    @field_validator(
        "data_dir",
        "upload_dir",
        "processed_dir",
        "chroma_path",
        "graph_dir",
        "benchmark_dir",
        "backup_dir",
        "user_registry_path",
        "user_audit_log_path",
    )
    @classmethod
    def validate_safe_path(cls, v: str) -> str:
        """Validate path does not contain dangerous path traversal sequences."""
        if not v or not v.strip():
            raise ValueError("Path setting cannot be empty.")
        v_clean = v.strip()
        parts = Path(v_clean).parts
        if ".." in parts:
            raise ValueError(f"Path setting contains forbidden traversal sequence '..': {v}")
        return v_clean

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, v: Any) -> str:
        """Convert log level string to uppercase."""
        if isinstance(v, str):
            v_upper = v.strip().upper()
            valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
            if v_upper not in valid_levels:
                raise ValueError(
                    f"Invalid LOG_LEVEL '{v}'. Must be one of: {', '.join(sorted(valid_levels))}"
                )
            return v_upper
        return v

    @model_validator(mode="after")
    def validate_hybrid_weights(self) -> "Settings":
        """Validate that both modality weights are not simultaneously zero."""
        if self.hybrid_vector_weight <= 0.0 and self.hybrid_graph_weight <= 0.0:
            raise ValueError(
                "Both hybrid_vector_weight and hybrid_graph_weight cannot be zero. "
                "At least one modality must have a positive weight."
            )
        return self

    def to_safe_dict(self) -> dict[str, Any]:
        """Return configuration dictionary with sensitive fields safely sanitized.

        Guarantees that credentials, tokens, or keys are never exposed in logs or dumps.
        """
        data = self.model_dump()
        sensitive_keywords = {"key", "secret", "token", "password", "auth", "credential"}
        for k in data:
            if any(kw in k.lower() for kw in sensitive_keywords):
                data[k] = "******"
        return data

    def __repr__(self) -> str:
        """Return safe string representation concealing secrets."""
        safe_items = [f"{k}={v!r}" for k, v in self.to_safe_dict().items()]
        return f"Settings({', '.join(safe_items)})"

    def __str__(self) -> str:
        return repr(self)


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton instance of validated application settings."""
    return Settings()


def reload_settings() -> Settings:
    """Clear cache and reload settings from environment and .env."""
    get_settings.cache_clear()
    return get_settings()
