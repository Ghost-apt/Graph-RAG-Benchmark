"""Graph RAG Benchmark - Core Package.

Phase 1: Environment Setup & Project Foundation.
Phase 2: Document Ingestion & Preprocessing Engine.
Phase 3: Vector Embeddings & ChromaDB Vector Search.
Phase 4: Knowledge Graph Construction Engine.
Phase 5: Graph Retrieval & Multi-Hop Reasoning.
Phase 6: Hybrid Retrieval & Reciprocal Rank Fusion.
Phase 7: LLM Answer Generation & Grounded Question Answering.
Phase 8: Formal Comparative Benchmark Evaluation.
Phase 9: Production Readiness, Operational Hardening & Disaster Recovery.
Phase 10: Real-User Pilot Deployment, Identity Management & Product Launch.
"""

from graphrag_bench.access_control import (
    ROLE_PERMISSIONS,
    Permission,
    PermissionDeniedError,
    Role,
    check_permission,
    has_permission,
    require_permission,
)
from graphrag_bench.answer_generator import AnswerGenerator, strip_think_tags
from graphrag_bench.answer_guardrails import (
    STANDARD_INSUFFICIENT_EVIDENCE_MESSAGE,
    AnswerGuardrails,
    PostCheckResult,
    PreCheckResult,
)
from graphrag_bench.answer_schemas import (
    AnswerCitation,
    AnswerClaim,
    AnswerGenerationRequest,
    AnswerStatus,
    EvidenceItem,
    GroundedAnswer,
)
from graphrag_bench.atomic_io import atomic_write_json, atomic_write_text
from graphrag_bench.auth_manager import AuthManager
from graphrag_bench.backup_manager import (
    BackupCategory,
    BackupManager,
    BackupManifest,
    BackupResult,
    FileChecksum,
    RestorePreview,
    RestoreResult,
    compute_sha256,
)
from graphrag_bench.benchmark_metrics import (
    aggregate_strategy_metrics,
    calculate_dcg_at_k,
    calculate_hit_rate_at_k,
    calculate_map_at_k,
    calculate_ndcg_at_k,
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_reciprocal_rank,
    evaluate_query_retrieval,
)
from graphrag_bench.benchmark_runner import BenchmarkRunner
from graphrag_bench.benchmark_schemas import (
    AnswerQualityMetrics,
    BenchmarkChunk,
    BenchmarkDocument,
    BenchmarkQuery,
    BenchmarkQueryCategory,
    BenchmarkQueryResult,
    BenchmarkRunManifest,
    BenchmarkSplit,
    LatencyTelemetry,
    RelevanceJudgment,
    RetrievalMetrics,
    RetrievalRankedItem,
    RetrievalStrategy,
    StrategyAggregateSummary,
)
from graphrag_bench.benchmark_statistics import (
    bootstrap_confidence_interval,
    calculate_paired_wilcoxon,
    compare_strategies_paired,
    compute_latency_summary,
)
from graphrag_bench.benchmark_storage import BenchmarkStorage
from graphrag_bench.chunking import TextChunker
from graphrag_bench.citation_validator import (
    CitationValidationResult,
    CitationValidator,
)
from graphrag_bench.config import Settings, get_settings
from graphrag_bench.document_loader import (
    DocumentExtractionError,
    DocumentLoader,
    ExtractedDocument,
    ExtractedPage,
)
from graphrag_bench.document_processor import DocumentProcessor
from graphrag_bench.document_repository import DocumentRepository
from graphrag_bench.document_validator import (
    DocumentValidator,
    ValidationError,
    ValidationResult,
    sanitize_filename,
)
from graphrag_bench.embedding_service import (
    DimensionMismatchError,
    EmbeddingError,
    EmbeddingResult,
    EmbeddingService,
    EmptyTextError,
    InvalidVectorError,
)
from graphrag_bench.entity_linker import (
    EntityLinker,
    EntityLinkingResult,
    LinkedEntity,
    MatchType,
)
from graphrag_bench.entity_resolver import (
    EntityResolver,
    generate_entity_id,
    normalize_text,
)
from graphrag_bench.evidence_preparer import EvidencePreparer, estimate_tokens
from graphrag_bench.faithfulness_checker import FaithfulnessChecker
from graphrag_bench.graph_builder import GraphBuilder
from graphrag_bench.graph_context_builder import (
    GraphContextBuilder,
    GraphContextEvidence,
)
from graphrag_bench.graph_extraction_service import (
    ExtractionError,
    GraphExtractionService,
)
from graphrag_bench.graph_ranker import GraphRanker
from graphrag_bench.graph_retrieval_schemas import (
    ExtractionMethod,
    GraphPath,
    GraphRetrievalResult,
    GraphSearchResponse,
    GraphSearchStatus,
    PathEdge,
    PathNode,
    QueryEntity,
    QueryEntityExtractionResult,
)
from graphrag_bench.graph_retriever import GraphRetriever
from graphrag_bench.graph_schemas import (
    ChunkGraphExtraction,
    ExtractedEntity,
    ExtractedRelationship,
    GraphEntity,
    GraphRelationship,
    normalize_entity_type,
    normalize_predicate,
)
from graphrag_bench.graph_search import (
    GraphSearchEngine,
    get_graph_search_engine,
    search_graph,
)
from graphrag_bench.graph_store import GraphStore
from graphrag_bench.graph_traversal import (
    GraphTraversalEngine,
)
from graphrag_bench.health import HealthChecker, HealthReport, HealthStatus
from graphrag_bench.hybrid_context_builder import HybridContextBuilder
from graphrag_bench.hybrid_retriever import (
    HybridRetriever,
    get_hybrid_retriever,
    search_hybrid,
)
from graphrag_bench.hybrid_schemas import (
    FusionCandidate,
    GraphCandidate,
    HybridSearchResponse,
    HybridSearchStatus,
    RetrievalSource,
    VectorCandidate,
)
from graphrag_bench.indexing_pipeline import IndexingError, IndexingPipeline, IndexingStats
from graphrag_bench.logger import get_logger, setup_logger
from graphrag_bench.observability import (
    ConcurrencyLimiter,
    JsonLogFormatter,
    TelemetryCollector,
    TelemetryEvent,
    configure_observability,
    trace_operation,
)
from graphrag_bench.ollama_client import (
    GenerateResponse,
    OllamaAPIError,
    OllamaClient,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
)
from graphrag_bench.production_health import (
    HealthCheckItem,
    ProductionHealthMonitor,
    ProductionHealthReport,
    ProductionHealthState,
)
from graphrag_bench.prompt_builder import PromptBuilder
from graphrag_bench.query_entity_extractor import QueryEntityExtractor
from graphrag_bench.rag_pipeline import RAGPipeline, answer_question
from graphrag_bench.reciprocal_rank_fusion import ReciprocalRankFusion
from graphrag_bench.result_normalizer import ResultNormalizer
from graphrag_bench.schemas import DocumentChunk, DocumentIngestionResult, DocumentMetadata
from graphrag_bench.semantic_search import SearchError, SearchResponse, SemanticSearch
from graphrag_bench.text_cleaner import TextCleaner
from graphrag_bench.user_audit import (
    UserAuditEvent,
    UserAuditLogger,
    sanitize_audit_details,
)
from graphrag_bench.user_registry import (
    UserAccount,
    UserRegistry,
    UserStatus,
    generate_user_id,
)
from graphrag_bench.vector_store import (
    SearchResultItem,
    VectorRecord,
    VectorStore,
    VectorStoreError,
)

__version__ = "0.10.0"
__all__ = [
    "ROLE_PERMISSIONS",
    "STANDARD_INSUFFICIENT_EVIDENCE_MESSAGE",
    "AnswerCitation",
    "AnswerClaim",
    "AnswerGenerationRequest",
    "AnswerGenerator",
    "AnswerGuardrails",
    "AnswerQualityMetrics",
    "AnswerStatus",
    "AuthManager",
    "BackupCategory",
    "BackupManager",
    "BackupManifest",
    "BackupResult",
    "BenchmarkChunk",
    "BenchmarkDocument",
    "BenchmarkQuery",
    "BenchmarkQueryCategory",
    "BenchmarkQueryResult",
    "BenchmarkRunManifest",
    "BenchmarkRunner",
    "BenchmarkSplit",
    "BenchmarkStorage",
    "ChunkGraphExtraction",
    "CitationValidationResult",
    "CitationValidator",
    "ConcurrencyLimiter",
    "DimensionMismatchError",
    "DocumentChunk",
    "DocumentExtractionError",
    "DocumentIngestionResult",
    "DocumentLoader",
    "DocumentMetadata",
    "DocumentProcessor",
    "DocumentRepository",
    "DocumentValidator",
    "EmbeddingError",
    "EmbeddingResult",
    "EmbeddingService",
    "EmptyTextError",
    "EntityLinker",
    "EntityLinkingResult",
    "EntityResolver",
    "EvidenceItem",
    "EvidencePreparer",
    "ExtractedDocument",
    "ExtractedEntity",
    "ExtractedPage",
    "ExtractedRelationship",
    "ExtractionError",
    "ExtractionMethod",
    "FaithfulnessChecker",
    "FileChecksum",
    "FusionCandidate",
    "GenerateResponse",
    "GraphBuilder",
    "GraphCandidate",
    "GraphContextBuilder",
    "GraphContextEvidence",
    "GraphEntity",
    "GraphExtractionService",
    "GraphPath",
    "GraphRanker",
    "GraphRelationship",
    "GraphRetrievalResult",
    "GraphRetriever",
    "GraphSearchEngine",
    "GraphSearchResponse",
    "GraphSearchStatus",
    "GraphStore",
    "GraphTraversalEngine",
    "GroundedAnswer",
    "HealthCheckItem",
    "HealthChecker",
    "HealthReport",
    "HealthStatus",
    "HybridContextBuilder",
    "HybridRetriever",
    "HybridSearchResponse",
    "HybridSearchStatus",
    "IndexingError",
    "IndexingPipeline",
    "IndexingStats",
    "InvalidVectorError",
    "JsonLogFormatter",
    "LatencyTelemetry",
    "LinkedEntity",
    "MatchType",
    "OllamaAPIError",
    "OllamaClient",
    "OllamaConnectionError",
    "OllamaModelNotFoundError",
    "OllamaTimeoutError",
    "PathEdge",
    "PathNode",
    "Permission",
    "PermissionDeniedError",
    "PostCheckResult",
    "PreCheckResult",
    "ProductionHealthMonitor",
    "ProductionHealthReport",
    "ProductionHealthState",
    "PromptBuilder",
    "QueryEntity",
    "QueryEntityExtractionResult",
    "QueryEntityExtractor",
    "RAGPipeline",
    "ReciprocalRankFusion",
    "RelevanceJudgment",
    "RestorePreview",
    "RestoreResult",
    "ResultNormalizer",
    "RetrievalMetrics",
    "RetrievalRankedItem",
    "RetrievalSource",
    "RetrievalStrategy",
    "Role",
    "SearchError",
    "SearchResponse",
    "SearchResultItem",
    "SemanticSearch",
    "Settings",
    "StrategyAggregateSummary",
    "TelemetryCollector",
    "TelemetryEvent",
    "TextChunker",
    "TextCleaner",
    "UserAccount",
    "UserAuditEvent",
    "UserAuditLogger",
    "UserRegistry",
    "UserStatus",
    "ValidationError",
    "ValidationResult",
    "VectorCandidate",
    "VectorRecord",
    "VectorStore",
    "VectorStoreError",
    "__version__",
    "aggregate_strategy_metrics",
    "answer_question",
    "atomic_write_json",
    "atomic_write_text",
    "bootstrap_confidence_interval",
    "calculate_dcg_at_k",
    "calculate_hit_rate_at_k",
    "calculate_map_at_k",
    "calculate_ndcg_at_k",
    "calculate_paired_wilcoxon",
    "calculate_precision_at_k",
    "calculate_recall_at_k",
    "calculate_reciprocal_rank",
    "check_permission",
    "compare_strategies_paired",
    "compute_latency_summary",
    "compute_sha256",
    "configure_observability",
    "estimate_tokens",
    "evaluate_query_retrieval",
    "generate_entity_id",
    "generate_user_id",
    "get_graph_search_engine",
    "get_hybrid_retriever",
    "get_logger",
    "get_settings",
    "has_permission",
    "normalize_entity_type",
    "normalize_predicate",
    "normalize_text",
    "require_permission",
    "sanitize_audit_details",
    "sanitize_filename",
    "search_graph",
    "search_hybrid",
    "setup_logger",
    "strip_think_tags",
    "trace_operation",
]
