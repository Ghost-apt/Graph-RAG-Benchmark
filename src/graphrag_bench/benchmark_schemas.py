"""Domain schemas and data models for Comparative Benchmark Evaluation (Phase 8).

Defines Pydantic v2 models for benchmark queries, graded relevance labels,
evaluation datasets, retrieval and generation metrics, timing statistics,
run manifests, and comparative analysis containers.
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BenchmarkQueryCategory(str, Enum):
    """Categorization taxonomy for benchmark queries."""

    SINGLE_HOP = "single_hop"
    MULTI_HOP = "multi_hop"
    ENTITY_RELATIONSHIP = "entity_relationship"
    COMPARISON = "comparison"
    UNANSWERABLE = "unanswerable"


class BenchmarkSplit(str, Enum):
    """Evaluation dataset splits."""

    DEV = "dev"
    VALIDATION = "validation"
    TEST = "test"


class RetrievalStrategy(str, Enum):
    """Retrieval architectures under comparative evaluation."""

    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"


class RelevanceJudgment(BaseModel):
    """Graded relevance judgment for a query-to-chunk pair."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    chunk_id: str = Field(..., min_length=1, description="Canonical chunk identifier")
    document_id: str = Field(..., min_length=1, description="Canonical source document identifier")
    grade: int = Field(
        ...,
        ge=0,
        le=3,
        description="Graded relevance: 0=Not relevant, 1=Related/partial, 2=Relevant supporting, 3=Direct answer",
    )
    rationale: str = Field(
        default="", description="Annotation rationale explaining the assigned grade"
    )

    @property
    def is_relevant(self) -> bool:
        """Binary relevance threshold: grade >= 2 (relevant supporting or direct answer)."""
        return self.grade >= 2


class BenchmarkQuery(BaseModel):
    """Gold-standard evaluation question with manually validated relevance judgments."""

    model_config = ConfigDict(extra="ignore")

    query_id: str = Field(..., description="Deterministic query identifier (e.g. 'Q01')")
    question: str = Field(..., min_length=3, description="Natural language benchmark question")
    category: BenchmarkQueryCategory = Field(..., description="Taxonomy classification")
    split: BenchmarkSplit = Field(default=BenchmarkSplit.TEST, description="Assigned dataset split")
    answerable: bool = Field(
        default=True, description="Whether the corpus contains sufficient evidence to answer"
    )
    expected_answer: str = Field(
        default="", description="Ground-truth reference answer or refusal statement"
    )
    relevant_chunk_ids: list[str] = Field(
        default_factory=list,
        description="List of canonical chunk IDs with grade >= 2 considered relevant",
    )
    relevance_judgments: list[RelevanceJudgment] = Field(
        default_factory=list,
        description="Comprehensive graded judgments (grades 0-3) for candidate chunks",
    )
    supporting_document_ids: list[str] = Field(
        default_factory=list,
        description="Canonical document IDs providing supporting context",
    )
    annotation_rationale: str = Field(
        default="", description="Human audit notes and validation reasoning"
    )
    annotation_status: Literal["verified", "candidate", "draft"] = Field(
        default="verified", description="Quality validation state of this annotation"
    )

    @field_validator("question")
    @classmethod
    def sanitize_question(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Question cannot be empty.")
        return s


class BenchmarkDocument(BaseModel):
    """Provenance record for a source document in the evaluation corpus."""

    model_config = ConfigDict(extra="ignore")

    document_id: str = Field(..., description="Canonical document identifier")
    title: str = Field(..., description="Document title or headline")
    filename: str = Field(..., description="Source filename")
    file_type: str = Field(..., description="File extension (e.g. 'md', 'txt')")
    source_url: str = Field(default="", description="Original publisher or canonical reference URL")
    author_or_publisher: str = Field(default="", description="Attributed creator or publisher")
    license_or_usage: str = Field(
        default="Research / Benchmark Use", description="Permitted usage terms"
    )
    acquisition_date: str = Field(default="", description="Date acquired (YYYY-MM-DD)")
    chunk_count: int = Field(default=0, ge=0, description="Number of canonical chunks generated")
    sha256_hash: str = Field(
        default="", description="Cryptographic SHA-256 hash of raw document content"
    )


class BenchmarkChunk(BaseModel):
    """Canonical chunk record within the evaluation corpus."""

    model_config = ConfigDict(extra="ignore")

    chunk_id: str = Field(..., description="Canonical unique chunk identifier")
    document_id: str = Field(..., description="Parent document identifier")
    document_title: str = Field(..., description="Document title")
    chunk_index: int = Field(..., ge=0, description="0-based ordinal chunk index")
    content: str = Field(..., min_length=1, description="Normalized chunk text content")
    token_count: int = Field(default=0, ge=0, description="Estimated token length")


class RetrievalRankedItem(BaseModel):
    """Single item returned in a ranked retrieval candidate list."""

    model_config = ConfigDict(extra="ignore")

    rank: int = Field(..., ge=1, description="1-based retrieval rank")
    chunk_id: str = Field(..., description="Canonical chunk identifier")
    document_id: str = Field(..., description="Canonical document identifier")
    document_title: str = Field(default="", description="Source document title")
    modality: Literal["vector", "graph", "hybrid"] = Field(
        ..., description="Retrieval modality attribution"
    )
    score: float = Field(default=0.0, description="Modality or fused relevance score")
    raw_score: float | None = Field(default=None, description="Underlying raw score if normalized")
    gold_grade: int = Field(
        default=0, ge=0, le=3, description="Matched gold-truth relevance grade (0-3)"
    )
    is_relevant: bool = Field(
        default=False, description="Whether item satisfies relevant threshold (grade >= 2)"
    )
    evidence_text: str = Field(default="", description="Text snippet of the retrieved item")


class RetrievalMetrics(BaseModel):
    """Standard information retrieval metrics evaluated at a specific cutoff K."""

    model_config = ConfigDict(extra="ignore")

    k: int = Field(..., ge=1, description="Evaluation rank cutoff K")
    precision: float = Field(..., ge=0.0, le=1.0, description="Precision@K: relevant retrieved / K")
    recall: float = Field(
        ..., ge=0.0, le=1.0, description="Recall@K: relevant retrieved / total gold relevant"
    )
    hit_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Hit Rate@K: 1.0 if at least one relevant item in top K"
    )
    mrr: float = Field(
        ..., ge=0.0, le=1.0, description="Reciprocal Rank: 1 / rank of first relevant item, or 0.0"
    )
    ndcg: float = Field(
        ..., ge=0.0, le=1.0, description="Normalized Discounted Cumulative Gain with graded gains"
    )
    map_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Mean Average Precision at K")


class AnswerQualityMetrics(BaseModel):
    """Evaluation metrics for grounded answer generation (Experiment B)."""

    model_config = ConfigDict(extra="ignore")

    answer_correctness: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Lexical or model-judged semantic alignment with expected answer",
    )
    answer_relevance: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Relevance of response to the user query"
    )
    faithfulness_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Supported factual claims / total evaluated claims",
    )
    citation_reference_precision: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Valid citations / total citations (None on 0 citations)",
    )
    citation_coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of gold relevant chunks cited in the answer",
    )
    refusal_accuracy: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="1.0 if unanswerable query was properly refused, 0.0 if hallucinated",
    )


class LatencyTelemetry(BaseModel):
    """Detailed timing telemetry for a single query execution."""

    model_config = ConfigDict(extra="ignore")

    retrieval_latency_ms: float = Field(default=0.0, ge=0.0)
    query_entity_extraction_ms: float = Field(default=0.0, ge=0.0)
    graph_traversal_ms: float = Field(default=0.0, ge=0.0)
    vector_search_ms: float = Field(default=0.0, ge=0.0)
    fusion_ms: float = Field(default=0.0, ge=0.0)
    generation_latency_ms: float = Field(default=0.0, ge=0.0)
    total_latency_ms: float = Field(default=0.0, ge=0.0)
    query_extraction_model_calls: int = Field(default=0, ge=0)
    answer_generation_model_calls: int = Field(default=0, ge=0)
    total_model_invocations: int = Field(default=0, ge=0)


class BenchmarkQueryResult(BaseModel):
    """Comprehensive evaluation record for a single query under a specific retrieval strategy."""

    model_config = ConfigDict(extra="ignore")

    query_id: str = Field(..., description="Benchmark query ID")
    strategy: RetrievalStrategy = Field(..., description="System tested: vector, graph, or hybrid")
    category: BenchmarkQueryCategory = Field(..., description="Query category")
    split: BenchmarkSplit = Field(..., description="Dataset split")
    answerable: bool = Field(..., description="Whether query was answerable")
    retrieved_items: list[RetrievalRankedItem] = Field(default_factory=list)
    metrics_by_k: dict[int, RetrievalMetrics] = Field(default_factory=dict)
    answer_quality: AnswerQualityMetrics | None = Field(default=None)
    latency: LatencyTelemetry = Field(default_factory=LatencyTelemetry)
    generated_answer: str = Field(
        default="", description="Generated textual response if Experiment B executed"
    )
    status: str = Field(default="SUCCESS", description="Execution outcome")
    error_message: str | None = Field(default=None)


class StrategyAggregateSummary(BaseModel):
    """Aggregated evaluation metrics for a specific retrieval strategy."""

    model_config = ConfigDict(extra="ignore")

    strategy: RetrievalStrategy = Field(...)
    query_count: int = Field(default=0, ge=0)
    mean_precision_at_k: dict[int, float] = Field(default_factory=dict)
    mean_recall_at_k: dict[int, float] = Field(default_factory=dict)
    mean_hit_rate_at_k: dict[int, float] = Field(default_factory=dict)
    mean_mrr: float = Field(default=0.0, ge=0.0, le=1.0)
    mean_ndcg_at_k: dict[int, float] = Field(default_factory=dict)
    mean_map_at_k: dict[int, float] = Field(default_factory=dict)
    mean_faithfulness: float | None = Field(default=None)
    mean_citation_precision: float | None = Field(default=None)
    refusal_accuracy: float | None = Field(default=None)
    mean_retrieval_latency_ms: float = Field(default=0.0)
    median_retrieval_latency_ms: float = Field(default=0.0)
    p95_retrieval_latency_ms: float = Field(default=0.0)
    mean_total_latency_ms: float = Field(default=0.0)
    total_model_invocations: int = Field(default=0)

    @property
    def total_queries(self) -> int:
        return self.query_count

    @property
    def mean_latency_ms(self) -> float:
        return self.mean_total_latency_ms


class PairedComparisonResult(BaseModel):
    """Statistical comparison between two retrieval strategies across identical queries."""

    model_config = ConfigDict(extra="ignore")

    strategy_a: RetrievalStrategy = Field(...)
    strategy_b: RetrievalStrategy = Field(...)
    metric_name: str = Field(...)
    mean_diff: float = Field(..., description="mean(A) - mean(B)")
    median_diff: float = Field(..., description="median(A) - median(B)")
    win_count_a: int = Field(default=0, ge=0)
    win_count_b: int = Field(default=0, ge=0)
    tie_count: int = Field(default=0, ge=0)
    bootstrap_ci_95: tuple[float, float] = Field(
        default=(0.0, 0.0), description="95% bootstrap confidence interval"
    )
    wilcoxon_stat: float | None = Field(default=None)
    wilcoxon_p_value: float | None = Field(default=None)
    is_statistically_significant: bool = Field(default=False)
    sample_size: int = Field(default=0, ge=0)
    notes: str = Field(default="")

    @property
    def baseline_strategy(self) -> RetrievalStrategy:
        return self.strategy_a

    @property
    def treatment_strategy(self) -> RetrievalStrategy:
        return self.strategy_b

    @property
    def wins(self) -> int:
        """Wins for treatment strategy_b over baseline strategy_a."""
        return self.win_count_b

    @property
    def losses(self) -> int:
        """Losses for treatment strategy_b against baseline strategy_a."""
        return self.win_count_a

    @property
    def ties(self) -> int:
        return self.tie_count

    @property
    def paired_query_count(self) -> int:
        return self.sample_size


class BenchmarkRunManifest(BaseModel):
    """Audit manifest recording complete experiment configuration, provenance, and environment."""

    model_config = ConfigDict(extra="ignore")

    run_id: str = Field(..., description="Unique run identifier (e.g. 'run_20261114_105822')")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    timestamp_utc: str = Field(default="")
    dataset_version: str = Field(default="1.0.0")
    dataset_name: str = Field(default="Phase 8 Canonical Benchmark Dataset")
    dataset_hash: str = Field(default="", description="SHA-256 hash of benchmark_dataset.json")
    system_version: str = Field(default="0.8.0")
    corpus_document_count: int = Field(default=0)
    corpus_chunk_count: int = Field(default=0)
    total_queries: int = Field(default=0)
    queries_evaluated: int = Field(default=0)
    splits_evaluated: list[str] = Field(default_factory=lambda: ["test"])
    strategies_evaluated: list[str] = Field(default_factory=lambda: ["vector", "graph", "hybrid"])
    eval_k_values: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    graph_max_hops: int = Field(default=2)
    rrf_k: int = Field(default=60)
    vector_weight: float = Field(default=1.0)
    graph_weight: float = Field(default=1.0)
    experiment_b_executed: bool = Field(default=False)
    llm_generation_model: str = Field(default="qwen3:8b")
    llm_temperature: float = Field(default=0.1)
    python_version: str = Field(default="")
    ollama_base_url: str = Field(default="")
    hardware_notes: str = Field(default="")
    strategy_summaries: dict[str, StrategyAggregateSummary] = Field(default_factory=dict)
    paired_comparisons: list[PairedComparisonResult] = Field(default_factory=list)
    status: Literal["completed", "interrupted", "failed", "in_progress"] = Field(
        default="completed"
    )

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, v: Any) -> str:
        if isinstance(v, str):
            val = v.lower()
            if val in ("completed", "interrupted", "failed", "in_progress"):
                return val
        return "completed"
