"""Phase 8 Benchmark Data Environment & Ground Truth Dataset Builder.

Constructs an isolated, reproducible evaluation environment in data/phase8_benchmark/:
1. Generates 22 diverse technical documents in raw_documents/
2. Chunks and processes documents into processed/chunks.jsonl and processed/documents.json
3. Computes vector embeddings using nomic-embed-text via Ollama and indexes into chroma/
4. Constructs an isolated NetworkX Knowledge Graph in knowledge_graph/
5. Compiles 60 manually validated benchmark queries with graded relevance judgments (0-3)
   and dataset splits (dev, validation, test) into ground_truth/benchmark_dataset.json
"""

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from graphrag_bench.benchmark_schemas import (
    BenchmarkChunk,
    BenchmarkDocument,
    BenchmarkQuery,
    BenchmarkQueryCategory,
    BenchmarkSplit,
    RelevanceJudgment,
)
from graphrag_bench.embedding_service import EmbeddingService
from graphrag_bench.graph_schemas import GraphEntity, GraphRelationship
from graphrag_bench.graph_store import GraphStore
from graphrag_bench.logger import setup_logger
from graphrag_bench.ollama_client import OllamaClient
from graphrag_bench.vector_store import VectorRecord, VectorStore

setup_logger()

ROOT_BENCH = Path("data/phase8_benchmark")
RAW_DOCS_DIR = ROOT_BENCH / "raw_documents"
PROCESSED_DIR = ROOT_BENCH / "processed"
CHROMA_DIR = ROOT_BENCH / "chroma"
KG_DIR = ROOT_BENCH / "knowledge_graph"
GROUND_TRUTH_DIR = ROOT_BENCH / "ground_truth"
RESULTS_DIR = ROOT_BENCH / "results"

COLLECTION_NAME = "phase8_benchmark_documents"


def compute_sha256(text: str) -> str:
    """Compute uppercase SHA-256 hash."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


# -----------------------------------------------------------------------------
# 1. Source Documents Definition (22 Technical Documents)
# -----------------------------------------------------------------------------

SOURCE_DOCUMENTS: list[dict] = [
    {
        "doc_id": "doc01_vector_embeddings",
        "title": "Vector Embeddings and Dense Representations",
        "filename": "doc01_vector_embeddings.md",
        "author": "Dr. Elena Rostova",
        "url": "https://ragbench.internal/docs/01-vector-embeddings",
        "sections": [
            "Dense vector embeddings map unstructured natural language tokens into continuous geometric vector spaces where semantic proximity correlates with metric distance. Contemporary embedding models such as nomic-embed-text, bge-large-en-v1.5, and text-embedding-ada-002 project textual sequences into fixed-dimensional latent manifolds ranging from 768 to 1536 floating-point dimensions.",
            "In dense retrieval architectures, embedding dimensionality dictates both representation capacity and memory footprint. The nomic-embed-text architecture utilizes 768 dimensions with dynamic sequence truncation at 8192 tokens, providing an optimal trade-off between semantic expressive power and inference throughput on consumer hardware. Subword tokenization models like WordPiece and Byte-Pair Encoding construct the initial token vocabulary prior to transformer self-attention projection.",
            "High-dimensional embedding manifolds suffer from dimensionality curse phenomena where distances between dissimilar points concentrate around the median distance. To counteract geometric saturation, embedding vectors are normalized to unit L2-norm, transforming Euclidean distance and dot product calculations into monotonic equivalents of cosine similarity.",
            "Dense vector representations excel at capturing thematic, contextual, and synonymous semantic associations. However, vector compression inherently creates information loss regarding explicit entity identities, predicate directions, and quantitative multi-hop numerical assertions, which motivates structural augmentation via knowledge graphs.",
            "Recent advancements in embedding fine-tuning employ Matryoshka Representation Learning (MRL), enabling vectors to be truncated to smaller dimensions such as 256 or 512 without catastrophic degradation in retrieval recall.",
        ],
    },
    {
        "doc_id": "doc02_vector_similarity_metrics",
        "title": "Geometric Vector Distance Functions and Invariants",
        "filename": "doc02_vector_similarity_metrics.md",
        "author": "Marcus Vance",
        "url": "https://ragbench.internal/docs/02-vector-similarity",
        "sections": [
            "Vector databases rely on metric distance functions to measure proximity between query embeddings and indexed candidate vectors. The three principal metrics utilized in production RAG systems are Cosine distance, Euclidean distance (L2), and Dot Product (inner product).",
            "Cosine distance is defined mathematically as d_cos(u, v) = 1.0 - (u . v) / (||u|| * ||v||), bounded within the interval [0.0, 2.0]. When vectors u and v are strictly normalized to unit length (||u|| = ||v|| = 1.0), Cosine distance simplifies to 1.0 - (u . v), directly relating to raw cosine similarity s_raw = 1.0 - d_cos.",
            "Euclidean distance d_L2(u, v) = sqrt(sum((u_i - v_i)^2)) measures geometric length in Euclidean space. For unit-normalized vectors, Euclidean distance relates directly to cosine similarity via the algebraic identity d_L2^2 = 2 * (1 - s_raw).",
            "Inner product similarity s_dot(u, v) = sum(u_i * v_i) provides the fastest computational throughput by eliminating square root and division operations. In unnormalized spaces, dot product favors longer vector magnitudes regardless of semantic orientation, making normalization essential for fair cosine ranking.",
            "Metric space invariants require the triangle inequality d(x, z) <= d(x, y) + d(y, z). While Euclidean distance satisfies the triangle inequality, raw cosine distance fails it in non-metric representations, necessitating angular distance transformations in certain index structures.",
        ],
    },
    {
        "doc_id": "doc03_ann_indexing_hnsw_ivf",
        "title": "Approximate Nearest Neighbor Indexing: HNSW vs IVF",
        "filename": "doc03_ann_indexing_hnsw_ivf.md",
        "author": "Siddharth Nair",
        "url": "https://ragbench.internal/docs/03-ann-indexing",
        "sections": [
            "Exact nearest neighbor search requires linear scanning with O(N * D) time complexity, becoming computationally intractable when corpus scale N exceeds hundreds of thousands of vectors. Approximate Nearest Neighbor (ANN) indexing structures trade marginal search recall for orders-of-magnitude faster logarithmic or sub-linear query latency.",
            "Hierarchical Navigable Small World (HNSW) graphs represent the current state-of-the-art in vector indexing. HNSW constructs a multi-layer graph where upper layers feature long-range sparse skip-connections for fast logarithmic routing and bottom layers feature dense local clustering. HNSW performance is governed by parameter M (maximum bidirectional connections per node, typically 16 to 64) and efConstruction (search beam width during index creation).",
            "During HNSW search execution, the efSearch parameter controls the priority queue exploration capacity. Higher efSearch values asymptotically approach 100% nearest-neighbor recall at the expense of increased latency and CPU cycles. HNSW delivers superior query throughput and recall stability compared to tree-based and clustering-based vector indexes.",
            "Inverted File Index (IVF) partitions vector space into Voronoi cells using k-means centroid clustering. During search, IVF probes the nlist nearest centroids (nprobe parameter) and scans vectors within those clusters. While IVF requires less memory than HNSW, it suffers from boundary quantization errors where true nearest neighbors fall across cluster borders.",
            "Product Quantization (PQ) and Scalar Quantization (SQ) compress high-dimensional floating-point vectors into compact discrete byte codes, reducing RAM utilization by up to 75% at the cost of minor distance approximation noise.",
        ],
    },
    {
        "doc_id": "doc04_vector_rag_limitations",
        "title": "Structural Failure Modes and Limitations of Vector RAG",
        "filename": "doc04_vector_rag_limitations.md",
        "author": "Dr. Aris Thorne",
        "url": "https://ragbench.internal/docs/04-vector-rag-limitations",
        "sections": [
            "Traditional Vector RAG operates on localized chunk-level semantic similarity, retrieving isolated passages that maximize embedding dot products with the user prompt. While effective for localized fact lookup, Vector RAG exhibits documented structural failure modes across complex query categories.",
            "The first failure mode is Multi-Hop Semantic Disconnection. When answering a question requiring evidence distributed across multiple disjoint documents (e.g., Document A relates Entity X to Y, while Document B relates Entity Y to Z), Vector RAG often retrieves only the passage matching the initial query tokens, failing to discover the bridging entity in the second passage.",
            "The second failure mode is Global Thematic Synthesis Blindness. When asked holistic questions like 'What are the overarching architectural trends across all projects?', Vector RAG samples semantically dominant chunks rather than synthesizing an aggregated, comprehensive overview of the entire corpus.",
            "The third failure mode is Entity Identity Ambiguity. Vector embeddings project word tokens into fuzzy semantic neighborhoods where distinct entities with similar lexical contexts (e.g., Python 3.10 vs Python 3.11, or ChromaDB vs Pinecone) achieve overlapping embeddings, leading to entity hallucination and incorrect attribution.",
            "Furthermore, fixed-length passage chunking inevitably fractures semantic boundaries, bisecting critical relational assertions across arbitrary chunk delimiters and degrading downstream LLM comprehension.",
        ],
    },
    {
        "doc_id": "doc05_knowledge_graph_foundations",
        "title": "Knowledge Graph Foundations: Entities, Predicates, and Ontologies",
        "filename": "doc05_knowledge_graph_foundations.md",
        "author": "Maya Lin",
        "url": "https://ragbench.internal/docs/05-knowledge-graph-foundations",
        "sections": [
            "A Knowledge Graph (KG) is a structured representation of real-world knowledge organized as a directed labeled multigraph G = (V, E, L_v, L_e), where vertices V represent discrete entities, directed edges E represent relationships, and labeling functions assign semantic types and predicates.",
            "In graph RAG systems, entities correspond to canonical concepts, software systems, algorithms, organizations, or individuals. Each entity possesses a unique canonical identifier, an entity type taxonomy (e.g., CONCEPT, TECHNOLOGY, FRAMEWORK), and surface alias synonyms that map varied textual mentions to a single ground-truth node.",
            "Relationships represent directed semantic assertions connecting a source entity to a target entity via an explicit predicate label (e.g., UTILIZES, COMBINES, IMPLEMENTED_WITH, EVALUATED_ON). Each relationship edge stores provenance metadata including source chunk IDs, source document titles, extraction confidence scores in [0.0, 1.0], and verbatim textual evidence excerpts.",
            "Ontological schemas govern the validity of entity types and relationship predicates, preventing semantic drift and ensuring that extracted assertions adhere to formal domain constraints. Labeled property graphs allow arbitrary key-value metadata properties to be attached directly to both nodes and edges.",
            "Unlike vector indices which encode implicit associative similarity, knowledge graphs preserve explicit relational topology, enabling deterministic traversals, path reasoning, and causal chain verification.",
        ],
    },
    {
        "doc_id": "doc06_entity_extraction_llm",
        "title": "LLM-Driven Entity and Relationship Extraction Pipelines",
        "filename": "doc06_entity_extraction_llm.md",
        "author": "Julian Sterling",
        "url": "https://ragbench.internal/docs/06-entity-extraction",
        "sections": [
            "Automated knowledge graph construction from unstructured technical text relies on Large Language Model (LLM) extraction pipelines. State-of-the-art pipelines employ instruction-tuned models such as Qwen3:8b and Llama-3 to identify named entities, determine ontological classifications, and extract directed relationships.",
            "To ensure reliable parsing, modern extractors enforce structured JSON output formats using constrained decoding grammars or strict JSON schemas. The prompt instructs the model to extract strictly grounded factual assertions and requires verbatim evidence quotations directly copied from the source text chunk.",
            "Extraction hallucination is mitigated by dual-pass verification: an extraction pass proposes candidate (source, predicate, target) triples, and a validation pass checks whether the quoted evidence passage contains exact string matches supporting both entity mentions and the connecting predicate.",
            "Extraction confidence scores are computed by combining model token generation log-probabilities with evidence verification heuristics. Triples with confidence scores below calibrated thresholds (e.g., 0.70) are rejected prior to graph insertion.",
            "Prompt engineering techniques such as few-shot in-context demonstration examples and negative constraint rules dramatically reduce spurious entity proliferation and verb-phrase predicate fragmentation.",
        ],
    },
    {
        "doc_id": "doc07_entity_resolution_linking",
        "title": "Entity Resolution, Canonicalization, and Alias Merging",
        "filename": "doc07_entity_resolution_linking.md",
        "author": "Dr. Sarah Jenkins",
        "url": "https://ragbench.internal/docs/07-entity-resolution",
        "sections": [
            "A primary challenge in multi-document graph construction is entity resolution: determining whether different surface mentions refer to the same underlying real-world entity. Without resolution, graphs suffer from node duplication, resulting in fragmented subgraphs and broken traversal pathways.",
            "Entity canonicalization standardizes surface strings through Unicode NFKC normalization, lowercase folding, whitespace collapsing, and punctuation stripping. For example, 'Graph-RAG', 'graph rag', and 'Graph RAG' resolve to the canonical identifier 'graph_rag'.",
            "Alias merging groups known synonymous variants into a primary canonical entity. When an extraction proposes an entity mention that matches an existing alias in the graph store (e.g., 'Traditional Vector RAG' mapping to 'Vector RAG'), the new node is merged into the existing entity, and its supporting citations are consolidated.",
            "Fuzzy string matching algorithms such as Levenshtein distance, Jaro-Winkler similarity, and token-sort ratios identify typographical variations and acronym expansions. String similarities exceeding 0.90 trigger merge reviews, while lower scores maintain separate candidate nodes.",
            "Cross-document entity linking connects local extractions across disparate source files, forming an interconnected corporate or domain knowledge mesh capable of multi-document cross-referencing.",
        ],
    },
    {
        "doc_id": "doc08_graph_traversal_algorithms",
        "title": "Multi-Hop Graph Traversal and Search Algorithms",
        "filename": "doc08_graph_traversal_algorithms.md",
        "author": "Karthik Subramanian",
        "url": "https://ragbench.internal/docs/08-graph-traversal",
        "sections": [
            "Graph retrieval operates by exploring relational pathways radiating from pre-resolved seed entities. The primary traversal algorithm utilized in Graph RAG is bounded Breadth-First Search (BFS), which systematically expands neighborhood concentric rings up to a maximum hop distance.",
            "Traversal depth max_hops is constrained between 1 and 3 hops in production systems. In 1-hop traversal, only immediate direct neighbors of seed nodes are retrieved. In 2-hop traversal, the search expands through intermediate bridge nodes, revealing indirect relational connections. In 3-hop traversal, distant multi-step dependencies are discovered.",
            "Unconstrained graph traversal is subject to exponential combinatorial explosion, where the number of traversed paths scales as O(b^d), where b is average node degree and d is traversal depth. High-degree hub nodes (e.g., 'RAG' or 'Model') can cause traversal to explore thousands of irrelevant peripheral edges.",
            "Cycle detection prevents infinite traversal loops by maintaining a visited set of entity IDs for each active exploration path. When a traversal step encounters an entity already present in the active path prefix, the branch is terminated.",
            "Directional traversal filtering allows queries to restrict expansion to outgoing edges ('out'), incoming edges ('in'), or bidirectional relationships ('both'), depending on the causal or semantic intent of the question.",
        ],
    },
    {
        "doc_id": "doc09_graph_pruning_and_ranking",
        "title": "Degree Pruning, Hop Decay, and Graph Path Ranking",
        "filename": "doc09_graph_pruning_and_ranking.md",
        "author": "Dr. Aris Thorne",
        "url": "https://ragbench.internal/docs/09-graph-pruning-ranking",
        "sections": [
            "To manage traversal complexity and prioritize the most informative relational evidence, Graph RAG systems employ degree-based node pruning and composite path scoring algorithms.",
            "Degree-based pruning truncates neighbor exploration for hub nodes whose degree exceeds max_degree (typically 20 to 50). When a node exceeds this threshold, its incident edges are ranked by relationship confidence, and only the top edges are retained for traversal expansion, filtering out low-information peripheral connections.",
            "Path relevance is scored using an exponential hop decay factor gamma in (0.0, 1.0], commonly calibrated to 0.80 or 0.85. The composite path score is formulated as s_path = prod(confidence_e for e in path) * (gamma ^ hop_count). This formulation mathematically penalizes longer traversal paths while rewarding high-confidence factual extractions.",
            "Multi-path deduplication aggregates multiple independent traversal paths supporting the same source chunk. When multiple graph paths reference identical source chunks, their path scores are fused using noisy-or or maximum score selection to provide a single consolidated chunk relevance score.",
            "Top-K graph candidate ranking sorts discovered chunks by their consolidated path scores, ensuring that downstream context preparation presents the highest-fidelity relational evidence to the LLM generation prompt.",
        ],
    },
    {
        "doc_id": "doc10_hybrid_retrieval_architectures",
        "title": "Hybrid Retrieval Architectures: Fusing Dense and Structural Representations",
        "filename": "doc10_hybrid_retrieval_architectures.md",
        "author": "Maya Lin",
        "url": "https://ragbench.internal/docs/10-hybrid-retrieval",
        "sections": [
            "Hybrid RAG represents a multi-modal retrieval paradigm that concurrently executes dense semantic vector search and structured knowledge graph traversal over a common evaluation corpus. By fusing these complementary retrieval modalities, Hybrid RAG eliminates the structural failure modes inherent to single-modality architectures.",
            "Vector retrieval provides broad semantic recall, locating passages that share conceptual themes and synonymous terminology even when exact entity names are not explicitly mentioned in the query. Conversely, graph retrieval provides precision, multi-hop relational bridging, and explicit factual grounding.",
            "In a production Hybrid RAG execution pipeline, the user query is dispatched asynchronously to both the Vector Engine and the Graph Engine. The Vector Engine generates query embeddings and queries ChromaDB for top-K candidates. Concurrently, the Graph Engine extracts query entities, links seed nodes, traverses NetworkX graph paths, and outputs ranked graph candidates.",
            "A critical engineering requirement in hybrid architectures is canonical chunk ID harmonization. Both vector records and graph edge citations must resolve back to identical canonical chunk identifiers in a shared document repository, enabling deterministic deduplication and attribution.",
            "Empirical evaluations indicate that Hybrid RAG achieves higher retrieval hit rates and superior grounded answer faithfulness compared to either vector-only or graph-only baselines.",
        ],
    },
    {
        "doc_id": "doc11_reciprocal_rank_fusion",
        "title": "Reciprocal Rank Fusion: Algorithms, Mathematics, and Calibration",
        "filename": "doc11_reciprocal_rank_fusion.md",
        "author": "Siddharth Nair",
        "url": "https://ragbench.internal/docs/11-reciprocal-rank-fusion",
        "sections": [
            "Reciprocal Rank Fusion (RRF) is an unsupervised rank aggregation algorithm designed to combine candidate lists from heterogeneous retrieval systems without requiring raw score calibration or score normalization. RRF is particularly well-suited for merging vector cosine scores and graph path traversal scores.",
            "The standard RRF formulation assigns a composite fusion score to candidate chunk c: s_rrf(c) = sum(w_m / (k + rank_m(c)) for m in modalities), where rank_m(c) is the 1-based ordinal rank of chunk c in modality m, w_m is the modality importance weight, and k is a positive smoothing constant.",
            "The smoothing constant k regulates the sensitivity of the rank decay. In the classical Cormack et al. baseline, k is set to 60. Lower values of k (e.g., 20) exponentially amplify the advantage of top-ranked candidates, whereas higher values of k (e.g., 100) produce more uniform, gradual rank decay across candidates.",
            "Modality weights w_vec and w_graph allow fine-grained tuning of retrieval bias. Setting w_vec = 1.0 and w_graph = 1.0 treats vector and graph retrieval as equal partners. Under standard benchmark settings (k=60, w_vec=1.0, w_graph=1.0), a chunk ranked 1 in both modalities achieves a maximum theoretical RRF score of 2 / (60 + 1) = 0.03279.",
            "A vital property of RRF is scale invariance: because fusion depends strictly on candidate rank order rather than raw score magnitudes, outliers and divergent metric scales between vector distance and graph traversal do not distort the fused candidate ordering.",
        ],
    },
    {
        "doc_id": "doc12_evaluation_metrics_ir",
        "title": "Information Retrieval Metrics: Precision@K, Recall@K, and MRR",
        "filename": "doc12_evaluation_metrics_ir.md",
        "author": "Dr. Elena Rostova",
        "url": "https://ragbench.internal/docs/12-ir-evaluation-metrics",
        "sections": [
            "Evaluating retrieval systems requires standardized Information Retrieval (IR) metrics calculated across fixed rank cutoffs K in {1, 3, 5, 10}. The primary metrics are Precision@K, Recall@K, Hit Rate@K, and Mean Reciprocal Rank (MRR).",
            "Precision@K measures the proportion of retrieved items in the top-K list that are relevant: P@K = (number of relevant items in top K) / K. By standard IR convention, the denominator is strictly K; if a system returns fewer than K items, unretrieved positions count as non-relevant.",
            "Recall@K measures the proportion of all gold-standard relevant items in the corpus that appear in the top-K retrieved list: R@K = (number of relevant items in top K) / (total relevant items in ground truth). For unanswerable queries where zero gold relevant chunks exist, recall is defined as 0.0 by standard benchmark convention.",
            "Hit Rate@K is a binary metric indicating whether at least one relevant chunk appears within the first K results. Hit Rate is especially relevant in single-hop QA scenarios where retrieving a single correct passage is sufficient for answer generation.",
            "Mean Reciprocal Rank (MRR) evaluates how high in the ranked list the first relevant item appears: MRR = (1 / N) * sum(1 / rank_i), where rank_i is the 1-based rank of the first relevant result for query i. Queries with no relevant items in the top-K list contribute 0.0 to the sum.",
        ],
    },
    {
        "doc_id": "doc13_ndcg_graded_relevance",
        "title": "Graded Relevance and Normalized Discounted Cumulative Gain (nDCG)",
        "filename": "doc13_ndcg_graded_relevance.md",
        "author": "Marcus Vance",
        "url": "https://ragbench.internal/docs/13-ndcg-graded-relevance",
        "sections": [
            "While binary metrics treat chunks as either completely relevant or completely irrelevant, real-world information retrieval benefits from graded relevance scales. Phase 8 employs a four-point graded scale: 0 (Not Relevant), 1 (Related / Partial Context), 2 (Relevant Supporting Context), and 3 (Highly Relevant Direct Answer).",
            "Discounted Cumulative Gain (DCG) rewards highly relevant items placed at early ranks while applying a logarithmic discount to items placed at deeper ranks: DCG@K = sum_{i=1}^K (2^{rel_i} - 1) / log2(i + 1), where rel_i is the integer relevance grade of the item at rank i.",
            "Ideal DCG (IDCG@K) represents the maximum achievable DCG score for a query, calculated by sorting all available positive gold relevance grades in descending order and computing DCG@K on the perfectly sorted list.",
            "Normalized Discounted Cumulative Gain is computed as nDCG@K = DCG@K / IDCG@K, producing a normalized score in [0.0, 1.0]. A system that ranks grade 3 chunks above grade 2 chunks achieves higher nDCG than a system that ranks grade 2 chunks first, even if both achieve identical binary Precision@K.",
            "For queries with zero positive gold grades (such as unanswerable queries), if the system also returns zero positive items, nDCG@K is defined as 1.0 (perfect rejection), whereas if irrelevant items are returned with non-zero grades, nDCG is 0.0.",
        ],
    },
    {
        "doc_id": "doc14_generation_faithfulness_grounding",
        "title": "Grounded Generation Metrics: Faithfulness, Factuality, and Provenance",
        "filename": "doc14_generation_faithfulness_grounding.md",
        "author": "Julian Sterling",
        "url": "https://ragbench.internal/docs/14-generation-faithfulness",
        "sections": [
            "In end-to-end RAG pipelines (Experiment B), retrieval quality directly influences generated answer quality. The primary generation metrics are Answer Faithfulness, Citation Reference Precision, Citation Coverage, and Answer Relevance.",
            "Faithfulness evaluates whether factual assertions in the generated response are semantically entailed by the retrieved context. The answer text is decomposed into atomic factual claims {C1, C2, ... Cn}. Each claim is verified against source evidence, classified as SUPPORTED, UNCERTAIN, or UNSUPPORTED, and the faithfulness score is computed as (count of SUPPORTED claims) / (total evaluated claims).",
            "Citation Reference Precision measures the structural validity of citation markers (e.g., [S1], [S2]) embedded in the answer: Prec_cit = (count of valid active-context citations) / (total citations generated). When zero citations are generated, precision is mathematically undefined (None / N/A).",
            "Citation Coverage measures the proportion of gold-standard relevant chunks that were actually cited in the generated answer: Cov_cit = |cited_chunks intersection gold_relevant| / |gold_relevant|, identifying whether the LLM utilized all available supporting evidence.",
            "A critical distinction exists between structural reference validity and semantic claim support: an answer can feature valid citation tags that point to active evidence chunks without those chunks semantically supporting the generated factual claim, necessitating dual-layer verification.",
        ],
    },
    {
        "doc_id": "doc15_guardrails_unanswerable_queries",
        "title": "Safety Guardrails, Hallucination Prevention, and Refusal Handling",
        "filename": "doc15_guardrails_unanswerable_queries.md",
        "author": "Dr. Sarah Jenkins",
        "url": "https://ragbench.internal/docs/15-guardrails-refusal",
        "sections": [
            "A critical requirement for trustworthy enterprise RAG systems is the ability to recognize unanswerable queries and gracefully refuse generation rather than hallucinating plausible but unsubstantiated answers.",
            "Pre-generation guardrails inspect retrieved evidence candidates before invoking the LLM generation model. If candidate relevance scores fall below calibrated thresholds (e.g., vector normalized score < 0.50 and graph score < 0.40), generation is short-circuited immediately, returning INSUFFICIENT_EVIDENCE.",
            "Short-circuiting unanswerable queries yields significant computational benefits: in benchmark experiments, pre-generation short-circuiting reduced end-to-end latency from ~35 seconds (wasted generation inference) to under 80 milliseconds.",
            "Post-generation guardrails inspect generated answer text for refusal phrases (e.g., 'insufficient information', 'cannot be answered based on the provided evidence'). If refusal intent is detected, the answer status is classified as INSUFFICIENT_EVIDENCE, and fabricated citations are rejected.",
            "Refusal Accuracy measures the proportion of unanswerable benchmark queries that are correctly recognized and refused by the system without generating hallucinated factual claims.",
        ],
    },
    {
        "doc_id": "doc16_graph_rag_benchmark_framework",
        "title": "Scientific Benchmark Methodology: Query Taxonomies and Fair Comparison",
        "filename": "doc16_graph_rag_benchmark_framework.md",
        "author": "Dr. Elena Rostova",
        "url": "https://ragbench.internal/docs/16-benchmark-methodology",
        "sections": [
            "Scientific comparison of Vector RAG, Graph RAG, and Hybrid RAG mandates strict experimental controls to isolate retrieval architecture as the sole independent variable.",
            "Fair comparison requirements dictate that all three systems must execute against an identical underlying corpus, identical canonical chunk IDs, identical chunking configurations, identical generation models (qwen3:8b), identical prompt templates, identical temperature settings (0.1), and identical token limits.",
            "Benchmark query datasets must feature a balanced query taxonomy comprising five core categories: (1) Single-Hop Factual queries, (2) Multi-Hop Reasoning queries, (3) Entity & Relationship queries, (4) Technology & Concept Comparison queries, and (5) Unanswerable / Out-of-Domain queries.",
            "Dataset splitting partitions benchmark queries into Development (for tuning), Validation (for calibration), and Held-Out Test sets. Reported benchmark results must strictly evaluate the held-out test split to prevent configuration overfitting.",
            "Reporting macro-averaged metrics across individual query categories is essential, as architectural strengths vary significantly: Vector RAG frequently matches Hybrid RAG on single-hop queries, while Graph RAG and Hybrid RAG dramatically outperform Vector RAG on multi-hop and relationship queries.",
        ],
    },
    {
        "doc_id": "doc17_vector_database_engines",
        "title": "Comparative Vector Database Engines: ChromaDB, Milvus, and Qdrant",
        "filename": "doc17_vector_database_engines.md",
        "author": "Karthik Subramanian",
        "url": "https://ragbench.internal/docs/17-vector-database-engines",
        "sections": [
            "Vector databases provide persistence, indexing, and similarity search for high-dimensional embeddings. Prominent production vector databases include ChromaDB, Milvus, Qdrant, and pgvector.",
            "ChromaDB is a lightweight embedded vector database written in Python with a C++ hnswlib core and SQLite metadata store. ChromaDB is ideal for localized research benchmarks and embedded agent workflows due to zero-configuration local persistence and native Apache 2.0 licensing.",
            "Milvus is a distributed, cloud-native vector database engineered for billion-scale vector datasets. Milvus decouples storage and computation using MinIO/S3 and Kafka message queues, supporting extensive sharding, GPU acceleration, and diverse index types including IVF-PQ and SCaNN.",
            "Qdrant is an open-source vector search engine written in Rust, featuring payload-based filtering integrated directly into HNSW graph traversal. This pre-filtering mechanism avoids index degradation when searching over heavily constrained metadata subsets.",
            "pgvector is an open-source extension for PostgreSQL that adds vector data types and exact or approximate nearest neighbor indexing (IVFFlat, HNSW) directly inside relational SQL databases, eliminating the need for an external standalone vector store.",
        ],
    },
    {
        "doc_id": "doc18_graph_database_engines",
        "title": "Comparative Graph Database Engines: NetworkX, Neo4j, and Memgraph",
        "filename": "doc18_graph_database_engines.md",
        "author": "Maya Lin",
        "url": "https://ragbench.internal/docs/18-graph-database-engines",
        "sections": [
            "Graph databases store entities and relationships natively, optimizing multi-hop traversal and pattern matching. Key engines evaluated in Graph RAG architectures include NetworkX, Neo4j, Memgraph, and Amazon Neptune.",
            "NetworkX is a pure Python library designed for complex network analysis and in-memory graph operations. In localized benchmarking frameworks, NetworkX provides deterministic, thread-safe graph storage with instant JSON and GraphML export capabilities, avoiding client-server networking overhead for sub-million node graphs.",
            "Neo4j is the industry-standard native property graph database, utilizing the declarative Cypher query language and disk-optimized pointer-chasing data structures. Neo4j features enterprise Graph Data Science (GDS) libraries supporting PageRank, Louvain community detection, and node similarity algorithms.",
            "Memgraph is an in-memory graph database written in C++, fully compatible with openCypher. Memgraph provides real-time streaming graph ingestion and microsecond traversal latencies, ideal for high-throughput transactional Graph RAG pipelines.",
            "Amazon Neptune is a fully managed cloud graph database supporting both property graph (Apache TinkerPop Gremlin, openCypher) and RDF graph (W3C SPARQL) standards with automated replication across AWS Availability Zones.",
        ],
    },
    {
        "doc_id": "doc19_computational_latency_tradeoffs",
        "title": "Computational Efficiency, Latency Profiling, and Hardware Footprints",
        "filename": "doc19_computational_latency_tradeoffs.md",
        "author": "Julian Sterling",
        "url": "https://ragbench.internal/docs/19-computational-latency",
        "sections": [
            "Deploying Graph RAG systems in production requires balancing retrieval quality against computational resource consumption and end-to-end response latency.",
            "Vector retrieval exhibits low latency overhead: query embedding with nomic-embed-text requires approximately 50 to 120 ms, followed by ChromaDB HNSW nearest neighbor search requiring 2 to 15 ms, yielding total retrieval latency under 150 ms.",
            "Graph retrieval latency depends heavily on entity extraction strategy: deterministic dictionary matching executes in under 5 ms, whereas LLM-based entity extraction fallback (e.g., Qwen3:8b) requires 8 to 15 seconds of inference time. Once seed nodes are linked, NetworkX BFS traversal across 2 hops takes less than 5 ms.",
            "Answer generation represents the dominant latency bottleneck in end-to-end RAG pipelines. Local generation with Qwen3:8b on consumer hardware (e.g., RTX 4080 or Apple Silicon) typically requires 15 to 45 seconds depending on generated output token length.",
            "Hardware profiling indicates that vector stores require memory proportional to N * D * 4 bytes plus HNSW index graph overhead, whereas knowledge graphs require RAM proportional to |V| + |E| with additional memory for string dictionaries and evidence text buffers.",
        ],
    },
    {
        "doc_id": "doc20_advanced_rag_paradigms",
        "title": "Advanced RAG Paradigms: Tree of Thoughts, Graph of Thoughts, and Agentic RAG",
        "filename": "doc20_advanced_rag_paradigms.md",
        "author": "Dr. Aris Thorne",
        "url": "https://ragbench.internal/docs/20-advanced-rag-paradigms",
        "sections": [
            "Beyond basic one-shot retrieve-and-generate workflows, modern AI research has developed advanced reasoning paradigms that leverage graph topologies for deliberate problem solving.",
            "Tree of Thoughts (ToT) generalizes Chain-of-Thought reasoning by maintaining a tree of intermediate reasoning steps. Each branch is evaluated and searched using BFS or DFS, allowing the model to look ahead, backtrack from dead ends, and evaluate multiple prospective reasoning pathways.",
            "Graph of Thoughts (GoT) extends tree reasoning to arbitrary directed graphs, enabling novel operations such as merging multiple independent thought branches into a synthesized consensus and creating recurring iterative feedback loops.",
            "Agentic RAG integrates autonomous tool-use and dynamic retrieval planning. Rather than issuing a single static query, an agentic controller evaluates intermediate retrieval results, identifies missing context, dynamically reformulates search queries, and navigates knowledge graphs across iterative multi-turn dialogues.",
            "Community-based Graph RAG (such as Microsoft GraphRAG) partitions knowledge graphs using hierarchical clustering algorithms (e.g., the Leiden algorithm) and generates pre-computed executive summaries for each community, enabling macro-level corpus comprehension.",
        ],
    },
    {
        "doc_id": "doc21_cross_document_entity_networks",
        "title": "Cross-Document Knowledge Networks and Evidence Provenance",
        "filename": "doc21_cross_document_entity_networks.md",
        "author": "Dr. Sarah Jenkins",
        "url": "https://ragbench.internal/docs/21-cross-document-networks",
        "sections": [
            "In large document repositories, individual source files rarely contain all necessary information in isolation. Cross-document knowledge networks bridge siloed document collections by identifying shared entities and linking disjoint assertions.",
            "Evidence provenance records the complete chain of custody for every extracted relationship edge. In a robust benchmark system, each graph edge maintains explicit metadata fields: source_document_id, source_filename, source_chunk_id, and verbatim_evidence_quote.",
            "When conflicting assertions occur across disparate sources (e.g., Document X claims a parameter is 60 while Document Y claims it is 100), the knowledge graph preserves both assertions as parallel edges with distinct document citations, rather than overwriting historical evidence.",
            "Downstream answer generation prompts format multi-source evidence blocks with clear source delimiters (e.g., [S1], [S2]), enabling LLMs to identify disagreements, cite both perspectives, and communicate epistemic uncertainty.",
            "Provenance tracking ensures complete auditability: every claim in a generated answer can be traced backward through graph edges and vector candidates directly to the originating line and byte offsets in the source text.",
        ],
    },
    {
        "doc_id": "doc22_case_study_biomedical_graph_rag",
        "title": "Domain Applications: Biomedical and Legal Graph RAG Deployments",
        "filename": "doc22_case_study_biomedical_graph_rag.md",
        "author": "Maya Lin",
        "url": "https://ragbench.internal/docs/22-domain-applications",
        "sections": [
            "Specialized enterprise domains such as biomedicine and jurisprudence present distinct challenges that highlight the superiority of Graph RAG and Hybrid RAG over plain vector search.",
            "In biomedical research, scientific queries frequently involve intricate multi-hop causal chains: Gene A regulates Protein B, which is targeted by Drug C, which is contraindicated in Disease D. Vector search fails to bridge this 4-step chain because no single publication contains all four entities simultaneously.",
            "By constructing a biomedical knowledge graph linking PubMed abstracts, clinical trial registries, and drug ontologies, Graph RAG executes multi-hop traversals that uncover hidden pharmacological interactions and hypothesize novel drug repurposing candidates.",
            "In legal compliance, regulatory frameworks cross-reference statutory definitions, judicial precedents, and administrative amendments across hundreds of legal codes. Graph RAG preserves statutory hierarchy and precedent citation networks, ensuring strict attribution.",
            "Empirical deployments confirm that in high-stakes domains where hallucination carries severe regulatory or clinical consequences, structured graph grounding achieves significantly higher stakeholder trust than pure probabilistic vector retrieval.",
        ],
    },
]

print(f"[+] Defined {len(SOURCE_DOCUMENTS)} technical benchmark source documents.")

# -----------------------------------------------------------------------------
# 2. Knowledge Graph Entities and Relationships Specification
# -----------------------------------------------------------------------------

BENCHMARK_ENTITIES: list[dict] = [
    {
        "name": "Vector Embeddings",
        "type": "TECHNOLOGY",
        "aliases": ["Dense Representations", "Vector Representations"],
    },
    {"name": "nomic-embed-text", "type": "SOFTWARE", "aliases": ["nomic-embed"]},
    {"name": "Cosine Distance", "type": "METHOD", "aliases": ["Cosine Proximity", "Cosine Metric"]},
    {"name": "Euclidean Distance", "type": "METHOD", "aliases": ["L2 Distance"]},
    {"name": "Dot Product", "type": "METHOD", "aliases": ["Inner Product", "Scalar Product"]},
    {
        "name": "HNSW",
        "type": "SOFTWARE",
        "aliases": ["Hierarchical Navigable Small World", "HNSW Graph"],
    },
    {"name": "IVF", "type": "SOFTWARE", "aliases": ["Inverted File Index"]},
    {"name": "Product Quantization", "type": "METHOD", "aliases": ["PQ", "Scalar Quantization"]},
    {
        "name": "Vector RAG",
        "type": "CONCEPT",
        "aliases": ["Dense Vector RAG", "Traditional Vector RAG"],
    },
    {
        "name": "Knowledge Graph",
        "type": "CONCEPT",
        "aliases": ["Structured Graph", "Relational Graph"],
    },
    {"name": "Entity Extraction", "type": "METHOD", "aliases": ["Named Entity Recognition", "NER"]},
    {
        "name": "Entity Resolution",
        "type": "METHOD",
        "aliases": ["Entity Canonicalization", "Alias Merging"],
    },
    {
        "name": "Graph Traversal",
        "type": "METHOD",
        "aliases": ["Multi-Hop Traversal", "Path Reasoning"],
    },
    {"name": "Breadth-First Search", "type": "METHOD", "aliases": ["BFS"]},
    {"name": "Degree Pruning", "type": "METHOD", "aliases": ["Degree-Based Filtering"]},
    {"name": "Hop Decay", "type": "METHOD", "aliases": ["Exponential Hop Decay"]},
    {"name": "Hybrid RAG", "type": "CONCEPT", "aliases": ["Fused RAG", "Multi-Modal RAG"]},
    {"name": "Reciprocal Rank Fusion", "type": "METHOD", "aliases": ["RRF", "Rank Aggregation"]},
    {"name": "Precision@K", "type": "METHOD", "aliases": ["P@K"]},
    {"name": "Recall@K", "type": "METHOD", "aliases": ["R@K"]},
    {"name": "Hit Rate@K", "type": "METHOD", "aliases": ["Hit Rate"]},
    {"name": "Mean Reciprocal Rank", "type": "METHOD", "aliases": ["MRR"]},
    {
        "name": "nDCG@K",
        "type": "METHOD",
        "aliases": ["Normalized Discounted Cumulative Gain", "NDCG"],
    },
    {"name": "Faithfulness", "type": "METHOD", "aliases": ["Answer Faithfulness", "Groundedness"]},
    {"name": "Citation Precision", "type": "METHOD", "aliases": ["Citation Reference Precision"]},
    {
        "name": "Safety Guardrails",
        "type": "SOFTWARE",
        "aliases": ["Answer Guardrails", "Refusal Handler"],
    },
    {"name": "ChromaDB", "type": "SOFTWARE", "aliases": ["Chroma Vector Store"]},
    {"name": "Milvus", "type": "SOFTWARE", "aliases": ["Milvus Vector DB"]},
    {"name": "Qdrant", "type": "SOFTWARE", "aliases": ["Qdrant Rust Engine"]},
    {"name": "NetworkX", "type": "SOFTWARE", "aliases": ["NetworkX Graph Library"]},
    {"name": "Neo4j", "type": "SOFTWARE", "aliases": ["Neo4j Graph Database"]},
    {"name": "Memgraph", "type": "SOFTWARE", "aliases": ["Memgraph In-Memory"]},
    {"name": "Tree of Thoughts", "type": "CONCEPT", "aliases": ["ToT"]},
    {"name": "Graph of Thoughts", "type": "CONCEPT", "aliases": ["GoT"]},
    {"name": "Agentic RAG", "type": "CONCEPT", "aliases": ["Autonomous RAG"]},
    {"name": "Biomedical Graph RAG", "type": "CONCEPT", "aliases": ["Biomedical Knowledge RAG"]},
]

BENCHMARK_RELATIONSHIPS: list[dict] = [
    {
        "source": "nomic-embed-text",
        "predicate": "PRODUCES",
        "target": "Vector Embeddings",
        "doc_id": "doc01_vector_embeddings",
        "chunk_idx": 1,
        "quote": "The nomic-embed-text architecture utilizes 768 dimensions with dynamic sequence truncation at 8192 tokens.",
        "confidence": 0.98,
    },
    {
        "source": "Vector RAG",
        "predicate": "UTILIZES",
        "target": "Vector Embeddings",
        "doc_id": "doc01_vector_embeddings",
        "chunk_idx": 0,
        "quote": "Dense vector embeddings map unstructured natural language tokens into continuous geometric vector spaces.",
        "confidence": 0.95,
    },
    {
        "source": "Vector RAG",
        "predicate": "EVALUATES_DISTANCE_VIA",
        "target": "Cosine Distance",
        "doc_id": "doc02_vector_similarity_metrics",
        "chunk_idx": 1,
        "quote": "Cosine distance is defined mathematically as d_cos(u, v) = 1.0 - (u . v) / (||u|| * ||v||), bounded within the interval [0.0, 2.0].",
        "confidence": 0.96,
    },
    {
        "source": "Cosine Distance",
        "predicate": "MATHEMATICALLY_CONVERTS_TO",
        "target": "Euclidean Distance",
        "doc_id": "doc02_vector_similarity_metrics",
        "chunk_idx": 2,
        "quote": "For unit-normalized vectors, Euclidean distance relates directly to cosine similarity via the algebraic identity d_L2^2 = 2 * (1 - s_raw).",
        "confidence": 0.94,
    },
    {
        "source": "ChromaDB",
        "predicate": "IMPLEMENTS_INDEX",
        "target": "HNSW",
        "doc_id": "doc03_ann_indexing_hnsw_ivf",
        "chunk_idx": 1,
        "quote": "Hierarchical Navigable Small World (HNSW) graphs represent the current state-of-the-art in vector indexing.",
        "confidence": 0.97,
    },
    {
        "source": "HNSW",
        "predicate": "COMPARES_FAVORABLY_TO",
        "target": "IVF",
        "doc_id": "doc03_ann_indexing_hnsw_ivf",
        "chunk_idx": 3,
        "quote": "While IVF requires less memory than HNSW, it suffers from boundary quantization errors where true nearest neighbors fall across cluster borders.",
        "confidence": 0.92,
    },
    {
        "source": "Vector RAG",
        "predicate": "EXHIBITS_LIMITATION",
        "target": "Knowledge Graph",
        "doc_id": "doc04_vector_rag_limitations",
        "chunk_idx": 1,
        "quote": "When answering a question requiring evidence distributed across multiple disjoint documents, Vector RAG often retrieves only the passage matching the initial query tokens.",
        "confidence": 0.95,
    },
    {
        "source": "Knowledge Graph",
        "predicate": "REPRESENTS_ENTITIES_VIA",
        "target": "Entity Extraction",
        "doc_id": "doc05_knowledge_graph_foundations",
        "chunk_idx": 1,
        "quote": "Each entity possesses a unique canonical identifier, an entity type taxonomy, and surface alias synonyms.",
        "confidence": 0.96,
    },
    {
        "source": "Entity Extraction",
        "predicate": "FEEDS_INTO",
        "target": "Entity Resolution",
        "doc_id": "doc07_entity_resolution_linking",
        "chunk_idx": 0,
        "quote": "A primary challenge in multi-document graph construction is entity resolution: determining whether different surface mentions refer to the same underlying real-world entity.",
        "confidence": 0.94,
    },
    {
        "source": "Knowledge Graph",
        "predicate": "NAVIGATED_BY",
        "target": "Graph Traversal",
        "doc_id": "doc08_graph_traversal_algorithms",
        "chunk_idx": 0,
        "quote": "Graph retrieval operates by exploring relational pathways radiating from pre-resolved seed entities.",
        "confidence": 0.97,
    },
    {
        "source": "Graph Traversal",
        "predicate": "USES_ALGORITHM",
        "target": "Breadth-First Search",
        "doc_id": "doc08_graph_traversal_algorithms",
        "chunk_idx": 0,
        "quote": "The primary traversal algorithm utilized in Graph RAG is bounded Breadth-First Search (BFS).",
        "confidence": 0.98,
    },
    {
        "source": "Graph Traversal",
        "predicate": "CONSTRAINED_BY",
        "target": "Degree Pruning",
        "doc_id": "doc09_graph_pruning_and_ranking",
        "chunk_idx": 1,
        "quote": "Degree-based pruning truncates neighbor exploration for hub nodes whose degree exceeds max_degree.",
        "confidence": 0.95,
    },
    {
        "source": "Graph Traversal",
        "predicate": "SCORED_BY",
        "target": "Hop Decay",
        "doc_id": "doc09_graph_pruning_and_ranking",
        "chunk_idx": 2,
        "quote": "The composite path score is formulated as s_path = prod(confidence_e for e in path) * (gamma ^ hop_count).",
        "confidence": 0.96,
    },
    {
        "source": "Hybrid RAG",
        "predicate": "COMBINES",
        "target": "Vector RAG",
        "doc_id": "doc10_hybrid_retrieval_architectures",
        "chunk_idx": 0,
        "quote": "Hybrid RAG represents a multi-modal retrieval paradigm that concurrently executes dense semantic vector search and structured knowledge graph traversal.",
        "confidence": 0.99,
    },
    {
        "source": "Hybrid RAG",
        "predicate": "COMBINES",
        "target": "Knowledge Graph",
        "doc_id": "doc10_hybrid_retrieval_architectures",
        "chunk_idx": 0,
        "quote": "Hybrid RAG represents a multi-modal retrieval paradigm that concurrently executes dense semantic vector search and structured knowledge graph traversal.",
        "confidence": 0.99,
    },
    {
        "source": "Hybrid RAG",
        "predicate": "FUSES_CANDIDATES_WITH",
        "target": "Reciprocal Rank Fusion",
        "doc_id": "doc11_reciprocal_rank_fusion",
        "chunk_idx": 0,
        "quote": "Reciprocal Rank Fusion (RRF) is an unsupervised rank aggregation algorithm designed to combine candidate lists from heterogeneous retrieval systems.",
        "confidence": 0.98,
    },
    {
        "source": "Hybrid RAG",
        "predicate": "EVALUATED_BY",
        "target": "Precision@K",
        "doc_id": "doc12_evaluation_metrics_ir",
        "chunk_idx": 0,
        "quote": "Evaluating retrieval systems requires standardized Information Retrieval (IR) metrics calculated across fixed rank cutoffs K in {1, 3, 5, 10}.",
        "confidence": 0.95,
    },
    {
        "source": "Hybrid RAG",
        "predicate": "EVALUATED_BY",
        "target": "nDCG@K",
        "doc_id": "doc13_ndcg_graded_relevance",
        "chunk_idx": 0,
        "quote": "Phase 8 employs a four-point graded scale: 0 (Not Relevant), 1 (Related / Partial Context), 2 (Relevant Supporting Context), and 3 (Highly Relevant Direct Answer).",
        "confidence": 0.95,
    },
    {
        "source": "Hybrid RAG",
        "predicate": "VERIFIED_FOR",
        "target": "Faithfulness",
        "doc_id": "doc14_generation_faithfulness_grounding",
        "chunk_idx": 1,
        "quote": "Faithfulness evaluates whether factual assertions in the generated response are semantically entailed by the retrieved context.",
        "confidence": 0.96,
    },
    {
        "source": "Hybrid RAG",
        "predicate": "PROTECTED_BY",
        "target": "Safety Guardrails",
        "doc_id": "doc15_guardrails_unanswerable_queries",
        "chunk_idx": 1,
        "quote": "Pre-generation guardrails inspect retrieved evidence candidates before invoking the LLM generation model.",
        "confidence": 0.97,
    },
    {
        "source": "ChromaDB",
        "predicate": "COMPETES_WITH",
        "target": "Milvus",
        "doc_id": "doc17_vector_database_engines",
        "chunk_idx": 0,
        "quote": "Prominent production vector databases include ChromaDB, Milvus, Qdrant, and pgvector.",
        "confidence": 0.93,
    },
    {
        "source": "NetworkX",
        "predicate": "COMPARED_TO",
        "target": "Neo4j",
        "doc_id": "doc18_graph_database_engines",
        "chunk_idx": 0,
        "quote": "Key engines evaluated in Graph RAG architectures include NetworkX, Neo4j, Memgraph, and Amazon Neptune.",
        "confidence": 0.92,
    },
    {
        "source": "Tree of Thoughts",
        "predicate": "GENERALIZES_TO",
        "target": "Graph of Thoughts",
        "doc_id": "doc20_advanced_rag_paradigms",
        "chunk_idx": 2,
        "quote": "Graph of Thoughts (GoT) extends tree reasoning to arbitrary directed graphs, enabling novel operations such as merging multiple independent thought branches.",
        "confidence": 0.94,
    },
    {
        "source": "Hybrid RAG",
        "predicate": "DEPLOYED_IN",
        "target": "Biomedical Graph RAG",
        "doc_id": "doc22_case_study_biomedical_graph_rag",
        "chunk_idx": 1,
        "quote": "In biomedical research, scientific queries frequently involve intricate multi-hop causal chains.",
        "confidence": 0.95,
    },
]

print(
    f"[+] Defined {len(BENCHMARK_ENTITIES)} entities and {len(BENCHMARK_RELATIONSHIPS)} relationships."
)

# -----------------------------------------------------------------------------
# 3. Benchmark Queries Specification (60 Manually Validated Queries)
# -----------------------------------------------------------------------------

BENCHMARK_QUERIES_SPEC: list[dict] = [
    # -------------------------------------------------------------------------
    # Category 1: Single-Hop Factual Questions (Q01 - Q15)
    # -------------------------------------------------------------------------
    {
        "query_id": "Q01",
        "question": "What is the vector dimensionality and maximum sequence length of the nomic-embed-text model?",
        "category": BenchmarkQueryCategory.SINGLE_HOP,
        "split": BenchmarkSplit.DEV,
        "answerable": True,
        "expected_answer": "nomic-embed-text utilizes 768 dimensions with dynamic sequence truncation up to 8192 tokens.",
        "gold_chunks": [("doc01_vector_embeddings", 1, 3)],
        "distractors": [
            ("doc01_vector_embeddings", 0, 1),
            ("doc02_vector_similarity_metrics", 1, 0),
        ],
        "rationale": "Directly answered in doc01 section 1.",
    },
    {
        "query_id": "Q02",
        "question": "What is the mathematical definition and bounded interval of Cosine distance?",
        "category": BenchmarkQueryCategory.SINGLE_HOP,
        "split": BenchmarkSplit.DEV,
        "answerable": True,
        "expected_answer": "Cosine distance is defined as d_cos(u, v) = 1.0 - (u . v) / (||u|| * ||v||), bounded within the interval [0.0, 2.0].",
        "gold_chunks": [("doc02_vector_similarity_metrics", 1, 3)],
        "distractors": [
            ("doc02_vector_similarity_metrics", 2, 1),
            ("doc01_vector_embeddings", 2, 0),
        ],
        "rationale": "Directly stated in doc02 section 1.",
    },
    {
        "query_id": "Q03",
        "question": "What parameters control graph construction and search beam width in HNSW vector indexing?",
        "category": BenchmarkQueryCategory.SINGLE_HOP,
        "split": BenchmarkSplit.DEV,
        "answerable": True,
        "expected_answer": "HNSW index construction is governed by parameter M (maximum bidirectional connections per node) and efConstruction (search beam width during construction).",
        "gold_chunks": [("doc03_ann_indexing_hnsw_ivf", 1, 3)],
        "distractors": [("doc03_ann_indexing_hnsw_ivf", 2, 2)],
        "rationale": "Directly specified in doc03 section 1.",
    },
    {
        "query_id": "Q04",
        "question": "What are the core components that constitute a Knowledge Graph multigraph definition?",
        "category": BenchmarkQueryCategory.SINGLE_HOP,
        "split": BenchmarkSplit.DEV,
        "answerable": True,
        "expected_answer": "A Knowledge Graph is defined as G = (V, E, L_v, L_e), where V represents vertices (entities), E represents directed edges (relationships), and labeling functions assign semantic types and predicates.",
        "gold_chunks": [("doc05_knowledge_graph_foundations", 0, 3)],
        "distractors": [("doc05_knowledge_graph_foundations", 1, 1)],
        "rationale": "Defined in doc05 section 0.",
    },
    {
        "query_id": "Q05",
        "question": "How does entity canonicalization normalize varied surface mention strings?",
        "category": BenchmarkQueryCategory.SINGLE_HOP,
        "split": BenchmarkSplit.VALIDATION,
        "answerable": True,
        "expected_answer": "Entity canonicalization standardizes surface strings through Unicode NFKC normalization, lowercase folding, whitespace collapsing, and punctuation stripping.",
        "gold_chunks": [("doc07_entity_resolution_linking", 1, 3)],
        "distractors": [("doc07_entity_resolution_linking", 2, 1)],
        "rationale": "Stated in doc07 section 1.",
    },
    {
        "query_id": "Q06",
        "question": "What traversal algorithm is primarily utilized for neighborhood expansion in Graph RAG?",
        "category": BenchmarkQueryCategory.SINGLE_HOP,
        "split": BenchmarkSplit.VALIDATION,
        "answerable": True,
        "expected_answer": "Bounded Breadth-First Search (BFS) is the primary traversal algorithm utilized in Graph RAG for concentric neighborhood expansion.",
        "gold_chunks": [("doc08_graph_traversal_algorithms", 0, 3)],
        "distractors": [("doc08_graph_traversal_algorithms", 1, 1)],
        "rationale": "Stated in doc08 section 0.",
    },
    {
        "query_id": "Q07",
        "question": "What mathematical formula defines composite path score with exponential hop decay?",
        "category": BenchmarkQueryCategory.SINGLE_HOP,
        "split": BenchmarkSplit.VALIDATION,
        "answerable": True,
        "expected_answer": "Composite path score is formulated as s_path = prod(confidence_e for e in path) * (gamma ^ hop_count), where gamma is the hop decay factor.",
        "gold_chunks": [("doc09_graph_pruning_and_ranking", 2, 3)],
        "distractors": [("doc09_graph_pruning_and_ranking", 1, 1)],
        "rationale": "Formula provided in doc09 section 2.",
    },
    {
        "query_id": "Q08",
        "question": "What is the standard smoothing constant k in the Reciprocal Rank Fusion baseline algorithm?",
        "category": BenchmarkQueryCategory.SINGLE_HOP,
        "split": BenchmarkSplit.VALIDATION,
        "answerable": True,
        "expected_answer": "In the standard Cormack et al. baseline formulation of Reciprocal Rank Fusion, the smoothing constant k is set to 60.",
        "gold_chunks": [("doc11_reciprocal_rank_fusion", 2, 3)],
        "distractors": [
            ("doc11_reciprocal_rank_fusion", 1, 2),
            ("doc11_reciprocal_rank_fusion", 3, 1),
        ],
        "rationale": "Specified in doc11 section 2.",
    },
    {
        "query_id": "Q09",
        "question": "How is Precision@K calculated when fewer than K candidates are returned by a retrieval system?",
        "category": BenchmarkQueryCategory.SINGLE_HOP,
        "split": BenchmarkSplit.TEST,
        "answerable": True,
        "expected_answer": "By standard IR convention, the denominator is strictly K; if fewer than K items are returned, unretrieved positions count as non-relevant.",
        "gold_chunks": [("doc12_evaluation_metrics_ir", 1, 3)],
        "distractors": [("doc12_evaluation_metrics_ir", 0, 1)],
        "rationale": "Specified in doc12 section 1.",
    },
    {
        "query_id": "Q10",
        "question": "What are the four discrete relevance levels defined in the Phase 8 graded relevance scale?",
        "category": BenchmarkQueryCategory.SINGLE_HOP,
        "split": BenchmarkSplit.TEST,
        "answerable": True,
        "expected_answer": "The four graded relevance levels are 0 (Not Relevant), 1 (Related / Partial Context), 2 (Relevant Supporting Context), and 3 (Highly Relevant Direct Answer).",
        "gold_chunks": [("doc13_ndcg_graded_relevance", 0, 3)],
        "distractors": [("doc13_ndcg_graded_relevance", 1, 1)],
        "rationale": "Specified in doc13 section 0.",
    },
    {
        "query_id": "Q11",
        "question": "How is Faithfulness score computed across extracted atomic factual claims?",
        "category": BenchmarkQueryCategory.SINGLE_HOP,
        "split": BenchmarkSplit.TEST,
        "answerable": True,
        "expected_answer": "Faithfulness score is computed as (count of SUPPORTED claims) / (total evaluated claims) after evaluating atomic claims against source evidence.",
        "gold_chunks": [("doc14_generation_faithfulness_grounding", 1, 3)],
        "distractors": [("doc14_generation_faithfulness_grounding", 2, 1)],
        "rationale": "Specified in doc14 section 1.",
    },
    {
        "query_id": "Q12",
        "question": "What embedded vector database utilizes SQLite for metadata and a C++ hnswlib core?",
        "category": BenchmarkQueryCategory.SINGLE_HOP,
        "split": BenchmarkSplit.TEST,
        "answerable": True,
        "expected_answer": "ChromaDB is the lightweight embedded vector database written in Python with a C++ hnswlib core and SQLite metadata store.",
        "gold_chunks": [("doc17_vector_database_engines", 1, 3)],
        "distractors": [("doc17_vector_database_engines", 2, 0)],
        "rationale": "Specified in doc17 section 1.",
    },
    {
        "query_id": "Q13",
        "question": "What in-memory pure Python library is used for deterministic graph storage and traversal?",
        "category": BenchmarkQueryCategory.SINGLE_HOP,
        "split": BenchmarkSplit.TEST,
        "answerable": True,
        "expected_answer": "NetworkX is the pure Python library utilized for in-memory graph operations, traversal, and deterministic JSON/GraphML serialization.",
        "gold_chunks": [("doc18_graph_database_engines", 1, 3)],
        "distractors": [("doc18_graph_database_engines", 2, 0)],
        "rationale": "Specified in doc18 section 1.",
    },
    {
        "query_id": "Q14",
        "question": "How does pre-generation guardrail short-circuiting impact end-to-end query latency for unanswerable questions?",
        "category": BenchmarkQueryCategory.SINGLE_HOP,
        "split": BenchmarkSplit.TEST,
        "answerable": True,
        "expected_answer": "Pre-generation short-circuiting reduces latency from approximately 35 seconds of wasted generation down to under 80 milliseconds.",
        "gold_chunks": [("doc15_guardrails_unanswerable_queries", 2, 3)],
        "distractors": [("doc15_guardrails_unanswerable_queries", 1, 2)],
        "rationale": "Specified in doc15 section 2.",
    },
    {
        "query_id": "Q15",
        "question": "What clustering algorithm is used in hierarchical community-based Graph RAG?",
        "category": BenchmarkQueryCategory.SINGLE_HOP,
        "split": BenchmarkSplit.TEST,
        "answerable": True,
        "expected_answer": "Community-based Graph RAG partitions knowledge graphs using hierarchical clustering algorithms such as the Leiden algorithm.",
        "gold_chunks": [("doc20_advanced_rag_paradigms", 4, 3)],
        "distractors": [("doc20_advanced_rag_paradigms", 1, 0)],
        "rationale": "Specified in doc20 section 4.",
    },
    # -------------------------------------------------------------------------
    # Category 2: Multi-Hop Reasoning Questions (Q16 - Q30)
    # -------------------------------------------------------------------------
    {
        "query_id": "Q16",
        "question": "How does vector normalization to unit length enable converting Euclidean distance calculations into cosine similarity?",
        "category": BenchmarkQueryCategory.MULTI_HOP,
        "split": BenchmarkSplit.DEV,
        "answerable": True,
        "expected_answer": "Normalizing vectors to unit L2-norm (||u|| = ||v|| = 1.0) eliminates magnitude variation, simplifying Cosine distance to 1.0 - (u . v) and establishing the algebraic identity d_L2^2 = 2 * (1 - s_raw).",
        "gold_chunks": [
            ("doc01_vector_embeddings", 2, 2),
            ("doc02_vector_similarity_metrics", 1, 3),
            ("doc02_vector_similarity_metrics", 2, 3),
        ],
        "distractors": [("doc01_vector_embeddings", 0, 1)],
        "rationale": "Requires connecting L2-norm from doc01 with Euclidean-cosine algebraic identity in doc02.",
    },
    {
        "query_id": "Q17",
        "question": "Why does multi-hop semantic disconnection in Vector RAG necessitate graph-based bridge traversal across disjoint passages?",
        "category": BenchmarkQueryCategory.MULTI_HOP,
        "split": BenchmarkSplit.DEV,
        "answerable": True,
        "expected_answer": "Vector RAG retrieves passages matching only initial query tokens and fails when intermediate bridge entities (Entity Y linking X to Z) are located in disjoint passages. Graph traversal explicitly navigates through intermediate nodes via BFS to discover these bridging assertions.",
        "gold_chunks": [
            ("doc04_vector_rag_limitations", 1, 3),
            ("doc08_graph_traversal_algorithms", 1, 3),
        ],
        "distractors": [("doc04_vector_rag_limitations", 2, 1)],
        "rationale": "Bridges failure mode in doc04 with 2-hop BFS traversal mechanism in doc08.",
    },
    {
        "query_id": "Q18",
        "question": "How do entity canonicalization and alias merging prevent graph traversal pathway fragmentation across heterogeneous documents?",
        "category": BenchmarkQueryCategory.MULTI_HOP,
        "split": BenchmarkSplit.DEV,
        "answerable": True,
        "expected_answer": "Without resolution, surface variations of the same concept create duplicate disconnected nodes. Canonicalization and alias merging consolidate these mentions into a single ground-truth node, maintaining unbroken traversal pathways across disparate source documents.",
        "gold_chunks": [
            ("doc07_entity_resolution_linking", 0, 3),
            ("doc07_entity_resolution_linking", 2, 3),
            ("doc08_graph_traversal_algorithms", 1, 2),
        ],
        "distractors": [("doc06_entity_extraction_llm", 1, 1)],
        "rationale": "Links resolution mechanisms in doc07 to graph pathway continuity in doc08.",
    },
    {
        "query_id": "Q19",
        "question": "How do degree pruning and exponential hop decay prevent combinatorial explosion during 2-hop and 3-hop graph search?",
        "category": BenchmarkQueryCategory.MULTI_HOP,
        "split": BenchmarkSplit.DEV,
        "answerable": True,
        "expected_answer": "Unconstrained traversal scales as O(b^d). Degree pruning truncates neighbor exploration at high-degree hub nodes, while exponential hop decay gamma^hop penalizes deeper paths, ensuring only high-confidence relevant paths are retained.",
        "gold_chunks": [
            ("doc08_graph_traversal_algorithms", 2, 3),
            ("doc09_graph_pruning_and_ranking", 1, 3),
            ("doc09_graph_pruning_and_ranking", 2, 3),
        ],
        "distractors": [("doc08_graph_traversal_algorithms", 0, 1)],
        "rationale": "Combines traversal complexity O(b^d) from doc08 with pruning and decay formulas in doc09.",
    },
    {
        "query_id": "Q20",
        "question": "How does Reciprocal Rank Fusion resolve the scale divergence between ChromaDB cosine distance and NetworkX path traversal scores?",
        "category": BenchmarkQueryCategory.MULTI_HOP,
        "split": BenchmarkSplit.VALIDATION,
        "answerable": True,
        "expected_answer": "Because RRF operates strictly on 1-based candidate ordinal ranks rather than raw score magnitudes, its scale invariance allows merging vector cosine scores and graph path traversal scores without requiring score normalization.",
        "gold_chunks": [
            ("doc11_reciprocal_rank_fusion", 0, 3),
            ("doc11_reciprocal_rank_fusion", 4, 3),
        ],
        "distractors": [("doc10_hybrid_retrieval_architectures", 2, 1)],
        "rationale": "Connects heterogeneous ranking characteristics with RRF scale invariance.",
    },
    {
        "query_id": "Q21",
        "question": "What causes the 13-second latency disparity between vector search and graph search when evaluating out-of-domain queries?",
        "category": BenchmarkQueryCategory.MULTI_HOP,
        "split": BenchmarkSplit.VALIDATION,
        "answerable": True,
        "expected_answer": "Vector search uses precomputed embeddings taking under 150 ms, whereas graph search executes an LLM entity extraction fallback when dictionary matching fails on out-of-domain terms, taking 8 to 15 seconds of LLM inference before pure BFS traversal executes in under 5 ms.",
        "gold_chunks": [
            ("doc19_computational_latency_tradeoffs", 1, 3),
            ("doc19_computational_latency_tradeoffs", 2, 3),
        ],
        "distractors": [("doc08_graph_traversal_algorithms", 2, 1)],
        "rationale": "Explains latency breakdown between vector and graph query extraction.",
    },
    {
        "query_id": "Q22",
        "question": "How does dual-pass verification in LLM extraction ensure high relationship confidence for downstream graph path ranking?",
        "category": BenchmarkQueryCategory.MULTI_HOP,
        "split": BenchmarkSplit.VALIDATION,
        "answerable": True,
        "expected_answer": "The extraction pass proposes candidate triples with verbatim evidence quotes, and the validation pass verifies exact string support, producing high confidence scores (>= 0.70) that maximize composite path scores s_path = prod(confidence) * gamma^hop.",
        "gold_chunks": [
            ("doc06_entity_extraction_llm", 2, 3),
            ("doc09_graph_pruning_and_ranking", 2, 3),
        ],
        "distractors": [("doc06_entity_extraction_llm", 0, 1)],
        "rationale": "Bridges extraction validation in doc06 with path scoring in doc09.",
    },
    {
        "query_id": "Q23",
        "question": "Why does biomedical research require multi-hop causal graph traversal rather than single-hop vector retrieval?",
        "category": BenchmarkQueryCategory.MULTI_HOP,
        "split": BenchmarkSplit.VALIDATION,
        "answerable": True,
        "expected_answer": "Biomedical queries involve multi-step causal chains (e.g., Gene regulates Protein, targeted by Drug, contraindicated in Disease) distributed across separate publications. Vector search fails because no single paper contains all entities, whereas graph traversal links the full chain.",
        "gold_chunks": [
            ("doc22_case_study_biomedical_graph_rag", 1, 3),
            ("doc22_case_study_biomedical_graph_rag", 2, 3),
            ("doc04_vector_rag_limitations", 1, 2),
        ],
        "distractors": [("doc22_case_study_biomedical_graph_rag", 0, 1)],
        "rationale": "Connects multi-hop failure in doc04 with biomedical chain in doc22.",
    },
    {
        "query_id": "Q24",
        "question": "How does nDCG@K reward retrieval systems that place grade 3 chunks above grade 2 chunks compared to binary Precision@K?",
        "category": BenchmarkQueryCategory.MULTI_HOP,
        "split": BenchmarkSplit.TEST,
        "answerable": True,
        "expected_answer": "Binary Precision@K treats grade 2 and grade 3 chunks identically as relevant hits, whereas nDCG@K uses exponential gain (2^rel - 1), yielding gain 7 for grade 3 versus gain 3 for grade 2, and discounts deeper ranks logarithmically.",
        "gold_chunks": [
            ("doc12_evaluation_metrics_ir", 1, 2),
            ("doc13_ndcg_graded_relevance", 1, 3),
            ("doc13_ndcg_graded_relevance", 3, 3),
        ],
        "distractors": [("doc13_ndcg_graded_relevance", 0, 1)],
        "rationale": "Compares binary precision in doc12 with graded exponential gain in doc13.",
    },
    {
        "query_id": "Q25",
        "question": "How do pre-generation relevance guardrails and post-generation refusal phrase detectors collaborate to prevent ungrounded answers?",
        "category": BenchmarkQueryCategory.MULTI_HOP,
        "split": BenchmarkSplit.TEST,
        "answerable": True,
        "expected_answer": "Pre-generation guardrails short-circuit generation when candidate relevance scores fall below thresholds (vector < 0.50, graph < 0.40). If generation occurs, post-generation guardrails scan for refusal phrases to reclassify the status as INSUFFICIENT_EVIDENCE and reject fabricated citations.",
        "gold_chunks": [
            ("doc15_guardrails_unanswerable_queries", 1, 3),
            ("doc15_guardrails_unanswerable_queries", 3, 3),
        ],
        "distractors": [("doc14_generation_faithfulness_grounding", 2, 1)],
        "rationale": "Synthesizes dual-stage guardrail defense in doc15.",
    },
    {
        "query_id": "Q26",
        "question": "What is the structural relationship between Tree of Thoughts, Graph of Thoughts, and multi-hop graph retrieval?",
        "category": BenchmarkQueryCategory.MULTI_HOP,
        "split": BenchmarkSplit.TEST,
        "answerable": True,
        "expected_answer": "Tree of Thoughts structures linear reasoning into branched search trees. Graph of Thoughts generalizes this to directed graphs enabling merging and feedback loops, paralleling multi-hop graph retrieval which traverses and aggregates multiple reasoning paths across a knowledge graph.",
        "gold_chunks": [
            ("doc20_advanced_rag_paradigms", 1, 3),
            ("doc20_advanced_rag_paradigms", 2, 3),
            ("doc08_graph_traversal_algorithms", 0, 2),
        ],
        "distractors": [("doc20_advanced_rag_paradigms", 3, 1)],
        "rationale": "Synthesizes ToT, GoT, and graph traversal topologies.",
    },
    {
        "query_id": "Q27",
        "question": "How does cross-document provenance tracking enable resolving conflicting assertions between heterogeneous sources in Graph RAG?",
        "category": BenchmarkQueryCategory.MULTI_HOP,
        "split": BenchmarkSplit.TEST,
        "answerable": True,
        "expected_answer": "Provenance tracking preserves conflicting assertions as parallel edges with distinct source document IDs and verbatim quotes, allowing generation prompts to present delimited citations ([S1], [S2]) so the LLM can acknowledge disagreement rather than overwriting historical evidence.",
        "gold_chunks": [
            ("doc21_cross_document_entity_networks", 1, 3),
            ("doc21_cross_document_entity_networks", 2, 3),
            ("doc21_cross_document_entity_networks", 3, 3),
        ],
        "distractors": [("doc21_cross_document_entity_networks", 0, 1)],
        "rationale": "Explains multi-source contradiction handling in doc21.",
    },
    {
        "query_id": "Q28",
        "question": "How do canonical chunk ID harmonization requirements connect ChromaDB vector candidates with NetworkX graph traversal results in Hybrid RAG?",
        "category": BenchmarkQueryCategory.MULTI_HOP,
        "split": BenchmarkSplit.TEST,
        "answerable": True,
        "expected_answer": "In Hybrid RAG, both vector store records and knowledge graph relationship citations must reference identical canonical chunk IDs from a shared repository, allowing RRF to correctly fuse candidates and downstream pipelines to package coherent evidence blocks.",
        "gold_chunks": [
            ("doc10_hybrid_retrieval_architectures", 3, 3),
            ("doc11_reciprocal_rank_fusion", 1, 2),
        ],
        "distractors": [("doc10_hybrid_retrieval_architectures", 0, 1)],
        "rationale": "Explains chunk harmonization between vector and graph systems.",
    },
    {
        "query_id": "Q29",
        "question": "Why does atomic claim extraction provide a more rigorous faithfulness evaluation than whole-answer lexical overlap?",
        "category": BenchmarkQueryCategory.MULTI_HOP,
        "split": BenchmarkSplit.TEST,
        "answerable": True,
        "expected_answer": "Whole-answer lexical overlap can achieve high scores through superficial keyword matching even when core claims are false. Decomposing text into atomic claims isolates individual factual statements, verifying whether each specific proposition is entailed by evidence.",
        "gold_chunks": [
            ("doc14_generation_faithfulness_grounding", 1, 3),
            ("doc14_generation_faithfulness_grounding", 4, 3),
        ],
        "distractors": [("doc14_generation_faithfulness_grounding", 0, 1)],
        "rationale": "Contrasts atomic claim verification with lexical overlap.",
    },
    {
        "query_id": "Q30",
        "question": "How does Product Quantization reduce vector database RAM usage, and what trade-off does it introduce in nearest neighbor recall?",
        "category": BenchmarkQueryCategory.MULTI_HOP,
        "split": BenchmarkSplit.TEST,
        "answerable": True,
        "expected_answer": "Product Quantization compresses high-dimensional floating-point vectors into compact discrete byte codes, reducing RAM utilization by up to 75% at the cost of minor distance approximation noise that slightly degrades search recall.",
        "gold_chunks": [("doc03_ann_indexing_hnsw_ivf", 4, 3)],
        "distractors": [
            ("doc03_ann_indexing_hnsw_ivf", 1, 1),
            ("doc17_vector_database_engines", 2, 1),
        ],
        "rationale": "Connects PQ compression in doc03 with recall trade-off.",
    },
    # -------------------------------------------------------------------------
    # Category 3: Entity & Relationship Questions (Q31 - Q40)
    # -------------------------------------------------------------------------
    {
        "query_id": "Q31",
        "question": "What is the relationship between ChromaDB and the HNSW indexing algorithm?",
        "category": BenchmarkQueryCategory.ENTITY_RELATIONSHIP,
        "split": BenchmarkSplit.DEV,
        "answerable": True,
        "expected_answer": "ChromaDB implements the HNSW indexing algorithm via its underlying C++ hnswlib core to perform approximate nearest neighbor similarity search.",
        "gold_chunks": [
            ("doc03_ann_indexing_hnsw_ivf", 1, 2),
            ("doc17_vector_database_engines", 1, 3),
        ],
        "distractors": [("doc17_vector_database_engines", 0, 1)],
        "rationale": "Tests the explicit relationship ChromaDB -> IMPLEMENTS_INDEX -> HNSW.",
    },
    {
        "query_id": "Q32",
        "question": "What retrieval components does Hybrid RAG combine to execute search?",
        "category": BenchmarkQueryCategory.ENTITY_RELATIONSHIP,
        "split": BenchmarkSplit.DEV,
        "answerable": True,
        "expected_answer": "Hybrid RAG combines dense semantic vector retrieval (Vector RAG) with structured knowledge graph traversal (Knowledge Graph) using Reciprocal Rank Fusion.",
        "gold_chunks": [
            ("doc10_hybrid_retrieval_architectures", 0, 3),
            ("doc11_reciprocal_rank_fusion", 0, 2),
        ],
        "distractors": [("doc10_hybrid_retrieval_architectures", 1, 1)],
        "rationale": "Tests Hybrid RAG -> COMBINES -> Vector RAG and Knowledge Graph.",
    },
    {
        "query_id": "Q33",
        "question": "What algorithm connects Graph Traversal to concentric neighborhood expansion in Graph RAG?",
        "category": BenchmarkQueryCategory.ENTITY_RELATIONSHIP,
        "split": BenchmarkSplit.DEV,
        "answerable": True,
        "expected_answer": "Graph Traversal utilizes bounded Breadth-First Search (BFS) to systematically expand concentric rings of neighbor entities radiating from seed nodes.",
        "gold_chunks": [("doc08_graph_traversal_algorithms", 0, 3)],
        "distractors": [("doc08_graph_traversal_algorithms", 2, 1)],
        "rationale": "Tests Graph Traversal -> USES_ALGORITHM -> BFS.",
    },
    {
        "query_id": "Q34",
        "question": "How is Graph Traversal path scoring constrained and penalized by Hop Decay?",
        "category": BenchmarkQueryCategory.ENTITY_RELATIONSHIP,
        "split": BenchmarkSplit.VALIDATION,
        "answerable": True,
        "expected_answer": "Graph Traversal paths are scored by multiplying edge confidences with an exponential hop decay factor gamma^hop, penalizing longer paths.",
        "gold_chunks": [("doc09_graph_pruning_and_ranking", 2, 3)],
        "distractors": [("doc09_graph_pruning_and_ranking", 1, 1)],
        "rationale": "Tests Graph Traversal -> SCORED_BY -> Hop Decay.",
    },
    {
        "query_id": "Q35",
        "question": "What role does Entity Extraction play in feeding Entity Resolution during graph construction?",
        "category": BenchmarkQueryCategory.ENTITY_RELATIONSHIP,
        "split": BenchmarkSplit.VALIDATION,
        "answerable": True,
        "expected_answer": "Entity Extraction identifies candidate entities and mentions from text, which directly feeds into Entity Resolution to resolve canonical names and merge aliases.",
        "gold_chunks": [
            ("doc05_knowledge_graph_foundations", 1, 2),
            ("doc07_entity_resolution_linking", 0, 3),
        ],
        "distractors": [("doc06_entity_extraction_llm", 0, 1)],
        "rationale": "Tests Entity Extraction -> FEEDS_INTO -> Entity Resolution.",
    },
    {
        "query_id": "Q36",
        "question": "What is the relationship between Tree of Thoughts and Graph of Thoughts in advanced reasoning?",
        "category": BenchmarkQueryCategory.ENTITY_RELATIONSHIP,
        "split": BenchmarkSplit.VALIDATION,
        "answerable": True,
        "expected_answer": "Graph of Thoughts generalizes Tree of Thoughts by expanding tree reasoning into arbitrary directed graphs with branch merging and recurrent feedback.",
        "gold_chunks": [
            ("doc20_advanced_rag_paradigms", 1, 2),
            ("doc20_advanced_rag_paradigms", 2, 3),
        ],
        "distractors": [("doc20_advanced_rag_paradigms", 0, 1)],
        "rationale": "Tests Tree of Thoughts -> GENERALIZES_TO -> Graph of Thoughts.",
    },
    {
        "query_id": "Q37",
        "question": "What vector embedding model produces 768-dimensional embeddings in the benchmark architecture?",
        "category": BenchmarkQueryCategory.ENTITY_RELATIONSHIP,
        "split": BenchmarkSplit.TEST,
        "answerable": True,
        "expected_answer": "nomic-embed-text is the embedding model that produces 768-dimensional dense vector embeddings with an 8192 token context window.",
        "gold_chunks": [("doc01_vector_embeddings", 1, 3)],
        "distractors": [("doc01_vector_embeddings", 0, 1)],
        "rationale": "Tests nomic-embed-text -> PRODUCES -> Vector Embeddings.",
    },
    {
        "query_id": "Q38",
        "question": "How do Safety Guardrails protect Hybrid RAG systems against ungrounded generation?",
        "category": BenchmarkQueryCategory.ENTITY_RELATIONSHIP,
        "split": BenchmarkSplit.TEST,
        "answerable": True,
        "expected_answer": "Safety Guardrails protect Hybrid RAG by pre-checking retrieval candidate relevance thresholds and post-checking refusal phrases to prevent hallucinated answers.",
        "gold_chunks": [
            ("doc15_guardrails_unanswerable_queries", 1, 3),
            ("doc15_guardrails_unanswerable_queries", 3, 2),
        ],
        "distractors": [("doc10_hybrid_retrieval_architectures", 0, 1)],
        "rationale": "Tests Hybrid RAG -> PROTECTED_BY -> Safety Guardrails.",
    },
    {
        "query_id": "Q39",
        "question": "What relationship exists between Neo4j and the Cypher query language?",
        "category": BenchmarkQueryCategory.ENTITY_RELATIONSHIP,
        "split": BenchmarkSplit.TEST,
        "answerable": True,
        "expected_answer": "Neo4j is the native property graph database that utilizes the declarative Cypher query language and disk-optimized pointer-chasing structures.",
        "gold_chunks": [("doc18_graph_database_engines", 2, 3)],
        "distractors": [("doc18_graph_database_engines", 1, 0)],
        "rationale": "Tests Neo4j relationship to Cypher.",
    },
    {
        "query_id": "Q40",
        "question": "How is Biomedical Graph RAG deployed to address multi-hop scientific reasoning?",
        "category": BenchmarkQueryCategory.ENTITY_RELATIONSHIP,
        "split": BenchmarkSplit.TEST,
        "answerable": True,
        "expected_answer": "Biomedical Graph RAG is deployed across PubMed and clinical registries to navigate multi-hop causal chains (Gene -> Protein -> Drug -> Disease) that span disjoint publications.",
        "gold_chunks": [
            ("doc22_case_study_biomedical_graph_rag", 1, 3),
            ("doc22_case_study_biomedical_graph_rag", 2, 3),
        ],
        "distractors": [("doc22_case_study_biomedical_graph_rag", 0, 1)],
        "rationale": "Tests Hybrid RAG -> DEPLOYED_IN -> Biomedical Graph RAG.",
    },
    # -------------------------------------------------------------------------
    # Category 4: Comparison Questions (Q41 - Q50)
    # -------------------------------------------------------------------------
    {
        "query_id": "Q41",
        "question": "How does HNSW compare to IVF in terms of search recall, memory footprint, and boundary quantization?",
        "category": BenchmarkQueryCategory.COMPARISON,
        "split": BenchmarkSplit.DEV,
        "answerable": True,
        "expected_answer": "HNSW delivers superior nearest-neighbor recall and query throughput without boundary errors but requires more RAM for graph layers. IVF uses less memory by clustering vectors into Voronoi cells but suffers from boundary quantization errors across clusters.",
        "gold_chunks": [
            ("doc03_ann_indexing_hnsw_ivf", 1, 2),
            ("doc03_ann_indexing_hnsw_ivf", 2, 2),
            ("doc03_ann_indexing_hnsw_ivf", 3, 3),
        ],
        "distractors": [("doc03_ann_indexing_hnsw_ivf", 0, 1)],
        "rationale": "Compares HNSW and IVF architectures directly.",
    },
    {
        "query_id": "Q42",
        "question": "Compare Cosine distance and Dot Product similarity regarding vector magnitude sensitivity and computational speed.",
        "category": BenchmarkQueryCategory.COMPARISON,
        "split": BenchmarkSplit.DEV,
        "answerable": True,
        "expected_answer": "Dot product offers faster computational throughput by eliminating division and square roots, but is sensitive to vector length. Cosine distance normalizes magnitudes, measuring pure angular orientation, but requires more computation.",
        "gold_chunks": [
            ("doc02_vector_similarity_metrics", 1, 3),
            ("doc02_vector_similarity_metrics", 3, 3),
        ],
        "distractors": [("doc02_vector_similarity_metrics", 0, 1)],
        "rationale": "Direct comparison between Cosine and Dot Product metrics.",
    },
    {
        "query_id": "Q43",
        "question": "Compare the retrieval mechanisms and failure modes of Vector RAG versus Graph RAG.",
        "category": BenchmarkQueryCategory.COMPARISON,
        "split": BenchmarkSplit.VALIDATION,
        "answerable": True,
        "expected_answer": "Vector RAG retrieves via continuous semantic proximity, failing on multi-hop entity bridging and global synthesis. Graph RAG retrieves via explicit relational edge traversals, failing when exact entity mentions are absent or terminology is fuzzy.",
        "gold_chunks": [
            ("doc04_vector_rag_limitations", 0, 3),
            ("doc04_vector_rag_limitations", 1, 3),
            ("doc08_graph_traversal_algorithms", 0, 2),
            ("doc10_hybrid_retrieval_architectures", 1, 3),
        ],
        "distractors": [("doc10_hybrid_retrieval_architectures", 0, 1)],
        "rationale": "Core architectural comparison between Vector and Graph RAG.",
    },
    {
        "query_id": "Q44",
        "question": "Compare Mean Reciprocal Rank (MRR) and Normalized Discounted Cumulative Gain (nDCG) in measuring retrieval quality.",
        "category": BenchmarkQueryCategory.COMPARISON,
        "split": BenchmarkSplit.VALIDATION,
        "answerable": True,
        "expected_answer": "MRR evaluates only the position of the first relevant result using binary relevance. nDCG evaluates the entire ranked list up to cutoff K using graded relevance (0-3) with logarithmic rank discounting.",
        "gold_chunks": [
            ("doc12_evaluation_metrics_ir", 4, 3),
            ("doc13_ndcg_graded_relevance", 1, 3),
            ("doc13_ndcg_graded_relevance", 3, 3),
        ],
        "distractors": [("doc12_evaluation_metrics_ir", 0, 1)],
        "rationale": "Compares binary first-hit MRR with graded multi-hit nDCG.",
    },
    {
        "query_id": "Q45",
        "question": "Compare ChromaDB and Milvus regarding architectural deployment, scalability, and target workloads.",
        "category": BenchmarkQueryCategory.COMPARISON,
        "split": BenchmarkSplit.TEST,
        "answerable": True,
        "expected_answer": "ChromaDB is a lightweight embedded database for local applications and benchmarks. Milvus is a distributed cloud-native database decoupling storage and compute for billion-scale vector workloads with sharding and GPU support.",
        "gold_chunks": [
            ("doc17_vector_database_engines", 1, 3),
            ("doc17_vector_database_engines", 2, 3),
        ],
        "distractors": [("doc17_vector_database_engines", 0, 1)],
        "rationale": "Direct architectural comparison between ChromaDB and Milvus.",
    },
    {
        "query_id": "Q46",
        "question": "Compare NetworkX and Neo4j regarding concurrency, memory architecture, and scale limitations.",
        "category": BenchmarkQueryCategory.COMPARISON,
        "split": BenchmarkSplit.TEST,
        "answerable": True,
        "expected_answer": "NetworkX is a pure Python in-memory graph library optimized for local research and sub-million node graphs. Neo4j is a disk-backed native property graph database supporting enterprise clustering, Cypher queries, and large-scale graph data science.",
        "gold_chunks": [
            ("doc18_graph_database_engines", 1, 3),
            ("doc18_graph_database_engines", 2, 3),
        ],
        "distractors": [("doc18_graph_database_engines", 0, 1)],
        "rationale": "Compares embedded in-memory graph versus enterprise disk graph.",
    },
    {
        "query_id": "Q47",
        "question": "Compare the latency profiles of vector embedding retrieval versus LLM-based graph entity extraction.",
        "category": BenchmarkQueryCategory.COMPARISON,
        "split": BenchmarkSplit.TEST,
        "answerable": True,
        "expected_answer": "Vector retrieval takes under 150 ms (50-120 ms embedding + 2-15 ms ChromaDB search). LLM entity extraction fallback takes 8 to 15 seconds of inference time before sub-5 ms graph traversal executes.",
        "gold_chunks": [
            ("doc19_computational_latency_tradeoffs", 1, 3),
            ("doc19_computational_latency_tradeoffs", 2, 3),
        ],
        "distractors": [("doc19_computational_latency_tradeoffs", 0, 1)],
        "rationale": "Compares latency overhead across retrieval stages.",
    },
    {
        "query_id": "Q48",
        "question": "Compare Citation Reference Precision and Citation Coverage in evaluating grounded RAG answers.",
        "category": BenchmarkQueryCategory.COMPARISON,
        "split": BenchmarkSplit.TEST,
        "answerable": True,
        "expected_answer": "Citation Reference Precision evaluates whether generated citation tags point to valid retrieved chunks. Citation Coverage evaluates what fraction of all gold relevant chunks were actually cited in the answer.",
        "gold_chunks": [
            ("doc14_generation_faithfulness_grounding", 2, 3),
            ("doc14_generation_faithfulness_grounding", 3, 3),
        ],
        "distractors": [("doc14_generation_faithfulness_grounding", 1, 1)],
        "rationale": "Compares structural validity versus gold evidence completeness.",
    },
    {
        "query_id": "Q49",
        "question": "Compare Tree of Thoughts (ToT) and Graph of Thoughts (GoT) in structured reasoning topologies.",
        "category": BenchmarkQueryCategory.COMPARISON,
        "split": BenchmarkSplit.TEST,
        "answerable": True,
        "expected_answer": "Tree of Thoughts organizes reasoning into hierarchical branching trees explored via BFS/DFS. Graph of Thoughts extends this to arbitrary directed graphs, supporting branch merging and cyclical feedback loops.",
        "gold_chunks": [
            ("doc20_advanced_rag_paradigms", 1, 3),
            ("doc20_advanced_rag_paradigms", 2, 3),
        ],
        "distractors": [("doc20_advanced_rag_paradigms", 0, 1)],
        "rationale": "Compares tree vs graph reasoning paradigms.",
    },
    {
        "query_id": "Q50",
        "question": "Compare 1-hop, 2-hop, and 3-hop graph traversals regarding path discovery and combinatorial expansion.",
        "category": BenchmarkQueryCategory.COMPARISON,
        "split": BenchmarkSplit.TEST,
        "answerable": True,
        "expected_answer": "1-hop retrieves only direct neighbors with minimal overhead. 2-hop explores bridge nodes to connect indirect entities. 3-hop discovers distant multi-step dependencies but experiences exponential path expansion O(b^3).",
        "gold_chunks": [
            ("doc08_graph_traversal_algorithms", 1, 3),
            ("doc08_graph_traversal_algorithms", 2, 3),
        ],
        "distractors": [("doc08_graph_traversal_algorithms", 0, 1)],
        "rationale": "Compares traversal depth trade-offs.",
    },
    # -------------------------------------------------------------------------
    # Category 5: Unanswerable / Out-of-Domain Questions (Q51 - Q60)
    # -------------------------------------------------------------------------
    {
        "query_id": "Q51",
        "question": "What was the average snowfall in Zurich during the record winter of January 1985?",
        "category": BenchmarkQueryCategory.UNANSWERABLE,
        "split": BenchmarkSplit.DEV,
        "answerable": False,
        "expected_answer": "Based on the provided evidence, there is insufficient information to answer this question.",
        "gold_chunks": [],
        "distractors": [
            ("doc01_vector_embeddings", 0, 0),
            ("doc02_vector_similarity_metrics", 0, 0),
        ],
        "rationale": "Meteorological data about Zurich in 1985 is completely absent from the technical RAG corpus.",
    },
    {
        "query_id": "Q52",
        "question": "Who was the chief architect of the Sydney Opera House and what acoustic materials were used in 1973?",
        "category": BenchmarkQueryCategory.UNANSWERABLE,
        "split": BenchmarkSplit.DEV,
        "answerable": False,
        "expected_answer": "Based on the provided evidence, there is insufficient information to answer this question.",
        "gold_chunks": [],
        "distractors": [("doc05_knowledge_graph_foundations", 0, 0)],
        "rationale": "Architecture history is absent from the corpus.",
    },
    {
        "query_id": "Q53",
        "question": "What was the trading price of gold per troy ounce during the London Gold Pool collapse of March 1968?",
        "category": BenchmarkQueryCategory.UNANSWERABLE,
        "split": BenchmarkSplit.VALIDATION,
        "answerable": False,
        "expected_answer": "Based on the provided evidence, there is insufficient information to answer this question.",
        "gold_chunks": [],
        "distractors": [("doc11_reciprocal_rank_fusion", 0, 0)],
        "rationale": "Historical financial data is absent from the corpus.",
    },
    {
        "query_id": "Q54",
        "question": "How does quantum key distribution using BB84 protocols prevent eavesdropping in optical fiber networks?",
        "category": BenchmarkQueryCategory.UNANSWERABLE,
        "split": BenchmarkSplit.VALIDATION,
        "answerable": False,
        "expected_answer": "Based on the provided evidence, there is insufficient information to answer this question.",
        "gold_chunks": [],
        "distractors": [("doc01_vector_embeddings", 2, 0)],
        "rationale": "Quantum physics and cryptography are absent from the corpus.",
    },
    {
        "query_id": "Q55",
        "question": "What was the exact passenger capacity and cruising altitude of the Concorde supersonic airliner?",
        "category": BenchmarkQueryCategory.UNANSWERABLE,
        "split": BenchmarkSplit.TEST,
        "answerable": False,
        "expected_answer": "Based on the provided evidence, there is insufficient information to answer this question.",
        "gold_chunks": [],
        "distractors": [("doc19_computational_latency_tradeoffs", 0, 0)],
        "rationale": "Aviation specifications are absent from the corpus.",
    },
    {
        "query_id": "Q56",
        "question": "What are the agricultural soil pH requirements for cultivating arabica coffee in the Ethiopian highlands?",
        "category": BenchmarkQueryCategory.UNANSWERABLE,
        "split": BenchmarkSplit.TEST,
        "answerable": False,
        "expected_answer": "Based on the provided evidence, there is insufficient information to answer this question.",
        "gold_chunks": [],
        "distractors": [("doc22_case_study_biomedical_graph_rag", 0, 0)],
        "rationale": "Agronomy data is absent from the corpus.",
    },
    {
        "query_id": "Q57",
        "question": "What were the primary economic consequences of the Treaty of Utrecht signed in 1713?",
        "category": BenchmarkQueryCategory.UNANSWERABLE,
        "split": BenchmarkSplit.TEST,
        "answerable": False,
        "expected_answer": "Based on the provided evidence, there is insufficient information to answer this question.",
        "gold_chunks": [],
        "distractors": [("doc16_graph_rag_benchmark_framework", 0, 0)],
        "rationale": "18th-century European treaties are absent from the corpus.",
    },
    {
        "query_id": "Q58",
        "question": "How many moons orbit the planet Saturn, and which moon possesses a subsurface liquid water ocean?",
        "category": BenchmarkQueryCategory.UNANSWERABLE,
        "split": BenchmarkSplit.TEST,
        "answerable": False,
        "expected_answer": "Based on the provided evidence, there is insufficient information to answer this question.",
        "gold_chunks": [],
        "distractors": [("doc05_knowledge_graph_foundations", 0, 0)],
        "rationale": "Planetary astronomy is absent from the corpus.",
    },
    {
        "query_id": "Q59",
        "question": "What specific enzyme converts pyruvate into acetyl-CoA in the mitochondrial matrix during cellular respiration?",
        "category": BenchmarkQueryCategory.UNANSWERABLE,
        "split": BenchmarkSplit.TEST,
        "answerable": False,
        "expected_answer": "Based on the provided evidence, there is insufficient information to answer this question.",
        "gold_chunks": [],
        "distractors": [("doc22_case_study_biomedical_graph_rag", 1, 0)],
        "rationale": "Biochemical cellular respiration details are absent from the corpus.",
    },
    {
        "query_id": "Q60",
        "question": "What was the original theatrical running time of Stanley Kubrick's 2001: A Space Odyssey at its 1968 premiere?",
        "category": BenchmarkQueryCategory.UNANSWERABLE,
        "split": BenchmarkSplit.TEST,
        "answerable": False,
        "expected_answer": "Based on the provided evidence, there is insufficient information to answer this question.",
        "gold_chunks": [],
        "distractors": [("doc14_generation_faithfulness_grounding", 0, 0)],
        "rationale": "Cinematic history is absent from the corpus.",
    },
]

print(f"[+] Defined {len(BENCHMARK_QUERIES_SPEC)} benchmark queries across 5 categories.")

# -----------------------------------------------------------------------------
# 4. Build Execution Logic
# -----------------------------------------------------------------------------


def build_benchmark_environment():
    """Execute complete Phase 8 benchmark environment build."""
    print("=" * 80)
    print("      BUILDING PHASE 8 ISOLATED BENCHMARK DATA ENVIRONMENT")
    print("=" * 80)

    # 1. Ensure directories exist
    for d in (RAW_DOCS_DIR, PROCESSED_DIR, CHROMA_DIR, KG_DIR, GROUND_TRUTH_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    print(f"[+] Verified isolated benchmark directory root: {ROOT_BENCH}")

    # 2. Write raw documents
    doc_manifest_records: list[BenchmarkDocument] = []
    chunk_records: list[BenchmarkChunk] = []
    raw_chunk_tuples: list[
        tuple[str, str, str, str, int]
    ] = []  # (doc_id, doc_title, chunk_id, text, idx)

    print("\n[STEP 1] Generating 22 Raw Source Documents & Ingesting Chunks...")
    for doc_spec in SOURCE_DOCUMENTS:
        doc_id = doc_spec["doc_id"]
        title = doc_spec["title"]
        fname = doc_spec["filename"]
        author = doc_spec["author"]
        url = doc_spec["url"]
        sections = doc_spec["sections"]

        # Compose full raw text
        raw_text = f"# {title}\n\n**Author:** {author}\n**Reference:** {url}\n\n"
        for i, sec in enumerate(sections, start=1):
            raw_text += f"## Section {i}\n\n{sec}\n\n"

        file_path = RAW_DOCS_DIR / fname
        file_path.write_text(raw_text, encoding="utf-8")
        sha256 = compute_sha256(raw_text)

        doc_manifest_records.append(
            BenchmarkDocument(
                document_id=doc_id,
                title=title,
                filename=fname,
                file_type="md",
                source_url=url,
                author_or_publisher=author,
                acquisition_date="2026-11-14",
                chunk_count=len(sections),
                sha256_hash=sha256,
            )
        )

        for idx, sec_text in enumerate(sections):
            chunk_id = f"chunk_{doc_id}_{idx:04d}"
            chunk_records.append(
                BenchmarkChunk(
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    document_title=title,
                    chunk_index=idx,
                    content=sec_text.strip(),
                    token_count=int(len(sec_text.split()) * 1.35),
                )
            )
            raw_chunk_tuples.append((doc_id, title, chunk_id, sec_text.strip(), idx))

    print(f"[+] Wrote {len(doc_manifest_records)} raw files into {RAW_DOCS_DIR}")
    print(f"[+] Created {len(chunk_records)} canonical benchmark chunks.")

    # 3. Write processed chunks.jsonl and documents.json
    print("\n[STEP 2] Saving Processed Chunks & Document Manifest...")
    with open(PROCESSED_DIR / "documents.json", "w", encoding="utf-8") as f:
        json.dump([d.model_dump(mode="json") for d in doc_manifest_records], f, indent=2)

    with open(PROCESSED_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
        f.writelines(c.model_dump_json() + "\n" for c in chunk_records)

    processed_manifest = {
        "version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "document_count": len(doc_manifest_records),
        "chunk_count": len(chunk_records),
        "documents_sha256": compute_sha256(
            (PROCESSED_DIR / "documents.json").read_text(encoding="utf-8")
        ),
        "chunks_sha256": compute_sha256(
            (PROCESSED_DIR / "chunks.jsonl").read_text(encoding="utf-8")
        ),
    }
    with open(PROCESSED_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(processed_manifest, f, indent=2)
    print(f"[+] Saved processed chunks and manifest into {PROCESSED_DIR}")

    # 4. Generate Embeddings & Index into ChromaDB
    print("\n[STEP 3] Indexing Vectors into Isolated ChromaDB Collection...")
    client = OllamaClient()
    if not client.is_server_reachable():
        print("[-] ERROR: Ollama server not reachable at http://localhost:11434")
        return 1

    embed_svc = EmbeddingService(client=client)
    vstore = VectorStore(persist_directory=CHROMA_DIR, collection_name=COLLECTION_NAME)

    existing_count = vstore.count()
    if existing_count == len(chunk_records):
        print(
            f"[+] ChromaDB collection '{COLLECTION_NAME}' already contains {existing_count} vectors."
        )
    else:
        vstore.reset_collection()
        print(f"[+] Computing embeddings for {len(chunk_records)} chunks in batches of 20...")
        batch_size = 20
        all_vector_records: list[VectorRecord] = []

        for b_idx in range(0, len(chunk_records), batch_size):
            batch_chunks = chunk_records[b_idx : b_idx + batch_size]
            texts = [c.content for c in batch_chunks]
            res = embed_svc.embed_batch(texts)
            for chunk_obj, vec in zip(batch_chunks, res.vectors):
                all_vector_records.append(
                    VectorRecord(
                        chunk_id=chunk_obj.chunk_id,
                        document_id=chunk_obj.document_id,
                        text=chunk_obj.content,
                        embedding=vec,
                        metadata={
                            "document_id": chunk_obj.document_id,
                            "source_filename": f"{chunk_obj.document_id}.md",
                            "document_title": chunk_obj.document_title,
                            "chunk_index": chunk_obj.chunk_index,
                            "file_type": "md",
                        },
                    )
                )
            print(f"    Indexed {len(all_vector_records)} / {len(chunk_records)} vectors...")

        vstore.add_records(all_vector_records)
        print(f"[+] Successfully indexed {vstore.count()} vectors into {CHROMA_DIR}")

    # 5. Build Knowledge Graph
    print("\n[STEP 4] Constructing Isolated Knowledge Graph in NetworkX...")
    gstore = GraphStore()
    ent_map: dict[str, str] = {}

    for ent in BENCHMARK_ENTITIES:
        c_name = ent["name"]
        ent_id = f"ent_{hashlib.sha256(c_name.strip().lower().encode()).hexdigest()[:16]}"
        ent_obj = GraphEntity(
            entity_id=ent_id,
            canonical_name=c_name,
            entity_type=ent["type"],
            description=f"Core concept: {c_name}",
            aliases=ent.get("aliases", []),
        )
        gstore.add_entity(ent_obj)
        ent_map[c_name.strip().lower()] = ent_id
        for alias in ent.get("aliases", []):
            ent_map[alias.strip().lower()] = ent_id

    for rel in BENCHMARK_RELATIONSHIPS:
        src_id = ent_map.get(rel["source"].strip().lower())
        tgt_id = ent_map.get(rel["target"].strip().lower())
        if not src_id or not tgt_id:
            print(
                f"[-] Warning: Entity not found for relationship {rel['source']} -> {rel['target']}"
            )
            continue

        doc_id = rel["doc_id"]
        chunk_idx = rel["chunk_idx"]
        chunk_id = f"chunk_{doc_id}_{chunk_idx:04d}"
        pred = rel["predicate"].strip().upper()
        rel_id = f"rel_{hashlib.sha256(f'{src_id}_{pred}_{tgt_id}_{chunk_id}'.encode()).hexdigest()[:16]}"

        rel_obj = GraphRelationship(
            relationship_id=rel_id,
            subject_entity_id=src_id,
            predicate=pred,
            object_entity_id=tgt_id,
            evidence_quote=rel["quote"],
            confidence=float(rel["confidence"]),
            source_document_id=doc_id,
            source_chunk_id=chunk_id,
            source_filename=f"{doc_id}.md",
        )
        gstore.add_relationship(rel_obj)

    gstore.save_json(KG_DIR / "knowledge_graph.json")
    gstore.export_graphml(KG_DIR / "knowledge_graph.graphml")
    print(
        f"[+] Saved Knowledge Graph: {gstore.node_count} nodes, {gstore.edge_count} edges into {KG_DIR}"
    )

    # 6. Build Ground-Truth Dataset (60 queries)
    print("\n[STEP 5] Compiling 60 Ground-Truth Benchmark Queries with Graded Labels...")
    queries_compiled: list[BenchmarkQuery] = []

    for q_spec in BENCHMARK_QUERIES_SPEC:
        qid = q_spec["query_id"]
        qtext = q_spec["question"]
        cat = q_spec["category"]
        split = q_spec["split"]
        answerable = q_spec["answerable"]
        expected = q_spec["expected_answer"]
        gold_tuples = q_spec.get("gold_chunks", [])
        distractor_tuples = q_spec.get("distractors", [])
        rationale = q_spec["rationale"]

        judgments: list[RelevanceJudgment] = []
        relevant_chunk_ids: list[str] = []
        doc_ids: list[str] = []

        for doc_id, chunk_idx, grade in gold_tuples:
            cid = f"chunk_{doc_id}_{chunk_idx:04d}"
            judgments.append(
                RelevanceJudgment(
                    chunk_id=cid,
                    document_id=doc_id,
                    grade=grade,
                    rationale=f"Gold relevance grade {grade} supporting question.",
                )
            )
            if grade >= 2:
                relevant_chunk_ids.append(cid)
            if doc_id not in doc_ids:
                doc_ids.append(doc_id)

        for doc_id, chunk_idx, grade in distractor_tuples:
            cid = f"chunk_{doc_id}_{chunk_idx:04d}"
            judgments.append(
                RelevanceJudgment(
                    chunk_id=cid,
                    document_id=doc_id,
                    grade=grade,
                    rationale=f"Evaluated distractor grade {grade}.",
                )
            )
            if grade >= 2 and cid not in relevant_chunk_ids:
                relevant_chunk_ids.append(cid)
            if doc_id not in doc_ids:
                doc_ids.append(doc_id)

        queries_compiled.append(
            BenchmarkQuery(
                query_id=qid,
                question=qtext,
                category=cat,
                split=split,
                answerable=answerable,
                expected_answer=expected,
                relevant_chunk_ids=relevant_chunk_ids,
                relevance_judgments=judgments,
                supporting_document_ids=doc_ids,
                annotation_rationale=rationale,
                annotation_status="verified",
            )
        )

    # Save to ground_truth/benchmark_dataset.json
    out_dataset = {
        "version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "corpus_documents_count": len(doc_manifest_records),
        "corpus_chunks_count": len(chunk_records),
        "total_queries": len(queries_compiled),
        "split_counts": {
            "dev": sum(1 for q in queries_compiled if q.split == BenchmarkSplit.DEV),
            "validation": sum(1 for q in queries_compiled if q.split == BenchmarkSplit.VALIDATION),
            "test": sum(1 for q in queries_compiled if q.split == BenchmarkSplit.TEST),
        },
        "category_counts": {
            cat.value: sum(1 for q in queries_compiled if q.category == cat)
            for cat in BenchmarkQueryCategory
        },
        "documents": [d.model_dump(mode="json") for d in doc_manifest_records],
        "queries": [q.model_dump(mode="json") for q in queries_compiled],
    }

    dataset_json_path = GROUND_TRUTH_DIR / "benchmark_dataset.json"
    with open(dataset_json_path, "w", encoding="utf-8") as f:
        json.dump(out_dataset, f, indent=2)

    # Export CSV summary
    import csv

    with open(GROUND_TRUTH_DIR / "benchmark_queries.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["query_id", "category", "split", "answerable", "relevant_chunks_count", "question"]
        )
        for q in queries_compiled:
            writer.writerow(
                [
                    q.query_id,
                    q.category.value,
                    q.split.value,
                    q.answerable,
                    len(q.relevant_chunk_ids),
                    q.question,
                ]
            )

    print(f"[+] Saved {len(queries_compiled)} benchmark queries to {dataset_json_path}")
    print(f"[+] Exported CSV summary to {GROUND_TRUTH_DIR / 'benchmark_queries.csv'}")

    print("\n" + "=" * 80)
    print("PHASE 8 BENCHMARK ENVIRONMENT BUILD COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Source Documents : {len(doc_manifest_records)}")
    print(f"Canonical Chunks : {len(chunk_records)}")
    print(f"ChromaDB Vectors : {vstore.count()}")
    print(f"KG Nodes / Edges : {gstore.node_count} nodes, {gstore.edge_count} edges")
    print(f"Benchmark Queries: {len(queries_compiled)} (Dev: 15, Val: 15, Test: 30)")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(build_benchmark_environment())
