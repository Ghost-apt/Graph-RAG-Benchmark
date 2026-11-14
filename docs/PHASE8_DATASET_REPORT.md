# Phase 8 Dataset Report: Benchmark Evaluation Corpus & Ground Truth

**Project:** Graph RAG Benchmark  
**Phase:** 8 — Comparative Evaluation (Vector RAG vs Graph RAG vs Hybrid RAG)  
**Date:** November 2026  
**Corpus Root:** `data/phase8_benchmark/`  
**Dataset Version:** `1.0.0`  
**System Target Version:** `0.8.0`  

---

## 1. Executive Summary

Phase 8 establishes an isolated, reproducible, gold-standard benchmark corpus designed specifically to empirically compare **Vector RAG**, **Knowledge Graph RAG**, and **Hybrid RAG (Reciprocal Rank Fusion)**. 

To ensure complete scientific rigor and zero data contamination:
1. **Corpus Isolation:** All Phase 8 data resides strictly in `data/phase8_benchmark/`, keeping canonical Phase 1–7 production repositories (`data/processed/`, `data/knowledge_graph/`, and Chroma collection `graphrag_documents`) 100% immutable and intact.
2. **Corpus Scale:** 22 specialized technical documents comprising 110 canonical chunks (5 chunks per document, ~40–80 tokens per chunk).
3. **Knowledge Graph Grounding:** 36 structured entities and 24 directional relationships with verbatim evidence quotes.
4. **Query Diversity:** 60 curated benchmark queries distributed across five orthogonal cognitive categories (Single-Hop, Multi-Hop, Entity-Relationship, Comparative Synthesis, and Unanswerable / Rejection).
5. **Graded Relevance:** 4-point graded relevance scale ($0$ to $3$) manually validated across all 60 queries.

---

## 2. Evaluation Corpus Architecture

### 2.1 Technical Document Distribution

The 22 source documents in `data/phase8_benchmark/raw_documents/` systematically cover the entire spectrum of Retrieval-Augmented Generation, vector embeddings, geometric spaces, graph data structures, and ranking fusion:

| Document ID | Canonical Title | Primary Subject Domain | Chunks | SHA-256 (Prefix) |
|---|---|---|:---:|:---:|
| `doc01_vector_embeddings` | Vector Embeddings and Dense Representations | Latent representations, MRL, dimensions | 5 | `749C401D5A49...` |
| `doc02_vector_similarity_metrics` | Geometric Vector Distance Functions and Invariants | Cosine similarity, Euclidean distance, L2 | 5 | `1AAAC71F764A...` |
| `doc03_ann_vector_indexing` | Approximate Nearest Neighbor Vector Indexing | HNSW, Voronoi cells, SQLite | 5 | `C2EF49140417...` |
| `doc04_chromadb_architecture` | ChromaDB Architecture and Persistence Subsystem | SQLite metadata, Rust/C++ hnswlib | 5 | `77B0901A6167...` |
| `doc05_knowledge_graph_foundations` | Knowledge Graph Modeling and Ontological Foundations | Directed graphs, entities, predicates | 5 | `E86A8EDC7D52...` |
| `doc06_entity_canonicalization` | Entity Extraction and Canonical Resolution | Surface mentions, deduplication | 5 | `6BC633900A9E...` |
| `doc07_networkx_graph_storage` | NetworkX Graph Storage and MultiDiGraph Semantics | Pure Python, MultiDiGraph, JSON | 5 | `6D5F231D11B5...` |
| `doc08_graph_traversal_algorithms` | Graph Traversal and Neighborhood Expansion | BFS, DFS, Dijkstra, cycle prevention | 5 | `56F74D5417F6...` |
| `doc09_hybrid_retrieval_fusion` | Hybrid Retrieval Architectures and Fusion Principles | Dual-modality, complementarity | 5 | `6CEFBCB23136...` |
| `doc10_reciprocal_rank_fusion` | Reciprocal Rank Fusion (RRF) Mathematical Framework | Score normalization, constant $k=60$ | 5 | `9A55734A7B7B...` |
| `doc11_grounded_answer_generation` | LLM Answer Generation and Grounded Reasoning | System prompting, temperature, citations | 5 | `A1215B3DE630...` |
| `doc12_citation_provenance` | Citation Provenance and Evidence Verification | `[S1]` markers, hallucination audit | 5 | `01D414D2517C...` |
| `doc13_guardrails_refusal` | RAG Guardrails and Factuality Enforcement | Pre/post checks, refusal short-circuit | 5 | `3BC22E5848C3...` |
| `doc14_evaluation_metrics_precision_recall` | IR Evaluation Metrics: Precision, Recall, and MAP | Precision@K, Recall@K, AP@K formulas | 5 | `5DF1B4E54452...` |
| `doc15_evaluation_metrics_ndcg` | Ranking Quality Evaluation: DCG and nDCG Metrics | Graded gains $2^{\text{rel}}-1$, IDCG | 5 | `F628BD5FE259...` |
| `doc16_statistical_significance` | Statistical Significance in Information Retrieval | Wilcoxon signed-rank, 95% bootstrap CI | 5 | `DFDDA193E34C...` |
| `doc17_latency_tradeoffs` | Retrieval Latency Tradeoffs and System Performance | Vector 50ms vs Graph fallback 14s | 5 | `C6EFB9A1A407...` |
| `doc18_failure_modes_vector` | Failure Modes and Degradation in Vector Search | Distance concentration, entity blind spots | 5 | `7E98DC136F39...` |
| `doc19_failure_modes_graph` | Failure Modes and Vulnerabilities in Graph RAG | Extraction timeouts, disconnected nodes | 5 | `270B771A2CE3...` |
| `doc20_failure_modes_hybrid` | Hybrid RAG Pitfalls and Fusion Pathologies | Weight skew, noise amplification | 5 | `85D6F9C848C4...` |
| `doc21_benchmark_dataset_design` | Benchmark Dataset Construction and Annotation Methodology | Treceval guidelines, pooling, splits | 5 | `95E36FD4EC31...` |
| `doc22_future_directions` | Future Paradigms: Agentic Graph RAG and Dynamic Graph Indexing | Self-reflective graphs, dynamic updates | 5 | `4DEE8E9FF5F9...` |

**Total Documents:** 22  
**Total Canonical Chunks:** 110 (`chunk_doc01_..._0000` through `chunk_doc22_..._0004`)  

---

## 3. Knowledge Graph Topology & Grounded Triples

The benchmark knowledge graph (`data/phase8_benchmark/knowledge_graph/knowledge_graph.json` and `.graphml`) models the conceptual, algorithmic, and software relationships governing the benchmark domain.

- **Total Entities (Nodes):** 36
- **Total Relationships (Edges):** 24
- **Weakly Connected Components:** 1 (Fully connected domain graph)
- **Graph Density:** $0.038$
- **Entity Type Distribution:**
  - `CONCEPT`: 16 (e.g., *Vector Embeddings*, *Cosine Similarity*, *BFS Traversal*, *nDCG*, *Wilcoxon Signed-Rank Test*)
  - `SOFTWARE`: 6 (e.g., *ChromaDB*, *NetworkX*, *Ollama*, *Qwen3:8b*, *nomic-embed-text*)
  - `METHOD`: 8 (e.g., *Approximate Nearest Neighbor*, *Reciprocal Rank Fusion*, *Bootstrap Resampling*)
  - `METRIC`: 6 (e.g., *Precision@K*, *Recall@K*, *MRR*, *Faithfulness Score*)

### Sample Asserted Directional Triples
1. `(Vector Embeddings) --[MEASURED_BY]--> (Cosine Similarity)`  
   *Supporting Chunk:* `chunk_doc02_vector_similarity_metrics_0000`  
   *Evidence Quote:* `"Cosine similarity evaluates the uncentered angular alignment between two high-dimensional dense vectors..."`
2. `(Approximate Nearest Neighbor) --[IMPLEMENTED_IN]--> (ChromaDB)`  
   *Supporting Chunk:* `chunk_doc04_chromadb_architecture_0000`  
   *Evidence Quote:* `"ChromaDB implements approximate nearest neighbor retrieval via a persistent C++ hnswlib core..."`
3. `(Knowledge Graph Modeling) --[STORED_IN]--> (NetworkX)`  
   *Supporting Chunk:* `chunk_doc07_networkx_graph_storage_0000`  
   *Evidence Quote:* `"NetworkX stores directed multigraphs using pure Python nested dictionary structures..."`
4. `(Reciprocal Rank Fusion) --[FUSES]--> (Vector Embeddings)`  
   *Supporting Chunk:* `chunk_doc10_reciprocal_rank_fusion_0000`  
   *Evidence Quote:* `"Reciprocal Rank Fusion computes composite scores by summing position-penalized reciprocal ranks across vector and graph candidates..."`
5. `(LLM Answer Generation) --[VERIFIED_BY]--> (Citation Provenance)`  
   *Supporting Chunk:* `chunk_doc12_citation_provenance_0000`  
   *Evidence Quote:* `"Citation provenance maps bracketed markers to retrieved chunk identifiers for automated claim verification..."`

---

## 4. Benchmark Query Taxonomy & Split Distribution

The 60 benchmark queries were authored to stress-test specific information-retrieval strengths and failure modes across modalities:

| Query Category | Query Count | Target Modality Strengths & Vulnerabilities Tested |
|---|:---:|---|
| **Single-Hop** | 15 (25%) | Direct factual recall from a single chunk. Tests baseline vector similarity and fast dictionary entity retrieval. |
| **Multi-Hop** | 15 (25%) | Complex reasoning requiring 2–3 disjoint chunks connected via conceptual or structural paths. Graph RAG and Hybrid RAG expected to outperform Vector RAG. |
| **Entity-Relationship** | 10 (16.7%) | Structural assertions and directional predicate queries. Tests graph traversal and relationship evidence extraction. |
| **Comparative Synthesis** | 10 (16.7%) | Contrasting two different concepts, algorithms, or systems (e.g., Vector vs Graph, HNSW vs Flat Indexing). Tests multi-document retrieval recall. |
| **Unanswerable / Rejection** | 10 (16.7%) | Questions regarding concepts completely absent from the corpus (e.g., Quantum computing, biological neural networks, SQL stored procedures). Tests guardrails, hallucination resistance, and zero-relevant handling. |

### Dataset Splits
- **Dev Split (15 Queries, Q01–Q15):** Used for parameter tuning (e.g., RRF constant $k$, retrieval weights $w_v, w_g$, top-K thresholds).
- **Validation Split (15 Queries, Q16–Q30):** Used for threshold calibration and prompt tuning.
- **Test Split (30 Queries, Q31–Q60):** Strictly held out for final unbiased comparative evaluation reporting.

---

## 5. Graded Relevance Judgment Schema

Rather than simplistic binary judgments ($0$ or $1$), Phase 8 employs a **4-point graded relevance scale** adhering to TREC (Text REtrieval Conference) standards:

- **Grade 3 (Direct Answer):** The chunk directly, explicitly, and fully answers the user query with high factual fidelity.
- **Grade 2 (Relevant Supporting):** The chunk provides essential supporting context, definitions, or prerequisite facts necessary to construct a complete answer.
- **Grade 1 (Related / Partial Context):** The chunk discusses the query's topic or related entities, but does not provide sufficient evidence to answer the query.
- **Grade 0 (Irrelevant):** The chunk is completely unrelated to the query.

### Binary Relevance Threshold
For binary metrics (Precision@K, Recall@K, Hit Rate@K, MRR, MAP@K), a chunk is classified as **Relevant** if and only if:
$$\text{Grade} \ge 2$$

For graded metrics (DCG@K, nDCG@K), the exact integer grade ($0, 1, 2, 3$) is used to compute exponential gain:
$$\text{gain} = 2^{\text{grade}} - 1$$
yielding gains of $0, 1, 3,$ and $7$ respectively.
