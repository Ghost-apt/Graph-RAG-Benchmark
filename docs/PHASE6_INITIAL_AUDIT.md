# Phase 6 Initial Architecture & Readiness Audit

**Project:** Graph RAG Benchmark  
**Phase:** Phase 6 — Hybrid Retrieval & Reciprocal Rank Fusion  
**Date:** 2026-11-13  
**Status:** COMPLETE & APPROVED  

---

## 1. Executive Overview

Phase 6 unites the independently developed and verified **Vector Retrieval Engine** (Phase 3) and **Knowledge Graph Retrieval Engine** (Phase 5) into a unified, high-precision **Hybrid Retrieval Engine**.

The core challenge of Hybrid RAG in this benchmark is fusing two fundamentally different retrieval paradigms:
1. **Dense Vector Retrieval:** Metric-space nearest-neighbor search (`nomic-embed-text` + ChromaDB cosine distance) returning topically similar text chunks.
2. **Graph-Based Retrieval:** Multi-hop relational path exploration (`NetworkX MultiDiGraph`) returning explicit entity relationships grounded by verbatim quotations.

This audit establishes the baseline interfaces, canonical data structures, score semantics, and preservation constraints required to implement Reciprocal Rank Fusion (RRF) safely and accurately.

---

## 2. Existing Retrieval Subsystem Interfaces

### 2.1 Vector Retriever Subsystem (Phase 3)
- **Primary Service:** `SemanticSearch` in `src/graphrag_bench/semantic_search.py`
- **Execution Method:**
  ```python
  def search(
      self,
      query: str,
      top_k: int | None = None,
      document_id: str | None = None,
      file_type: str | None = None,
  ) -> SearchResponse
  ```
- **Output Container (`SearchResponse`):**
  - `query: str`
  - `top_k: int`
  - `total_found: int`
  - `duration_ms: float`
  - `items: list[SearchResultItem]`
  - `model_used: str`
  - `applied_filter: dict[str, Any] | None`
- **Item Schema (`SearchResultItem`):**
  - `rank: int` (1-based rank from vector search)
  - `chunk_id: str` (Canonical Phase 2 chunk identifier)
  - `document_id: str` (Canonical Phase 2 document identifier)
  - `text: str` (Original chunk text)
  - `distance: float` (ChromaDB cosine distance in $[0.0, 2.0]$)
  - `cosine_similarity: float` (Raw cosine similarity: $1.0 - \text{distance}$)
  - `normalized_score: float` (Normalized cosine score: $1.0 - \text{distance}/2.0$)
  - `metadata: dict[str, Any]` (`source_filename`, `file_type`, `page_number`, `chunk_index`, etc.)

### 2.2 Graph Retriever Subsystem (Phase 5)
- **Primary Service:** `GraphSearchEngine` in `src/graphrag_bench/graph_search.py` (and `search_graph()` helper)
- **Execution Method:**
  ```python
  def search_graph(
      self,
      query: str,
      max_hops: int | None = None,
      direction: Literal["both", "out", "in"] = "both",
      top_k: int = 10,
      predicates: list[str] | set[str] | None = None,
      doc_id_filter: str | None = None,
  ) -> GraphSearchResponse
  ```
- **Output Container (`GraphSearchResponse`):**
  - `query: str`
  - `status: GraphSearchStatus` (`SUCCESS`, `NO_MATCHING_ENTITIES`, `NO_GRAPH_RESULTS`, etc.)
  - `message: str`
  - `extraction_result: QueryEntityExtractionResult | None`
  - `linked_entities: list[LinkedEntity]`
  - `results: list[GraphRetrievalResult]`
  - `structured_context: str`
  - `context_evidence: GraphContextEvidence | None`
  - `statistics: dict[str, Any]`
  - `latency_ms: float`
- **Item Schema (`GraphRetrievalResult`):**
  - `rank: int` (1-based rank of the reasoning path)
  - `seed_entity_name: str`
  - `path: GraphPath` (Contains `nodes: list[PathNode]` and `edges: list[PathEdge]`)
  - `score: float` (Composite path score in $[0.0, 1.0]$)
  - `score_explanation: str`
  - `source_document_ids: list[str]`
  - `source_chunk_ids: list[str]`
  - `source_filenames: list[str]`
  - `evidence_quotes: list[str]`
- **Edge Schema (`PathEdge`):**
  - `relationship_id: str`
  - `subject_entity_id: str`
  - `predicate: str`
  - `object_entity_id: str`
  - `direction: Literal["outgoing", "incoming"]`
  - `evidence_quote: str`
  - `confidence: float`
  - `source_document_id: str`
  - `source_chunk_id: str`
  - `source_filename: str`

---

## 3. Canonical Chunk Mapping & Fusion Key Behavior

### 3.1 Single Source of Truth: `chunk_id`
In Phase 2, chunk identifiers were made strictly deterministic:
$$\text{chunk\_id} = \text{doc\_id} + \text{"\_c"} + \text{index:04d} + \text{"\_"} + \text{content\_hash[:8]}$$
Example: `doc_6234cf74cb459ff7_c0000_34c56be6`.

- **Vector Store Alignment:** Every record in ChromaDB is keyed by `chunk_id`.
- **Knowledge Graph Alignment:** Every relationship edge in NetworkX stores `source_chunk_id` and `source_document_id`.
- **Document Repository:** `DocumentRepository` maintains all chunks in `data/processed/chunks.jsonl` with complete text, offsets, and source metadata.

### 3.2 Key Graph-to-Chunk Mapping Insight
While Vector Retrieval directly returns ranked **chunks**, Graph Retrieval returns ranked **relational paths**. A single graph path can traverse 1 to 3 edges, each referencing a `source_chunk_id`.

**Critical Deduplication & Ranking Rule for Graph Candidates:**
1. A single chunk may appear across multiple edges or across multiple retrieved graph paths.
2. In graph candidate normalization, each unique `chunk_id` must appear **exactly once** in the normalized graph ranking list.
3. The graph rank for that chunk is determined by the **best (lowest numerical) rank** among all graph paths that reference it.
4. All distinct supporting graph paths and all distinct supporting evidence quotes for that chunk are aggregated into the candidate record.
5. This prevents awarding multiple RRF rank credits to the same chunk from repeated graph path occurrences.

---

## 4. Score Semantics & Discrepancy Prevention

| Retrieval Modality | Native Metric | Range | Semantics | Incompatibility Warning |
|:---|:---|:---:|:---|:---|
| **Vector Retrieval** | Cosine Distance ($d$) | $[0.0, 2.0]$ | $0.0 =$ identical, $1.0 =$ orthogonal | Cannot be added to graph score |
| **Vector Retrieval** | Cosine Similarity ($1 - d$) | $[-1.0, 1.0]$ | Angular similarity | Cannot be added to graph score |
| **Vector Retrieval** | Normalized Score ($1 - d/2$) | $[0.0, 1.0]$ | Bounded linear scale | Incommensurate distribution with graph score |
| **Graph Retrieval** | Composite Ranker Score | $[0.0, 1.0]$ | Linear multi-factor heuristic | Incommensurate distribution with vector score |
| **Reciprocal Rank Fusion (RRF)** | $RRF(d) = \sum_{m} \frac{w_m}{k + r_m(d)}$ | $[0.0, \frac{w_v + w_g}{k + 1}]$ | Order-based rank reciprocal sum | **Strictly uses 1-based ranks $r_m$, never raw scores** |

> [!CAUTION]
> **Mathematical Rule:** Under no circumstances should ChromaDB distance or similarity be directly combined with GraphRanker scores. RRF operates purely on ordinal ranks. RRF scores must never be described or displayed as probabilities.

---

## 5. Strict Data Preservation Rules

The following directories and canonical files are strictly **immutable and read-only** during Phase 6 development:
1. `data/processed/documents.json` (Phase 2 canonical metadata)
2. `data/processed/chunks.jsonl` (Phase 2 canonical chunks)
3. `data/processed/manifest.json` (Phase 2 manifest)
4. `data/chroma/` (Phase 3 ChromaDB collections)
5. `data/knowledge_graph/knowledge_graph.json` (Phase 4 canonical graph: 22 nodes, 20 edges)
6. `data/knowledge_graph/knowledge_graph.graphml` (Phase 4 GraphML export)
7. `data/knowledge_graph/manifest.json` (Phase 4 extraction manifest)
8. `data/knowledge_graph/extractions.jsonl` (Phase 4 extraction log)

No operation in Phase 6 will re-index, rebuild, mutate, or delete these assets.

---

## 6. Phase 6 Implementation Blueprint

1. **Configuration (`config.py`):**
   - Add `hybrid_top_k`, `hybrid_vector_top_k`, `hybrid_graph_top_k`, `hybrid_rrf_k`, `hybrid_vector_weight`, `hybrid_graph_weight`, `hybrid_min_rrf_score`, `hybrid_parallel_retrieval`.
   - Implement strict Pydantic validators.
2. **Schemas (`hybrid_schemas.py`):**
   - Define `VectorCandidate`, `GraphCandidate`, `FusionCandidate`, `HybridSearchResponse`, `RetrievalSource`.
3. **Normalization (`result_normalizer.py`):**
   - Normalize `SearchResponse` -> `list[VectorCandidate]`.
   - Normalize `GraphSearchResponse` -> `list[GraphCandidate]` (mapped through `DocumentRepository` with path deduplication).
4. **Reciprocal Rank Fusion (`reciprocal_rank_fusion.py`):**
   - Implement standard and weighted RRF:
     $$RRF(d) = \frac{w_v}{k + r_v(d)} + \frac{w_g}{k + r_g(d)}$$
   - Deterministic tie-breaking by RRF score descending, then presence in both modalities, then best single modality rank, then canonical `chunk_id`.
5. **Context Builder (`hybrid_context_builder.py`):**
   - Format unified evidence blocks with source attribution (`VECTOR`, `GRAPH`, `VECTOR_AND_GRAPH`), vector similarity diagnostics, and grounded graph paths.
6. **Orchestrator (`hybrid_retriever.py`):**
   - Implement `search_hybrid()` coordinating both pipelines, measuring per-modality latency, and handling partial failures honestly.
7. **Streamlit UI (`app.py`):**
   - Add Tab 6: `⚖️ Hybrid Retrieval Lab` preserving Tabs 1–5.
8. **Automated Test Suite (`tests/`):**
   - Complete unit and integration tests across all Phase 6 modules.
