# GRAPH RAG BENCHMARK
## PHASE 7 INITIAL ARCHITECTURE AUDIT & IMPLEMENTATION BLUEPRINT

**Date:** 2026-11-13  
**Status:** AUDIT COMPLETE  
**Current Version:** `0.6.0`  
**Target Version:** `0.7.0`  
**Workspace:** `C:\VS CODE\Graph RAG Benchmark`  

---

### 1. Existing System Architecture Overview

The **Graph RAG Benchmark** system comprises six completed, verified, and passing architectural phases:

1. **Phase 1 (Environment Foundation):**
   - Python 3.11.9 isolated virtual environment (`.venv`).
   - Pydantic v2 `Settings` with `.env` loading and secret masking.
   - Resilient `OllamaClient` supporting server health verification, model catalog checks, `/api/generate` for text generation, and `/api/embed` for embeddings.
   - Comprehensive system health checks (`HealthChecker`).

2. **Phase 2 (Document Ingestion & Preprocessing):**
   - Multi-format ingestion (`pypdf`, `python-docx`, TXT, Markdown).
   - Deterministic Unicode NFKC text cleaning and normalization (`TextCleaner`).
   - Sliding-window word chunker with character-offset and page mapping (`TextChunker`).
   - Local JSON/JSONL repository with SHA-256 duplicate detection (`DocumentRepository`).
   - Canonical dataset: 3 documents, 3 chunks in `data/processed/`.

3. **Phase 3 (Vector Embeddings & ChromaDB Semantic Search):**
   - `EmbeddingService` wrapping `nomic-embed-text` (768 dimensions).
   - `VectorStore` using persistent ChromaDB with cosine distance (`data/chroma/`).
   - `SemanticSearch` executing natural-language query embedding and nearest-neighbor vector retrieval with dual cosine metrics ($1.0 - \text{distance}$ and $1.0 - \text{distance}/2$) and metadata filtering (`document_id`, `file_type`).

4. **Phase 4 (Knowledge Graph Construction Engine):**
   - `GraphExtractionService` using local `qwen3:8b` via Ollama with JSON-constrained decoding, reasoning tokens bypassed, and chunk ground-truth verification.
   - `EntityResolver` providing NFKC normalization, alias registry, and deterministic entity IDs (`ent_<type>_<hash[:12]>`).
   - `GraphStore` managing NetworkX `MultiDiGraph` with atomic JSON persistence (`knowledge_graph.json`) and primitive-sanitized GraphML export (`knowledge_graph.graphml`).
   - Canonical graph: 22 nodes, 20 relationships.

5. **Phase 5 (Graph Retrieval & Multi-Hop Reasoning Engine):**
   - `QueryEntityExtractor` with sub-millisecond fast-path dictionary scanning and bounded local Qwen fallback.
   - `EntityLinker` mapping query mentions to verified GraphStore nodes with ambiguity detection.
   - `GraphTraversalEngine` performing bounded direction-aware BFS path exploration (1 to 3 hops) with cycle prevention.
   - `GraphRanker` computing deterministic scores incorporating seed match quality, hop economy ($0.85^{\text{hops}-1}$), and edge confidence.
   - `GraphSearchEngine` high-level facade.

6. **Phase 6 (Hybrid Retrieval & Reciprocal Rank Fusion Engine):**
   - `ResultNormalizer` mapping vector results and graph paths to canonical `chunk_id` keys via `DocumentRepository`.
   - `ReciprocalRankFusion` computing standard and weighted RRF ($RRF(d) = \frac{w_v}{k + r_v} + \frac{w_g}{k + r_g}$) over ordinal ranks with 4-tier deterministic tie-breaking.
   - `HybridContextBuilder` formatting RRF candidates into structured Markdown evidence blocks.
   - `HybridRetriever` coordinator managing parallel/sequential retrieval, error isolation (`PARTIAL_SUCCESS`), and multi-field filtering.
   - Streamlit Tab 6 (`⚖️ Hybrid Retrieval Lab`).
   - 266 total passing tests (257 unit + 9 live integration).

---

### 2. Verified Interfaces & Return Contracts

#### 2.1 Hybrid Retrieval API (`src/graphrag_bench/hybrid_retriever.py`)
- **Method Signature:**
  ```python
  HybridRetriever.search_hybrid(
      query: str,
      vector_top_k: int | None = None,
      graph_top_k: int | None = None,
      final_top_k: int | None = None,
      rrf_k: int | None = None,
      vector_weight: float | None = None,
      graph_weight: float | None = None,
      max_hops: int | None = None,
      direction: str = "both",
      predicates: list[str] | set[str] | None = None,
      document_ids: list[str] | set[str] | str | None = None,
      file_types: list[str] | set[str] | str | None = None,
  ) -> HybridSearchResponse
  ```
- **Alias:** `HybridRetriever.search = search_hybrid`
- **Output Container (`HybridSearchResponse`):**
  - `status: HybridSearchStatus` (`SUCCESS`, `PARTIAL_SUCCESS`, `NO_RESULTS`, `INVALID_QUERY`, `ALL_ENGINES_FAILED`)
  - `results: list[FusionCandidate]` (also accessible via property `.fused_candidates`)
  - `vector_search_status: str`, `graph_search_status: str`
  - `vector_results_count: int`, `graph_results_count: int`, `fused_results_count: int`
  - `total_latency_ms: float`, `vector_latency_ms: float`, `graph_latency_ms: float`, `fusion_latency_ms: float`
  - `warnings: list[str]`, `errors: list[str]`
  - `structured_context: str`

#### 2.2 Fusion Candidate Provenance (`src/graphrag_bench/hybrid_schemas.py`)
Each `FusionCandidate` contains:
- `chunk_id: str`: Canonical Phase 2 chunk identifier (`<doc_id>_c<index>_<hash>`)
- `document_id: str`: Canonical Phase 2 document identifier (`doc_<hash>`)
- `source_filename: str`: Cleaned original filename
- `chunk_text: str`: Verbatim canonical text content
- `fused_rank: int`: 1-based composite ordinal rank
- `retrieval_sources: RetrievalSource`: `VECTOR`, `GRAPH`, `VECTOR_AND_GRAPH`
- `vector_rank: int | None`: Ordinal rank in vector search
- `graph_rank: int | None`: Ordinal rank in graph search
- `vector_contribution: float`: Weighted RRF component $w_v / (k + r_v)$
- `graph_contribution: float`: Weighted RRF component $w_g / (k + r_g)$
- `rrf_score: float`: Composite RRF summation score
- `supporting_graph_paths: list[dict[str, Any]]`: Associated graph reasoning paths
- `evidence_quotes: list[str]`: Verbatim relationship grounding quotes
- `source_metadata: dict[str, Any]`: Page number, cosine distance, cosine similarity, matched entities, min hops

#### 2.3 Ollama Generation API (`src/graphrag_bench/ollama_client.py`)
- **Method Signature:**
  ```python
  OllamaClient.generate(
      prompt: str,
      model: str | None = None,
      system: str | None = None,
      options: dict[str, Any] | None = None,
      format: str | dict[str, Any] | None = None,
      think: bool | None = None,
      timeout: float | None = None,
  ) -> GenerateResponse
  ```
- **Output Container (`GenerateResponse`):**
  - `model: str`
  - `response: str`
  - `done: bool`
  - `total_duration_ns: int | None`
  - `eval_count: int | None`
  - `prompt_eval_count: int | None`
  - `raw: dict[str, Any]`

---

### 3. Proposed Phase 7 Architecture & Module Plan

To implement grounded question answering without disrupting Phases 1–6, Phase 7 introduces the following modules:

```
src/graphrag_bench/
├── answer_schemas.py         # Pydantic v2 data models for answers, citations, claims, requests
├── evidence_preparer.py      # Budget-aware evidence packager from HybridSearchResponse
├── prompt_builder.py         # Grounded system & user prompt builder with citation registry
├── answer_generator.py       # Ollama Qwen3:8b generation wrapper with latency & error handling
├── citation_validator.py     # Deterministic citation marker validator against retrieved evidence
├── answer_guardrails.py      # Insufficient evidence, prompt injection defense, uncertainty handler
├── faithfulness_checker.py   # Claim extraction & deterministic claim-to-evidence support checker
└── rag_pipeline.py           # Master RAG pipeline orchestrator (answer_question API)
```

#### Module Descriptions:
1. **`answer_schemas.py`:**
   - Enums: `AnswerStatus` (`SUCCESS`, `INSUFFICIENT_EVIDENCE`, `NO_RETRIEVAL_RESULTS`, `GENERATION_FAILED`, `INVALID_QUERY`, `CITATION_VALIDATION_FAILED`, `PARTIAL_VALIDATION`).
   - Models: `AnswerCitation`, `AnswerClaim`, `GroundedAnswer`, `AnswerGenerationRequest`.
2. **`evidence_preparer.py`:**
   - Extracts and deduplicates canonical chunks from `HybridSearchResponse`.
   - Conservative token estimation: `ceil(len(text) / 3.2)` or `ceil(len(words) * 1.35)`.
   - Enforces `rag_max_context_tokens` budget, prioritizes higher-ranked candidates, avoids mid-sentence cutting when possible, reports omitted items.
3. **`prompt_builder.py`:**
   - Grounded system prompt enforcing: (1) no unsupported facts, (2) untrusted context (prompt injection defense), (3) strict citations `[S1]`, `[S2]`, (4) reporting insufficiency/contradiction, (5) technical terminology preservation.
   - Maps each evidence item to deterministic source identifier `[S1]`, `[S2]`, etc.
4. **`answer_generator.py`:**
   - Wraps `OllamaClient.generate()` for local `qwen3:8b`.
   - Supports configurable temperature (`0.1`), token limit (`1200`), timeout (`600s`), and streaming (optional).
   - Handles connection errors, timeouts, and empty responses.
5. **`citation_validator.py`:**
   - Extracts markers `[S1]`, `[S2]` from generated answer.
   - Validates markers against evidence registry (rejects unknown markers).
   - Verifies citation metadata mapping to canonical Phase 2 chunks.
6. **`answer_guardrails.py`:**
   - Detects insufficient evidence (returns `INSUFFICIENT_EVIDENCE` without LLM invocation).
   - Detects prompt injection patterns in retrieved text.
   - Recommends explicit uncertainty phrasing ("The available evidence indicates...").
7. **`faithfulness_checker.py`:**
   - Decomposes generated answer into factual claims.
   - Verifies each claim against cited evidence chunks (SUPPORTED, UNSUPPORTED, UNCERTAIN).
   - Purely preliminary grounding heuristic (NOT formal Phase 8 benchmarking).
8. **`rag_pipeline.py`:**
   - Main application coordinator exposing `answer_question()`.
   - Executes Step 1 (Validation) $\to$ Step 2 (Hybrid Retrieval) $\to$ Step 3 (Guardrails / Insufficient Evidence Check) $\to$ Step 4 (Evidence Preparation) $\to$ Step 5 (Prompt Construction) $\to$ Step 6 (Qwen Generation) $\to$ Step 7 (Citation Validation) $\to$ Step 8 (Faithfulness Checks) $\to$ Step 9 (GroundedAnswer Packaging).

---

### 4. Preservation Strategy & Immutability Certification

- **Strictly Read-Only Directories:**
  - `data/processed/`
  - `data/chroma/`
  - `data/knowledge_graph/`
- **Existing Retrieval Pipelines:**
  - `HybridRetriever`, `SemanticSearch`, and `GraphSearchEngine` are invoked as read-only services.
  - Zero modifications to Phase 1–6 algorithms, data structures, or scoring logic.
- **Streamlit App:**
  - Preserve Tabs 1 through 6 exactly as they exist.
  - Add Tab 7 (`💬 Grounded RAG Chat`) with session-state caching to prevent rerun model calls.
