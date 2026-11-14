# Phase 8 Initial Audit & Baseline Preservation Report

**Workspace:** `C:\VS CODE\Graph RAG Benchmark`  
**Current Version:** `0.7.0`  
**Target Version:** `0.8.0`  
**Phase:** 8 — Formal Comparative Benchmark Evaluation: Vector RAG vs Graph RAG vs Hybrid RAG  
**Date:** 2026-11-14  
**Audit Status:** **BASELINE SECURED & PRESERVED**

---

## 1. Executive Baseline Summary

This audit establishes the pre-implementation state of the Graph RAG Benchmark workspace before initiating Phase 8 development. All seven foundational phases are verified fully operational, with 332 automated tests (320 unit tests, 12 integration tests) passing at 100% success rate.

The canonical data repository and indexes (`data/processed/`, `data/knowledge_graph/`, and ChromaDB collection `graphrag_documents`) are verified immutable and will remain completely untouched. Phase 8 will operate in a strictly isolated directory structure: `data/phase8_benchmark/`.

---

## 2. Canonical Data Inventory & Baseline Integrity

### Canonical Files SHA-256 Checksums
| File Path | SHA-256 Checksum | Records / Elements | Status |
| :--- | :--- | :--- | :---: |
| `data/processed/chunks.jsonl` | `F236374B00197003B1F9933CA8BEEF576CEDDE41BD67E5DF12718AB15FAB16F4` | 3 Canonical Chunks | IMMUTABLE |
| `data/processed/documents.json` | `CFE029D3AC803D9D7AAFC697037C5296F491B5CE7F658C541A393317A4D54EAB` | 3 Source Documents | IMMUTABLE |
| `data/processed/manifest.json` | `69BD862C213332BEEB0E6FA0A047B9790AD3400240FDB8371B2F77F57ECDAF39` | Ingestion Manifest | IMMUTABLE |
| `data/knowledge_graph/knowledge_graph.json` | `2E0C8783848B69C730825C20AB6C0D96D674C4195C809E5F4AA71CA89B0AF6DD` | 22 Nodes, 20 Edges | IMMUTABLE |
| `data/knowledge_graph/knowledge_graph.graphml` | `4BA25CAA4C8259218FEBB874B28EE278352FF7E1A6FF7CA455D41AB5A9F198ED` | 22 Nodes, 20 Edges | IMMUTABLE |
| `data/knowledge_graph/manifest.json` | `66ABAD41A937157D9D711082AFD39A9CB0A3D77B7B274FCAA0DCD3AD5CAABE8D` | KG Build Manifest | IMMUTABLE |
| `data/knowledge_graph/extractions.jsonl` | `7769F803E3EFF7AC37565DDF9EE6BF2C2AC4EB6449A4EDB0DC52CE69E42E44C8` | Canonical Extractions | IMMUTABLE |

### Canonical Index Inventory
- **ChromaDB (`data/chroma/`):**
  - Collection: `graphrag_documents`
  - Record Count: 3 vectors
  - Dimension: 768 (`nomic-embed-text`)
  - Chunk IDs:
    - `doc_16d32f9171c906cb_c0000_dcfc5bb1` (`rag_architectures_overview.txt`)
    - `doc_8125aaecfd279966_c0000_afd4f832` (`graph_rag_benchmarking_guide.md`)
    - `doc_e4c105b4b9f713cc_c0000_21a64384` (`sample_research_paper.docx`)
- **NetworkX Graph (`data/knowledge_graph/`):**
  - Entities: 22 entities across CONCEPT, TECHNOLOGY, SOFTWARE categories
  - Relationships: 20 relationships with verbatim evidence quotes and confidences $\ge 0.85$

---

## 3. Existing Services and Interfaces Map

The Phase 8 benchmark runner and metrics engine will cleanly wrap and reuse these existing production components:

1. **Vector Retrieval:**
   - Class: `SemanticSearch` (`src/graphrag_bench/semantic_search.py`)
   - Backend: `VectorStore` (`src/graphrag_bench/vector_store.py`) backed by ChromaDB
   - Key Method: `search(query: str, top_k: int, ...)` returning `SearchResponse` containing `SearchResultItem` list
2. **Graph Retrieval:**
   - Class: `GraphSearchEngine` (`src/graphrag_bench/graph_search.py`) & `GraphRetriever` (`src/graphrag_bench/graph_retriever.py`)
   - Backend: `GraphStore` (`src/graphrag_bench/graph_store.py`) backed by NetworkX DiGraph
   - Key Method: `search_graph(query: str, max_hops: int, ...)` returning `GraphSearchResponse` containing `GraphRetrievalResult` list
3. **Hybrid Fusion Retrieval:**
   - Class: `HybridRetriever` (`src/graphrag_bench/hybrid_retriever.py`)
   - Engine: `ReciprocalRankFusion` (`src/graphrag_bench/reciprocal_rank_fusion.py`)
   - Key Method: `search_hybrid(query: str, vector_top_k: int, graph_top_k: int, final_top_k: int, rrf_k: int, vector_weight: float, graph_weight: float)` returning `HybridSearchResponse`
4. **Grounded Answer Generation & Verification (Phase 7):**
   - Master Coordinator: `RAGPipeline` (`src/graphrag_bench/rag_pipeline.py`)
   - LLM Generator: `AnswerGenerator` (`src/graphrag_bench/answer_generator.py`)
   - Citation Validator: `CitationValidator` (`src/graphrag_bench/citation_validator.py`)
   - Safety Guardrails: `AnswerGuardrails` (`src/graphrag_bench/answer_guardrails.py`)
   - Faithfulness Checker: `FaithfulnessChecker` (`src/graphrag_bench/faithfulness_checker.py`)
   - Key Method: `answer_question(request: str | AnswerGenerationRequest, ...)` returning `GroundedAnswer`
5. **Interactive Dashboard:**
   - Entry Point: `app.py` (Streamlit runtime with Tabs 1 through 7 operational)

---

## 4. Phase 1–7 Test Suite Baseline

- **Total Collected Tests:** 332
- **Unit Tests (`-m "not integration"`):** 320 passed, 0 failed, 249 warnings (6.52s)
- **Live Integration Tests:** 12 passed, 0 failed, 25 warnings (225.79s)
- **All Pytest Warnings (274 total):** 100% upstream ChromaDB Pydantic v2 `model_fields` deprecation (`chromadb/types.py:144`)
- **Code Linting:** `ruff check .` passes with 0 errors
- **Code Formatting:** `ruff format --check .` passes with 123 files formatted
- **Streamlit AppTest:** `AppTest.from_file('app.py')` passes with 0 exceptions

---

## 5. Phase 8 Isolated Environment Architecture

To strictly guarantee zero contamination of canonical Phase 1–7 datasets, Phase 8 operates in a completely separate root:

```text
data/phase8_benchmark/
├── raw_documents/         # Ingested evaluation source files (20+ docs)
├── processed/             # Phase 8 normalized chunks and documents.json
│   ├── documents.json
│   ├── chunks.jsonl
│   └── manifest.json
├── chroma/                # Phase 8 isolated ChromaDB persistence directory
├── knowledge_graph/       # Phase 8 isolated NetworkX graph & extractions
│   ├── knowledge_graph.json
│   ├── knowledge_graph.graphml
│   └── extractions.jsonl
├── ground_truth/          # Benchmark queries & graded relevance labels
│   ├── benchmark_dataset.json
│   └── benchmark_dataset.csv
├── results/               # Unique timestamped run directories
│   └── <run_id>/
│       ├── run_manifest.json
│       ├── retrieval_results.jsonl
│       ├── answer_results.jsonl
│       ├── metrics.json
│       ├── latency.json
│       ├── comparison.json
│       └── run.log
├── snapshots/             # Data snapshots for run reproducibility
└── logs/                  # Execution & debugging telemetry logs
```

---

## 6. Risk Analysis & Mitigation Strategy

1. **Risk:** Model throughput bottlenecks when evaluating 50+ queries across 3 retrieval modes + answer generation.
   - **Mitigation:**
     - Experiment A (Retrieval-Only): Runs in seconds since embeddings and graph traversals are fast; evaluate all queries across Vector, Graph, and Hybrid.
     - Experiment B (End-to-End RAG): Uses a balanced, representative subset of benchmark queries (or test split) to avoid multi-hour generation runs while preserving statistical validity.
     - Sequential execution prevents GPU/RAM contention on Ollama.
     - Resume functionality in `BenchmarkStorage` allows interrupted runs to pick up without re-running finished queries.
2. **Risk:** Canonical chunk ID divergence between Vector and Graph results.
   - **Mitigation:** `BenchmarkRunner` maps all graph results to their underlying canonical chunk ID using the Phase 8 chunk registry, logging any mapping ambiguities.
3. **Risk:** Small-sample statistical misinterpretation.
   - **Mitigation:** Report non-parametric paired tests (Wilcoxon signed-rank), bootstrap 95% confidence intervals, and explicit sample size caveats as mandated in Part 9.

---

## 7. Baseline Sign-Off

The baseline audit confirms that Phases 1 through 7 are secure, completely verified, and reproducible. Phase 8 implementation is authorized to proceed within `data/phase8_benchmark/`.
