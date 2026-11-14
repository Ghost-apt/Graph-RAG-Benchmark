# Phase 3 Initial Audit & Architectural Plan

**Project:** Graph RAG Benchmark  
**Phase:** Phase 3 — Vector Embeddings & ChromaDB Vector Store Engine  
**Date:** 2026-11-13  
**Status:** **AUDIT COMPLETE — READY FOR IMPLEMENTATION**

---

## 1. Current Project State & Reusable Components

### Phase 1 Components (Preserved & Active)
- **`src/graphrag_bench/ollama_client.py`**:
  - Already supports `/api/embed` with batching, timeouts, error taxonomy (`OllamaConnectionError`, `OllamaModelNotFoundError`, `OllamaTimeoutError`, `OllamaAPIError`), and validation.
  - Verified working locally against `nomic-embed-text` (768 dimensions) and `qwen3:8b`.
- **`src/graphrag_bench/config.py`**:
  - Loads environment settings via `pydantic-settings`.
  - Need to extend with ChromaDB settings: `chroma_path`, `chroma_collection_name`, `embedding_batch_size`, `retrieval_top_k`.
- **`src/graphrag_bench/health.py`**:
  - Automated environment checks. Need to add ChromaDB dependency and storage directory check.

### Phase 2 Components (Preserved & Active)
- **`src/graphrag_bench/schemas.py`**:
  - `DocumentChunk` contains all necessary attributes: `chunk_id`, `document_id`, `chunk_index`, `text`, `word_count`, `char_count`, `source_filename`, `file_type`, `page_number`, `start_char`, `end_char`.
- **`src/graphrag_bench/document_repository.py`**:
  - Master persistent store with `chunks.jsonl`, `documents.json`, and `manifest.json`.
  - Single source of truth for canonical document and chunk records.
- **`data/sample_documents/` & `data/processed/`**:
  - Populated with clean benchmark documents (`rag_architectures_overview.txt`, `graph_rag_benchmarking_guide.md`, `sample_research_paper.docx`) with 1 chunk each (total 3 chunks).

---

## 2. Dependencies Assessment for Phase 3

- **Target Library:** `chromadb`
- **Compatibility:** Requires Python 3.11 compatibility.
- **Requirements Update:** Add `chromadb>=0.4.22,<1.0.0` to `requirements.txt` and `pyproject.toml`.
- **Virtual Environment:** Install into `.venv` using `pip install chromadb`.

---

## 3. Planned Architecture for Phase 3

```
src/graphrag_bench/
├── embedding_service.py     # Wraps OllamaClient.embed with latency tracking, dimension validation
├── vector_store.py          # ChromaDB PersistentClient interface, cosine distance, metadata filtering
├── indexing_pipeline.py     # Incremental chunk indexer, batch manager, index metadata tracking
└── semantic_search.py       # Natural-language query retrieval, top-K ranking, similarity scoring
```

### 3.1 `EmbeddingService` (`embedding_service.py`)
- Reuses the existing `OllamaClient` (does not create an unnecessary second client).
- Dynamically validates output vector dimensions without assuming 768 unconditionally.
- Measures latency (ms) per call.
- Validates vector numerical types (floats, non-null, finite).

### 3.2 `VectorStore` (`vector_store.py`)
- Uses `chromadb.PersistentClient(path=chroma_path)`.
- Default collection: `graphrag_documents` with `metadata={"hnsw:space": "cosine"}`.
- Sanitizes chunk metadata to ChromaDB-compatible primitives (`str`, `int`, `float`, `bool`).
- Safe null handling: omits `None` fields or converts them safely.
- Stores:
  - `id`: `chunk.chunk_id`
  - `document`: `chunk.text`
  - `embedding`: generated float vector
  - `metadata`: `chunk.metadata`
- Supports similarity query with cosine distance and metadata filters (`document_id`, `file_type`).

### 3.3 `IndexingPipeline` (`indexing_pipeline.py`)
- Reads canonical chunks from `DocumentRepository`.
- Computes difference: `unindexed_chunks = all_chunks - indexed_chunk_ids`.
- Batches embeddings (`batch_size=32`).
- Inserts vectors into `VectorStore`.
- Tracks embedding model and dimensions in collection metadata.
- Prevents cross-model corruption (detects model mismatch).

### 3.4 `SemanticSearch` (`semantic_search.py`)
- Accepts query string, validates non-empty.
- Generates query embedding via `EmbeddingService`.
- Queries `VectorStore` for top-K neighbors.
- Formats results into structured retrieval objects containing: rank, score/distance, text, citations, chunk ID, doc ID, page.

---

## 4. Safety & Scope Rules
- **No ChromaDB deletions of Phase 2 data:** Vector store is a derived index; `data/processed/` remains canonical.
- **No LLM Answer Generation in Phase 3:** Retrieval only; LLM QA benchmarking is reserved for Phase 5.
- **No Knowledge Graph / NetworkX in Phase 3:** Reserved strictly for Phase 4.
