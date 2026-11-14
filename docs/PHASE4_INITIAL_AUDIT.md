# Phase 4 Initial Project Audit & Architecture Blueprint

**Project:** Graph RAG Benchmark  
**Phase:** Phase 4 — Knowledge Graph Construction Engine  
**Audit Date:** 2026-11-13  
**Status:** **AUDIT COMPLETE — READY FOR IMPLEMENTATION**

---

## 1. Executive Summary

Phase 4 of the **Graph RAG Benchmark** project transitions from dense vector retrieval (Phase 3) into structured symbolic knowledge representation by building a comprehensive **Knowledge Graph Construction Engine**.

The engine will consume canonical document chunks produced in Phase 2, send them to local LLM inference (`qwen3:8b`), extract typed entities and grounded relationships, enforce strict evidence verification against source text, conservatively resolve entity identities across chunks, and construct an in-memory, persistent `networkx.MultiDiGraph` with rich provenance.

This audit establishes the baseline architectural state, confirms strict preservation boundaries for Phases 1–3, verifies dependencies, and details the implementation blueprint for Phase 4.

---

## 2. Audit of Existing Components (Phases 1, 2, 3)

### 2.1 Phase 1: Environment & Foundation
- **Configuration (`config.py`):** Centralized `Settings` via `pydantic-settings` reading `.env`. Masks sensitive tokens, normalizes URLs, validates positive integer parameters.
- **Logging (`logger.py`):** Configures colored terminal logging and file logging at `logs/graphrag.log`.
- **Ollama Client (`ollama_client.py`):** Synchronous HTTP client handling `/api/tags`, `/api/generate`, and `/api/embed` with timeout and exception taxonomy (`OllamaConnectionError`, `OllamaTimeoutError`, `OllamaModelNotFoundError`, `OllamaAPIError`).
- **Health Checks (`health.py`):** Validates Python 3.11, virtual environment, package dependencies, storage directories, and model availability.

### 2.2 Phase 2: Document Ingestion & Preprocessing
- **Data Schemas (`schemas.py`):** Typed models `DocumentMetadata`, `DocumentChunk`, `DocumentIngestionResult`.
- **Validation & Extraction (`document_validator.py`, `document_loader.py`):** Ingests PDF (`pypdf`), DOCX (`python-docx`), TXT, and Markdown with path sanitization, size checks, and stream verification.
- **Normalization & Chunking (`text_cleaner.py`, `chunking.py`):** NFKC cleaning, sliding-window word chunker (default 400 words, 60 overlap), exact character offsets (`full_text[start_char:end_char] == chunk.text`), and PDF page mapping.
- **Repository (`document_repository.py`):** Canonical storage at `data/uploads/`, metadata at `data/processed/documents.json`, chunks at `data/processed/chunks.jsonl`, and auto-rebuilding `data/processed/manifest.json`.

### 2.3 Phase 3: Embeddings & Vector Database
- **Embedding Service (`embedding_service.py`):** Wraps Ollama `/api/embed` using `nomic-embed-text` (768 dimensions), dynamic dimension check, NaN verification, and batching.
- **Vector Store (`vector_store.py`):** Persistent ChromaDB (`data/chroma/`, collection `graphrag_documents`) using HNSW cosine distance metric. Distinguishes raw cosine similarity ($1 - d$) and normalized score ($1 - d/2$).
- **Indexing Pipeline (`indexing_pipeline.py`):** Incremental diffing against `DocumentRepository`, skipping previously indexed chunk IDs.
- **Semantic Search (`semantic_search.py`):** Top-K retrieval with document and file-type filters.

---

## 3. Storage & Integration Analysis

### 3.1 Canonical Chunk Storage & Retrieval
- Phase 2 canonical chunks reside in `data/processed/chunks.jsonl` and metadata in `data/processed/documents.json`.
- `DocumentRepository.list_documents()` returns all ingested document metadata.
- `DocumentRepository.get_chunks(document_id)` returns ordered `list[DocumentChunk]` for any document.
- **Crucial Rule:** The Phase 4 Knowledge Graph Construction Engine will read chunks solely through `DocumentRepository` without re-reading or re-chunking raw source files.

### 3.2 Ollama LLM Generation & Structured Output
- Current `OllamaClient.generate` accepts `prompt`, `model`, `system`, and `options`.
- To support robust entity-relationship extraction, `OllamaClient.generate` will be extended with an optional `format` parameter (`format="json"` or a JSON schema dictionary).
- This addition will be strictly backward-compatible with Phase 1 and Phase 3 generation calls.

### 3.3 ChromaDB State & Preservation
- ChromaDB is populated with 3 canonical document vectors in `data/chroma/`.
- **Scope Boundary:** Phase 4 must **never** modify, reset, or overwrite `data/chroma/` or Phase 2 canonical storage. Graph data will be strictly isolated in `data/knowledge_graph/`.

---

## 4. Dependencies & Environmental Requirements

### 4.1 Required New Dependencies
1. **`networkx` (`networkx>=3.0,<4.0.0`):** Graph data structure (`MultiDiGraph`), graph algorithms, and GraphML serialization.
2. **`plotly` (`plotly>=5.18.0,<6.0.0`):** Interactive 2D graph visualization in Streamlit.

### 4.2 Dependency File Updates
- Add `networkx>=3.0,<4.0.0` and `plotly>=5.18.0,<6.0.0` to `requirements.txt`.
- Add identical constraints to `pyproject.toml`.
- Install into `.venv`.

---

## 5. Phase 4 Target Architecture

```
                 [DocumentRepository (Phase 2 Chunks)]
                                   │
                                   ▼
                       [GraphBuildingPipeline]
                                   │
                 ┌─────────────────┴─────────────────┐
                 │ (Unprocessed chunks)              │ (Manifest diff)
                 ▼                                   ▼
   [GraphExtractionService] ──────────► [ExtractionValidator]
        (Qwen3:8b via Ollama)              │ (Exact quote & support checks)
                                           ▼
                                   [EntityResolver]
                                   │ (NFKC, case, type-safe aliases)
                                   ▼
                              [GraphStore]
                       (networkx.MultiDiGraph)
                                   │
            ┌──────────────────────┼──────────────────────┐
            ▼                      ▼                      ▼
  [knowledge_graph.json] [knowledge_graph.graphml] [extractions.jsonl]
            ▲                      ▲                      ▲
            └──────────────────────┴──────────────────────┘
                                   │
                                   ▼
                   [Streamlit Graph Explorer Tab]
```

### 5.1 New Components
1. **`graph_schemas.py`:** Pydantic models `ExtractedEntity`, `ExtractedRelationship`, `ChunkGraphExtraction`, `GraphEntity`, `GraphRelationship`.
2. **`graph_extraction_service.py`:** Prompt construction, Qwen3:8b invocation with JSON structured format, fallback handling, latency recording.
3. **`extraction_validator.py` (or integrated in service):** Strict evidence validation: quote must be exact substring of chunk; mentions must appear in chunk; confidence in $[0, 1]$; valid entity types.
4. **`entity_resolver.py`:** Canonicalization, NFKC/case-insensitive matching, type-compatibility gating, deterministic entity ID generation.
5. **`graph_store.py`:** `networkx.MultiDiGraph` wrapper, node/edge management, statistics, JSON serialization, primitive GraphML export, atomic persistence.
6. **`graph_builder.py`:** Pipeline coordinator consuming canonical chunks, diffing against manifest, batching, updating graph, supporting incremental updates and rebuilds.
7. **Streamlit Integration (`app.py`):** Fourth tab `🕸️ Knowledge Graph Explorer` with overview, extraction triggers, Plotly interactive visualization, entity explorer, relationship explorer, and export controls.

---

## 6. Implementation Plan & Milestones

1. **Phase 4 Initial Audit & Cleanup Reports:** Document architecture and clean up temporary files.
2. **Dependency Management:** Add `networkx` and `plotly` to `requirements.txt` and install into `.venv`.
3. **Data Schemas (`graph_schemas.py`):** Define validated Pydantic models for extraction and graph storage.
4. **LLM Extraction Service & Prompting (`graph_extraction_service.py`):** Design grounded extraction prompt and Ollama structured JSON interface.
5. **Evidence Validation & Filtering:** Ensure zero hallucinated edges; enforce exact substring match for supporting evidence.
6. **Entity Resolution Engine (`entity_resolver.py`):** Implement conservative type-safe entity normalization and deduplication.
7. **Graph Store & MultiDiGraph (`graph_store.py`):** NetworkX multi-edge graph manager, JSON and GraphML export.
8. **Graph Building Pipeline (`graph_builder.py`):** Incremental builder with manifest tracking and deletion synchronization.
9. **Configuration & Health Updates:** Update `config.py` and `health.py` for graph paths and dependencies.
10. **Streamlit Graph Explorer (`app.py`):** Implement Phase 4 interactive tab with Plotly visualization.
11. **Unit & Integration Testing:** Author comprehensive test suites (`test_graph_schemas.py`, `test_graph_extraction_service.py`, `test_entity_resolver.py`, `test_graph_store.py`, `test_graph_builder.py`, `test_integration_phase4.py`).
12. **End-to-End Verification & Documentation:** Extract real graph from canonical chunks, verify visualization, execute regression suite, and generate completion reports.
