# Phase 6 Safe Cleanup & Data Preservation Report

**Project:** Graph RAG Benchmark  
**Phase:** Phase 6 — Hybrid Retrieval & Reciprocal Rank Fusion  
**Date:** 2026-11-13  
**Status:** VERIFIED & IMMUTABLE  

---

## 1. Executive Summary

Phase 6 implements the Hybrid Retrieval Engine by combining the existing Vector Retriever (Phase 3) and Graph Retriever (Phase 5). To ensure reproducibility and integrity, all pre-existing datasets and assets are strictly protected against modification, deletion, or recomputation.

- **Phase 2 Canonical Chunks:** Completely preserved in `data/processed/` (SHA-256 content hashes, chunk offsets, and manifest verified).
- **Phase 3 ChromaDB Vector Store:** Completely preserved in `data/chroma/` (3 vectors in `graphrag_documents` untouched).
- **Phase 4 Canonical Knowledge Graph:** Completely preserved in `data/knowledge_graph/` (22 entities, 20 relationships untouched).
- **Phase 1–5 Retrieval Components:** Completely preserved without architectural redesign.

---

## 2. Protected Baseline Inventory

| Subsystem | Canonical Path | Description | Immutability Status |
|:---|:---|:---|:---:|
| **Phase 2 Documents** | `data/processed/documents.json` | Master document registry | Read-Only |
| **Phase 2 Chunks** | `data/processed/chunks.jsonl` | Canonical text chunks | Read-Only |
| **Phase 2 Manifest** | `data/processed/manifest.json` | Repository master index | Read-Only |
| **Phase 3 ChromaDB** | `data/chroma/chroma.sqlite3` | Persistent vector index storage | Read-Only |
| **Phase 4 Graph JSON** | `data/knowledge_graph/knowledge_graph.json` | Canonical Knowledge Graph (22 nodes, 20 edges) | Read-Only |
| **Phase 4 GraphML** | `data/knowledge_graph/knowledge_graph.graphml` | Sanitized GraphML export | Read-Only |
| **Phase 4 Manifest** | `data/knowledge_graph/manifest.json` | Construction state manifest | Read-Only |
| **Phase 4 Extractions** | `data/knowledge_graph/extractions.jsonl` | Granular extraction log | Read-Only |

---

## 3. Planned File Additions and Modifications

### 3.1 Existing Files to be Modified Safely
- `src/graphrag_bench/config.py`: Add Phase 6 hybrid configuration parameters with validators.
- `src/graphrag_bench/__init__.py`: Export Phase 6 schemas, normalizer, RRF engine, and hybrid retriever; bump package version to `0.6.0`.
- `.env.example`: Add Phase 6 environment variable templates.
- `app.py`: Add Tab 6: `⚖️ Hybrid Retrieval Lab` preserving existing Tabs 1–5.
- `README.md`: Update architecture documentation, quickstart, and test summaries.

### 3.2 New Phase 6 Source Modules to be Created
- `src/graphrag_bench/hybrid_schemas.py`: Pydantic models for normalized candidates, RRF candidates, and responses.
- `src/graphrag_bench/result_normalizer.py`: Vector and Graph result normalization with canonical chunk mapping.
- `src/graphrag_bench/reciprocal_rank_fusion.py`: Standard and weighted Reciprocal Rank Fusion implementation.
- `src/graphrag_bench/hybrid_context_builder.py`: Unified multi-source evidence packaging for downstream RAG.
- `src/graphrag_bench/hybrid_retriever.py`: Hybrid search coordinator (`search_hybrid`).

### 3.3 New Phase 6 Test Modules to be Created
- `tests/test_hybrid_schemas.py`: Schema validation, numerical bounds, and serialization tests.
- `tests/test_result_normalizer.py`: Normalization tests for vector and graph results, chunk ID validation, and deduplication.
- `tests/test_reciprocal_rank_fusion.py`: RRF calculation tests, weighted fusion, tie-breaking, and manual score verification.
- `tests/test_hybrid_context_builder.py`: Multi-source context assembly and citation consolidation tests.
- `tests/test_hybrid_retriever.py`: End-to-end coordinator tests, partial failure handling, and filter tests.

---

## 4. Verification Checkpoint

Zero destructive operations have been executed. All canonical data remains in its validated Phase 1–5 state.
