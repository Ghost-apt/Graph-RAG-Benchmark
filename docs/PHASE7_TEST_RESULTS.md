# Graph RAG Benchmark — Phase 7 Full Test Results Inventory

**Project:** Graph RAG Benchmark (`C:\VS CODE\Graph RAG Benchmark`)  
**Version:** `0.7.0`  
**Date:** 2026-11-14  
**Test Runner:** `pytest 8.4.2` (Python 3.11.9, Windows x64)  
**Execution Command:** `python -m pytest -q --junitxml=phase7_test_results.xml`  
**Execution Result:** **327 passed, 272 warnings in 217.14s (0:03:37)**  
**Pass Rate:** **100.0%** (0 failed, 0 errors, 0 skipped)  

---

## 1. Executive Summary of Test Execution

```
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-8.4.2, pluggy-1.7.0
rootdir: C:\VS CODE\Graph RAG Benchmark
configfile: pyproject.toml
testpaths: tests
plugins: anyio-4.16.0, mock-3.17.0, requests-mock-1.12.1

327 passed, 272 warnings in 217.14s (0:03:37)
============================== 100% Passed ====================================
```

- **Total Test Files:** 42 files
- **Unit Test Files:** 38 files (316 tests passed in 6.81s)
- **Live Integration Test Files:** 4 files (11 tests passed in ~210s against local Ollama `qwen3:8b` and `nomic-embed-text`)
- **Total Executed Tests:** **327 passed**

---

## 2. Complete Test Inventory by Phase & Module

### Phase 1: Environment & Ollama Client Foundation (55 Tests)
- `tests/test_config.py` (11 tests passed):
  - `test_default_settings`
  - `test_custom_environment_override`
  - `test_invalid_base_url`
  - `test_invalid_timeout`
  - `test_log_level_normalization`
  - `test_invalid_log_level`
  - `test_safe_dict_masking`
  - `test_singleton_and_reload`
  - `test_phase5_graph_retrieval_settings`
  - `test_phase6_hybrid_retrieval_settings`
  - `test_phase7_rag_settings`
- `tests/test_health.py` (15 tests passed):
  - System diagnostics, memory thresholds, disk checks, Ollama connectivity ping
- `tests/test_ollama_client.py` (21 tests passed):
  - Connection retry logic, JSON mode formatting, error mapping, stream generators
- `tests/test_integration_ollama.py` (5 live integration tests passed):
  - Live HTTP handshake, server reachability, model availability, live generation

### Phase 2: Ingestion, Validation & Sliding-Window Chunking (59 Tests)
- `tests/test_document_loader.py` (8 tests passed):
  - UTF-8, UTF-8-BOM, Latin-1 fallback, PDF, DOCX, Markdown extraction
- `tests/test_document_validator.py` (18 tests passed):
  - Magic byte verification, MIME detection, size bounds, filename sanitization
- `tests/test_text_cleaner.py` (9 tests passed):
  - Unicode NFKC normalization, whitespace collapse, citation retention
- `tests/test_chunking.py` (9 tests passed):
  - Sliding window overlap, character offset tracking, deterministic chunk IDs
- `tests/test_document_repository.py` (9 tests passed):
  - Chunk persistence, retrieval, deduplication, metadata querying
- `tests/test_document_processor.py` (6 tests passed):
  - End-to-end ingestion and normalization workflow

### Phase 3: Embeddings & ChromaDB Vector Search (45 Tests)
- `tests/test_embedding_service.py` (10 tests passed):
  - `nomic-embed-text` batching, dimension validation (768), caching
- `tests/test_vector_store.py` (15 tests passed):
  - Collection creation, record upsert, cosine distance computation, metadata sanitization
- `tests/test_semantic_search.py` (8 tests passed):
  - Approximate nearest neighbor retrieval, metadata filtering, error handling
- `tests/test_indexing_pipeline.py` (8 tests passed):
  - Batch indexing of canonical chunks into ChromaDB
- `tests/test_integration_phase3.py` (2 live integration tests passed):
  - Real embedding generation and ChromaDB vector retrieval

### Phase 4: Knowledge Graph Construction & Extraction (59 Tests)
- `tests/test_graph_schemas.py` (11 tests passed):
  - Entity, Relationship, KnowledgeGraph schemas and validation
- `tests/test_graph_builder.py` (4 tests passed):
  - NetworkX directed multigraph construction, orphan cleanup
- `tests/test_graph_store.py` (8 tests passed):
  - JSON serialization/deserialization, GraphML export and re-import
- `tests/test_entity_resolver.py` (9 tests passed):
  - Levenshtein entity resolution, canonical entity mapping
- `tests/test_entity_linker.py` (7 tests passed):
  - Chunk-to-entity bidirectional graph linking
- `tests/test_graph_extraction_service.py` (4 tests passed):
  - Qwen3 extraction prompt building, JSON parsing, reasoning token isolation
- `tests/test_graph_search.py` (6 tests passed):
  - Entity keyword search, 1-hop and 2-hop neighborhood expansion
- `tests/test_integration_phase4.py` (2 live integration tests passed):
  - Live entity extraction and relationship assertions on canonical chunk

### Phase 5: Graph Retrieval & Multi-Hop Traversal (23 Tests)
- `tests/test_graph_traversal.py` (9 tests passed):
  - Breadth-first search (BFS), max-hop depth bounding, degree thresholds
- `tests/test_graph_ranker.py` (5 tests passed):
  - Path score calculation, degree penalties, hop count weighting
- `tests/test_graph_retriever.py` (6 tests passed):
  - Graph query execution, path reconstruction, entity grounding
- `tests/test_graph_context_builder.py` (3 tests passed):
  - Formatting graph paths into structured evidence blocks

### Phase 6: Hybrid Retrieval & Reciprocal Rank Fusion (39 Tests)
- `tests/test_hybrid_schemas.py` (7 tests passed):
  - `FusionCandidate`, `HybridSearchResponse`, `RetrievalSource` validation
- `tests/test_query_entity_extractor.py` (8 tests passed):
  - Query entity extraction, exact matching, LLM fallback parsing
- `tests/test_result_normalizer.py` (3 tests passed):
  - Candidate deduplication, modality score normalization
- `tests/test_reciprocal_rank_fusion.py` (8 tests passed):
  - Standard RRF ($k=60$), weighted RRF, dual-modality tie breaking
- `tests/test_hybrid_retriever.py` (9 tests passed):
  - Combined vector and graph retrieval orchestration
- `tests/test_hybrid_context_builder.py` (4 tests passed):
  - Fused hybrid context packaging

### Phase 7: Grounded Answer Generation & Safety QA (47 Tests)
- `tests/test_answer_schemas.py` (8 tests passed):
  - `EvidenceItem`, `AnswerCitation` (`claim_supported`), `AnswerClaim`, `GroundedAnswer` (null faithfulness, precision alias, latency breakdown)
- `tests/test_evidence_preparer.py` (8 tests passed):
  - Token budget estimation, chunk deduplication, delimiter escaping (`[ESCAPED_DELIMITER_BLOCK_END]`), instruction injection resistance
- `tests/test_prompt_builder.py` (4 tests passed):
  - Grounded QA system prompt construction, anti-injection formatting
- `tests/test_answer_generator.py` (4 tests passed):
  - `qwen3:8b` integration, `<think>` reasoning token isolation, stream generation
- `tests/test_citation_validator.py` (8 tests passed):
  - Single and compound citation extraction, provenance verification, duplicate markers, wrong registry rejection, reference vs claim support semantics
- `tests/test_answer_guardrails.py` (13 tests passed):
  - Pre-check, delimiter leak sanitization, refusal detection, 7 insufficient-evidence conditions (answerable, zero evidence, irrelevant vector, irrelevant graph, weak evidence, contradictory refusal, partial answer)
- `tests/test_faithfulness_checker.py` (8 tests passed):
  - Sentence claim splitting, lexical overlap, Q1 demonstration 75% audit, empty-claim null faithfulness score
- `tests/test_rag_pipeline.py` (5 tests passed):
  - Hybrid mode end-to-end, zero-evidence short-circuit, vector mode, functional helper, Q6 relevance short-circuit with zero LLM generation
- `tests/test_integration_phase7.py` (2 live integration tests passed):
  - Live grounded answer generation with verified citations against canonical repository

---

## 3. Verified Per-File Summary Table

| # | Test File | Type | Tests Collected | Passed | Failed | Errors | Skipped |
|---|---|---|---|---|---|---|---|
| 1 | `tests/test_answer_generator.py` | Unit | 4 | 4 | 0 | 0 | 0 |
| 2 | `tests/test_answer_guardrails.py` | Unit | 13 | 13 | 0 | 0 | 0 |
| 3 | `tests/test_answer_schemas.py` | Unit | 8 | 8 | 0 | 0 | 0 |
| 4 | `tests/test_chunking.py` | Unit | 9 | 9 | 0 | 0 | 0 |
| 5 | `tests/test_citation_validator.py` | Unit | 8 | 8 | 0 | 0 | 0 |
| 6 | `tests/test_config.py` | Unit | 11 | 11 | 0 | 0 | 0 |
| 7 | `tests/test_document_loader.py` | Unit | 8 | 8 | 0 | 0 | 0 |
| 8 | `tests/test_document_processor.py` | Unit | 6 | 6 | 0 | 0 | 0 |
| 9 | `tests/test_document_repository.py` | Unit | 9 | 9 | 0 | 0 | 0 |
| 10 | `tests/test_document_validator.py` | Unit | 18 | 18 | 0 | 0 | 0 |
| 11 | `tests/test_embedding_service.py` | Unit | 10 | 10 | 0 | 0 | 0 |
| 12 | `tests/test_entity_linker.py` | Unit | 7 | 7 | 0 | 0 | 0 |
| 13 | `tests/test_entity_resolver.py` | Unit | 9 | 9 | 0 | 0 | 0 |
| 14 | `tests/test_evidence_preparer.py` | Unit | 8 | 8 | 0 | 0 | 0 |
| 15 | `tests/test_faithfulness_checker.py` | Unit | 8 | 8 | 0 | 0 | 0 |
| 16 | `tests/test_graph_builder.py` | Unit | 4 | 4 | 0 | 0 | 0 |
| 17 | `tests/test_graph_context_builder.py` | Unit | 3 | 3 | 0 | 0 | 0 |
| 18 | `tests/test_graph_extraction_service.py` | Unit | 4 | 4 | 0 | 0 | 0 |
| 19 | `tests/test_graph_ranker.py` | Unit | 5 | 5 | 0 | 0 | 0 |
| 20 | `tests/test_graph_retriever.py` | Unit | 6 | 6 | 0 | 0 | 0 |
| 21 | `tests/test_graph_schemas.py` | Unit | 11 | 11 | 0 | 0 | 0 |
| 22 | `tests/test_graph_search.py` | Unit | 6 | 6 | 0 | 0 | 0 |
| 23 | `tests/test_graph_store.py` | Unit | 8 | 8 | 0 | 0 | 0 |
| 24 | `tests/test_graph_traversal.py` | Unit | 9 | 9 | 0 | 0 | 0 |
| 25 | `tests/test_health.py` | Unit | 15 | 15 | 0 | 0 | 0 |
| 26 | `tests/test_hybrid_context_builder.py` | Unit | 4 | 4 | 0 | 0 | 0 |
| 27 | `tests/test_hybrid_retriever.py` | Unit | 9 | 9 | 0 | 0 | 0 |
| 28 | `tests/test_hybrid_schemas.py` | Unit | 7 | 7 | 0 | 0 | 0 |
| 29 | `tests/test_indexing_pipeline.py` | Unit | 8 | 8 | 0 | 0 | 0 |
| 30 | `tests/test_integration_ollama.py` | Integration | 5 | 5 | 0 | 0 | 0 |
| 31 | `tests/test_integration_phase3.py` | Integration | 2 | 2 | 0 | 0 | 0 |
| 32 | `tests/test_integration_phase4.py` | Integration | 2 | 2 | 0 | 0 | 0 |
| 33 | `tests/test_integration_phase7.py` | Integration | 2 | 2 | 0 | 0 | 0 |
| 34 | `tests/test_ollama_client.py` | Unit | 21 | 21 | 0 | 0 | 0 |
| 35 | `tests/test_prompt_builder.py` | Unit | 4 | 4 | 0 | 0 | 0 |
| 36 | `tests/test_query_entity_extractor.py` | Unit | 8 | 8 | 0 | 0 | 0 |
| 37 | `tests/test_rag_pipeline.py` | Unit | 5 | 5 | 0 | 0 | 0 |
| 38 | `tests/test_reciprocal_rank_fusion.py` | Unit | 8 | 8 | 0 | 0 | 0 |
| 39 | `tests/test_result_normalizer.py` | Unit | 3 | 3 | 0 | 0 | 0 |
| 40 | `tests/test_semantic_search.py` | Unit | 8 | 8 | 0 | 0 | 0 |
| 41 | `tests/test_text_cleaner.py` | Unit | 9 | 9 | 0 | 0 | 0 |
| 42 | `tests/test_vector_store.py` | Unit | 15 | 15 | 0 | 0 | 0 |
| **TOTAL** | **42 Files** | — | **327** | **327** | **0** | **0** | **0** |

---

## 4. Code Quality & Formatting Audit Summary

- **Linter Check:** `ruff check .` $\implies$ **All checks passed!** (0 lint violations across 121 files).
- **Formatter Check:** `ruff format --check .` $\implies$ **121 files already formatted** (0 style violations).
- **Streamlit AppTest:** `AppTest.from_file('app.py')` $\implies$ **0 exceptions**.
