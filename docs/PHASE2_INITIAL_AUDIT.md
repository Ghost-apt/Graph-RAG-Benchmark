# Phase 2 Initial Project Audit

**Project:** Graph RAG Benchmark  
**Phase:** Phase 2 — Document Ingestion & Preprocessing Engine  
**Audit Timestamp:** 2026-11-13T12:21:00+05:30  
**Workspace Path:** `C:\VS CODE\Graph RAG Benchmark`  

---

## 1. Executive Summary

A comprehensive recursive audit of the workspace at `C:\VS CODE\Graph RAG Benchmark` was conducted prior to beginning Phase 2 implementation. Phase 1 environment setup and foundation is complete, verified, and operational.

Key audit findings:
- **Phase 1 Modules:** Stable, well-tested foundation in `src/graphrag_bench/` (`config.py`, `logger.py`, `ollama_client.py`, `health.py`, `__init__.py`).
- **Runtime Environment:** Python 3.11.9 in `.venv/` with all Phase 1 packages installed and passing 49 unit and integration tests.
- **Existing Document Processing Code:** None currently present. Document parsing, validation, cleaning, chunking, and storage must be built cleanly from the ground up for Phase 2.
- **Dependency State:** Need to add `pypdf>=4.0.0` and `python-docx>=1.1.0` to `requirements.txt` and install into `.venv`.
- **Target Formats:** PDF, DOCX, TXT, MD (Markdown).

---

## 2. Inventory of Existing Code Assets & Reusability

| Component | Path | Current State | Reusability in Phase 2 |
|-----------|------|---------------|-------------------------|
| Application Entry | `app.py` | Complete Phase 1 dashboard & CLI | Extend with dedicated Document Ingestion tab/views |
| Configuration | `src/graphrag_bench/config.py` | Validated Pydantic Settings | Add storage paths, chunk sizes, upload limits |
| Logging Subsystem | `src/graphrag_bench/logger.py` | Unified console & file logger | Direct reuse for document processing audit logs |
| Health Check Engine | `src/graphrag_bench/health.py` | 8 system & model checks | Add `pypdf` & `python-docx` dependency validation |
| Ollama REST Client | `src/graphrag_bench/ollama_client.py` | Complete Qwen & nomic-embed | Preserved intact for Phase 3 embeddings |
| Verification Script | `scripts/verify_environment.py` | Complete Phase 1 verifier | Preserved |
| Test Suite | `tests/` | 44 unit + 5 live tests | Preserved; add 6 new test modules for Phase 2 |

---

## 3. Dependency Audit & Assessment

Existing packages:
- `requests>=2.31.0`
- `python-dotenv>=1.0.0`
- `pydantic>=2.6.0`
- `pydantic-settings>=2.2.0`
- `streamlit>=1.32.0`
- `pytest`, `pytest-mock`, `requests-mock`, `ruff`

Required additions for Phase 2:
- `pypdf>=4.0.0,<6.0.0`: Pure Python PDF parsing, text extraction, page tracking, encryption check.
- `python-docx>=1.1.0,<2.0.0`: Pure Python DOCX extraction for paragraphs and tables.
- Standard libraries: `hashlib` (SHA-256), `unicodedata`, `re`, `json`, `csv`, `pathlib`, `tempfile`.

No dependency conflicts detected. Both `pypdf` and `python-docx` are lightweight, pure-Python, and fully compatible with Python 3.11.

---

## 4. Missing Components for Phase 2

1. **Schemas (`src/graphrag_bench/schemas.py`):**
   - `DocumentMetadata`: Typed document metadata with SHA-256, page count, word count, timestamps.
   - `DocumentChunk`: Typed chunk model with document ID, chunk index, word count, offsets, source filename, page reference.
   - `DocumentIngestionResult`: Overall ingestion outcome with warnings, errors, and duplicate indicators.

2. **Document Validator (`src/graphrag_bench/document_validator.py`):**
   - File extension, size (25 MB max), magic byte/signature checks, malformed file detection, path traversal prevention, encrypted PDF detection, scanned PDF detection.

3. **Document Loader (`src/graphrag_bench/document_loader.py`):**
   - PDF extractor (`pypdf`), DOCX extractor (`python-docx`), TXT extractor (UTF-8, UTF-8 BOM, legacy fallback), Markdown extractor (structure preserving).
   - Unified `ExtractedDocument` and `ExtractedPage` data models.

4. **Text Cleaner (`src/graphrag_bench/text_cleaner.py`):**
   - Unicode NFKC normalization, null/control character stripping, newline normalization (`\n`), whitespace collapsing, paragraph preservation, markdown syntax preservation.

5. **Chunking Engine (`src/graphrag_bench/chunking.py`):**
   - Word-based text chunker (default: 400 words, 60 overlap), offset preservation, page number mapping, deterministic chunk ID generation.

6. **Storage & Repository (`src/graphrag_bench/document_repository.py`):**
   - `data/uploads/`: Original uploaded documents.
   - `data/processed/`: `documents.json`, `chunks.jsonl`, `manifest.json`.
   - Atomic updates, duplicate index, corrupt manifest recovery, safe deletion.

7. **Document Processing Pipeline (`src/graphrag_bench/document_processor.py`):**
   - End-to-end pipeline linking validation, extraction, cleaning, chunking, and persistence.

8. **Dashboard UI Integration (`app.py`):**
   - Retain Phase 1 health/diagnostic tabs and add Phase 2 Ingestion, Chunk Explorer, Document Library, and Export views.

9. **Automated Testing Suite (`tests/`):**
   - `test_document_validator.py`
   - `test_document_loader.py`
   - `test_text_cleaner.py`
   - `test_chunking.py`
   - `test_document_repository.py`
   - `test_document_processor.py`

---

## 5. Recommended Implementation Plan

1. Create `docs/PHASE2_INITIAL_AUDIT.md` and `docs/PHASE2_CLEANUP_REPORT.md`.
2. Add `pypdf` and `python-docx` to `requirements.txt` and install into `.venv`.
3. Implement `schemas.py` with strict Pydantic v2 data models.
4. Implement `document_validator.py` with robust signature and security checks.
5. Implement `document_loader.py` covering PDF, DOCX, TXT, and Markdown.
6. Implement `text_cleaner.py` with deterministic normalization.
7. Implement `chunking.py` with word-based sliding window and page mapping.
8. Implement `document_repository.py` for persistent local storage in `data/`.
9. Implement `document_processor.py` for orchestrated execution.
10. Update `config.py` and `health.py` for Phase 2 settings and dependency verification.
11. Update `app.py` with the complete Phase 2 UI tabs while preserving Phase 1 diagnostics.
12. Build and run comprehensive unit tests in `tests/`.
13. Run `ruff` linting and formatting.
14. Perform live Streamlit application verification.
15. Produce `docs/PHASE2_COMPLETION_REPORT.md` and update `README.md`.
