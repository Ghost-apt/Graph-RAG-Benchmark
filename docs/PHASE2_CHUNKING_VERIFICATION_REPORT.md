# Phase 2 Chunking Verification & Audit Report

**Project:** Graph RAG Benchmark  
**Task:** Phase 2 Chunking Audit & Controlled Verification  
**Date:** 2026-11-13  
**Status:** **VERIFIED & PASSING**

---

## 1. Executive Summary & Root Cause Investigation

### The Reported Issue
The Phase 2 demonstration logs reported:
- **Document A (`rag_architectures_overview.txt`):** Word count = 163, Chunks = 3
- **Document B (`graph_rag_benchmarking_guide.md`):** Word count = 157, Chunks = 3
- **Document C (`sample_research_paper.docx`):** Word count = 60, Chunks = 2

Under the standard default configuration:
$$\text{CHUNK\_SIZE} = 400 \text{ words}, \quad \text{CHUNK\_OVERLAP} = 60 \text{ words}$$
documents with fewer than 400 words are expected to produce exactly **1 chunk**.

### Investigation Findings
A thorough inspection of:
- `src/graphrag_bench/config.py`
- `src/graphrag_bench/chunking.py`
- `src/graphrag_bench/document_processor.py`
- `data/processed/chunks.jsonl`
- Historical step logs in `transcript.jsonl`

revealed the **exact root cause**:

1. **No algorithmic defect exists in `TextChunker`**:
   The sliding-window word chunking algorithm in `src/graphrag_bench/chunking.py` correctly produces 1 chunk for any document whose word count $\le \text{chunk\_size}$ (400 words).
2. **The demonstration test executed custom chunk arguments**:
   During the Phase 2 end-to-end integration test run, custom chunk parameters were intentionally passed to test non-default chunk sizes:
   ```python
   # Demonstration test call from previous session:
   processor.process_file(txt_bytes, "rag_architectures_overview.txt", chunk_size=80, chunk_overlap=20)
   processor.process_file(md_bytes, "graph_rag_benchmarking_guide.md", chunk_size=70, chunk_overlap=15)
   processor.process_file(docx_bytes, "sample_research_paper.docx", chunk_size=50, chunk_overlap=10)
   ```
   - For `rag_architectures_overview.txt` (163 words): With `chunk_size=80, chunk_overlap=20` (stride 60), the windows are `[0..80]`, `[60..140]`, and `[120..163]` $\to$ **3 chunks** (80, 80, 43 words).
   - For `graph_rag_benchmarking_guide.md` (157 words): With `chunk_size=70, chunk_overlap=15` (stride 55), the windows are `[0..70]`, `[55..125]`, and `[110..157]` $\to$ **3 chunks** (70, 70, 47 words).
   - For `sample_research_paper.docx` (60 words): With `chunk_size=50, chunk_overlap=10` (stride 40), the windows are `[0..50]` and `[40..60]` $\to$ **2 chunks** (50, 20 words).
3. **Inspection of `data/processed/chunks.jsonl`**:
   Line-by-line inspection confirmed:
   - Chunk 0 of Document A had `word_count=80`.
   - Chunk 0 of Document B had `word_count=70`.
   - Chunk 0 of Document C had `word_count=50`.
   This directly confirms that the custom parameters from the test call were stored.

---

## 2. Controlled Verification on Default Configuration (400 / 60)

When processed under the actual default settings (`chunk_size=400, chunk_overlap=60`, $\text{stride} = 340$):
- `rag_architectures_overview.txt` (163 words) $\to$ **1 chunk** (163 words)
- `graph_rag_benchmarking_guide.md` (157 words) $\to$ **1 chunk** (157 words)
- `sample_research_paper.docx` (60 words) $\to$ **1 chunk** (60 words)

The sample documents in `data/processed/` were cleanly re-ingested under the default configuration, resulting in **3 registered documents and exactly 3 total chunks** in [manifest.json](file:///c:/VS%20CODE/Graph%20RAG%20Benchmark/data/processed/manifest.json).

---

## 3. Controlled Word Count Benchmark Matrix

A dedicated test suite was added to `tests/test_chunking.py` evaluating controlled document word counts under `chunk_size=400, chunk_overlap=60`:

| Word Count | Expected Chunks | Actual Chunks | Chunk Word Allocations | Verification Status |
|---|---|---|---|---|
| **60 words** | 1 | 1 | `[60]` | **PASS** |
| **157 words** | 1 | 1 | `[157]` | **PASS** |
| **163 words** | 1 | 1 | `[163]` | **PASS** |
| **400 words** | 1 | 1 | `[400]` | **PASS** |
| **401 words** | 2 | 2 | `[400, 61]` | **PASS** |
| **800 words** | 3* | 3 | `[400, 400, 120]` | **PASS** |
| **1,000 words** | 3 | 3 | `[400, 400, 320]` | **PASS** |

### Mathematical Analysis for 800 Words
- With $\text{chunk\_size}=400$ and $\text{chunk\_overlap}=60$, the forward stride is $\text{stride} = 400 - 60 = 340$ words.
- Chunk 0: words `0..400` (400 words).
- Chunk 1: words `340..740` (400 words).
- At word 740, there are 60 unread words remaining (`740..800`).
- To ensure no document text is dropped and no chunk exceeds the 400-word limit, Chunk 2 covers words `680..800` (120 words with 60-word overlap).
- Hence, 800 words mathematically requires 3 chunks to maintain both complete text coverage and the strict 400-word ceiling.

---

## 4. Integrity Checks Verified

1. **Character Offset Accuracy:**
   For every chunk in all tests:
   $$\text{full\_text}[\text{start\_char} : \text{end\_char}] == \text{chunk.text}$$
2. **Overlap Preservation:**
   Consecutive chunks in multi-chunk documents share exactly the configured number of overlapping words.
3. **Deterministic Chunk Identification:**
   Chunk IDs maintain the schema: `doc_<sha256[:16]>_c<index:04d>_<content_hash[:8]>`.
4. **Source Page Mapping:**
   Midpoint character offsets accurately resolve to 1-based PDF source page numbers.

---

## 5. Regression Test Results

```powershell
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-8.4.2
collected 108 items / 5 deselected / 103 selected
103 passed, 5 deselected in 1.33s
```
- `ruff check .` $\to$ **All checks passed! (0 errors)**
- `ruff format --check .` $\to$ **34 files already formatted (0 changes required)**

---

## 6. Audit Conclusion & Phase 3 Sign-Off

- **Root Cause Identified:** The demonstration script used explicit custom arguments (`80/20`, `70/15`, `50/10`) to test custom slicing; the underlying chunking algorithm adheres strictly to sliding-window specifications.
- **Repository Cleaned:** `data/processed/` has been updated with the 400/60 default ingestions (1 chunk each).
- **Regression Passed:** 103 unit tests pass without error.
- **Phase 3 Readiness:** Objective A is **100% complete**. We now proceed to Objective B: Phase 3 (Embeddings and ChromaDB Vector Search).
