# Phase 8 Completion Report: Formal Comparative Benchmark Evaluation

**Project:** Graph RAG Benchmark  
**Phase:** 8 — Comparative Evaluation: Vector RAG vs. Graph RAG vs. Hybrid RAG  
**Date:** November 2026  
**Workspace:** `C:\VS CODE\Graph RAG Benchmark`  
**Current System Version:** `0.8.0`  
**Evaluation Corpus Root:** `data/phase8_benchmark/`  
**Phase Status:** COMPLETED, AUDITED & RECONCILED  
**Next Phase Authorization:** Phase 9 STRICTLY NOT AUTHORIZED — Strictly Halted  

---

## 1. Executive Summary & Success Criteria Fulfillment

Phase 8 implements a scientifically defensible, reproducible comparative benchmarking engine that empirically evaluates **Vector RAG**, **Knowledge Graph RAG**, and **Hybrid RAG (Reciprocal Rank Fusion)** over an isolated 22-document evaluation corpus.

| Objective / Deliverable | Status | Verification & Evidence |
|---|:---:|---|
| **Strict Preservation of Phases 1–7** | **100% Verified** | Canonical data (`data/processed/`, `data/knowledge_graph/`, Chroma collection `graphrag_documents`) untouched. All 356 automated tests passing. |
| **Isolated Benchmark Environment** | **100% Verified** | Standalone directory `data/phase8_benchmark/` with 22 source docs, 110 canonical chunks, 768-d Chroma vectors, 36-node/24-edge KG, and 60 ground-truth queries. |
| **Benchmark Schemas (`benchmark_schemas.py`)** | **100% Verified** | Pydantic v2 schemas for queries, graded judgments (0–3), IR metrics, answer quality, latency telemetry, paired contrasts, and run manifests. |
| **Benchmark Metrics Engine (`benchmark_metrics.py`)** | **100% Verified** | P@K, R@K, HitRate@K, MRR, nDCG@K ($2^{\text{rel}}-1$ gains), MAP@K at $K \in \{1, 3, 5, 10\}$ with strict denominator conventions and IDCG edge cases. |
| **Benchmark Statistics Engine (`benchmark_statistics.py`)** | **100% Verified** | Latency stats (mean, p50, p95, min, max, std dev), Wilcoxon signed-rank paired tests, 2,000-sample bootstrap 95% CIs, and win/loss/tie accounting. |
| **Benchmark Storage Engine (`benchmark_storage.py`)** | **100% Verified** | Atomic run directories, incremental JSONL appending, resume capabilities, run manifests, and flat CSV exports. |
| **Benchmark Runner Engine (`benchmark_runner.py`)** | **100% Verified** | Modular orchestrator running Experiment A (retrieval-only across 60 queries * 3 strategies = 180 runs) and Experiment B (grounded QA generation). |
| **Streamlit Dashboard (Tab 8 in `app.py`)** | **100% Verified** | Sections A–I implemented: Dataset Overview, Experiment Controls, Retrieval Results, Answer Quality, Latency Breakdown, Statistical Contrasts, Query Inspector, Run History, and Artifact Export. Verified via `AppTest`. |
| **Comprehensive Documentation** | **100% Verified** | `PHASE8_INITIAL_AUDIT.md`, `PHASE8_DATASET_REPORT.md`, `PHASE8_METHODOLOGY.md`, `PHASE8_METRICS_REFERENCE.md`, `PHASE8_COMPLETION_REPORT.md`, `PHASE8_HELDOUT_TEST_RESULTS.md`, `PHASE8_FINAL_VALIDATION_REPORT.md`. |
| **Version Bump & QA Compliance** | **100% Verified** | Bumped to `0.8.0` in `pyproject.toml` and `__init__.py`. Ruff linting and formatting 100% clean. 356 tests passing. |

---

## 2. Strict Preservation Audit

Before and after implementing Phase 8, the canonical Phase 1–7 production repositories and interfaces were audited:
- `data/processed/chunks.jsonl`: Unmodified, exactly 3 canonical chunks (SHA-256 verified).
- `data/processed/documents.json`: Unmodified, exactly 2 canonical documents (SHA-256 verified).
- `data/processed/manifest.json`: Unmodified.
- `data/knowledge_graph/knowledge_graph.json`: Unmodified, exactly **22 entities and 20 relationships** (SHA-256: `2E0C8783848B69C730825C20AB6C0D96D674C4195C809E5F4AA71CA89B0AF6DD` certified 100% bit-for-bit identical to Phase 4 remediation baseline).
- `data/knowledge_graph/knowledge_graph.graphml`: Unmodified, exactly 22 nodes and 20 edges (SHA-256: `4BA25CAA4C8259218FEBB874B28EE278352FF7E1A6FF7CA455D41AB5A9F198ED`).
- ChromaDB collection `graphrag_documents`: Exactly 3 vectors (unmodified).
- Full regression suite across Phases 1–7: **332/332 passed (100%)** (320 unit tests + 12 live integration tests).

---

## 3. Benchmark Dataset Architecture

- **Root Location:** `data/phase8_benchmark/`
  - `raw_documents/`: 22 Markdown documents (`doc01_vector_embeddings.md` to `doc22_case_study_biomedical_graph_rag.md`).
  - `processed/`: 110 canonical chunks (`chunk_doc01_..._0000` to `chunk_doc22_..._0004`), document metadata dictionary, and manifest.
  - `chroma/`: Dedicated ChromaDB collection `phase8_benchmark_documents` (110 vectors, 768 dimensions via `nomic-embed-text`).
  - `knowledge_graph/`: NetworkX graph with 36 entities and 24 relationships with verbatim chunk evidence quotations.
  - `ground_truth/`: `benchmark_dataset.json` and `benchmark_queries.csv` containing 60 queries:
    - 15 Single-Hop queries (25.0%)
    - 15 Multi-Hop queries (25.0%)
    - 10 Entity-Relationship queries (16.7%)
    - 10 Comparative Synthesis queries (16.7%)
    - 10 Unanswerable Rejection queries (16.7%)
    - Split into Dev (15), Validation (15), and Test (30) partitions.
    - 4-point graded relevance scale: Grade 3 (Direct Answer), Grade 2 (Relevant Supporting), Grade 1 (Related/Partial), Grade 0 (Irrelevant).

---

## 4. Empirical Evaluation Findings

### Experiment A: Retrieval-Only Comparative Results (All 60 Queries)
*Macro-averaged evaluation metrics across all 60 benchmark queries (`run_20261114_060224`):*

| Retrieval Modality | P@5 | R@5 | HitRate@5 | MRR | nDCG@5 | nDCG@10 | MAP@10 | Mean Total Latency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Vector RAG** | $0.2633$ | $0.7403$ | $\mathbf{0.8333}$ | $0.7118$ | $\mathbf{0.8877}$ | $\mathbf{0.9161}$ | $\mathbf{0.7118}$ | $\mathbf{94.62\text{ ms}}$ |
| **Graph RAG** | $0.1000$ | $0.2708$ | $0.4167$ | $0.2576$ | $0.4619$ | $0.4634$ | $0.2576$ | $7,542.01\text{ ms}$ |
| **Hybrid RAG (RRF)** | $\mathbf{0.2733}$ | $\mathbf{0.7597}$ | $\mathbf{0.8333}$ | $\mathbf{0.7320}$ | $0.8106$ | $0.8316$ | $0.6002$ | $7,701.63\text{ ms}$ |

#### Confirmatory Held-Out Test Evaluation ($N=30$ Locked Test Queries)
*As documented in `docs/PHASE8_HELDOUT_TEST_RESULTS.md`, on the locked 30-query test set:*
- **All 30 Test Queries:** Vector RAG nDCG@10: $\mathbf{0.9244}$, Hybrid RAG nDCG@10: $0.8381$, Graph RAG nDCG@10: $0.3477$.
- **Answerable-Only Test Partition ($N=24$ Queries):**
  - **Recall@5:** Hybrid RAG achieves highest recall: $\mathbf{0.9236}$ vs $0.9167$ for Vector RAG and $0.1389$ for Graph RAG.
  - **Hit Rate@5:** Both Vector RAG and Hybrid RAG achieve $\mathbf{1.0000}$ (100% success).
- **Unanswerable Test Partition ($N=6$ Queries):**
  - **Graph RAG:** Achieves a $\mathbf{0.0\%}$ false-positive retrieval rate ($6/6$ empty results, $0$ items retrieved).
  - **Vector / Hybrid RAG:** Return nearest neighbors, but all are safely rejected by Phase 7 relevance guardrails with $\mathbf{100.0\%}$ refusal accuracy.

---

### Experiment B: Grounded Answer Generation (Representative Evaluation Sample)
Evaluated with local LLM `qwen3:8b` ($T=0.10$, top $K=5$) on a representative subset of 5 queries from the Validation split (Q05, Q20, Q34, Q43, Q53):

| Metric | Vector RAG | Graph RAG | Hybrid RAG | Methodological Classification |
|---|:---:|:---:|:---:|---|
| **Heuristic Faithfulness Score** | $0.6583$ | $0.6667$ | $\mathbf{0.7292}$ | Automated sentence-level lexical/entity overlap heuristic (`FaithfulnessChecker`). Not formal semantic entailment. |
| **Citation Reference Precision** | $\mathbf{1.0000}$ | $\mathbf{1.0000}$ | $\mathbf{1.0000}$ | Valid citation markers `[S1]`, `[S2]` mapping to active context chunks. |
| **Multi-Hop Citation Coverage (Q20)** | $0.5000$ | $0.0000$ | $\mathbf{1.0000}$ | Hybrid RAG cited both required disjoint source chunks ($100\%$). |
| **Comparative Citation Coverage (Q43)** | $0.2500$ | $0.2500$ | $\mathbf{0.7500}$ | Hybrid RAG synthesized evidence across 3 distinct documents. |
| **Refusal Accuracy (Q53 Unanswerable)** | $\mathbf{1.0000}$ | $\mathbf{1.0000}$ | $\mathbf{1.0000}$ | Correct refusal on unanswerable query with zero hallucination. |

---

## 5. Statistical Significance & Paired Contrasts

### Confirmatory Held-Out Test Set Contrasts ($N=30$ Queries, nDCG@10):
*Evaluated with 2,000 bootstrap resamples and two-sided paired Wilcoxon signed-rank tests with Holm-Bonferroni step-down correction:*

| Contrast ($T$ vs $B$) | Mean Diff ($\Delta$) | Median Diff | 95% Bootstrap CI | Wilcoxon $W$ | Raw $p$-value | Holm-Adjusted $p$-value | Win / Loss / Tie | Statistically Significant? |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Hybrid vs. Vector** | $-0.0863$ | $+0.0000$ | $[-0.1455, -0.0308]$ | $6.0$ | $0.009633$ | $\mathbf{0.009633}$ | 2 Wins / 10 Losses / 18 Ties | **YES ($p < 0.01$, Favors Vector)** |
| **Hybrid vs. Graph** | $+0.4905$ | $+0.5095$ | $[+0.3668, +0.6149]$ | $0.0$ | $0.000027$ | $\mathbf{0.000081}$ | 23 Wins / 0 Losses / 7 Ties | **YES ($p < 0.001$, Favors Hybrid)** |
| **Vector vs. Graph** | $+0.5767$ | $+0.7239$ | $[+0.4327, +0.7187]$ | $0.0$ | $0.000027$ | $\mathbf{0.000081}$ | 23 Wins / 0 Losses / 7 Ties | **YES ($p < 0.001$, Favors Vector)** |

#### Statistical Interpretation:
- Both Vector RAG and Hybrid RAG achieve statistically significant retrieval ranking improvements over Graph RAG ($p_{\text{adj}} = 0.000081$).
- Vector RAG achieves a statistically significant advantage over Hybrid RAG specifically on **nDCG@10 ranking position** ($p_{\text{adj}} = 0.009633$). However, on **Recall@5 on answerable queries**, Hybrid RAG achieves higher recall ($0.9236$ vs $0.9167$) by bridging disjoint documents.

---

## 6. Latency Disparity & Hardware Profiling Analysis

| Metric | Vector RAG | Graph RAG | Hybrid RAG |
|---|:---:|:---:|:---:|
| **Mean Retrieval Latency** | $94.62\text{ ms}$ | $7,542.01\text{ ms}$ | $7,701.63\text{ ms}$ |
| **Median (p50) Retrieval Latency** | $37.68\text{ ms}$ | **$1.35\text{ ms}$** | $41.50\text{ ms}$ |
| **95th Percentile (p95)** | $90.14\text{ ms}$ | $25,641.19\text{ ms}$ | $25,428.03\text{ ms}$ |
| **Minimum Latency** | $22.84\text{ ms}$ | **$0.69\text{ ms}$** | $24.60\text{ ms}$ |

### Attribution of the Bimodal Latency Distribution:
- **Fast Path (38 of 60 queries = 63.3%):** Direct dictionary seed matching executes in-memory NetworkX traversal in **$1.25\text{ ms}$** on average.
- **LLM Fallback Path (22 of 60 queries = 36.7%):** Out-of-domain, unanswerable, or complex queries trigger local Qwen3:8b extraction fallback, incurring **$20,566.96\text{ ms}$** on average.
- **Production Recommendation:** Gazetteers and vector-guided candidate filtering should always gate LLM extraction fallbacks in low-latency systems.

---

## 7. Streamlit Dashboard Integration (Tab 8 in `app.py`)

Tab 8 (`📊 Comparative Benchmark Lab`) provides an enterprise-grade UI:
- **Section A:** Benchmark Corpus Architecture & Metrics Cards (22 docs, 110 chunks, 36 nodes, 24 edges, 60 queries). Expander with table of all 22 technical source documents.
- **Section B:** Experiment Controls (Split selector, strategy checkboxes, K cutoff selector, Run Experiment A / Run Experiment B buttons).
- **Section C:** Retrieval Performance (Summary table comparing P@5, R@5, HitRate@5, MRR, nDCG@5, nDCG@10, MAP@10; Plotly nDCG@K bar chart; Plotly category breakdown).
- **Section D:** Grounded Answer Quality (Faithfulness, citation precision, citation coverage, refusal accuracy table).
- **Section E:** Latency Analysis (Summary table with Mean, p50, p95, Min, Max, Std Dev).
- **Section F:** Statistical Significance Table (Paired contrasts, bootstrap 95% CIs, Wilcoxon $p$-values, win/loss/tie, significance badges).
- **Section G:** Granular Query Inspector (Dropdown selector for all 60 queries with side-by-side tabs for Vector, Graph, and Hybrid retrieved items with rank, score, gold grade, and relevance badge).
- **Section H:** Historical Run Selector (Dropdown to view and reload past experiment runs).
- **Section I:** Artifact Export (Download buttons for JSON manifest, CSV results, and JSONL retrieval logs).

---

## 8. Verification & QA Automation Suite

- **Total Test Suite:** **356 / 356 automated tests passing (100% pass rate)** in $268.50\text{s}$.
  - Phase 1–7 Regression Unit Tests: 320 tests
  - Phase 1–7 Live Integration Tests: 12 tests
  - Phase 8 Specific Tests: 24 tests
- **UI Verification:** `tests/test_benchmark_app.py` passed using Streamlit `AppTest.from_file("app.py")` confirming all 8 tabs render cleanly with 0 exceptions.
- **Linting & Formatting:**
  - `ruff check src/ tests/ scripts/ app.py`: 0 errors.
  - `ruff format --check src/ tests/ scripts/ app.py`: 100% compliant across 106 files.

---

## 9. Final Sign-off Statement

Phase 8 is hereby complete, verified, independently audited, and strictly halted.

```
================================================================================
PHASE 8 COMPARATIVE EVALUATION ENGINE — FORMAL COMPLETION SIGN-OFF
Version: 0.8.0
Phase: 8 — Formal Comparative Benchmark Evaluation
Status: COMPLETED, AUDITED, AND VERIFIED
Regression State: 0 Failures across 356 tests (100% Pass)
Next Action: AWAITING USER REVIEW — PHASE 9 STRICTLY NOT AUTHORIZED
================================================================================
```
