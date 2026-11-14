# Graph RAG Benchmark

Empirical benchmark comparing **Vector RAG**, **Graph RAG**, and **Hybrid RAG** for retrieval quality, context richness, hallucination reduction, and end-to-end question-answering accuracy.

> **Status:** Phase 1 Complete (Environment Foundation), Phase 2 Complete (Document Ingestion & Preprocessing Engine), Phase 3 Complete (Embeddings & ChromaDB Vector Search Engine), Phase 4 Complete (Knowledge Graph Construction Engine), Phase 5 Complete (Graph Retrieval & Multi-Hop Reasoning Engine), Phase 6 Complete (Hybrid Retrieval & Reciprocal Rank Fusion Engine), Phase 7 Complete (LLM Answer Generation & Grounded Question Answering Engine)

---

## 1. Project Overview

The **Graph RAG Benchmark** project systematically evaluates three retrieval-augmented generation paradigms:
1. **Vector RAG:** Dense vector semantic similarity via embedding models (`nomic-embed-text`) and vector stores (ChromaDB - Phase 3).
2. **Graph RAG:** Structured knowledge graph traversal and entity-relation extraction (NetworkX - Phase 4 & Phase 5).
3. **Hybrid RAG:** Fused retrieval synthesizing dense semantic matching with structured multi-hop graph hops (Phase 6).

### Phase 1 Capabilities (Foundation)
- Automated project workspace auditing and safe foundation setup.
- Python 3.11 isolated virtual environment configuration (`.venv`).
- Robust environment configuration system with secret masking via `pydantic-settings` and `.env`.
- Ollama local REST client with full support for `/api/tags`, `/api/generate`, and modern `/api/embed`.
- Verified local connectivity to `qwen3:8b` (LLM) and `nomic-embed-text` (768-dim embeddings).
- Comprehensive environment health checks returning structured `PASS`, `WARNING`, and `FAIL` metrics.
- Automated test suite with mocked unit tests and live integration tests.

### Phase 2 Capabilities (Document Ingestion & Preprocessing Engine)
- **Multi-Format Ingestion:** Native extraction support for PDF (`pypdf`), DOCX (`python-docx`), TXT (UTF-8 with legacy encodings fallback), and Markdown.
- **Multi-Layer Validation:** Automatic file size enforcement (configurable 25 MB default), path traversal sanitization, magic byte header verification, corrupted stream detection, encrypted PDF rejection, and binary executable screening.
- **Text Normalization:** Deterministic Unicode NFKC normalization, control character stripping, standard Unix `\n` line endings, and whitespace collapsing while preserving citations `[1]`, code blocks, lists, and formulas.
- **Sliding-Window Word Chunker:** Configurable chunk size (default 400 words) and chunk overlap (default 60 words) with exact character span tracking (`full_text[start_char:end_char] == chunk.text`) and PDF page number mapping.
- **Deterministic ID Generation:** Document IDs derived from content hash (`doc_<sha256[:16]>`) and chunk IDs (`<doc_id>_c<index:04d>_<content_hash[:8]>`).
- **Cryptographic Duplicate Detection:** Fast byte-level SHA-256 duplicate detection avoiding duplicate ingestion and redundant storage.
- **Local Persistence & Manifest:** Uploads preserved in `data/uploads/`, document metadata in `data/processed/documents.json`, chunks in `data/processed/chunks.jsonl`, and auto-rebuilding master index in `data/processed/manifest.json`.
- **Export Utilities:** Instant multi-format export to JSON, JSONL, and CSV for downstream vector indexing and graph extraction.
- **Interactive Streamlit UI:** Tabbed dashboard for document upload, real-time chunk configuration, batch processing, Chunk Explorer, Document Library, and Export downloads.

### Phase 3 Capabilities (Embeddings & ChromaDB Vector Search Engine)
- **Embedding Generation Service:** High-level `EmbeddingService` wrapping Ollama `/api/embed` using `nomic-embed-text` with dimension discovery (768), latency profiling, numerical validation (finite floats, NaNs detection), and batching.
- **ChromaDB Persistent Vector Store:** `VectorStore` backed by `chromadb.PersistentClient` in `data/chroma/` using cosine distance (`hnsw:space: cosine`). Preserves Phase 2 `chunk_id`, `document_id`, and primitive citation metadata.
- **Incremental Indexing Pipeline:** `IndexingPipeline` maintaining single-source-of-truth alignment with `DocumentRepository`. Caches existing chunk IDs to eliminate duplicate computation; supports single-document indexing and full re-indexing.
- **Dense Semantic Retrieval:** `SemanticSearch` executing natural-language query embedding and nearest-neighbor vector retrieval. Supports Top-K ranking (1–20), raw cosine similarity ($1.0 - \text{distance}$), normalized score ($1.0 - \text{distance}/2$), and combined metadata filtering (`document_id`, `file_type`).
- **Index Management & Safety:** Safe collection resetting and re-indexing leaving Phase 2 canonical data untouched in `data/uploads/` and `data/processed/`.
- **Interactive Streamlit Search Interface:** Dedicated `🔍 Embeddings & Vector Search` dashboard tab with collection statistics, batch indexing progress bar, search query box, ranked result cards, similarity percentage badges, citations, and index maintenance tools.

### Phase 4 Capabilities (Knowledge Graph Construction Engine)
- **Structured Knowledge Graph Schemas:** Rigorous Pydantic v2 schemas supporting an 11-category entity taxonomy (`PERSON`, `ORGANIZATION`, `CONCEPT`, `TECHNOLOGY`, `PROGRAMMING_LANGUAGE`, `FRAMEWORK`, `SOFTWARE`, `DATASET`, `METHOD`, `LOCATION`, `OTHER`). Directed relationships with verbatim textual evidence quotes, confidence scores, and origin chunk tracking.
- **LLM-Driven Extraction Service:** `GraphExtractionService` utilizing local `qwen3:8b` via Ollama with JSON-constrained decoding, reasoning tokens bypassed for high-throughput extraction, markdown code-fence stripping, regex fallback parsing, and strict chunk ground-truth verification (hallucinated entities and ungrounded evidence quotes rejected).
- **Entity Resolution & Disambiguation Engine:** `EntityResolver` providing NFKC normalization, lowercase folding, whitespace collapsing, and strict type-compatibility checks. Deterministic entity IDs (`ent_<type>_<hash[:12]>`), alias registry, and multi-mention provenance consolidation.
- **NetworkX MultiDiGraph Storage:** `GraphStore` managing nodes and directed multi-edges, comprehensive graph metrics, ego-subgraph extraction, and document-level deletion with orphan node purging.
- **Dual Serialization Formats:** Atomic canonical JSON persistence (`data/knowledge_graph/knowledge_graph.json`) and primitive-sanitized standard GraphML export (`data/knowledge_graph/knowledge_graph.graphml`).
- **Resilient Build Pipeline:** `GraphBuilder` coordinating document repository chunks, incremental manifest tracking (`data/knowledge_graph/manifest.json`), failure recovery, and extraction logging (`data/knowledge_graph/extractions.jsonl`).
- **Interactive Knowledge Graph Explorer (Streamlit Tab 1):** Real-time graph metrics, extraction controller with progress tracking, interactive 2D Plotly node-link network visualization, Entity Explorer, Triples table, and one-click JSON/GraphML downloads.

### Phase 5 Capabilities (Graph Retrieval & Multi-Hop Reasoning Engine)
- **Query Entity Extraction Engine (`QueryEntityExtractor`):** Multi-strategy entity identification featuring sub-millisecond fast-path dictionary scanning against known GraphStore canonical names and aliases, complemented by bounded local Qwen LLM extraction fallback for complex natural-language phrasing.
- **Entity Linking & Disambiguation (`EntityLinker`):** Multi-tiered mapping of query mentions to verified GraphStore nodes. Enforces exact canonical matches, alias lookups, type-aware disambiguation, and conservative partial match ambiguity detection (`is_ambiguous=True`) without mutating graph state.
- **Bounded Multi-Hop Traversal Engine (`GraphTraversalEngine`):** Direction-aware breadth-first path exploration across NetworkX `MultiDiGraph` supporting depths from 1 to 3 hops, directional filtering (`outgoing`, `incoming`, `bidirectional`), predicate whitelisting, document ID scoping, strict cycle prevention, and super-node degree bounds (`graph_max_degree`).
- **Transparent Multi-Factor Path Ranker (`GraphRanker`):** Deterministic, explainable scoring integrating seed match quality, hop economy penalties ($0.85^{\text{hops}-1}$), query lexical coverage, and edge confidence with full mathematical breakdown transparency.
- **Structured RAG Context Assembler (`GraphContextBuilder`):** Assembles retrieved reasoning paths into structured Markdown prompts incorporating relational triples (`(A) --[PREDICATE]--> (B)`), entity definitions, and deduplicated verbatim source quotes mapped directly to canonical Phase 2 chunks.
- **Unified Graph Search API (`GraphSearchEngine` / `search_graph`):** Production-ready facade managing the complete retrieval lifecycle with exhaustive operational status codes (`SUCCESS`, `NO_MATCHING_ENTITIES`, `NO_GRAPH_RESULTS`, `AMBIGUOUS_ENTITY`, `INVALID_QUERY`, `GRAPH_UNAVAILABLE`, `INTERNAL_ERROR`).
- **Interactive Graph Retrieval Lab (Streamlit Tab 5):** Complete interactive laboratory featuring query input, dynamic hop and direction controls, predicate and document filters, extracted mention badges, step-by-step visual path chains, verbatim quotation evidence inspector, latency profiling, and timestamped JSON exports.

### Phase 6 Capabilities (Hybrid Retrieval & Reciprocal Rank Fusion Engine)
- **Reciprocal Rank Fusion Engine (`ReciprocalRankFusion`):** Implements both standard ($RRF(d) = \sum \frac{1}{k + r_m}$) and weighted ($RRF(d) = \frac{w_v}{k + r_v} + \frac{w_g}{k + r_g}$) fusion over ordinal candidate ranks. Enforces rank isolation without combining uncalibrated metric distances or heuristic score scales.
- **Canonical Chunk Normalization (`ResultNormalizer`):** Maps vector and graph candidate outputs to canonical Phase 2 `chunk_id` identifiers via `DocumentRepository`. Aggregates multiple graph paths reaching the same chunk, assigning the best path rank and collecting all unique supporting paths and verbatim quotations.
- **Deterministic Multi-Tier Tie-Breaking:** Resolves score collisions with strict deterministic rules: (1) composite RRF score, (2) dual-modality presence (`VECTOR_AND_GRAPH` prioritized over single modality), (3) lowest individual modality rank ($\min(r_v, r_g)$), and (4) lexicographical `chunk_id`.
- **Structured Hybrid Context Builder (`HybridContextBuilder`):** Formats reciprocal-rank-fused candidates into an evidence block containing provenance headers, modality attribution badges, ranking metrics (vector rank, graph rank, cosine similarity, traversal hops), and deduplicated verbatim quotations.
- **Unified Hybrid Retriever Coordinator (`HybridRetriever` / `search_hybrid`):** End-to-end orchestrator executing parallel/sequential vector and graph search with partial failure isolation, top-k bounding, and cross-modal metadata filtering (`document_ids`, `file_types`).
- **Interactive Hybrid Retrieval Lab (Streamlit Tab 6):** 6th dashboard tab featuring 7 dedicated sections: Query & Fusion Controls, Modality Status Cards, Ranked Results Table, Chunk & Provenance Inspector, Mathematical Explanation Expander, Performance Diagnostics, and JSON/CSV Exports.

### Phase 7 Capabilities (LLM Answer Generation & Grounded Question Answering Engine)
- **Grounded Answer Generator (`AnswerGenerator`):** Executes inference with local `qwen3:8b` via `OllamaClient`, enforcing strict low-temperature decoding (`temperature=0.1`), thinking token isolation (`<think>...</think>`), prompt/answer token tracking, and sub-millisecond generation latency profiling.
- **Evidence Context Preparer & Budgeter (`EvidencePreparer`):** Adapts and deduplicates multi-modal retrieval candidates (Hybrid, Vector, or Knowledge Graph) into deterministic `EvidenceItem` structures (`[S1]`, `[S2]`, ...). Enforces strict upper-bound context token budgets with conservative word/character token heuristics (`estimate_tokens`) and clean boundary truncation.
- **Anti-Injection Prompt Builder (`PromptBuilder`):** Delimits context blocks with boundary guards (`<<<EVIDENCE_BLOCK_START>>> ... <<<EVIDENCE_BLOCK_END>>>`), enforces passive data treatment to prevent prompt injection, mandates bracketed citation markers immediately adjacent to supported assertions, and instructs explicit refusal upon insufficient context.
- **Citation Provenance Validator (`CitationValidator`):** Extracts single (`[S1]`) and compound (`[S1, S2]`) citation markers from generated answers using regex, maps them to canonical chunk and document identifiers, flags hallucinated or non-existent citations as invalid, computes mathematical citation precision ($\frac{\text{valid}}{\text{total}}$), and detects uncited context chunks.
- **Safety Guardrails & Refusal Policies (`AnswerGuardrails`):** Enforces pre-generation input validation, short-circuits to `AnswerStatus.INSUFFICIENT_EVIDENCE` upon zero or empty retrieval without making unnecessary LLM calls, detects model refusal language, and strips accidental prompt delimiter leakage.
- **Claim-Level Faithfulness Checker (`FaithfulnessChecker`):** Decomposes generated answers into atomic factual assertions, evaluates grounding against source context via lexical overlap and entity overlap heuristics, and computes an overall faithfulness score.
- **End-to-End Grounded QA Orchestrator (`RAGPipeline` / `answer_question`):** Unified master pipeline managing the full question answering lifecycle across Hybrid, Vector, and Graph retrieval modes with comprehensive latency breakdowns and full provenance reporting.
- **Interactive Grounded RAG Chat (Streamlit Tab 7):** Complete 7-section chat dashboard: (A) Query & Parameter Controls, (B) Grounded Answer & Attribution Card with highlighted citation pills, (C) Interactive Footnotes & Evidence Inspector, (D) Claim Grounding Matrix, (E) Delimited Context Inspector, (F) Guardrail Logs, and (G) JSON/CSV Artifact Exports.

---

## 2. Technology Stack

- **Runtime:** Python 3.11.9
- **Inference Server:** [Ollama](https://ollama.com/) (Local HTTP REST API)
- **Generative LLM:** `qwen3:8b`
- **Embedding Model:** `nomic-embed-text` (768-dimensional dense vector embeddings)
- **Vector Database:** `chromadb>=0.4.22` (Local persistent storage with HNSW cosine distance)
- **Knowledge Graph & Traversal:** `networkx>=3.0` (MultiDiGraph data structure, BFS traversal)
- **Interactive Visualizations:** `plotly>=5.18` (WebGL/SVG 2D network graph rendering)
- **Document Extractors:** `pypdf>=4.0`, `python-docx>=1.1`
- **Data Modeling & Validation:** `pydantic>=2.6`, `pydantic-settings>=2.2`, `python-dotenv>=1.0`
- **HTTP Client:** `requests>=2.31`
- **Testing & Quality:** `pytest>=8.0`, `pytest-mock`, `requests-mock`, `ruff>=0.3`
- **Application Dashboard:** `streamlit>=1.32`

---

## 3. Directory Structure

```
Graph RAG Benchmark/
├── .env.example                     # Environment template (Phases 1-5 settings)
├── .gitignore                       # Git ignore rules
├── app.py                           # Dual-mode application (CLI & Streamlit 5-tab dashboard)
├── pyproject.toml                   # Project metadata and Ruff/Pytest config
├── requirements.txt                 # Core dependencies
├── requirements-dev.txt             # Development & testing dependencies
├── README.md                        # Project Documentation
├── data/                            # Persistent data directory
│   ├── chroma/                      # ChromaDB SQLite & vector index storage
│   ├── knowledge_graph/             # Knowledge Graph persistence directory
│   │   ├── knowledge_graph.json     # Canonical JSON knowledge graph format
│   │   ├── knowledge_graph.graphml  # Sanitized GraphML export for Gephi / Cytoscape
│   │   ├── manifest.json            # Graph construction state & chunk mapping
│   │   └── extractions.jsonl        # Granular extraction execution history
│   ├── sample_documents/            # Benchmark sample documents (PDF, DOCX, TXT, MD)
│   ├── uploads/                     # Preserved original document files
│   └── processed/                   # Processed metadata, chunks, and manifest
│       ├── documents.json           # Ingested documents metadata registry
│       ├── chunks.jsonl             # Extracted text chunks
│       └── manifest.json            # Master repository index
├── docs/                            # Audit and completion reports
│   ├── PHASE1_INITIAL_AUDIT.md      # Initial Phase 1 audit report
│   ├── PHASE1_CLEANUP_REPORT.md     # Phase 1 cleanup log
│   ├── PHASE1_COMPLETION_REPORT.md  # Phase 1 sign-off report
│   ├── PHASE2_INITIAL_AUDIT.md      # Initial Phase 2 audit report
│   ├── PHASE2_CLEANUP_REPORT.md     # Phase 2 safety & cleanup log
│   ├── PHASE2_COMPLETION_REPORT.md  # Phase 2 completion report
│   ├── PHASE2_CHUNKING_VERIFICATION_REPORT.md # Phase 2 chunking audit & verification
│   ├── PHASE3_INITIAL_AUDIT.md      # Phase 3 readiness & architecture audit
│   ├── PHASE3_COMPLETION_REPORT.md  # Phase 3 sign-off completion report
│   ├── PHASE4_INITIAL_AUDIT.md      # Phase 4 readiness audit report
│   ├── PHASE4_CLEANUP_REPORT.md     # Phase 4 environment cleanup log
│   ├── PHASE4_COMPLETION_REPORT.md  # Phase 4 completion sign-off report
│   ├── PHASE4_FINAL_REGRESSION_REPORT.md # Phase 4 gate verification report
│   ├── PHASE5_INITIAL_AUDIT.md      # Phase 5 architecture audit
│   ├── PHASE5_CLEANUP_REPORT.md     # Phase 5 environment cleanup log
│   ├── PHASE5_COMPLETION_REPORT.md  # Phase 5 completion sign-off report
│   ├── PHASE6_INITIAL_AUDIT.md      # Phase 6 hybrid retrieval architecture audit
│   ├── PHASE6_CLEANUP_REPORT.md     # Phase 6 environment & data safety log
│   └── PHASE6_COMPLETION_REPORT.md  # Phase 6 completion sign-off report
├── logs/                            # Application log files
│   └── graphrag.log                 # Continuous runtime log
├── scripts/                         # Maintenance and verification scripts
│   ├── verify_environment.py        # Environment health checking script
│   ├── build_knowledge_graph.py     # Knowledge graph CLI build script
│   ├── verify_phase5_retrieval.py   # Multi-hop graph retrieval verification script
│   └── verify_phase6_hybrid.py      # End-to-end hybrid retrieval verification script
├── src/                             # Source code root
│   └── graphrag_bench/              # Benchmark package (v0.6.0)
│       ├── __init__.py              # Package exports & versioning
│       ├── chunking.py              # Word-based sliding window text chunker
│       ├── config.py                # Pydantic Settings & environment config
│       ├── document_loader.py       # Multi-format document extractor (PDF, DOCX, TXT, MD)
│       ├── document_processor.py    # Ingestion pipeline orchestrator
│       ├── document_repository.py   # Persistent storage, duplicate index & export
│       ├── document_validator.py    # Multi-layer file validation & sanitization
│       ├── embedding_service.py     # Ollama embedding client & validation
│       ├── entity_linker.py         # Type-safe entity linking & disambiguation
│       ├── entity_resolver.py       # Entity resolution, normalization & deduplication
│       ├── graph_builder.py         # Knowledge graph construction pipeline
│       ├── graph_context_builder.py # Traceable RAG structured prompt builder
│       ├── graph_extraction_service.py # Qwen LLM structured triple extraction
│       ├── graph_ranker.py          # Deterministic multi-factor path scoring
│       ├── graph_retrieval_schemas.py # Pydantic data models for retrieval & search
│       ├── graph_retriever.py       # Multi-hop path retrieval orchestrator
│       ├── graph_schemas.py         # Pydantic schemas for entities and relations
│       ├── graph_search.py          # Unified search facade & execution API
│       ├── graph_store.py           # NetworkX MultiDiGraph storage & export
│       ├── graph_traversal.py       # Bounded BFS multi-hop graph explorer
│       ├── health.py                # Health checker subsystem
│       ├── hybrid_context_builder.py # Structured Markdown hybrid prompt builder
│       ├── hybrid_retriever.py      # Unified hybrid retrieval coordinator
│       ├── hybrid_schemas.py        # Pydantic data models for hybrid RAG & RRF
│       ├── indexing_pipeline.py     # Batch & incremental ChromaDB indexing
│       ├── logger.py                # Structured logging configuration
│       ├── ollama_client.py         # Ollama HTTP client (LLM & Embeddings)
│       ├── query_entity_extractor.py # Fast-path & LLM entity detection
│       ├── reciprocal_rank_fusion.py # Standard & weighted RRF engine
│       ├── result_normalizer.py     # Canonical chunk mapping & deduplication
│       ├── schemas.py               # Pydantic data schemas (Doc, Chunk, Result)
│       ├── semantic_search.py       # Nearest-neighbor dense vector search
│       ├── text_cleaner.py          # Deterministic Unicode NFKC text normalizer
│       └── vector_store.py          # Persistent ChromaDB client & cosine space
└── tests/                           # Automated test suite (266 tests)
    ├── __init__.py
    ├── test_chunking.py             # Word chunker & offset tests (9 tests)
    ├── test_config.py               # Settings & masking tests (10 tests)
    ├── test_document_loader.py      # Format extractor tests (8 tests)
    ├── test_document_processor.py   # Ingestion pipeline tests (6 tests)
    ├── test_document_repository.py  # Persistence, duplicate & export tests (9 tests)
    ├── test_document_validator.py   # Validation & security tests (18 tests)
    ├── test_embedding_service.py    # EmbeddingService unit tests (10 tests)
    ├── test_entity_linker.py        # Entity linking & disambiguation tests (7 tests)
    ├── test_entity_resolver.py      # Entity resolution unit tests (9 tests)
    ├── test_graph_builder.py        # GraphBuilder pipeline unit tests (4 tests)
    ├── test_graph_context_builder.py# Structured context builder tests (3 tests)
    ├── test_graph_extraction_service.py # Extraction service unit tests (4 tests)
    ├── test_graph_ranker.py         # Multi-factor path ranker tests (5 tests)
    ├── test_graph_retrieval_schemas.py # Retrieval schema validation tests
    ├── test_graph_retriever.py      # GraphRetriever orchestrator tests (6 tests)
    ├── test_graph_schemas.py        # Graph schema validation unit tests (11 tests)
    ├── test_graph_search.py         # GraphSearchEngine facade tests (6 tests)
    ├── test_graph_store.py          # NetworkX GraphStore unit tests (8 tests)
    ├── test_graph_traversal.py      # Multi-hop traversal unit tests (9 tests)
    ├── test_health.py               # Health checker unit tests (15 tests)
    ├── test_hybrid_context_builder.py # Hybrid context formatting tests (4 tests)
    ├── test_hybrid_retriever.py      # Hybrid retriever orchestration tests (9 tests)
    ├── test_hybrid_schemas.py        # Hybrid schema validation tests (7 tests)
    ├── test_indexing_pipeline.py    # Indexing pipeline unit tests (8 tests)
    ├── test_integration_ollama.py   # Live Ollama integration tests (5 tests)
    ├── test_integration_phase3.py   # Live ChromaDB & vector tests (2 tests)
    ├── test_integration_phase4.py   # Live Qwen & Knowledge Graph tests (2 tests)
    ├── test_ollama_client.py        # Ollama client mock tests (21 tests)
    ├── test_query_entity_extractor.py # Fast-path & LLM extractor tests (8 tests)
    ├── test_reciprocal_rank_fusion.py # RRF math & tie-breaking tests (8 tests)
    ├── test_result_normalizer.py    # Normalization & deduplication tests (3 tests)
    ├── test_semantic_search.py      # SemanticSearch unit tests (8 tests)
    ├── test_text_cleaner.py         # Text cleaning & normalization tests (9 tests)
    └── test_vector_store.py         # ChromaDB VectorStore unit tests (15 tests)
    ├── test_integration_ollama.py   # Live Ollama integration tests (5 tests)
    ├── test_integration_phase3.py   # Live ChromaDB & vector tests (2 tests)
    ├── test_integration_phase4.py   # Live Qwen & Knowledge Graph tests (2 tests)
    ├── test_ollama_client.py        # Ollama client mock tests (21 tests)
    ├── test_query_entity_extractor.py # Fast-path & LLM extractor tests (8 tests)
    ├── test_semantic_search.py      # SemanticSearch unit tests (8 tests)
    ├── test_text_cleaner.py         # Text cleaning & normalization tests (9 tests)
    └── test_vector_store.py         # ChromaDB VectorStore unit tests (15 tests)
```

---

## 4. Quickstart Guide

### 4.1 Prerequisites
- Python 3.11 installed.
- Ollama installed and running (`ollama serve`).
- Models pulled:
  ```powershell
  ollama pull qwen3:8b
  ollama pull nomic-embed-text
  ```

### 4.2 Setup Environment
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment (PowerShell)
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install -r requirements-dev.txt
```

### 4.3 Configure Environment Variables
Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```

---

## 5. Execution Modes

### 5.1 CLI Health Check
To run the automated terminal environment verification:
```powershell
python app.py
```
Output displays the status of Python runtime, dependencies, directories, Ollama connectivity, and model availability.

### 5.2 Streamlit Web Application
To launch the interactive benchmark dashboard:
```powershell
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

The dashboard features five interactive tabs:
1. **🕸️ Knowledge Graph Explorer (Phase 4):**
   - Real-time graph statistics (Nodes, Edges, Categories, Predicates).
   - Graph Construction Controller with background chunk extraction.
   - Interactive 2D Plotly network graph visualization.
   - Entity Explorer and Triples Table with source chunk citations.
   - Export Knowledge Graph to canonical JSON and GraphML.
2. **🔍 Embeddings & Vector Search (Phase 3):**
   - Vector Database Overview (`graphrag_documents`, 768-dim, cosine space).
   - Incremental batch indexing with real-time progress.
   - Semantic Search with Top-K slider, metadata filters, dual cosine metrics, and citations.
   - Safe collection reset maintenance tool.
3. **📄 Document Ingestion & Preprocessing (Phase 2):**
   - Multi-file drag-and-drop upload supporting `.pdf`, `.docx`, `.txt`, `.md`.
   - Dynamic chunk size and chunk overlap sliders with validation.
   - Batch ingestion summary metrics and Chunk Explorer.
   - Document Library and one-click export (JSON, JSONL, CSV).
4. **⚡ System Health & Diagnostics (Phase 1):**
   - Ollama engine status, installed models, and environment check summary.
   - Interactive testing of Qwen text generation and Nomic vector embeddings.
5. **🧭 Graph Retrieval Lab (Phase 5):**
   - **Section A (Query Interface):** Natural-language query box, search execution button, max hops selector (1, 2, 3), traversal direction (`both`, `out`, `in`), top-k slider, predicate filters, and source document scoping.
   - **Section B (Entity Matching):** Visual cards of detected query entities (mention, canonical name, confidence, detection method) and matched graph seed nodes.
   - **Section C (Retrieved Graph Paths):** Ranked reasoning chains with step-by-step visual node-and-arrow diagrams, hop counts, and multi-factor score breakdown.
   - **Section D (Evidence Inspector):** Verbatim source quotations, chunk IDs, document IDs, and source filenames.
   - **Section E (Performance Metrics):** Overview cards for Seed Entities, Paths Retrieved, Configured Hops, Latency (ms), and Evidence Records.
   - **Section F (Export):** One-click formatted JSON export of complete retrieval response.
6. **⚖️ Hybrid Retrieval Lab (Phase 6):**
   - **Section A (Query & Fusion Configuration):** Query input, vector top-k (1–20), graph top-k (1–20), final top-k (1–20), RRF constant $k$ (1–100), vector weight $w_v$, graph weight $w_g$, max hops, direction, document filter, and file type filter.
   - **Section B (Retrieval Status):** Real-time operational status cards for Vector Engine, Graph Engine, Overall Fusion, and execution latency breakdown.
   - **Section C (Ranked Fusion Results):** Interactive table detailing final rank, modality source attribution (`VECTOR`, `GRAPH`, `VECTOR_AND_GRAPH`), chunk ID, filename, individual modality ranks, and weighted RRF contributions.
   - **Section D (Evidence & Chunk Inspector):** Verbatim canonical chunk text viewer, document/page provenance, diagnostic cosine similarity, graph traversal hops, matched entities, supporting paths, and verbatim quotes.
   - **Section E (Mathematical Explanation):** Detailed formula attribution breakdown showing exact mathematical substitution for each ranked candidate.
   - **Section F (Performance Diagnostics):** Sub-millisecond latency profiling across Vector Search, Graph Search, RRF Fusion, and Total Execution, plus structured prompt context preview.
   - **Section G (Export):** Full retrieval response JSON and CSV export download buttons.

---

## 6. Testing & Quality Assurance

### Run Unit Tests (297 tests)
```powershell
python -m pytest -v -m "not integration"
```

### Run Live Integration Tests (11 tests against local Ollama)
```powershell
python -m pytest -v -m "integration"
```

### Run Complete Test Suite (308 tests)
```powershell
python -m pytest -v
```

### Run Phase 7 Grounded RAG Benchmark Verification
```powershell
python scripts/verify_phase7_rag.py
```

### Run Phase 6 Hybrid Retrieval Verification
```powershell
python scripts/verify_phase6_hybrid.py
```

### Code Formatting and Linting (Ruff)
```powershell
# Check lint rules
ruff check .

# Check formatting
ruff format --check .

# Auto-format
ruff format .
```

---

## 7. Next Phase: Phase 8

In Phase 8, we will implement:
- **Comprehensive Multi-Modal Benchmark Suite:** Empirical comparison between Vector RAG, Graph RAG, and Hybrid RAG across diverse domain query sets.
- **Automated Metric Evaluation:** Latency, context precision, context recall, faithfulness, answer relevancy, and token consumption matrices.
- **Ablation Studies & Scaling Analysis:** Evaluation of hop depths (1, 2, 3), RRF constant $k$, and hybrid modality weightings ($w_v, w_g$).
- **Final Research Report & Publication Artifacts:** Automated chart generation and exportable benchmark reports.
