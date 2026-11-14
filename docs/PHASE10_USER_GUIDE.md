# Graph RAG Benchmark — Phase 10 User Guide

**Release Version:** `v0.10.0` (Phase 10 — Real-User Pilot Deployment & Product Launch)  
**Target Audience:** Pilot Evaluators, AI Researchers, NLP Engineers, Domain Analysts  
**Document Status:** Official User Documentation  

---

## 1. Introduction to Graph RAG Benchmark

Welcome to the **Graph RAG Benchmark** pilot platform! 

Graph RAG Benchmark is a scientifically grounded, production-hardened platform designed to empirically compare and contrast three retrieval paradigms for Retrieval-Augmented Generation (RAG):
1. **Vector RAG:** ChromaDB semantic similarity search using high-dimensional dense embeddings (`nomic-embed-text`).
2. **Graph RAG:** NetworkX multi-hop entity traversal and knowledge-grounded subgraph extraction (`qwen3:8b`).
3. **Hybrid RAG:** Unified multi-modal retrieval using Reciprocal Rank Fusion (RRF) with reciprocal rank fusion score normalization.

Phase 10 introduces enterprise **OpenID Connect (OIDC)** authentication, granular **Role-Based Access Control (RBAC)**, and **attributable audit logging**, enabling multiple researchers and evaluators to work concurrently while maintaining strict session privacy and reproducibility.

---

## 2. Getting Started & Authentication

### 2.1 System Requirements & Browser Compatibility
The web dashboard is optimized for modern web browsers supporting WebSockets and WebGL:
- **Google Chrome** v110+ (Recommended)
- **Microsoft Edge** v110+
- **Mozilla Firefox** v115+
- **Apple Safari** v16+

### 2.2 Accessing the Pilot Deployment
1. Navigate to your organization's pilot deployment URL (e.g. `https://graphrag-pilot.internal.organization.com` or local pilot `http://localhost:8501`).
2. If unauthenticated, you will see the **Welcome & Secure Sign-In** screen.

### 2.3 Signing In
- **In Enterprise OIDC Mode:** Click **"🔑 Sign in with Enterprise OIDC"**. You will be redirected to your organization's identity provider (Google, Microsoft Entra ID, Okta). Authenticate using your enterprise credentials and MFA. Upon successful verification, you will be redirected back to the dashboard.
- **In Development Mode:** If testing locally, use the **"Select Dev Persona"** dropdown to choose between `Admin`, `Researcher`, or `Viewer` and click **"Sign In as Selected Persona"**.

### 2.4 Your User Badge & Role
In the sidebar, locate the **SIGNED IN AS** badge:
- Displays your display name and email address.
- Highlights your active role:
  - 🔴 **ADMIN:** Full administrative, configuration, and disaster recovery access.
  - 🔵 **RESEARCHER:** Document ingestion, retrieval experimentation, grounded QA, and 60-query benchmark execution.
  - 🟢 **VIEWER:** Read-only access to knowledge graphs, search queries, grounded chat, and benchmark comparison metrics.
- Click **"🚪 Sign Out"** at any time to terminate your session and flush state.

---

## 3. Comprehensive Feature Guide: The 9 Navigation Tabs

```
+---------------------------------------------------------------------------------------------------+
|                                  GRAPH RAG BENCHMARK DASHBOARD                                    |
+---------------------------------------------------------------------------------------------------+
| [Tab 1: KG Explorer] [Tab 2: Vector Search] [Tab 3: Doc Ingestion] [Tab 4: Health Diagnostics]   |
| [Tab 5: Graph Lab]   [Tab 6: Hybrid Lab]    [Tab 7: Grounded Chat] [Tab 8: Benchmark Lab]        |
| [Tab 9: Operations & SRE]                                                                         |
+---------------------------------------------------------------------------------------------------+
```

---

### Tab 1: 🕸️ Knowledge Graph Explorer
Explore the synthesized entity-relationship network extracted from canonical ingested documents.
- **Interactive 2D Network Visualization:** Built with Plotly. Drag, zoom, and inspect nodes and edges.
- **Color-Coded Entity Types:**
  - `CONCEPT` (Cyan)
  - `TECHNOLOGY` (Purple)
  - `SOFTWARE` (Emerald)
  - `FRAMEWORK` (Amber)
  - `METHOD` (Indigo)
  - `DATASET` (Teal)
  - `PERSON` (Orange)
  - `ORGANIZATION` (Blue)
- **Graph Statistics:** Displays exact node counts, edge counts, graph density, and average degree.
- **Node Inspector:** Click or select an entity from the dropdown to view all connected edges, relationship predicates, confidence scores, and source document chunks.

---

### Tab 2: 🔍 Embeddings & Vector Search
Perform semantic vector search across dense ChromaDB collections.
- **Search Query Input:** Enter natural language phrases.
- **Cosine Similarity Threshold:** Filter out noisy chunks below minimum cosine similarity.
- **Top-K Tuning:** Select number of candidate chunks to retrieve ($k = 1$ to $50$).
- **Metadata Filters:** Filter candidates by source document filename or file extension (`.pdf`, `.docx`, `.txt`, `.md`).
- **Ranked Results View:** Inspect chunk text, relevance scores, and word count.

---

### Tab 3: 📄 Document Ingestion & Preprocessing
Upload and process research papers, guides, and enterprise documents.
*(Available to RESEARCHER and ADMIN roles)*
- **Supported Formats:** PDF, DOCX, TXT, MD.
- **Configurable Sliding-Window Chunking:**
  - **Chunk Size:** Target words per chunk ($50$ to $2,000$ words, recommended: $300$–$500$).
  - **Chunk Overlap:** Boundary overlap words ($0$ to $500$ words) to preserve narrative continuity across chunk boundaries.
- **Batch Processing:** Click **"🚀 Process & Ingest Documents"** to extract text, strip control characters, normalize NFKC Unicode, and generate dense embeddings.
- **Document Repository:** View active corpus library, download chunk payloads, or delete documents (Admin only).

---

### Tab 4: ⚡ System Health & Diagnostics
Real-time diagnostic health check across the full platform foundation.
- **Ollama Engine Status:** Validates local Ollama server connectivity, model installation (`qwen3:8b`, `nomic-embed-text`), and API responsiveness.
- **Vector Store Status:** Confirms ChromaDB collection integrity, path accessibility, and cosine index availability.
- **Storage Directory Diagnostics:** Verifies existence and write permissions for `data/`, `data/processed/`, `data/chroma/`, `data/knowledge_graph/`, and `logs/`.
- **Memory & GPU Telemetry:** Monitors available VRAM and concurrent execution headroom.

---

### Tab 5: 🧭 Graph Retrieval Lab
Perform multi-hop reasoning over the entity knowledge graph.
- **Entity Extraction:** Extracts query entities using fuzzy match and Qwen-based entity linkage.
- **Multi-Hop Traversal:** Traverses paths between linked query entities up to $H$ hops ($H = 1, 2, 3$).
- **Path Ranking:** Ranks retrieved paths based on edge extraction confidence and shortest-path distance.
- **Sub-graph Evidence Extraction:** Converts traversed triples into grounded natural language evidence statements.

---

### Tab 6: ⚖️ Hybrid Retrieval Lab
Evaluate multi-modal retrieval fusing Vector search with Graph traversal.
- **Reciprocal Rank Fusion (RRF):** Evaluates formula:
  $$\text{RRF}(d) = \sum_{m \in \{\text{vec}, \text{graph}\}} \frac{w_m}{k_{\text{rrf}} + r_m(d)}$$
- **Weight Calibration:** Adjust relative importance sliders between Vector weight ($w_{\text{vec}}$) and Graph weight ($w_{\text{graph}}$).
- **Tie-Breaking Semantics:** Chunks supported by *both* modalities receive higher reciprocal ranking than single-modality candidates.
- **Modality Attribution:** Badges clearly indicate whether a chunk originated from `VECTOR`, `GRAPH`, or `DUAL_MODALITY`.

---

### Tab 7: 💬 Grounded RAG Chat with Citation Footnotes
Generate attributable, hallucination-resistant answers backed by verified evidence.
- **Input Form:** Enter research questions. Select retrieval mode: `Hybrid (RRF)`, `Vector Search`, or `Graph Traversal`.
- **Grounded Answer Display:** Answer generated with bracketed inline citation footnotes (e.g. `[1]`, `[2]`).
- **Insufficient Evidence Guardrails:** If retrieved evidence does not meet relevance or completeness thresholds, the system short-circuits and politely states insufficient evidence, avoiding hallucinations.
- **Citation Footnotes & Provenance Inspector:**
  - Displays chunk origin, page numbers, and similarity scores for every cited claim.
  - Highlights claims verified by faithfulness checking.
- **Claim-by-Claim Verification:** Inspect individual factual claims extracted from the answer and their supporting evidence sentences.

---

### Tab 8: 📊 Comparative Benchmark Lab
The empirical core of the platform: scientifically compare Vector RAG vs Graph RAG vs Hybrid RAG.
*(Execution available to RESEARCHER and ADMIN roles)*
- **Standardized Evaluation Corpus:** 60 queries across 4 categories:
  - 15 Factual Queries
  - 15 Multi-Hop Reasoning Queries
  - 15 Thematic Synthesis Queries
  - 15 Unanswerable / Negative Control Queries
- **Evaluation Splits:** Run on Train (30 queries), Validation (15 queries), Test (15 queries), or All 60 queries.
- **Comparative Information Retrieval Metrics:**
  - Mean Reciprocal Rank (MRR)
  - Precision@3, Precision@5
  - Recall@3, Recall@5
  - Normalized Discounted Cumulative Gain (NDCG@3, NDCG@5)
  - Hit Rate@5
- **Paired Statistical Significance Testing:**
  - Non-parametric **Wilcoxon Signed-Rank Test** across paired query runs.
  - $p$-value computation with hypothesis rejection at $\alpha = 0.05$.
  - 95% Bootstrap Confidence Intervals ($B = 1,000$ resamples).
- **Export Capabilities:** Download CSV or JSON summary tables with one click.

---

### Tab 9: 🛡️ Production Operations & Identity Governance
Site reliability, operational controls, and administrative governance.
- **Section A: Operational Status & System Health** (Service state, uptime, port binding).
- **Section B: Active Telemetry & Latency Metrics** ($p50$, $p95$, $p99$ response times).
- **Section C: Concurrency Limiter** (Active generation slots preventing GPU VRAM exhaustion).
- **Section D: Structured JSON Logs** (Live tail of application logs).
- **Section E: Cryptographic Backups** (Create verified `.tar.gz` snapshots with SHA-256 manifests).
- **Section F: Disaster Recovery & Restore** (Dry-run preview and atomic restoration, Admin only).
- **Section G: Release & Architectural Metadata** (Version `0.10.0` status and runbooks).
- **Section H: User Management & Security Audit** (User directory, allowlist onboarding, role updates, and real-time attributable audit stream, Admin only).

---

## 4. Best Practices for Evaluators

1. **Specific vs Broad Questions:** Factual queries perform best with dense Vector retrieval; relationship and structural questions ("How does X connect to Y?") benefit significantly from Knowledge Graph retrieval.
2. **Reviewing Footnotes:** Always inspect the cited evidence tabs below generated answers to verify that claims match source document statements.
3. **Reproducibility:** When reporting evaluation metrics, record the Benchmark Run ID and active model temperatures.
4. **Session Hygiene:** Always click **"Sign Out"** when completing an evaluation session on shared workstations.

---

## 5. Frequently Asked Questions (FAQ)

**Q: Why was my account denied access at login?**  
A: The pilot operates on an invited allowlist. If your email has not been pre-registered by an administrator, access is restricted. Contact your pilot administrator to be added to the allowlist.

**Q: How do I know if an answer is grounded or hallucinated?**  
A: Look for green citation badges `[1]`, `[2]`. Every cited bracket links to an exact text chunk in Section C. If evidence was insufficient, the model will explicitly state so rather than fabricating an answer.

**Q: Can other pilot users see my draft queries?**  
A: No. Each browser session possesses an isolated Streamlit state thread. However, executed queries are logged in `logs/user_audit.jsonl` for evaluation auditability and benchmark reproducibility.
