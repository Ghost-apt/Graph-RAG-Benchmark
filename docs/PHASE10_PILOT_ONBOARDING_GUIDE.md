# Graph RAG Benchmark — Phase 10 Pilot Evaluator Onboarding Guide

**Release Version:** `v0.10.0` (Phase 10 — Real-User Pilot Deployment & Product Launch)  
**Target Audience:** Pilot Evaluators, AI Researchers, NLP Specialists, Data Scientists  
**Status:** Approved for Evaluator Distribution  

---

## 1. Welcome to the Pilot Evaluation Program!

Thank you for participating in the **Graph RAG Benchmark Phase 10 Pilot Evaluation**. 

Your feedback and empirical testing are vital to verifying whether the hybrid integration of **Knowledge Graphs** and **Dense Vector Embeddings** achieves higher factual precision, superior multi-hop reasoning, and reduced hallucination rates compared to standard vector-only RAG pipelines.

---

## 2. Pilot Evaluation Objectives

During this controlled real-user pilot, evaluators will assess:
1. **Retrieval Completeness:** Does Hybrid RAG retrieve critical structural connections that Vector search misses?
2. **Attribution & Grounding:** Are generated answers backed by verifiable inline citation footnotes `[1]`, `[2]` linked directly to source document chunks?
3. **Refusal Reliability:** When asked unanswerable or adversarial questions, does the system safely refuse to speculate rather than hallucinate?
4. **Latency & Usability:** Are response times acceptable for interactive research workflows under single-host concurrency controls?

---

## 3. Step-by-Step Evaluator Onboarding Checklist

```
+-----------------------------------------------------------------------------------+
|                           PILOT ONBOARDING WORKFLOW                               |
+-----------------------------------------------------------------------------------+

[ 1. Check Invitation ] ---> [ 2. Open Pilot URL ] ---> [ 3. Authenticate with OIDC ]
                                                                   |
[ 6. Run Benchmark Split ] <-- [ 5. Test Tab 7 Chat ] <-- [ 4. Verify Role Badge ]
```

### Step 1: Confirm Pilot Authorization
Ensure your enterprise email has been authorized by your pilot administrator. If your account is not yet on the allowlist, you will receive an "Access Restricted" message upon login.

### Step 2: Navigate to Dashboard
Open your designated pilot URL in Google Chrome, Microsoft Edge, or Firefox:
- Enterprise Pilot URL: `https://graphrag-pilot.internal.organization.com` (or local `http://localhost:8501`)

### Step 3: Authenticate
Click **"🔑 Sign in with Enterprise OIDC"** and complete authentication with your organization's identity provider.

### Step 4: Verify Your Role Badge
Look at the top of the sidebar on the left:
- Your email and display name should appear.
- Your role badge should display **RESEARCHER** (or **VIEWER**).
- If your role shows unexpected permissions, contact your administrator.

### Step 5: Execute Your First Grounded Query (Tab 7)
1. Click **"💬 Grounded RAG Chat"** in the top navigation tabs.
2. Select retrieval mode: **`Hybrid (RRF)`**.
3. Enter a domain question (e.g. *"What is the difference between Vector RAG and Graph RAG?"* or *"How does reciprocal rank fusion combine search results?"*).
4. Click **"🚀 Ask Grounded Question"**.
5. Verify the generated answer:
   - Notice the bracketed numbers like `[1]` and `[2]`.
   - Scroll down to **Section C (Citation Footnotes & Evidence Provenance)** to inspect the exact document chunk and similarity score supporting each claim.

### Step 6: Explore the Knowledge Graph (Tab 1)
1. Click **"🕸️ Knowledge Graph Explorer"**.
2. Drag, zoom, and click on entities in the interactive Plotly graph.
3. Select an entity from the dropdown to inspect its relationship triples and extraction confidence scores.

### Step 7: Run a Comparative Benchmark Evaluation (Tab 8)
1. Click **"📊 Comparative Benchmark Lab"**.
2. Select the **Validation Split (15 queries)**.
3. Click **"🚀 Run Comparative Benchmark"**.
4. Observe parallel evaluation across Vector, Graph, and Hybrid RAG.
5. Review the paired Wilcoxon significance test results and NDCG@5 metrics.

---

## 4. Evaluation Criteria & Scoring Guidelines

When providing feedback on grounded answers, please evaluate across four primary dimensions:

| Dimension | Evaluation Question | Scoring Scale |
| :--- | :--- | :---: |
| **Factual Grounding** | Is every statement directly supported by the cited source chunk? | 1 (Hallucinated) — 5 (Fully Grounded) |
| **Citation Precision** | Do the bracketed footnotes point to the exact sentence containing the evidence? | 1 (Mismatched) — 5 (Exact Match) |
| **Multi-Hop Synthesis** | Did the system successfully connect disparate entities across multiple documents? | 1 (Isolated Chunks) — 5 (Synthesized) |
| **Response Latency** | Was retrieval and answer generation sufficiently fast? | 1 (Too Slow >30s) — 5 (Fast <5s) |

---

## 5. Reporting Issues & Anomaly Submissions

If you encounter an unexpected error, a hallucinated claim without supporting evidence, or a UI issue:
1. Note the exact query text and selected retrieval mode.
2. Note the timestamp (UTC) of the execution.
3. Take a screenshot of the answer and the Citation Footnotes section.
4. Email your feedback report to the benchmark engineering team or log an issue in the pilot tracker.
