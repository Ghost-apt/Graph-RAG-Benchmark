# Phase 8 Methodology: Scientific Benchmark Evaluation Protocol

**Project:** Graph RAG Benchmark  
**Phase:** 8 — Formal Comparative Benchmark Evaluation  
**Date:** November 2026  
**Document Target Version:** `0.8.0`  

---

## 1. Experimental Rationale & Research Questions

The primary objective of Phase 8 is to provide an empirical, reproducible, and mathematically rigorous comparison of three core Retrieval-Augmented Generation paradigms:
1. **Vector RAG (Dense Embedding Retrieval):** Semantic similarity in continuous latent vector spaces.
2. **Knowledge Graph RAG (Structured Multi-Hop Traversal):** Deterministic entity linking and relational graph traversal.
3. **Hybrid RAG (Reciprocal Rank Fusion):** Multi-modal rank fusion combining dense vector representations with structured graph paths.

### Primary Research Questions (RQs)

- **RQ1 (Retrieval Effectiveness):** Does Hybrid RAG achieve statistically significant retrieval quality gains (measured by nDCG@10 and MAP@10) over standalone Vector RAG and Graph RAG across a diverse query distribution?
- **RQ2 (Multi-Hop Superiority):** Under what query topologies (e.g., multi-hop reasoning and entity-relationship assertions) does Knowledge Graph traversal outperform dense vector nearest-neighbor search?
- **RQ3 (Grounded Generation & Faithfulness):** Does structured graph-grounded evidence improve LLM answer faithfulness and reduce hallucination rates compared to unstructured text chunks?
- **RQ4 (Latency & Telemetry Trade-offs):** What is the exact computational and latency overhead introduced by graph entity extraction, graph traversal, and rank fusion relative to pure vector retrieval?

---

## 2. Tested Retrieval Modalities & Architecture

All three systems operate over the identical canonical benchmark evaluation corpus (22 documents, 110 canonical chunks).

### Modality 1: Vector RAG
- **Embedding Model:** `nomic-embed-text` (768 floating-point dimensions).
- **Index Subsystem:** Persistent ChromaDB with HNSW index (`space: cosine`).
- **Retrieval Metric:** Cosine similarity computed from ChromaDB cosine distance:
  $$\text{Cosine Similarity} = 1.0 - \text{distance}$$
  $$\text{Normalized Score} = 1.0 - \frac{\text{distance}}{2.0}$$
- **Top-K Cutoff:** Evaluated at $K \in \{1, 3, 5, 10\}$.

### Modality 2: Knowledge Graph RAG
- **Graph Storage:** NetworkX `MultiDiGraph` serialized to JSON and GraphML.
- **Seed Entity Linking:** Dual-path resolution combining fast exact/alias dictionary lookup with LLM extraction fallback for complex surface forms.
- **Traversal Engine:** Directional bounded Breadth-First Search (BFS) up to $\text{max\_hops} = 2$.
- **Path Scoring:** Composite ranking combining edge confidence, traversal depth penalty, and node degree centrality:
  $$\text{Score}(\text{path}) = \left(\prod_{e \in \text{edges}} \text{conf}(e)\right) \times \gamma^{\text{hop\_count}-1}$$
- **Candidate Normalization:** Normalized by `ResultNormalizer` to canonical Phase 8 chunk IDs via edge evidence quote references.

### Modality 3: Hybrid RAG (Reciprocal Rank Fusion)
- **Coordinator:** `HybridRetriever` coordinating `SemanticSearch` and `GraphSearchEngine`.
- **Normalization:** Canonical chunk resolution and intra-modality deduplication.
- **Fusion Algorithm:** Reciprocal Rank Fusion (RRF) with smoothing constant $k=60$ and equal modality weights ($w_v = 1.0, w_g = 1.0$):
  $$\text{RRF Score}(c) = \frac{w_v}{k + r_v(c)} + \frac{w_g}{k + r_g(c)}$$
  where $r_v(c)$ and $r_g(c)$ represent the 1-based ranks in the vector and graph candidate lists respectively.

---

## 3. Evaluation Protocol & Experiments

### Experiment A: Retrieval-Only Comparative Evaluation
- **Corpus:** 22 source documents, 110 canonical chunks.
- **Queries:** All 60 benchmark queries (Dev: 15, Validation: 15, Test: 30).
- **Total Runs:** 180 query executions ($60 \text{ queries} \times 3 \text{ strategies}$).
- **Evaluation Cutoffs:** $K \in \{1, 3, 5, 10\}$.
- **Metrics Computed:**
  - Precision@K
  - Recall@K
  - Hit Rate@K
  - Mean Reciprocal Rank (MRR)
  - Normalized Discounted Cumulative Gain (nDCG@K) with graded relevance ($0$–$3$)
  - Mean Average Precision (MAP@K)

### Experiment B: End-to-End Grounded Answer Generation
- **Generator Model:** `qwen3:8b` via local Ollama server.
- **Parameters:** Temperature $T = 0.1$, generation timeout 120s, citation enforcement enabled.
- **Evaluation Subset:** Representative sample across all 5 query categories.
- **Metrics Computed:**
  - Answer Faithfulness Score (supported atomic claims / total claims)
  - Citation Reference Precision (valid registry citations / total citations)
  - Citation Coverage (gold relevant chunks cited / total gold relevant chunks)
  - Refusal Accuracy (binary correct rejection for unanswerable queries)
  - Latency Telemetry (retrieval ms, traversal ms, generation ms, total ms)

---

## 4. Statistical Significance Framework

To ensure that reported improvements are not artifacts of query sampling variance:

1. **Paired Experimental Design:** Each query $q_i \in Q$ is evaluated under both baseline and treatment strategies. Differences are evaluated per query:
   $$d_i = \text{Metric}_{\text{treatment}}(q_i) - \text{Metric}_{\text{baseline}}(q_i)$$
2. **Wilcoxon Signed-Rank Test:** Non-parametric paired hypothesis testing evaluating whether the distribution of paired differences is symmetric around zero. Selected over Student's paired t-test because information retrieval metrics (nDCG, MAP) are non-normally distributed and bounded in $[0, 1]$.
   - Significance threshold: $\alpha = 0.05$.
3. **Bootstrap Confidence Intervals:** 95% Confidence Intervals for mean difference $\Delta$ computed via 2,000 non-parametric bootstrap resamples with replacement.
4. **Win / Loss / Tie Accounting:** Discrete query count where treatment outperformed, underperformed, or tied with baseline by a threshold $\epsilon = 10^{-4}$.
5. **Sample-Size Transparency:** In accordance with empirical IR reporting standards, sample sizes $N < 30$ are explicitly flagged as exploratory, while $N \ge 30$ (Test split) provide standard statistical power.
