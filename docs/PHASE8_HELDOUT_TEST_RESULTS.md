# Phase 8: Held-Out Test Evaluation & Experimental Data Splits Report

**Project:** Graph RAG Benchmark  
**Phase:** 8 — Comparative Evaluation: Vector RAG vs. Graph RAG vs. Hybrid RAG  
**Document Type:** Empirical Research Audit & Held-Out Test Validation  
**Date:** November 2026  
**Workspace:** `C:\VS CODE\Graph RAG Benchmark`  
**Current System Version:** `0.8.0`  
**Evaluation Run ID:** `run_20261114_heldout_eval`  
**Baseline Run ID:** `run_20261114_060224`  

---

## 1. Executive Summary & Experimental Integrity

This report establishes the empirical findings of the **Phase 8 Comparative Benchmark Evaluation** broken down by experimental data splits:
- **Development Split:** 15 queries (used for early diagnostic checks)
- **Validation Split:** 15 queries (used for generation pipeline verification)
- **Held-Out Test Split:** 30 queries (**locked, unseen during tuning; primary source of confirmatory statistical inferences**)

### Core Audit Takeaways:
1. **Zero Data Leakage:** The 30 held-out test queries were strictly preserved and held out during configuration calibration. System hyperparameters (RRF $k=60$, vector/graph weights $1.0/1.0$, traversal max hops $2$, decay $\gamma=0.8$, relevance threshold $0.5$) were frozen from Phases 5–7 prior to test set evaluation.
2. **Held-Out Retrieval Inferences ($N=30$ Test Queries):**
   - On the complete held-out test set (30 queries), **Vector RAG** achieves higher nDCG@10 ($0.9244$) than **Hybrid RAG** ($0.8381$) and **Graph RAG** ($0.3477$).
   - On the **Answerable-Only** test partition (24 queries), **Hybrid RAG** achieves the highest Recall@5 ($\mathbf{0.9236}$ vs $0.9167$ for Vector RAG and $0.1389$ for Graph RAG) and perfect Hit Rate@5 ($1.0000$).
   - On **Unanswerable** queries (6 queries), **Graph RAG** achieves a **$0.0\%$ false-positive retrieval rate** (100% empty results, 0 items returned), whereas Vector RAG and Hybrid RAG have a $100\%$ false-positive retrieval rate (retrieving 10 non-relevant items by vector similarity).
3. **Statistical Significance (Holm-Bonferroni Corrected at $\alpha=0.05$):**
   - **Hybrid vs. Graph:** $\Delta = +0.4905$, $95\%\text{ CI } [0.3668, 0.6149]$, Wilcoxon $p = 0.000027$, Holm-adjusted $p = \mathbf{0.000081}$ (**23 Wins / 0 Losses / 7 Ties** — Statistically Significant favoring Hybrid).
   - **Vector vs. Graph:** $\Delta = +0.5767$, $95\%\text{ CI } [0.4327, 0.7187]$, Wilcoxon $p = 0.000027$, Holm-adjusted $p = \mathbf{0.000081}$ (**23 Wins / 0 Losses / 7 Ties** — Statistically Significant favoring Vector).
   - **Hybrid vs. Vector:** $\Delta = -0.0863$, $95\%\text{ CI } [-0.1455, -0.0308]$, Wilcoxon $p = 0.009633$, Holm-adjusted $p = \mathbf{0.009633}$ (**2 Wins / 10 Losses / 18 Ties** — Statistically Significant favoring Vector on nDCG@10 ranking).

---

## 2. Experimental Data Split Architecture

The 60 benchmark queries in `data/phase8_benchmark/ground_truth/benchmark_dataset.json` are stratified across 5 distinct query categories:

| Category | Dev Split ($N=15$) | Validation Split ($N=15$) | Held-Out Test Split ($N=30$) | Total Queries ($N=60$) |
|---|:---:|:---:|:---:|:---:|
| **Single-Hop Fact Retrieval** | 4 (Q01–Q04) | 4 (Q05–Q08) | 7 (Q09–Q15) | 15 (25.0%) |
| **Multi-Hop Graph Reasoning** | 4 (Q16–Q19) | 4 (Q20–Q23) | 7 (Q24–Q30) | 15 (25.0%) |
| **Entity-Relationship Path** | 3 (Q31–Q33) | 3 (Q34–Q36) | 4 (Q37–Q40) | 10 (16.7%) |
| **Comparative Synthesis** | 2 (Q41–Q42) | 2 (Q43–Q44) | 6 (Q45–Q50) | 10 (16.7%) |
| **Unanswerable Rejection** | 2 (Q51–Q52) | 2 (Q53–Q54) | 6 (Q55–Q60) | 10 (16.7%) |
| **Total Queries** | **15** | **15** | **30** | **60 (100%)** |
| **Answerable Subtotal** | **13** | **13** | **24** | **50 (83.3%)** |
| **Unanswerable Subtotal** | **2** | **2** | **6** | **10 (16.7%)** |

---

## 3. Empirical Results by Split

### A. Development Split ($N=15$ Queries)
Evaluated across all 15 Dev queries (13 answerable, 2 unanswerable):

| Strategy | P@5 | R@5 | HitRate@5 | MRR | nDCG@5 | nDCG@10 | MAP@10 | Mean Total Latency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Vector RAG** | $0.2933$ | $0.7556$ | $0.8667$ | $0.7361$ | $0.8933$ | $\mathbf{0.9201}$ | $\mathbf{0.7361}$ | $267.23\text{ ms}$ |
| **Graph RAG** | $0.2133$ | $0.5556$ | $0.8000$ | $0.5139$ | $0.7082$ | $0.7082$ | $0.5139$ | $2,558.29\text{ ms}$ |
| **Hybrid RAG (RRF)** | $\mathbf{0.3200}$ | $\mathbf{0.8111}$ | $\mathbf{0.8667}$ | $0.6593$ | $0.8605$ | $0.8640$ | $0.6593$ | $2,661.24\text{ ms}$ |

*Key Dev Observation:* Hybrid RAG achieves the highest Precision@5 ($0.3200$) and Recall@5 ($0.8111$), while Vector RAG achieves highest nDCG@10 ($0.9201$).

---

### B. Validation Split ($N=15$ Queries)
Evaluated across all 15 Validation queries (13 answerable, 2 unanswerable):

| Strategy | P@5 | R@5 | HitRate@5 | MRR | nDCG@5 | nDCG@10 | MAP@10 | Mean Total Latency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Vector RAG** | $0.2667$ | $0.7389$ | $0.8667$ | $0.7199$ | $0.8428$ | $\mathbf{0.8953}$ | $\mathbf{0.7199}$ | $34.94\text{ ms}$ |
| **Graph RAG** | $0.1067$ | $0.3056$ | $0.4667$ | $0.2944$ | $0.4441$ | $0.4500$ | $0.2944$ | $7,865.23\text{ ms}$ |
| **Hybrid RAG (RRF)** | $\mathbf{0.2800}$ | $\mathbf{0.7500}$ | $\mathbf{0.8667}$ | $0.6108$ | $0.7345$ | $0.7860$ | $0.6108$ | $7,935.39\text{ ms}$ |

*Key Validation Observation:* Precision and recall mirror the development split. Graph RAG latency increases due to LLM extraction fallback calls on complex validation queries.

---

### C. Confirmatory Held-Out Test Split ($N=30$ Locked Queries)
Evaluated across all 30 Test queries (24 answerable, 6 unanswerable):

| Strategy | P@5 | R@5 | HitRate@5 | MRR | nDCG@5 | nDCG@10 | MAP@10 | Mean Total Latency | Median (p50) | p95 Latency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Vector RAG** | $\mathbf{0.2467}$ | $0.7333$ | $\mathbf{0.8000}$ | $\mathbf{0.6956}$ | $\mathbf{0.9072}$ | $\mathbf{0.9244}$ | $\mathbf{0.6956}$ | $38.16\text{ ms}$ | $34.50\text{ ms}$ | $62.10\text{ ms}$ |
| **Graph RAG** | $0.0400$ | $0.1111$ | $0.2000$ | $0.1111$ | $0.3477$ | $0.3477$ | $0.1111$ | $9,872.25\text{ ms}$ | $1.20\text{ ms}$ | $26,100\text{ ms}$ |
| **Hybrid RAG (RRF)** | $\mathbf{0.2467}$ | $\mathbf{0.7389}$ | $\mathbf{0.8000}$ | $0.5654$ | $0.8238$ | $0.8381$ | $0.5654$ | $10,104.94\text{ ms}$ | $41.20\text{ ms}$ | $26,250\text{ ms}$ |

---

## 4. Disaggregated Performance: Answerable vs. Unanswerable

Standard IR metrics (Recall, Precision, nDCG) exhibit mathematical edge cases when applied to unanswerable queries where $|G| = 0$. To provide complete transparency, metrics are disaggregated below:

### A. Answerable-Only Test Queries ($N=24$ Queries)
*Denominator includes only queries with at least one gold relevant chunk ($|G| \ge 1$):*

| Strategy | P@5 | R@5 | HitRate@5 | MRR | nDCG@5 | nDCG@10 | MAP@10 | Mean Latency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Vector RAG** | $\mathbf{0.3083}$ | $0.9167$ | $\mathbf{1.0000}$ | $\mathbf{0.8694}$ | $\mathbf{0.8840}$ | $\mathbf{0.9055}$ | $\mathbf{0.8694}$ | $36.21\text{ ms}$ |
| **Graph RAG** | $0.0500$ | $0.1389$ | $0.2500$ | $0.1389$ | $0.1846$ | $0.1846$ | $0.1389$ | $7,195.55\text{ ms}$ |
| **Hybrid RAG (RRF)** | $\mathbf{0.3083}$ | $\mathbf{0.9236}$ | $\mathbf{1.0000}$ | $0.7068$ | $0.7797$ | $0.7977$ | $0.7068$ | $7,539.66\text{ ms}$ |

#### Insights on Answerable Queries:
- **Recall Supremacy:** Hybrid RAG achieves higher Recall@5 ($0.9236$) than Vector RAG ($0.9167$) by retrieving bridging documents that vector similarity misses.
- **Hit Rate Equality:** Both Vector and Hybrid achieve $100\%$ HitRate@5 (retrieving at least one relevant chunk in top 5 for every answerable query).
- **Ranking Dynamics:** Vector RAG achieves higher nDCG@10 because its top result is consistently a high-similarity chunk ($2^3-1=7$ gain at rank 1), whereas RRF combines vector and graph lists, occasionally interleaving a lower-relevance graph path before a secondary vector chunk.

---

### B. Unanswerable Test Queries ($N=6$ Queries: Q55–Q60)
*Evaluates behavior when no relevant evidence exists in the corpus:*

| Strategy | Total Queries | Empty Results | False Positive Retrieval Rate | Mean Items Retrieved | Correct Refusal Rate (Pipeline) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Vector RAG** | 6 | 0 (0.0%) | **100.0%** (6/6) | 10.0 items | **100.0%** (via Guardrails) |
| **Graph RAG** | 6 | **6 (100.0%)** | **0.0%** (0/6) | **0.0 items** | **100.0%** (via Short-Circuit) |
| **Hybrid RAG (RRF)** | 6 | 0 (0.0%) | **100.0%** (6/6) | 10.0 items | **100.0%** (via Guardrails) |

#### Critical Epistemological Clarification:
- **Graph RAG Empty Result:** When Graph RAG returns 0 items for unanswerable queries, this establishes that **no matching entities or traversal paths were indexed within this specific knowledge graph**. It does **NOT** constitute philosophical proof that the fact does not exist in the real world.
- **Refusal Mechanism:** While Vector RAG and Hybrid RAG always return 10 items (ChromaDB top-K nearest neighbors), the Phase 7 RAG Pipeline successfully refused all unanswerable queries because their retrieved relevance scores fell below `rag_min_relevance_score` ($0.50$).

---

## 5. Paired Statistical Comparisons on Held-Out Test Set

Statistical comparisons were computed strictly on the **30 held-out test queries** using:
- Metric: **nDCG@10**
- Bootstrap: **2,000 resamples** ($95\%$ percentile CI, seed=42)
- Hypothesis Test: **Two-sided paired Wilcoxon signed-rank test**
- Multiple Testing Correction: **Holm-Bonferroni step-down procedure** across all 3 contrasts ($\alpha = 0.05$)

| Contrast ($T$ vs $B$) | Mean Diff ($\Delta$) | Median Diff | 95% Bootstrap CI | Wilcoxon $W$ | Raw $p$-value | Holm-Adjusted $p$-value | Win / Loss / Tie | Statistically Significant? |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Hybrid vs. Vector** | $-0.0863$ | $+0.0000$ | $[-0.1455, -0.0308]$ | $6.0$ | $0.009633$ | $\mathbf{0.009633}$ | 2 Wins / 10 Losses / 18 Ties | **YES (Favors Vector on nDCG@10)** |
| **Hybrid vs. Graph** | $+0.4905$ | $+0.5095$ | $[+0.3668, +0.6149]$ | $0.0$ | $0.000027$ | $\mathbf{0.000081}$ | 23 Wins / 0 Losses / 7 Ties | **YES (Favors Hybrid)** |
| **Vector vs. Graph** | $+0.5767$ | $+0.7239$ | $[+0.4327, +0.7187]$ | $0.0$ | $0.000027$ | $\mathbf{0.000081}$ | 23 Wins / 0 Losses / 7 Ties | **YES (Favors Vector)** |

### Statistical Limitations & Discussion:
1. **Sample Size ($N=30$):** While $N=30$ satisfies the central limit theorem heuristic, the non-zero paired differences count is $N=12$ for Hybrid vs Vector and $N=23$ for comparisons against Graph. Both yield asymptotic $p$-values well below the $\alpha=0.05$ threshold after Holm-Bonferroni correction.
2. **Metric Specificity:** Vector RAG's statistical superiority over Hybrid RAG applies specifically to **nDCG@10 ranking position**. When evaluated on **Recall@5 on answerable queries**, Hybrid RAG achieves higher coverage ($0.9236$ vs $0.9167$).

---

## 6. Reconciled Experiment B Answer Quality (Representative Subset)

Experiment B was executed on a **representative sample of 5 queries** from the **Validation Split** (1 per category) across all 3 strategies (15 generations total):

| Query ID | Category | Question Text | Answerable? |
|---|---|---|:---:|
| **Q05** | Single-Hop | *How does entity canonicalization normalize varied surface mention strings?* | YES |
| **Q20** | Multi-Hop | *How does Reciprocal Rank Fusion resolve the scale divergence between ChromaDB cosine distance and NetworkX path traversal scores?* | YES |
| **Q34** | Entity-Relationship | *How is Graph Traversal path scoring constrained and penalized by Hop Decay?* | YES |
| **Q43** | Comparative Synthesis | *Compare the retrieval mechanisms and failure modes of Vector RAG versus Graph RAG.* | YES |
| **Q53** | Unanswerable | *What was the trading price of gold per troy ounce during the London Gold Pool collapse of March 1968?* | NO |

### Empirical Generation Results ($N=5$ Queries per Strategy):

| Metric | Vector RAG | Graph RAG | Hybrid RAG | Calculation / Methodology |
|---|:---:|:---:|:---:|---|
| **Mean Heuristic Faithfulness** | $0.6583$ | $0.6667$ | $\mathbf{0.7292}$ | Heuristic claim token/n-gram overlap against evidence chunks. |
| **Citation Reference Precision** | $\mathbf{1.0000}$ | $\mathbf{1.0000}$ | $\mathbf{1.0000}$ | Fraction of citations `[S1]`, `[S2]` mapping to valid retrieved context. |
| **Multi-Hop Citation Coverage (Q20)** | $0.5000$ | $0.0000$ | $\mathbf{1.0000}$ | Fraction of required multi-hop source chunks cited in answer. |
| **Comparative Citation Coverage (Q43)** | $0.2500$ | $0.2500$ | $\mathbf{0.7500}$ | Fraction of distinct comparative documents synthesized. |
| **Correct Refusal Accuracy (Q53)** | $\mathbf{1.0000}$ | $\mathbf{1.0000}$ | $\mathbf{1.0000}$ | Correct refusal on unanswerable query without hallucination. |

*Crucial Caveat:* The faithfulness metric is computed via `FaithfulnessChecker`, an automated sentence-level lexical overlap heuristic. It must **NOT** be interpreted as formal proof of counterfactual semantic truth or mathematical entailment.

---

## 7. Artifact Integrity & Reproducibility Directory

All raw outputs, per-query records, metrics, and manifests for the held-out test evaluation are preserved at:
- **Directory:** `data/phase8_benchmark/results/run_20261114_heldout_eval/`
- **Files:**
  - `run_manifest.json`: Configuration and metadata
  - `retrieval_results.jsonl`: 90 per-query retrieval executions (30 test queries * 3 strategies)
  - `metrics.json`: Stratified metrics (all test queries and answerable-only)
  - `comparison.json`: Holm-corrected paired statistical contrasts
  - `splits_summary.json`: Multi-split comparative summary (Dev, Val, Test)
  - `results.csv`: Tabular export for spreadsheet analysis
