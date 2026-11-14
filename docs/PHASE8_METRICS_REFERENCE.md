# Phase 8 Metrics Reference: Information Retrieval & Evaluation Formulas

**Project:** Graph RAG Benchmark  
**Phase:** 8 — Comparative Evaluation (Vector RAG vs Graph RAG vs Hybrid RAG)  
**Date:** November 2026  
**Module Reference:** `src/graphrag_bench/benchmark_metrics.py`, `src/graphrag_bench/benchmark_statistics.py`  
**Document Target Version:** `0.8.0`  

---

## 1. Information Retrieval Metrics (Experiment A)

All retrieval metrics are evaluated at rank cutoffs $K \in \{1, 3, 5, 10\}$. Let $R$ denote the set of ground-truth relevant chunks for a query ($\text{grade} \ge 2$), and let $I_1, I_2, \dots, I_K$ denote the ranked candidate chunks returned by the system.

### 1.1 Precision@K
Measures the fraction of retrieved items in the top-$K$ list that are relevant.

$$\text{Precision@K} = \frac{|\{I_i \in R : 1 \le i \le K\}|}{K}$$

- **Strict Denominator Convention:** The denominator is strictly $K$ (not $\min(K, \text{retrieved})$). If a system returns fewer than $K$ items, unretrieved slots are treated as non-relevant.
- **Range:** $[0.0, 1.0]$. If $K < 1$, a `ValueError` is raised.

### 1.2 Recall@K
Measures the fraction of all corpus-relevant items that appear in the top-$K$ list.

$$\text{Recall@K} = \frac{|\{I_i \in R : 1 \le i \le K\}|}{|R|}$$

- **Zero-Gold Handling:** If $|R| = 0$ (e.g., an unanswerable query), Recall@K is defined as $1.0$ if no items were retrieved, or $0.0$ if non-relevant items were retrieved.
- **Range:** $[0.0, 1.0]$.

### 1.3 Hit Rate@K (Success@K)
Binary indicator of whether at least one relevant chunk is present in the top-$K$ retrieved list.

$$\text{Hit Rate@K} = \begin{cases} 1.0 & \text{if } \exists i \in \{1, \dots, K\} \text{ such that } I_i \in R \\ 0.0 & \text{otherwise} \end{cases}$$

- **Use Case:** Reflects first-page user utility where finding at least one relevant source enables answering the query.

### 1.4 Mean Reciprocal Rank (MRR)
Measures the reciprocal of the 1-based rank position of the *first* relevant chunk retrieved within cutoff $K$.

$$\text{RR@K} = \begin{cases} \frac{1}{\text{rank}_1} & \text{if first relevant item is at } \text{rank}_1 \le K \\ 0.0 & \text{if no relevant item is found within top } K \end{cases}$$

$$\text{MRR@K} = \frac{1}{|Q|} \sum_{q \in Q} \text{RR@K}(q)$$

- **Range:** $[0.0, 1.0]$.

### 1.5 Normalized Discounted Cumulative Gain (nDCG@K)
Evaluates ranking quality using graded relevance assessments ($g_i \in \{0, 1, 2, 3\}$) with logarithmic rank discounting.

$$\text{DCG@K} = \sum_{i=1}^K \frac{2^{g_i} - 1}{\log_2(i + 1)}$$

$$\text{IDCG@K} = \sum_{i=1}^{\min(K, |G^+|)} \frac{2^{g_i^*} - 1}{\log_2(i + 1)}$$

$$\text{nDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

where:
- $g_i$ is the graded relevance of the $i$-th retrieved item.
- $g_i^*$ is the $i$-th grade in the sorted sequence of all positive corpus grades for this query in descending order ($g_1^* \ge g_2^* \ge \dots$).
- **Exponential Gain:** $2^{\text{grade}} - 1$ produces gain values $0, 1, 3, 7$ for grades $0, 1, 2, 3$ respectively.
- **Boundary Conventions:**
  - If $\text{IDCG@K} = 0.0$ and $\text{DCG@K} = 0.0$ (e.g., unanswerable query with zero corpus relevance): $\text{nDCG@K} = 1.0$ (perfect rejection).
  - If $\text{IDCG@K} = 0.0$ and $\text{DCG@K} > 0.0$: $\text{nDCG@K} = 0.0$.
  - Output is strictly clamped to $[0.0, 1.0]$.

### 1.6 Mean Average Precision (MAP@K)
Computes Average Precision (AP@K) for each query, normalized by the number of ground-truth relevant documents up to cutoff $K$:

$$\text{AP@K} = \frac{1}{\min(K, |R|)} \sum_{i=1}^K \text{Precision@}i \times \mathbb{I}(I_i \in R)$$

$$\text{MAP@K} = \frac{1}{|Q|} \sum_{q \in Q} \text{AP@K}(q)$$

- **Range:** $[0.0, 1.0]$.

---

## 2. Answer Quality & Generation Metrics (Experiment B)

Evaluates grounded responses synthesized by `qwen3:8b` in the end-to-end RAG pipeline.

### 2.1 Answer Faithfulness Score
The fraction of atomic factual claims extracted from the response that are semantically supported by the cited evidence items:

$$\text{Faithfulness} = \frac{|\{c \in \text{Claims} : \text{Verdict}(c) = \text{SUPPORTED}\}|}{|\text{Claims}|}$$

- **Range:** $[0.0, 1.0]$ (returns `None` if zero factual claims were asserted).

### 2.2 Citation Reference Precision
Evaluates whether bracketed citation references (`[S1]`, `[S2]`) map to genuine retrieved chunks in the active evidence registry:

$$\text{Citation Precision} = \frac{|\{s \in \text{Citations} : \text{Valid}(s) = \text{True}\}|}{|\text{Citations}|}$$

### 2.3 Citation Coverage
The fraction of gold relevant chunks that were cited in the synthesized answer:

$$\text{Citation Coverage} = \frac{|\text{Cited Chunks} \cap R|}{|R|}$$

### 2.4 Refusal Accuracy
For unanswerable queries (where `answerable = False`), measures whether the system correctly refused to answer rather than hallucinating:

$$\text{Refusal Accuracy} = \begin{cases} 1.0 & \text{if system refuses or cites insufficient evidence} \\ 0.0 & \text{if system hallucinates an answer} \end{cases}$$

---

## 3. Statistical Testing & Significance Framework

### 3.1 Wilcoxon Signed-Rank Test
Non-parametric test for paired differences $d_i = \text{Metric}_{\text{treatment}}(q_i) - \text{Metric}_{\text{baseline}}(q_i)$:

1. Compute absolute differences $|d_i|$ for all $d_i \ne 0$.
2. Rank $|d_i|$ from smallest ($1$) to largest ($N$). Assign average ranks to ties.
3. Compute sum of positive ranks $W^+$ and negative ranks $W^-$:
   $$W^+ = \sum_{d_i > 0} \text{Rank}(|d_i|), \quad W^- = \sum_{d_i < 0} \text{Rank}(|d_i|)$$
   $$\text{Statistic } W = \min(W^+, W^-)$$
4. For $N \ge 10$, evaluate two-sided $p$-value via standard normal approximation:
   $$\mu_W = \frac{N(N+1)}{4}, \quad \sigma_W = \sqrt{\frac{N(N+1)(2N+1)}{24}}$$
   $$z = \frac{W - \mu_W}{\sigma_W}, \quad p = 2 \times \left(1 - \Phi(|z|)\right)$$

### 3.2 95% Bootstrap Confidence Intervals
1. From paired differences $\{d_1, \dots, d_N\}$, draw $B = 2,000$ bootstrap resamples with replacement of size $N$.
2. Compute the mean difference $\bar{d}^*_b$ for each bootstrap sample $b \in \{1, \dots, B\}$.
3. Sort the resampled means: $\bar{d}^*_{(1)} \le \dots \le \bar{d}^*_{(B)}$.
4. The 95% percentile interval is given by:
   $$\text{CI}_{95} = \left[\bar{d}^*_{(0.025 \times B)}, \, \bar{d}^*_{(0.975 \times B)}\right]$$
