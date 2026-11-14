# Graph RAG Benchmarking & Evaluation Methodology

## 1. Core Evaluation Metrics
Benchmarking RAG systems requires rigorous quantitative evaluation across retrieval quality and generation accuracy.

### 1.1 Retrieval Metrics
- **Mean Reciprocal Rank (MRR@K)**: Evaluates the rank position of the first relevant chunk or node.
- **Normalized Discounted Cumulative Gain (NDCG@K)**: Measures ranking quality accounting for graded relevance.
- **Context Precision**: The proportion of retrieved context that directly answers the user query.
- **Context Recall**: The proportion of ground-truth facts successfully retrieved.

### 1.2 Generation Metrics
- **Faithfulness / Groundedness**: Ensures LLM responses do not introduce hallucinations beyond retrieved context.
- **Answer Relevance**: Semantic similarity between query intention and generated answer.

## 2. Experimental Configurations
The benchmark evaluates three primary pipelines:
1. **Vector RAG**: ChromaDB dense vector indexing using nomic-embed-text.
2. **Graph RAG**: NetworkX knowledge graph with entity extraction via Qwen 8B.
3. **Hybrid RAG**: Reciprocal Rank Fusion (RRF) combining dense similarity scores with graph traversal distances.
