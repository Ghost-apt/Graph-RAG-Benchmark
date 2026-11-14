# Phase 9 Centralized Configuration Reference

**Project:** Graph RAG Benchmark  
**Current Version:** `0.8.0`  
**Target Version:** `0.9.0`  
**Phase:** 9 — Production Readiness & Deployment  
**Status:** VALIDATED SPECIFICATION  

---

## 1. Overview

The Graph RAG Benchmark platform uses a single centralized configuration specification powered by **Pydantic v2 `BaseSettings`** (`Settings` in `src/graphrag_bench/config.py`). Settings are loaded with the following precedence order:
1. Explicit instantiation arguments (`Settings(...)`)
2. Operating system environment variables
3. `.env` file in the application working root
4. Built-in defaults

Sensitive parameters (passwords, tokens, keys) are automatically masked in log dumps, string representations, and the Streamlit dashboard via `.to_safe_dict()`.

---

## 2. Environment Variable Reference Table

| Variable Name | Description | Default | Accepted Values | Status | Security Sensitivity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`APP_ENV`** | Runtime environment stage | `development` | `development`, `testing`, `production` | Optional | Low |
| **`APP_HOST`** | Network interface bind address | `127.0.0.1` | Valid IPv4/IPv6 address (e.g. `127.0.0.1`, `0.0.0.0`) | Optional | Medium (0.0.0.0 exposes service) |
| **`APP_PORT`** | TCP port for Streamlit interface | `8501` | Integer `1` to `65535` | Optional | Low |
| **`AUTH_MODE`** | Production authentication mode | `none` | `none`, `basic`, `header` | Optional | High |
| **`LOG_LEVEL`** | Minimum logger verbosity level | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` | Optional | Low |
| **`LOG_FILE`** | Persistent log file destination | `logs/graphrag.log` | Valid filesystem path | Optional | Low |
| **`LOG_FORMAT`** | Structured logging output format | `text` | `text`, `json` | Optional | Low |
| **`LOG_MAX_BYTES`** | Max log file size before rotation | `10485760` (10 MB) | Positive integer $\ge 1024$ | Optional | Low |
| **`LOG_BACKUP_COUNT`** | Number of rotated log files kept | `5` | Integer `1` to `50` | Optional | Low |
| **`OLLAMA_BASE_URL`** | Local Ollama REST server endpoint | `http://localhost:11434` | Valid HTTP/HTTPS URL | Required | Low |
| **`OLLAMA_LLM_MODEL`** | LLM model for text generation & QA | `qwen3:8b` | Valid Ollama model identifier | Required | Low |
| **`OLLAMA_EMBED_MODEL`** | Dense vector embedding model | `nomic-embed-text` | Valid Ollama model identifier | Required | Low |
| **`OLLAMA_TIMEOUT`** | Ollama HTTP request timeout (sec) | `120` | Positive integer $> 0$ | Optional | Low |
| **`DATA_DIR`** | Root data storage directory | `data` | Valid filesystem directory | Optional | Low |
| **`UPLOAD_DIR`** | Original document upload directory | `data/uploads` | Valid filesystem directory | Optional | Low |
| **`PROCESSED_DIR`** | Processed documents & chunks path | `data/processed` | Valid filesystem directory | Optional | Low |
| **`MAX_UPLOAD_SIZE_MB`** | Maximum allowed upload size (MB) | `25` | Positive integer $> 0$ | Optional | Low |
| **`DEFAULT_CHUNK_SIZE`** | Chunk token/word length | `400` | Positive integer $> 0$ | Optional | Low |
| **`DEFAULT_CHUNK_OVERLAP`**| Sliding window overlap length | `60` | Positive integer $\ge 0$ | Optional | Low |
| **`CHROMA_PATH`** | ChromaDB persistence directory | `data/chroma` | Valid filesystem directory | Optional | Low |
| **`CHROMA_COLLECTION_NAME`**| Production vector collection name | `graphrag_documents` | Valid string identifier | Optional | Low |
| **`EMBEDDING_BATCH_SIZE`**| Max embeddings computed per batch | `32` | Positive integer $> 0$ | Optional | Low |
| **`RETRIEVAL_TOP_K`** | Top-K vector semantic neighbors | `5` | Positive integer $> 0$ | Optional | Low |
| **`GRAPH_DIR`** | Root knowledge graph directory | `data/knowledge_graph` | Valid filesystem directory | Optional | Low |
| **`GRAPH_PATH`** | Canonical JSON knowledge graph | `data/knowledge_graph/knowledge_graph.json` | Valid filesystem path | Optional | Low |
| **`GRAPH_EXPORT_PATH`** | Canonical GraphML export path | `data/knowledge_graph/knowledge_graph.graphml` | Valid filesystem path | Optional | Low |
| **`GRAPH_MAX_HOPS`** | Max traversal hops for graph search | `2` | Integer `1` to `3` | Optional | Low |
| **`GRAPH_MAX_RESULTS`** | Max candidate graph traversal paths| `10` | Integer `1` to `100` | Optional | Low |
| **`GRAPH_MAX_DEGREE`** | Hub node degree expansion limit | `50` | Integer `1` to `500` | Optional | Low |
| **`HYBRID_TOP_K`** | Final fused results returned by RRF | `10` | Integer `1` to `100` | Optional | Low |
| **`HYBRID_RRF_K`** | Reciprocal Rank Fusion constant $k$ | `60` | Integer `1` to `1000` | Optional | Low |
| **`HYBRID_VECTOR_WEIGHT`**| Modality weight for Vector in RRF | `1.0` | Float $\ge 0.0$ | Optional | Low |
| **`HYBRID_GRAPH_WEIGHT`** | Modality weight for Graph in RRF | `1.0` | Float $\ge 0.0$ | Optional | Low |
| **`RAG_GENERATION_MODEL`**| LLM identifier for grounded QA | `qwen3:8b` | Valid Ollama model identifier | Optional | Low |
| **`RAG_TEMPERATURE`** | Sampling temperature for QA | `0.1` | Float `0.0` to `2.0` | Optional | Low |
| **`RAG_MAX_CONTEXT_TOKENS`**| Strict evidence context token budget| `3000` | Integer `500` to `16000` | Optional | Low |
| **`RAG_MAX_GENERATION_TOKENS`**| Maximum generation length | `1024` | Integer `64` to `4096` | Optional | Low |
| **`RAG_MIN_RELEVANCE_SCORE`**| Minimum score before refusal trigger| `0.50` | Float `0.0` to `1.0` | Optional | Low |
| **`RAG_REQUIRE_CITATIONS`**| Enforce source citation brackets | `true` | Boolean (`true`/`false`) | Optional | Low |
| **`BENCHMARK_DIR`** | Root directory for Phase 8 data | `data/phase8_benchmark`| Valid filesystem directory | Optional | Low |
| **`BACKUP_DIR`** | Persistence path for system backups | `data/backups` | Valid filesystem directory | Optional | Low |
| **`MAX_CONCURRENT_GENERATIONS`**| Max in-flight LLM generations | `2` | Integer `1` to `10` | Optional | Medium |
| **`MAX_CONCURRENT_BENCHMARKS`**| Max concurrent benchmark runs | `1` | Integer `1` to `2` | Optional | Medium |
| **`REQUEST_TIMEOUT`** | Default general HTTP request timeout | `60` | Integer `5` to `600` | Optional | Low |
| **`ENABLE_TELEMETRY`** | Collect structured execution logs | `true` | Boolean (`true`/`false`) | Optional | Low |

---

## 3. Configuration Validation & Fail-Fast Guarantees

1. **Path Traversal Shield:** Every path parameter is inspected to ensure it contains no `..` traversal components.
2. **Positive Integer Bounds:** Timeouts, batch sizes, and token limits must be positive non-zero integers.
3. **Weight Boundary Enforcement:** `HYBRID_VECTOR_WEIGHT` and `HYBRID_GRAPH_WEIGHT` cannot simultaneously equal `0.0`.
4. **URL Normalization:** `OLLAMA_BASE_URL` must specify `http://` or `https://` schema, and trailing slashes are automatically stripped.
