# Graph RAG Benchmark

Empirical benchmark comparing **Vector RAG**, **Graph RAG**, and **Hybrid RAG** for retrieval quality, context richness, hallucination reduction, and end-to-end question-answering accuracy.

> **Status:** Phase 1 Complete (Environment Setup & Project Foundation)

---

## 1. Project Overview

The **Graph RAG Benchmark** project evaluates retrieval-augmented generation paradigms:
1. **Vector RAG:** Dense vector semantic similarity via embedding models and vector databases.
2. **Graph RAG:** Structured knowledge graph traversal and entity-relation extraction.
3. **Hybrid RAG:** Fused retrieval synthesizing dense semantic matching with structured graph hops.

### Phase 1 Objectives
- Automated project workspace auditing and safe foundation setup.
- Python 3.11 isolated virtual environment configuration (`.venv`).
- Robust environment configuration system with secret masking via `pydantic-settings` and `.env`.
- Ollama local REST client with full support for `/api/tags`, `/api/generate`, and modern `/api/embed`.
- Verified local connectivity to `qwen3:8b` (LLM) and `nomic-embed-text` (768-dim embeddings).
- Comprehensive environment health checks returning structured `PASS`, `WARNING`, and `FAIL` metrics.
- Automated test suite with mocked unit tests and live integration tests.
- Interactive Phase 1 diagnostics via CLI and Streamlit dashboard.

---

## 2. Technology Stack

- **Runtime:** Python 3.11.9
- **Inference Server:** [Ollama](https://ollama.com/) (Local HTTP REST API)
- **Generative LLM:** `qwen3:8b`
- **Embedding Model:** `nomic-embed-text` (768-dimensional dense vector embeddings)
- **Configuration & Validation:** `pydantic>=2.6`, `pydantic-settings>=2.2`, `python-dotenv>=1.0`
- **HTTP Client:** `requests>=2.31`
- **Testing & Quality:** `pytest>=8.0`, `pytest-mock`, `requests-mock`, `ruff>=0.3`
- **Application Dashboard:** `streamlit>=1.32`

---

## 3. Prerequisites

- **Operating System:** Windows 10/11, Linux, or macOS (Windows instructions provided below).
- **Python:** Python 3.11 installed and available in PATH.
- **Ollama:** Installed from [ollama.com](https://ollama.com).
- **GPU:** Optional but recommended for local model inference.

---

## 4. Windows Installation & Setup Guide

### Step 4.1: Clone or Navigate to Project
Open PowerShell and navigate to the project directory:
```powershell
cd "c:\VS CODE\Graph RAG Benchmark"
```

### Step 4.2: Create and Activate Virtual Environment
```powershell
# Create Python 3.11 virtual environment
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1
```

> **Note:** If PowerShell execution policy restricts scripts, run:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Step 4.3: Install Dependencies
```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install production and development dependencies
python -m pip install -r requirements-dev.txt
```

---

## 5. Ollama Setup & Model Pulling

Ensure Ollama is running locally:
```powershell
# Start Ollama service if not already active
ollama serve
```

In a new PowerShell window, pull the required LLM and Embedding models:
```powershell
# Pull Qwen3 8B parameter model for LLM generation
ollama pull qwen3:8b

# Pull Nomic Embed Text model for vector embeddings
ollama pull nomic-embed-text
```

Verify installed models:
```powershell
ollama list
```

---

## 6. Environment Configuration

Copy the example environment file to create `.env`:
```powershell
Copy-Item .env.example .env
```

Verify or customize variables in `.env`:
```env
# Ollama Server Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=qwen3:8b
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_TIMEOUT=120

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/graphrag.log

# Application Configuration
APP_ENV=development
```

> **Security Note:** `.env` and `.venv/` are excluded from Git via `.gitignore`. Never commit credentials.

---

## 7. Running Verification & Health Checks

Run the automated environment verification script:
```powershell
python scripts/verify_environment.py
```

Expected output:
```
============================================================
GRAPH RAG BENCHMARK
PHASE 1 ENVIRONMENT VERIFICATION
============================================================
Project Version : v0.1.0
Python Version  : 3.11.9
Ollama Server   : http://localhost:11434
Configured LLM  : qwen3:8b
Configured Embed: nomic-embed-text
------------------------------------------------------------
[PASS]  Python environment       - Python 3.11.9 | Venv: PASS | Deps: PASS
[PASS]  Configuration            - Settings loaded and validated
[PASS]  Ollama connection        - Ollama server is reachable at http://localhost:11434.
[PASS]  Qwen model               - qwen3:8b presence verified
[PASS]  Embedding model          - nomic-embed-text presence verified
[PASS]  LLM generation           - Generated 17 chars
[PASS]  Embedding generation     - Generated 768-dim vector successfully
============================================================
FINAL STATUS: PHASE 1 READY
============================================================
```

---

## 8. Running Tests

### Run Unit Tests (Mocked - No Ollama Server Required)
```powershell
python -m pytest -v -m "not integration"
```

### Run Live Integration Tests (Requires Running Ollama Server)
```powershell
python -m pytest -v -m "integration"
```

### Run All Tests
```powershell
python -m pytest -v
```

### Run Code Quality & Formatting Checks
```powershell
python -m ruff check .
python -m ruff format --check .
```

---

## 9. Starting the Phase 1 Application

### Command-Line Interface (CLI Mode)
```powershell
python app.py
```

### Streamlit Web Dashboard Mode
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501` to view interactive model health metrics, run live generation tests, and inspect vector embeddings.

---

## 10. Project Structure

```
graphrag-benchmark/
│
├── app.py                      # Phase 1 diagnostic entry point (CLI & Streamlit)
├── requirements.txt            # Runtime dependencies
├── requirements-dev.txt        # Development, testing, and linting dependencies
├── pyproject.toml              # Build metadata and tool configs (pytest, ruff)
├── .env.example                # Configuration template
├── .env                        # Local runtime configuration (git-ignored)
├── .gitignore                  # Git exclusion rules
├── README.md                   # Project documentation
│
├── src/
│   └── graphrag_bench/
│       ├── __init__.py         # Package entry and exports
│       ├── config.py           # Pydantic settings management & secret masking
│       ├── logger.py           # Unified console & rotating file logging
│       ├── ollama_client.py    # Robust Ollama REST client (/api/tags, /generate, /embed)
│       └── health.py           # System and model health check engine
│
├── scripts/
│   └── verify_environment.py   # Comprehensive Phase 1 verification script
│
├── tests/
│   ├── __init__.py             # Test package marker
│   ├── test_config.py          # Configuration loading & validation unit tests
│   ├── test_ollama_client.py   # Ollama client mocked unit tests
│   ├── test_health.py          # Health checker unit tests
│   └── test_integration_ollama.py # Live Ollama integration tests
│
├── docs/
│   ├── PHASE1_INITIAL_AUDIT.md # Initial workspace state and inventory
│   ├── PHASE1_CLEANUP_REPORT.md# Safe cleanup action log
│   └── PHASE1_COMPLETION_REPORT.md # Final Phase 1 milestone verification report
│
├── logs/
│   └── .gitkeep                # Log directory placeholder (logs/graphrag.log)
│
└── data/
    └── .gitkeep                # Data directory placeholder for future phases
```

---

## 11. Troubleshooting & FAQ

1. **`Ollama connection failed: Connection refused`**
   - Ensure Ollama is running. In PowerShell: `Get-Process ollama*` or launch the Ollama desktop app.
   - Test directly: `curl http://localhost:11434/api/tags`.

2. **`Model 'qwen3:8b' not found`**
   - Run `ollama pull qwen3:8b`. Verify with `ollama list`.

3. **`Model 'nomic-embed-text' not found`**
   - Run `ollama pull nomic-embed-text`. Verify with `ollama list`.

4. **PowerShell script execution error (`Activate.ps1`)**
   - Run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`.

5. **Timeout during model generation**
   - The first request loads the model into VRAM/RAM which can take 10-30 seconds.
   - Increase timeout in `.env`: `OLLAMA_TIMEOUT=180`.
