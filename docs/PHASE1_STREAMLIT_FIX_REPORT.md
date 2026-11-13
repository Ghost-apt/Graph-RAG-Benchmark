# Phase 1 Streamlit Blank Page Bug Fix & Verification Report

**Project:** Graph RAG Benchmark  
**Phase:** Phase 1 — Environment Setup & Project Foundation  
**Issue:** Streamlit application showing a completely blank page when accessed via browser  
**Status:** **RESOLVED & VERIFIED**  
**Date:** 2026-11-13  

---

## 1. Executive Summary

When executing `python -m streamlit run app.py`, the Streamlit HTTP server started and reported `Local URL: http://localhost:8501`. However, navigating to `http://localhost:8501` in Google Chrome rendered a completely blank white/dark screen with no dashboard elements, titles, cards, or metrics.

An in-depth investigation revealed a logic flaw in how `app.py` attempted to distinguish between direct CLI invocations (`python app.py`) and Streamlit runtime invocations (`streamlit run app.py` / `python -m streamlit run app.py`). The runtime detection heuristic evaluated to `False` under Streamlit, causing `app.py` to invoke the terminal text printer (`run_cli()`) instead of the Streamlit UI renderer (`run_streamlit()`). Consequently, zero Streamlit rendering commands (`st.*`) were dispatched to the browser frontend, causing Streamlit to render an empty document.

The issue has been completely fixed using Streamlit's official `streamlit.runtime.exists()` runtime detection API, complemented by a modern dark navy, electric blue, and cyan Phase 1 diagnostic dashboard.

---

## 2. Files Inspected

1. `app.py`: Primary application entry point containing both CLI and Streamlit routing logic.
2. `src/graphrag_bench/config.py`: Configuration models, environment loaders, and secret sanitizers.
3. `src/graphrag_bench/health.py`: Diagnostic check engine.
4. `src/graphrag_bench/ollama_client.py`: Local Ollama REST client.
5. `src/graphrag_bench/logger.py`: Logging configuration and file handlers.
6. `requirements.txt` & `pyproject.toml`: Runtime package versions and test configurations.

---

## 3. Root Cause Analysis

### The Flawed Logic:
In the initial implementation of `app.py`, runtime detection was implemented as:
```python
if __name__ == "__main__":
    is_streamlit_run = any(
        "streamlit" in Path(arg).stem.lower() for arg in [sys.argv[0]] + sys.argv[:2]
    )

    if is_streamlit_run:
        run_streamlit()
    else:
        run_cli()
```

### The Mechanism of Failure:
1. When a user runs `python -m streamlit run app.py` or `streamlit run app.py`, Streamlit's internal `ScriptRunner` executes `app.py`.
2. Streamlit's `ScriptRunner` parses out Streamlit CLI arguments and sets `sys.argv` for the executed script to:
   ```python
   sys.argv = ["app.py"]  # or ['C:\\VS CODE\\Graph RAG Benchmark\\app.py']
   ```
3. Evaluating `Path(sys.argv[0]).stem.lower()` yields `'app'`.
4. Checking `"streamlit" in "app"` evaluates to `False`.
5. Since no other argument contained the substring `"streamlit"`, `is_streamlit_run` was evaluated as `False`.
6. As a result, the code branch diverted into `run_cli()`:
   - `run_cli()` executed standard Python `print(...)` statements directed to the terminal's `stdout`.
   - `run_streamlit()` was never called.
   - Not a single `st.title()`, `st.metric()`, or `st.*` widget command was called in the execution thread.
7. Streamlit's frontend React application connected over WebSocket (`ws://localhost:8501/_stcore/stream`), received a script execution finished event with 0 UI elements, and rendered a **completely blank page**.

---

## 4. Fixes Implemented

### 4.1. Official Runtime Detection Engine
Streamlit provides an official, purpose-built function specifically for detecting raw Python execution vs Streamlit execution: `streamlit.runtime.exists()`.
As documented in Streamlit's codebase:
> *"True if the singleton Runtime instance has been created. When a Streamlit app is running in 'raw mode' - that is, when the app is run via `python app.py` instead of `streamlit run app.py` - the Runtime will not exist."*

We refactored `app.py` to use:
```python
def is_running_in_streamlit() -> bool:
    """Detect whether execution is running inside the active Streamlit runtime."""
    try:
        from streamlit.runtime import exists

        return exists()
    except (ImportError, ModuleNotFoundError):
        return False


def main() -> None:
    if is_running_in_streamlit():
        run_streamlit()
    else:
        run_cli()


if __name__ == "__main__":
    main()
```
- When executed via `python app.py`: `exists()` returns `False` $\rightarrow$ triggers `run_cli()` to print the terminal scorecard.
- When executed via `python -m streamlit run app.py` or `streamlit run app.py`: `exists()` returns `True` $\rightarrow$ triggers `run_streamlit()` and renders the rich dashboard.

### 4.2. Phase 1 Modern Dashboard Implementation
Implemented an interactive dashboard adhering to all visual and functional specifications:
- **Color Palette & Theme:** Dark navy background (`#0b0f19`, `#0d1527`), electric blue (`#3b82f6`, `#0066ff`), and cyan accents (`#00e5ff`, `#38bdf8`).
- **Section A: Project Overview:**
  - Project version (`v0.1.0`), Python runtime (`3.11.9`), Virtual environment isolation status, Phase 1 operational readiness.
- **Section B: Ollama Integration:**
  - Active server endpoint (`http://localhost:11434`), connection status (Connected 200 OK badge), configured LLM (`qwen3:8b`), configured embedding (`nomic-embed-text`), and dynamic tags for all installed models (`qwen3:8b`, `nomic-embed-text:latest`, `mistral:latest`, `qwen3:14b`).
- **Section C: System Health:**
  - Styled HTML table containing all 8 health check components with distinct `PASS`, `WARNING`, and `FAIL` status badges.
- **Section D: Model Diagnostics & Testing:**
  - Interactive Qwen text generation test with customizable prompt input and duration reporting.
  - Interactive vector embedding test with customizable text input and 768-dimension vector preview.
  - **State Isolation:** Test outputs are persisted in `st.session_state` and only invoked upon explicit button clicks, preventing expensive LLM or embedding re-computation during page refreshes or unrelated UI interactions.
  - **Error Defense:** Unreachable Ollama instances or missing models trigger clear instructional warning cards (`ollama serve`, `ollama pull`) without crashing the dashboard. Secrets and environment credentials remain sanitized.

---

## 5. Regression Testing & Verification

### A. Unit Tests (Mocked Transport)
```powershell
python -m pytest -v -m "not integration"
```
- **Result:** **44 passed, 5 deselected in 0.69s (100% PASS)**

### B. Live Integration Tests (Real Ollama Server)
```powershell
python -m pytest -v -m "integration"
```
- **Result:** **5 passed, 44 deselected in 18.71s (100% PASS)**
  - Live server reachable: PASS
  - Live model catalog verification: PASS
  - Live Qwen LLM generation: PASS
  - Live single & batch embedding generation (768 dimensions): PASS

### C. Code Formatting & Linting
```powershell
python -m ruff check .
python -m ruff format --check .
```
- **Result:** **All checks passed! 16 files clean and formatted.**

---

## 6. Live Browser Verification Results

The running application on `http://localhost:8501` was verified end-to-end:
1. Streamlit server started cleanly on port 8501.
2. Web page loads immediately with zero blank screen issues.
3. Dark navy background with electric blue and cyan accent gradients properly applied.
4. Main title `GRAPH RAG BENCHMARK` and subtitle rendered prominently.
5. All 4 sections (Project Overview, Ollama Integration, System Health, Model Diagnostics) clearly visible.
6. Qwen generation button clicked $\rightarrow$ spinner displayed $\rightarrow$ genuine inference generated from `qwen3:8b`.
7. Embedding generation button clicked $\rightarrow$ spinner displayed $\rightarrow$ genuine 768-dimension dense vector calculated and previewed.
8. Page refreshed $\rightarrow$ dashboard re-renders instantly without blank screen or session failure.
9. No critical exceptions logged in `logs/graphrag.log`.

---

## 7. Conclusion & Status

**BUG FIX STATUS: COMPLETE & VERIFIED**
The blank page defect has been resolved. Both CLI and Streamlit interfaces are fully functional, resilient, and ready for Phase 2 approval.
