# Phase 9 Operations & Incident Response Runbook

**Project:** Graph RAG Benchmark  
**Current Version:** `0.8.0`  
**Target Version:** `0.9.0`  
**Phase:** 9 — Production Readiness & Deployment  
**Author:** Principal Site Reliability Engineer  
**Status:** PRODUCTION OPERATIONAL SPECIFICATION  

---

## 1. Overview & Operational Philosophy

This runbook provides actionable, step-by-step standard operating procedures (SOPs) for site reliability engineers, system operators, and developers maintaining the **Graph RAG Benchmark** application in production environments.

### Golden Signals of Monitoring:
1. **Latency:** Monitored via `src/graphrag_bench/observability.py` and Streamlit Tab 9 Section C.
2. **Traffic & Concurrency:** Throttled via `ConcurrencyLimiter` (max 2 LLM generations, max 1 benchmark job).
3. **Errors:** Logged in structured JSONL format (`logs/graphrag.jsonl`) and rotating text logs (`logs/graphrag.log`).
4. **Saturation:** Filesystem disk headroom (alert at $< 2\text{ GB}$) and GPU VRAM context pressure.

---

## 2. Fast Triage Checklist

When responding to an alert or anomaly, run the following triage commands in order:

```powershell
# 1. Evaluate overall system health (13-point diagnostic probe)
.\.venv\Scripts\python.exe scripts/health_check.py

# 2. Inspect latest 50 operational log entries
Get-Content logs\graphrag.log -Tail 50

# 3. Check local Ollama REST server reachability
curl.exe -s http://localhost:11434/api/tags | jq .

# 4. Check free disk space
Get-PSDrive -PSProvider FileSystem
```

---

## 3. Incident Response Procedures

### 3.1 Incident: Application Process Unavailable (Crash / OOM)
- **Symptom:** Web browser returns `Connection Refused` on port 8501.
- **Root Cause Analysis:**
  - Check Windows Event Viewer or console logs for memory exhaustion.
- **Remediation:**
  1. Restart via production launcher:
     ```powershell
     .\scripts\start_production.ps1
     ```
  2. In Docker environments:
     ```bash
     docker compose restart graphrag-bench
     ```

### 3.2 Incident: Ollama Server Unavailable
- **Symptom:** `scripts/health_check.py` reports `Ollama Connectivity: DEGRADED`. Streamlit UI loads, but RAG Chat and Benchmark execution fail.
- **Remediation:**
  1. Check whether Ollama service is running:
     ```powershell
     Get-Process -Name "ollama" -ErrorAction SilentlyContinue
     ```
  2. If stopped, start Ollama:
     ```powershell
     Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
     ```
  3. Verify tags:
     ```powershell
     curl.exe http://localhost:11434/api/tags
     ```

### 3.3 Incident: Model Unavailable (`qwen3:8b` or `nomic-embed-text` missing)
- **Symptom:** Health check indicates model tag missing.
- **Remediation:**
  Pull missing models directly:
  ```powershell
  ollama pull qwen3:8b
  ollama pull nomic-embed-text
  ```

### 3.4 Incident: Corrupted Knowledge Graph
- **Symptom:** `check_knowledge_graph` reports JSON corruption or missing nodes/edges.
- **Remediation:**
  Restore canonical knowledge graph from the latest verified backup:
  ```powershell
  .\.venv\Scripts\python.exe scripts/restore.py --restore <latest_canonical_backup_id> --confirm
  ```

### 3.5 Incident: Low Filesystem Disk Space ($< 2\text{ GB}$ Free)
- **Symptom:** Health check warns `Filesystem Disk Space: DEGRADED`.
- **Remediation:**
  1. Inspect log sizes and archive older logs:
     ```powershell
     Remove-Item logs\*.log.* -Force
     ```
  2. Clean up temporary upload cache in `data\uploads\` (guaranteed non-canonical):
     ```powershell
     Remove-Item data\uploads\tmp_* -Force -Recurse
     ```
  3. Inspect older backup archives in `data\backups\` and prune archives older than retention policy.

### 3.6 Incident: Slow Graph Retrieval Latency ($> 20\text{ seconds}$)
- **Symptom:** Graph RAG queries take $\sim 20\text{ seconds}$ to return.
- **Root Cause:**
  - Query terms do not match entity alias dictionaries in `data/knowledge_graph/knowledge_graph.json`, causing `query_entity_extractor.py` to trigger local LLM fallback extraction via Qwen3:8b.
- **Remediation:**
  1. This is normal bimodal behavior for novel terms not in dictionaries.
  2. To accelerate frequent queries, register aliases in `data/knowledge_graph/knowledge_graph.json` or ingest documents introducing new canonical entities.

### 3.7 Incident: Failed Benchmark Execution
- **Symptom:** Streamlit UI reports benchmark timeout or worker disconnect.
- **Remediation:**
  1. Verify concurrency semaphore released:
     ```powershell
     .\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'src'); from graphrag_bench.observability import ConcurrencyLimiter; print(ConcurrencyLimiter.get_instance().active_benchmarks)"
     ```
  2. Inspect partial results in `data\phase8_benchmark\results\`.
  3. The runner writes atomically to `results.csv` and `retrieval_results.jsonl`; partially finished runs are preserved without corruption.

---

## 4. Routine Maintenance Procedures

### 4.1 Creating Scheduled Weekly Backups
Schedule weekly execution of:
```powershell
.\.venv\Scripts\python.exe scripts/backup.py --category full --description "Scheduled maintenance snapshot"
```

### 4.2 Verifying Disaster Recovery Integrity (Monthly Drill)
```powershell
# 1. List available backups
.\.venv\Scripts\python.exe scripts/restore.py --list

# 2. Select latest backup and run dry-run preview
.\.venv\Scripts\python.exe scripts/restore.py --preview <backup_id>

# 3. Test restoration to an isolated scratch folder
.\.venv\Scripts\python.exe scripts/restore.py --restore <backup_id> --target-dir data/scratch_drill --confirm
Remove-Item -Recurse -Force data/scratch_drill
```

### 4.3 Updating Python Dependencies Safely
1. Run existing regression test suite:
   ```powershell
   .\.venv\Scripts\python.exe -m pytest -q
   ```
2. Update specific package in virtual environment.
3. Re-run complete test suite and AppTest.
4. Export updated `requirements.txt`.
