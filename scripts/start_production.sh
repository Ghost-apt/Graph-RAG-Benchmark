#!/usr/bin/env bash
# Production Launcher for Graph RAG Benchmark (Linux / Container Profile B)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${WORKSPACE_ROOT}"

echo "============================================================"
echo "  GRAPH RAG BENCHMARK - PRODUCTION LAUNCHER (PROFILE B)     "
echo "============================================================"
echo "Workspace: ${WORKSPACE_ROOT}"

# Detect Python executable
PYTHON_CMD="python3"
if [ -f ".venv/bin/python" ]; then
    PYTHON_CMD=".venv/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
fi

echo "[OK] Python executable: ${PYTHON_CMD}"

# Ensure required storage directories exist
mkdir -p data/processed data/chroma data/knowledge_graph data/backups logs

# Run health check
echo "Running pre-flight health evaluation..."
${PYTHON_CMD} scripts/health_check.py

HOST="${APP_HOST:-0.0.0.0}"
PORT="${APP_PORT:-8501}"

echo "Starting Streamlit dashboard on ${HOST}:${PORT}..."
exec ${PYTHON_CMD} -m streamlit run app.py \
    --server.address="${HOST}" \
    --server.port="${PORT}" \
    --server.headless=true
