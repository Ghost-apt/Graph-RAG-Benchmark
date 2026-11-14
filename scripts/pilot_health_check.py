"""End-to-End Pilot Deployment Health Check.

Probes Ollama engine, ChromaDB vector store, NetworkX knowledge graph,
user registry, audit logging, and concurrency control.

Usage:
    python scripts/pilot_health_check.py [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure src is on Python search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from graphrag_bench.config import get_settings
from graphrag_bench.logger import setup_logger
from graphrag_bench.observability import ConcurrencyLimiter
from graphrag_bench.ollama_client import OllamaClient
from graphrag_bench.user_audit import UserAuditLogger
from graphrag_bench.user_registry import UserRegistry
from graphrag_bench.vector_store import VectorStore


def run_pilot_health_check(as_json: bool = False) -> int:
    setup_logger()
    settings = get_settings()
    results = {}
    all_healthy = True

    # 1. Ollama Connectivity
    client = OllamaClient(settings=settings)
    t0 = time.time()
    try:
        reachable = client.is_server_reachable()
        installed_models = client.list_models() if reachable else []
        has_llm = any(settings.ollama_llm_model in m for m in installed_models)
        has_embed = any(settings.ollama_embed_model in m for m in installed_models)
        ollama_ok = reachable and has_llm and has_embed
        results["ollama"] = {
            "status": "HEALTHY" if ollama_ok else "UNHEALTHY",
            "reachable": reachable,
            "llm_model_present": has_llm,
            "embed_model_present": has_embed,
            "latency_ms": round((time.time() - t0) * 1000, 2),
        }
        if not ollama_ok:
            all_healthy = False
    except Exception as e:
        results["ollama"] = {"status": "ERROR", "error": str(e)}
        all_healthy = False

    # 2. ChromaDB Vector Store
    t0 = time.time()
    try:
        vs = VectorStore(settings=settings)
        count = vs.count()
        results["vector_store"] = {
            "status": "HEALTHY",
            "indexed_vectors": count,
            "collection": vs.collection_name,
            "latency_ms": round((time.time() - t0) * 1000, 2),
        }
    except Exception as e:
        results["vector_store"] = {"status": "ERROR", "error": str(e)}
        all_healthy = False

    # 3. Knowledge Graph
    t0 = time.time()
    try:
        kg_path = Path(settings.graph_dir) / "knowledge_graph.json"
        if kg_path.exists():
            file_size = kg_path.stat().st_size
            results["knowledge_graph"] = {
                "status": "HEALTHY" if file_size > 0 else "WARNING",
                "file_size_bytes": file_size,
                "path": str(kg_path),
                "latency_ms": round((time.time() - t0) * 1000, 2),
            }
        else:
            results["knowledge_graph"] = {
                "status": "WARNING",
                "message": f"Graph file not found at {kg_path}",
            }
    except Exception as e:
        results["knowledge_graph"] = {"status": "ERROR", "error": str(e)}
        all_healthy = False

    # 4. User Registry
    t0 = time.time()
    try:
        reg = UserRegistry(settings=settings)
        users = reg.list_users()
        results["user_registry"] = {
            "status": "HEALTHY",
            "total_users": len(users),
            "path": str(reg.registry_path),
            "latency_ms": round((time.time() - t0) * 1000, 2),
        }
    except Exception as e:
        results["user_registry"] = {"status": "ERROR", "error": str(e)}
        all_healthy = False

    # 5. User Audit Log
    t0 = time.time()
    try:
        audit = UserAuditLogger(settings=settings)
        stats = audit.get_summary_statistics()
        results["user_audit_log"] = {
            "status": "HEALTHY",
            "total_events": stats["total_events"],
            "path": str(audit.log_path),
            "latency_ms": round((time.time() - t0) * 1000, 2),
        }
    except Exception as e:
        results["user_audit_log"] = {"status": "ERROR", "error": str(e)}
        all_healthy = False

    # 6. Concurrency Limiter
    try:
        limiter = ConcurrencyLimiter.get_instance(settings=settings)
        status_limiter = limiter.get_status()
        results["concurrency_limiter"] = {
            "status": "HEALTHY",
            "max_concurrent_generations": status_limiter["max_concurrent_generations"],
            "active_generations": status_limiter["active_generations"],
        }
    except Exception as e:
        results["concurrency_limiter"] = {"status": "ERROR", "error": str(e)}
        all_healthy = False

    if as_json:
        print(json.dumps({"healthy": all_healthy, "checks": results}, indent=2))
        return 0 if all_healthy else 1

    print("=" * 65)
    print("🔬 GRAPH RAG BENCHMARK - PILOT HEALTH CHECK (v0.10.0)")
    print("=" * 65)
    for comp, data in results.items():
        status = data.get("status", "UNKNOWN")
        icon = "✅" if status == "HEALTHY" else ("⚠️" if status == "WARNING" else "❌")
        print(f"{icon} {comp.upper():<22} [{status}]")
        for k, v in data.items():
            if k != "status":
                print(f"     • {k}: {v}")
    print("-" * 65)
    if all_healthy:
        print("🎉 ALL CRITICAL SYSTEMS OPERATIONAL - PILOT READY")
        return 0
    else:
        print("⚠️ ONE OR MORE HEALTH CHECKS REPORTED ISSUES")
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Pilot Health Check CLI")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    args = parser.parse_args()
    code = run_pilot_health_check(as_json=args.json)
    sys.exit(code)


if __name__ == "__main__":
    main()
