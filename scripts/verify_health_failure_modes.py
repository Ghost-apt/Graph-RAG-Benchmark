"""Verification of Production Health Monitor failure modes and non-blocking degradation."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from graphrag_bench.config import Settings
from graphrag_bench.ollama_client import OllamaClient, OllamaConnectionError
from graphrag_bench.production_health import (
    ProductionHealthMonitor,
    ProductionHealthState,
)


def verify_failure_modes() -> None:
    print("============================================================")
    print("  PHASE 9 HEALTH FAILURE MODES & DEGRADATION VERIFICATION   ")
    print("============================================================")

    # 1. Ollama Server Unavailable
    mock_client = MagicMock(spec=OllamaClient)
    mock_client.list_models.side_effect = OllamaConnectionError("Connection refused to Ollama")
    monitor = ProductionHealthMonitor(ollama_client=mock_client)
    c1, c2, c3 = monitor.check_ollama()
    print(f"1. Ollama Unavailable: {c1.state.value} -> {c1.message}")
    assert c1.state == ProductionHealthState.DEGRADED
    assert c2.state == ProductionHealthState.DEGRADED
    assert c3.state == ProductionHealthState.DEGRADED

    # 2. Qwen3:8b Missing from Ollama
    mock_client2 = MagicMock(spec=OllamaClient)
    mock_client2.list_models.return_value = ["nomic-embed-text:latest", "llama3:8b"]
    monitor2 = ProductionHealthMonitor(ollama_client=mock_client2)
    c1, c2, c3 = monitor2.check_ollama()
    print(f"2. Qwen3:8b Missing: {c2.state.value} -> {c2.message}")
    assert c1.state == ProductionHealthState.HEALTHY
    assert c2.state == ProductionHealthState.DEGRADED
    assert c3.state == ProductionHealthState.HEALTHY

    # 3. nomic-embed-text Missing from Ollama
    mock_client3 = MagicMock(spec=OllamaClient)
    mock_client3.list_models.return_value = ["qwen3:8b"]
    monitor3 = ProductionHealthMonitor(ollama_client=mock_client3)
    c1, c2, c3 = monitor3.check_ollama()
    print(f"3. nomic-embed-text Missing: {c3.state.value} -> {c3.message}")
    assert c1.state == ProductionHealthState.HEALTHY
    assert c2.state == ProductionHealthState.HEALTHY
    assert c3.state == ProductionHealthState.DEGRADED

    # 4. ChromaDB Directory Missing
    tmp = Path(tempfile.mkdtemp())
    cfg_tmp = Settings(chroma_path="non_existent_chroma", _env_file=None)
    monitor4 = ProductionHealthMonitor(settings=cfg_tmp, base_dir=tmp)
    c_chroma, _c_coll = monitor4.check_chromadb()
    print(f"4. ChromaDB Missing: {c_chroma.state.value} -> {c_chroma.message}")
    assert c_chroma.state == ProductionHealthState.DEGRADED

    # 5. Missing Knowledge Graph
    cfg_kg = Settings(graph_path="non_existent_kg.json", _env_file=None)
    monitor5 = ProductionHealthMonitor(settings=cfg_kg, base_dir=tmp)
    c_kg = monitor5.check_knowledge_graph()
    print(f"5. Knowledge Graph Missing: {c_kg.state.value} -> {c_kg.message}")
    assert c_kg.state == ProductionHealthState.DEGRADED

    # 6. Missing Document Repository
    cfg_doc = Settings(processed_dir="non_existent_processed", _env_file=None)
    monitor6 = ProductionHealthMonitor(settings=cfg_doc, base_dir=tmp)
    c_doc = monitor6.check_document_repository()
    print(f"6. Document Repository Missing: {c_doc.state.value} -> {c_doc.message}")
    assert c_doc.state == ProductionHealthState.DEGRADED

    print("\nPASS: All 6 failure degradation modes verified without unhandled exceptions!")


if __name__ == "__main__":
    verify_failure_modes()
