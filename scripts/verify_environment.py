"""Environment Verification Script for Graph RAG Benchmark.

Executes end-to-end Phase 1 validation:
- Inspects Python runtime & configuration
- Connects to Ollama REST server
- Inspects model catalog
- Performs real text generation with configured Qwen LLM
- Performs real embedding generation and vector validation with nomic-embed-text
- Prints structured pass/fail scorecard and exits with appropriate status code
"""

import sys
from pathlib import Path

# Add project src directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from graphrag_bench import __version__
from graphrag_bench.config import get_settings
from graphrag_bench.health import HealthChecker, HealthStatus
from graphrag_bench.logger import setup_logger
from graphrag_bench.ollama_client import OllamaClient, OllamaError


def main() -> int:
    """Run full Phase 1 environment verification."""
    setup_logger()
    settings = get_settings()
    client = OllamaClient(settings=settings)
    checker = HealthChecker(settings=settings, ollama_client=client)

    print("=" * 60)
    print("GRAPH RAG BENCHMARK")
    print("PHASE 1 ENVIRONMENT VERIFICATION")
    print("=" * 60)
    print(f"Project Version : v{__version__}")
    print(f"Python Version  : {sys.version.split()[0]} ({sys.executable})")
    print(f"Ollama Server   : {settings.ollama_base_url}")
    print(f"Configured LLM  : {settings.ollama_llm_model}")
    print(f"Configured Embed: {settings.ollama_embed_model}")
    print("-" * 60)

    scorecard = []

    # 1. Python Environment Check
    py_res = checker.check_python_version()
    venv_res = checker.check_virtual_environment()
    dep_res = checker.check_dependencies()
    py_env_ok = py_res.status != HealthStatus.FAIL and dep_res.status != HealthStatus.FAIL
    scorecard.append(
        (
            "Python environment",
            py_env_ok,
            f"Python {sys.version.split()[0]} | Venv: {venv_res.status.value} | Deps: {dep_res.status.value}",
        )
    )

    # 2. Configuration Check
    cfg_res = checker.check_configuration()
    cfg_ok = cfg_res.status == HealthStatus.PASS
    scorecard.append(("Configuration", cfg_ok, "Settings loaded and validated"))

    # 3. Ollama Server Connection
    server_res = checker.check_ollama_server()
    server_ok = server_res.status == HealthStatus.PASS
    scorecard.append(("Ollama connection", server_ok, server_res.message))

    if not server_ok:
        print("\n[FAIL] Ollama server is not reachable!")
        print("\nActionable Steps to Resolve:")
        print("1. Ensure Ollama is installed (https://ollama.com).")
        print("2. Start the Ollama server: open Ollama or run 'ollama serve' in a terminal.")
        print("3. Verify the server is running by visiting http://localhost:11434 in a browser.")
        print("-" * 60)
        _print_scorecard(scorecard)
        print("\nFINAL STATUS: PHASE 1 BLOCKED (Ollama Unreachable)")
        return 1

    # 4. List Models
    try:
        models = client.list_models()
        print(f"Installed Ollama models ({len(models)}): {', '.join(models)}")
    except OllamaError as exc:
        print(f"[FAIL] Could not list models: {exc}")
        models = []

    # 5. LLM Model Check
    llm_res = checker.check_llm_model()
    llm_ok = llm_res.status == HealthStatus.PASS
    scorecard.append(("Qwen model", llm_ok, f"{settings.ollama_llm_model} presence verified"))

    if not llm_ok:
        print(f"\n[FAIL] LLM Model '{settings.ollama_llm_model}' is not installed in Ollama.")
        print(f"Run in PowerShell: ollama pull {settings.ollama_llm_model}")

    # 6. Embedding Model Check
    embed_res = checker.check_embed_model()
    embed_ok = embed_res.status == HealthStatus.PASS
    scorecard.append(
        ("Embedding model", embed_ok, f"{settings.ollama_embed_model} presence verified")
    )

    if not embed_ok:
        print(
            f"\n[FAIL] Embedding Model '{settings.ollama_embed_model}' is not installed in Ollama."
        )
        print(f"Run in PowerShell: ollama pull {settings.ollama_embed_model}")

    # 7. LLM Generation Test
    gen_ok = False
    gen_msg = ""
    if llm_ok:
        print(f"\nTesting LLM inference on '{settings.ollama_llm_model}'...")
        try:
            gen_resp = client.generate(
                prompt="Briefly state 'Phase 1 verified' in five words or less.",
                system="You are an automated verification agent. Be extremely concise.",
            )
            output_text = gen_resp.response.strip()
            print(f'  Response: "{output_text}"')
            if output_text:
                gen_ok = True
                gen_msg = (
                    f"Generated {len(output_text)} chars in {gen_resp.total_duration_ns or 0}ns"
                )
            else:
                gen_msg = "Model returned empty text."
        except OllamaError as exc:
            gen_msg = f"Inference failed: {exc}"
            print(f"  [ERROR] {gen_msg}")
    else:
        gen_msg = "Skipped (model missing)"
    scorecard.append(("LLM generation", gen_ok, gen_msg))

    # 8. Embedding Generation Test
    emb_gen_ok = False
    emb_msg = ""
    if embed_ok:
        print(f"\nTesting vector embeddings on '{settings.ollama_embed_model}'...")
        try:
            emb_resp = client.embed("Knowledge Graph RAG benchmark verification sample text.")
            dims = emb_resp.dimensions
            vec = emb_resp.embeddings[0]
            print(
                f"  Vector dimension: {dims} floats (first 3 values: {[round(v, 4) for v in vec[:3]]})"
            )
            if dims > 0 and len(vec) == dims:
                emb_gen_ok = True
                emb_msg = f"Generated {dims}-dim vector successfully"
            else:
                emb_msg = f"Invalid dimension: expected > 0, got {dims}"
        except OllamaError as exc:
            emb_msg = f"Embedding generation failed: {exc}"
            print(f"  [ERROR] {emb_msg}")
    else:
        emb_msg = "Skipped (model missing)"
    scorecard.append(("Embedding generation", emb_gen_ok, emb_msg))

    # Print Summary Scorecard
    print("\n" + "=" * 60)
    print("PHASE 1 VERIFICATION SCORECARD")
    print("=" * 60)
    _print_scorecard(scorecard)

    all_passed = all(item[1] for item in scorecard)
    print("=" * 60)
    if all_passed:
        print("FINAL STATUS: PHASE 1 READY")
        print("=" * 60)
        return 0
    else:
        print("FINAL STATUS: PHASE 1 INCOMPLETE (Issues Detected)")
        print("=" * 60)
        return 1


def _print_scorecard(scorecard):
    for name, passed, detail in scorecard:
        tag = "[PASS]" if passed else "[FAIL]"
        print(f"{tag:<7} {name:<24} - {detail}")


if __name__ == "__main__":
    sys.exit(main())
