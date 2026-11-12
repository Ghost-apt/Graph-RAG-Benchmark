"""Graph RAG Benchmark - Phase 1 Application Entry Point.

Provides a diagnostic health-check interface that works both as:
1. A Streamlit interactive web dashboard (`streamlit run app.py`)
2. A direct command-line health overview (`python app.py`)
"""

import sys
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from graphrag_bench import __version__
from graphrag_bench.config import get_settings
from graphrag_bench.health import HealthChecker, HealthStatus
from graphrag_bench.logger import get_logger, setup_logger
from graphrag_bench.ollama_client import OllamaClient, OllamaError

logger = get_logger("app")


def run_cli():
    """Execute direct terminal health check overview."""
    setup_logger()
    settings = get_settings()
    client = OllamaClient(settings=settings)
    checker = HealthChecker(settings=settings, ollama_client=client)
    report = checker.run_all_checks()

    print("\n" + "=" * 65)
    print("      GRAPH RAG BENCHMARK - PHASE 1 FOUNDATION")
    print("=" * 65)
    print(f"Project Version   : v{__version__}")
    print(f"Python Runtime    : {sys.version.split()[0]}")
    print(f"Virtual Env       : {sys.prefix}")
    print(f"Ollama Base URL   : {settings.ollama_base_url}")
    print(f"Target LLM Model  : {settings.ollama_llm_model}")
    print(f"Target Embed Model: {settings.ollama_embed_model}")
    print("-" * 65)
    print("SYSTEM HEALTH CHECKS:")
    print("-" * 65)

    for chk in report.checks:
        tag = f"[{chk.status.value}]"
        print(f" {tag:<10} {chk.name:<22} - {chk.message}")

    print("=" * 65)
    print(f"OVERALL STATUS: {report.overall_status.value}")
    print(f"SUMMARY       : {report.summary}")
    print("=" * 65)
    print("\nHint: To launch interactive web dashboard, run:")
    print("      streamlit run app.py\n")


def run_streamlit():
    """Render interactive Streamlit health and diagnostic dashboard."""
    import streamlit as st

    setup_logger()
    settings = get_settings()
    client = OllamaClient(settings=settings)
    checker = HealthChecker(settings=settings, ollama_client=client)

    st.set_page_config(
        page_title="Graph RAG Benchmark - Phase 1 Foundation",
        page_icon="🔍",
        layout="wide",
    )

    st.title("🔍 Graph RAG Benchmark")
    st.caption(f"Phase 1: Environment Setup & Project Foundation (v{__version__})")

    # Overview Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Project Version", f"v{__version__}")
    with m2:
        st.metric("Python Version", sys.version.split()[0])
    with m3:
        st.metric("Target LLM", settings.ollama_llm_model)
    with m4:
        st.metric("Target Embed", settings.ollama_embed_model)

    st.divider()

    # Environment & Health Check Section
    st.subheader("Environment Health Checks")
    report = checker.run_all_checks()

    status_color = {
        HealthStatus.PASS: "🟢 PASS",
        HealthStatus.WARNING: "🟡 WARNING",
        HealthStatus.FAIL: "🔴 FAIL",
    }

    if report.overall_status == HealthStatus.PASS:
        st.success(f"**Overall Status: {status_color[report.overall_status]}** — {report.summary}")
    elif report.overall_status == HealthStatus.WARNING:
        st.warning(f"**Overall Status: {status_color[report.overall_status]}** — {report.summary}")
    else:
        st.error(f"**Overall Status: {status_color[report.overall_status]}** — {report.summary}")

    # Health Checks Data Display
    check_rows = []
    for c in report.checks:
        check_rows.append(
            {
                "Status": status_color.get(c.status, c.status.value),
                "Component Check": c.name,
                "Details": c.message,
            }
        )
    st.dataframe(check_rows, use_container_width=True, hide_index=True)

    st.divider()

    # Interactive Verification Diagnostics
    st.subheader("Live Model Diagnostics")
    col_llm, col_emb = st.columns(2)

    with col_llm:
        st.markdown(f"#### 🤖 LLM Verification (`{settings.ollama_llm_model}`)")
        test_prompt = st.text_input(
            "Test Prompt",
            value="Explain Graph RAG in one sentence.",
            key="llm_prompt",
        )
        if st.button("Run LLM Generation Test", type="primary"):
            with st.spinner("Generating inference from Ollama..."):
                try:
                    res = client.generate(test_prompt)
                    st.success(f"Response ({res.eval_count or 'N/A'} tokens):")
                    st.write(res.response)
                except OllamaError as exc:
                    st.error(f"LLM generation failed: {exc}")

    with col_emb:
        st.markdown(f"#### 📐 Vector Embedding (`{settings.ollama_embed_model}`)")
        test_text = st.text_input(
            "Text to Embed",
            value="Vector and Knowledge Graph Hybrid Benchmark.",
            key="emb_text",
        )
        if st.button("Generate Embedding Vector"):
            with st.spinner("Calculating dense vector..."):
                try:
                    emb_res = client.embed(test_text)
                    st.success(
                        f"Embedding generated: {emb_res.dimensions} dimensions "
                        f"(model: {emb_res.model})"
                    )
                    st.write("First 10 vector values:")
                    st.code(str([round(v, 4) for v in emb_res.embeddings[0][:10]]))
                except OllamaError as exc:
                    st.error(f"Embedding generation failed: {exc}")

    st.divider()
    with st.expander("Active Configuration"):
        st.json(settings.to_safe_dict())


if __name__ == "__main__":
    # Check if executed via `streamlit run app.py`
    is_streamlit_run = any(
        "streamlit" in Path(arg).stem.lower() for arg in [sys.argv[0]] + sys.argv[:2]
    )

    if is_streamlit_run:
        run_streamlit()
    else:
        run_cli()
