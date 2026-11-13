"""Graph RAG Benchmark - Phase 1 Application Entry Point.

Provides a dual-mode entry point:
1. Streamlit Interactive Web Dashboard (`python -m streamlit run app.py` or `streamlit run app.py`)
2. Command-Line Interface Health Overview (`python app.py`)
"""

import sys
from pathlib import Path
from typing import Any

# Ensure src is on Python search path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from graphrag_bench import __version__
from graphrag_bench.config import Settings, get_settings
from graphrag_bench.health import HealthChecker, HealthStatus
from graphrag_bench.logger import get_logger, setup_logger
from graphrag_bench.ollama_client import OllamaClient, OllamaError

logger = get_logger("app")


def is_running_in_streamlit() -> bool:
    """Detect whether execution is running inside the active Streamlit runtime.

    Streamlit sets up a global Runtime singleton whenever invoked via:
    `streamlit run app.py` or `python -m streamlit run app.py`.
    In pure CLI mode (`python app.py`), this runtime does not exist.
    """
    try:
        from streamlit.runtime import exists

        return exists()
    except (ImportError, ModuleNotFoundError):
        return False


def run_cli() -> None:
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
    print("      python -m streamlit run app.py\n")


def run_streamlit() -> None:
    """Render interactive Streamlit health and diagnostic dashboard."""
    import streamlit as st

    setup_logger()
    settings: Settings = get_settings()
    client = OllamaClient(settings=settings)
    checker = HealthChecker(settings=settings, ollama_client=client)

    # 1. Page Configuration (Must be first Streamlit call)
    st.set_page_config(
        page_title="Graph RAG Benchmark - Phase 1 Foundation",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 2. Custom CSS styling (Dark navy, electric blue & cyan accents)
    st.markdown(
        """
        <style>
        /* Base page background */
        .stApp {
            background-color: #0b0f19;
            color: #e2e8f0;
        }

        /* Benchmark Header Container */
        .benchmark-header {
            background: linear-gradient(135deg, #0d1527 0%, #111c38 100%);
            border: 1px solid #1e293b;
            border-radius: 14px;
            padding: 24px 28px;
            margin-bottom: 24px;
            box-shadow: 0 4px 20px rgba(0, 102, 255, 0.12);
        }
        .benchmark-title {
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            background: linear-gradient(90deg, #00e5ff 0%, #3b82f6 50%, #6366f1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0 0 6px 0;
        }
        .benchmark-subtitle {
            color: #94a3b8;
            font-size: 1.05rem;
            font-weight: 400;
            margin: 0;
        }

        /* Overview Stat Card */
        .overview-card {
            background-color: #10192e;
            border: 1px solid #1e2d4d;
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 12px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25);
        }
        .overview-card-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748b;
            margin-bottom: 4px;
        }
        .overview-card-value {
            font-size: 1.35rem;
            font-weight: 700;
            color: #00e5ff;
        }

        /* Status Badge */
        .status-badge-pass {
            display: inline-block;
            padding: 3px 12px;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 700;
            background-color: rgba(16, 185, 129, 0.15);
            color: #10b981;
            border: 1px solid #10b981;
        }
        .status-badge-warning {
            display: inline-block;
            padding: 3px 12px;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 700;
            background-color: rgba(245, 158, 11, 0.15);
            color: #f59e0b;
            border: 1px solid #f59e0b;
        }
        .status-badge-fail {
            display: inline-block;
            padding: 3px 12px;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 700;
            background-color: rgba(239, 68, 68, 0.15);
            color: #ef4444;
            border: 1px solid #ef4444;
        }

        /* Model Pill Tags */
        .model-pill {
            display: inline-block;
            background-color: #1e293b;
            color: #38bdf8;
            border: 1px solid #0284c7;
            padding: 3px 10px;
            border-radius: 6px;
            font-size: 0.85rem;
            font-family: monospace;
            margin: 3px 4px 3px 0;
        }

        /* Card container */
        .section-card {
            background-color: #0d1527;
            border: 1px solid #1e293b;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 3. Application Title Header
    st.markdown(
        f"""
        <div class="benchmark-header">
            <h1 class="benchmark-title">GRAPH RAG BENCHMARK</h1>
            <p class="benchmark-subtitle">
                Retrieval-Augmented Generation &amp; Knowledge Graph Evaluation Platform
                &bull; <span style="color: #00e5ff; font-weight: 600;">Phase 1 Foundation v{__version__}</span>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar: Quick Navigation & Diagnostics Summary
    with st.sidebar:
        st.markdown("### ⚡ System Status")
        server_live = client.is_server_reachable()
        if server_live:
            st.success("Ollama Engine: **ONLINE**")
        else:
            st.error("Ollama Engine: **OFFLINE**")

        st.markdown("---")
        st.markdown("### ⚙️ Target Models")
        st.caption(f"**LLM:** `{settings.ollama_llm_model}`")
        st.caption(f"**Embed:** `{settings.ollama_embed_model}`")

        st.markdown("---")
        st.markdown("### 📁 Configuration")
        st.caption(f"**Host:** `{settings.ollama_base_url}`")
        st.caption(f"**Timeout:** `{settings.ollama_timeout}s`")
        st.caption(f"**Env:** `{settings.app_env}`")

        if st.button("🔄 Rerun Health Checks", use_container_width=True):
            st.rerun()

    # SECTION A: PROJECT OVERVIEW
    st.markdown("### A. Project Overview")
    col1, col2, col3, col4 = st.columns(4)

    is_venv = (
        hasattr(sys, "base_prefix") and sys.prefix != sys.base_prefix
    ) or "VIRTUAL_ENV" in sys.path
    venv_status = "Active (.venv)" if is_venv else "System / Global"

    with col1:
        st.markdown(
            f"""
            <div class="overview-card">
                <div class="overview-card-label">Project Version</div>
                <div class="overview-card-value">v{__version__}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="overview-card">
                <div class="overview-card-label">Python Version</div>
                <div class="overview-card-value">{sys.version.split()[0]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="overview-card">
                <div class="overview-card-label">Virtual Environment</div>
                <div class="overview-card-value">{venv_status}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        app_status_text = "READY" if server_live else "OLLAMA OFFLINE"
        status_color = "#10b981" if server_live else "#ef4444"
        st.markdown(
            f"""
            <div class="overview-card">
                <div class="overview-card-label">Phase 1 Status</div>
                <div class="overview-card-value" style="color: {status_color};">{app_status_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # SECTION B: OLLAMA INTEGRATION
    st.markdown("### B. Ollama Integration")
    col_b1, col_b2 = st.columns([1, 1])

    installed_models = []
    if server_live:
        try:
            installed_models = client.list_models()
        except OllamaError as exc:
            logger.error("Failed to list models in UI: %s", exc)
            installed_models = []

    with col_b1:
        st.markdown(
            f"""
            <div class="section-card">
                <h4 style="margin-top:0; color:#38bdf8;">🌐 Server Connectivity</h4>
                <p><strong>Base Endpoint:</strong> <code>{settings.ollama_base_url}</code></p>
                <p><strong>Connection Status:</strong>
                    {"<span class='status-badge-pass'>CONNECTED (200 OK)</span>" if server_live else "<span class='status-badge-fail'>DISCONNECTED</span>"}
                </p>
                <p><strong>Configured LLM:</strong> <code>{settings.ollama_llm_model}</code></p>
                <p><strong>Configured Embedding:</strong> <code>{settings.ollama_embed_model}</code></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b2:
        if server_live:
            if installed_models:
                pills_html = "".join(
                    [f"<span class='model-pill'>{m}</span>" for m in installed_models]
                )
            else:
                pills_html = "<p style='color:#94a3b8;'>No models currently detected on host.</p>"

            st.markdown(
                f"""
                <div class="section-card">
                    <h4 style="margin-top:0; color:#38bdf8;">📦 Installed Model Catalog ({len(installed_models)})</h4>
                    <div style="margin-top: 10px;">{pills_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning(
                "**Ollama server is currently unreachable at http://localhost:11434.**\n\n"
                "To start Ollama:\n"
                "1. Open the Ollama application from the Start Menu, or\n"
                "2. Run `ollama serve` in a new PowerShell terminal.\n"
                "3. Once running, refresh this page."
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # SECTION C: SYSTEM HEALTH CHECKS
    st.markdown("### C. System Health")
    report = checker.run_all_checks()

    # Table representation of health checks
    health_data = []
    for c in report.checks:
        badge_class = {
            HealthStatus.PASS: "status-badge-pass",
            HealthStatus.WARNING: "status-badge-warning",
            HealthStatus.FAIL: "status-badge-fail",
        }.get(c.status, "status-badge-warning")

        health_data.append(
            {
                "Status": f"<span class='{badge_class}'>{c.status.value}</span>",
                "Component": f"<strong>{c.name}</strong>",
                "Diagnostic Details": c.message,
            }
        )

    # Render table as HTML for styled badges
    table_html = """
    <table style="width:100%; border-collapse:collapse; background-color:#10192e; border:1px solid #1e293b; border-radius:10px; overflow:hidden;">
        <thead>
            <tr style="background-color:#172554; color:#38bdf8; text-align:left; border-bottom:1px solid #1e293b;">
                <th style="padding:12px 16px; width:120px;">Status</th>
                <th style="padding:12px 16px; width:220px;">Component</th>
                <th style="padding:12px 16px;">Diagnostic Message</th>
            </tr>
        </thead>
        <tbody>
    """
    for row in health_data:
        table_html += f"""
            <tr style="border-bottom: 1px solid #1e293b;">
                <td style="padding:12px 16px;">{row["Status"]}</td>
                <td style="padding:12px 16px;">{row["Component"]}</td>
                <td style="padding:12px 16px; color:#cbd5e1;">{row["Diagnostic Details"]}</td>
            </tr>
        """
    table_html += "</tbody></table>"

    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # SECTION D: MODEL TESTING (Isolated interactive triggers)
    st.markdown("### D. Model Diagnostics & Testing")
    st.caption(
        "Interactive validation of text generation and dense vector embeddings via local Ollama."
    )

    # Initialize session state storage so tests do not run automatically on rerun
    if "qwen_output" not in st.session_state:
        st.session_state["qwen_output"] = None
    if "embed_output" not in st.session_state:
        st.session_state["embed_output"] = None

    test_col1, test_col2 = st.columns(2)

    with test_col1:
        st.markdown(
            f"""
            <div class="section-card" style="min-height: 400px;">
                <h4 style="margin-top:0; color:#38bdf8;">🤖 Qwen Generation Test</h4>
                <p style="color:#94a3b8; font-size:0.9rem;">Target Model: <code>{settings.ollama_llm_model}</code></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        prompt_input = st.text_input(
            "Inference Prompt",
            value="Explain Knowledge Graph RAG in one clear sentence.",
            key="input_qwen_prompt",
        )

        if st.button("🚀 Test Qwen Generation", type="primary", use_container_width=True):
            if not server_live:
                st.error("Cannot run generation: Ollama server is offline.")
            else:
                with st.spinner(f"Requesting inference from {settings.ollama_llm_model}..."):
                    try:
                        gen_resp = client.generate(
                            prompt=prompt_input,
                            system="You are an expert AI research assistant. Provide concise, clear answers.",
                        )
                        st.session_state["qwen_output"] = {
                            "response": gen_resp.response,
                            "tokens": gen_resp.eval_count,
                            "duration_ns": gen_resp.total_duration_ns,
                            "model": gen_resp.model,
                        }
                    except OllamaError as exc:
                        logger.exception("LLM generation error")
                        st.error(f"Generation failed: {exc}")

        # Display stored LLM result if available
        if st.session_state["qwen_output"]:
            out: dict[str, Any] = st.session_state["qwen_output"]
            dur_ms = (out["duration_ns"] or 0) / 1_000_000
            st.success(
                f"Generated response from `{out['model']}` ({out.get('tokens', 'N/A')} tokens in {dur_ms:.1f}ms):"
            )
            st.info(out["response"])

    with test_col2:
        st.markdown(
            f"""
            <div class="section-card" style="min-height: 400px;">
                <h4 style="margin-top:0; color:#38bdf8;">📐 Vector Embedding Test</h4>
                <p style="color:#94a3b8; font-size:0.9rem;">Target Model: <code>{settings.ollama_embed_model}</code></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        embed_input = st.text_input(
            "Text to Embed",
            value="Graph RAG Benchmark hybrid retrieval evaluation.",
            key="input_embed_text",
        )

        if st.button("⚡ Test Embedding Generation", type="secondary", use_container_width=True):
            if not server_live:
                st.error("Cannot generate embeddings: Ollama server is offline.")
            else:
                with st.spinner(f"Computing vector using {settings.ollama_embed_model}..."):
                    try:
                        emb_resp = client.embed(embed_input)
                        dims = emb_resp.dimensions
                        vector = emb_resp.embeddings[0]
                        st.session_state["embed_output"] = {
                            "dimensions": dims,
                            "preview": [round(val, 4) for val in vector[:8]],
                            "model": emb_resp.model,
                        }
                    except OllamaError as exc:
                        logger.exception("Embedding generation error")
                        st.error(f"Embedding failed: {exc}")

        # Display stored Embedding result if available
        if st.session_state["embed_output"]:
            e_out: dict[str, Any] = st.session_state["embed_output"]
            st.success(
                f"Generated vector from `{e_out['model']}`: **{e_out['dimensions']} dimensions**"
            )
            st.write("First 8 vector dimensions:")
            st.code(str(e_out["preview"]))

    st.markdown("---")
    with st.expander("🔒 Active Environment Configuration (Sanitized)"):
        st.json(settings.to_safe_dict())


def main() -> None:
    """Application main entry router."""
    # Check if executed inside Streamlit runtime
    if is_running_in_streamlit():
        run_streamlit()
    else:
        run_cli()


if __name__ == "__main__":
    main()
