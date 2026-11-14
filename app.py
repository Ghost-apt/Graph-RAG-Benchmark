"""Graph RAG Benchmark - Application Entry Point.

Provides a dual-mode entry point:
1. Streamlit Interactive Web Dashboard (`python -m streamlit run app.py` or `streamlit run app.py`)
2. Command-Line Interface Health Overview (`python app.py`)

Includes:
- Phase 1: Environment Setup, Ollama Connectivity, Health Diagnostics, Model Verification
- Phase 2: Document Ingestion, Validation, Extraction, Cleaning, Chunking, Local Repository & Export
"""

import csv
import io
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Ensure src is on Python search path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import networkx as nx
import plotly.graph_objects as go

from graphrag_bench import __version__
from graphrag_bench.access_control import Permission, Role, has_permission
from graphrag_bench.answer_schemas import AnswerStatus, GroundedAnswer
from graphrag_bench.auth_manager import AuthManager
from graphrag_bench.backup_manager import BackupCategory, BackupManager
from graphrag_bench.benchmark_metrics import (
    aggregate_strategy_metrics,
)
from graphrag_bench.benchmark_runner import BenchmarkRunner
from graphrag_bench.benchmark_schemas import (
    BenchmarkQueryCategory,
    BenchmarkQueryResult,
    BenchmarkSplit,
    RetrievalStrategy,
)
from graphrag_bench.benchmark_statistics import (
    compare_strategies_paired,
    compute_latency_summary,
)
from graphrag_bench.benchmark_storage import BenchmarkStorage
from graphrag_bench.config import Settings, get_settings
from graphrag_bench.document_processor import DocumentProcessor
from graphrag_bench.document_repository import DocumentRepository
from graphrag_bench.embedding_service import EmbeddingService
from graphrag_bench.entity_resolver import EntityResolver
from graphrag_bench.graph_builder import GraphBuilder
from graphrag_bench.graph_extraction_service import GraphExtractionService
from graphrag_bench.graph_retrieval_schemas import (
    ExtractionMethod,
    GraphSearchResponse,
    GraphSearchStatus,
)
from graphrag_bench.graph_search import GraphSearchEngine
from graphrag_bench.graph_store import GraphStore
from graphrag_bench.health import HealthChecker, HealthStatus
from graphrag_bench.hybrid_retriever import HybridRetriever
from graphrag_bench.hybrid_schemas import (
    HybridSearchResponse,
    HybridSearchStatus,
    RetrievalSource,
)
from graphrag_bench.indexing_pipeline import IndexingPipeline
from graphrag_bench.logger import get_logger, setup_logger
from graphrag_bench.observability import ConcurrencyLimiter, TelemetryCollector
from graphrag_bench.ollama_client import OllamaClient, OllamaError
from graphrag_bench.production_health import (
    ProductionHealthMonitor,
    ProductionHealthState,
)
from graphrag_bench.rag_pipeline import RAGPipeline
from graphrag_bench.semantic_search import SearchResponse, SemanticSearch
from graphrag_bench.user_audit import UserAuditLogger
from graphrag_bench.user_registry import UserRegistry, UserStatus
from graphrag_bench.vector_store import VectorStore

ENTITY_TYPE_COLORS: dict[str, str] = {
    "CONCEPT": "#38bdf8",
    "TECHNOLOGY": "#a855f7",
    "SOFTWARE": "#10b981",
    "FRAMEWORK": "#f59e0b",
    "PROGRAMMING_LANGUAGE": "#ec4899",
    "METHOD": "#6366f1",
    "DATASET": "#14b8a6",
    "PERSON": "#f97316",
    "ORGANIZATION": "#3b82f6",
    "LOCATION": "#e11d48",
    "OTHER": "#94a3b8",
}

logger = get_logger("app")


def is_running_in_streamlit() -> bool:
    """Detect whether execution is running inside the active Streamlit runtime."""
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
    print("      GRAPH RAG BENCHMARK - SYSTEM STATUS")
    print("=" * 65)
    print(f"Project Version   : v{__version__}")
    print(f"Python Runtime    : {sys.version.split()[0]}")
    print(f"Virtual Env       : {sys.prefix}")
    print(f"Ollama Base URL   : {settings.ollama_base_url}")
    print(f"Target LLM Model  : {settings.ollama_llm_model}")
    print(f"Target Embed Model: {settings.ollama_embed_model}")
    print(f"Uploads Directory : {settings.upload_dir}")
    print(f"Processed Storage : {settings.processed_dir}")
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
    """Render interactive Streamlit dashboard for Ingestion and Diagnostics."""
    import streamlit as st

    setup_logger()
    settings: Settings = get_settings()
    client = OllamaClient(settings=settings)
    checker = HealthChecker(settings=settings, ollama_client=client)
    repo = DocumentRepository(base_dir=Path.cwd())
    processor = DocumentProcessor(repository=repo)
    embed_service = EmbeddingService(client=client, settings=settings)
    vector_store = VectorStore(settings=settings)
    indexing_pipeline = IndexingPipeline(
        repository=repo,
        embedding_service=embed_service,
        vector_store=vector_store,
    )
    search_engine = SemanticSearch(
        vector_store=vector_store,
        embedding_service=embed_service,
        settings=settings,
    )
    resolver = EntityResolver()
    graph_store = GraphStore()
    graph_builder = GraphBuilder(
        doc_repo=repo,
        extraction_service=GraphExtractionService(
            client=client,
            settings=settings,
            min_confidence=settings.graph_min_confidence,
        ),
        resolver=resolver,
        store=graph_store,
        graph_dir=settings.graph_dir,
    )
    graph_search_engine = GraphSearchEngine(store=graph_store, settings=settings)
    hybrid_retriever = HybridRetriever(
        vector_search=search_engine,
        graph_search=graph_search_engine,
        repository=repo,
        settings=settings,
    )
    user_registry = UserRegistry(settings=settings)
    user_audit = UserAuditLogger(settings=settings)
    auth_manager = AuthManager(
        settings=settings,
        registry=user_registry,
        audit_logger=user_audit,
    )

    # 1. Page Configuration (Must be first Streamlit call)
    st.set_page_config(
        page_title="Graph RAG Benchmark",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 2. Custom CSS styling (Dark navy, electric blue & cyan accents)
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0b0f19;
            color: #e2e8f0;
        }
        .benchmark-header {
            background: linear-gradient(135deg, #0d1527 0%, #111c38 100%);
            border: 1px solid #1e293b;
            border-radius: 14px;
            padding: 22px 28px;
            margin-bottom: 20px;
            box-shadow: 0 4px 20px rgba(0, 102, 255, 0.12);
        }
        .benchmark-title {
            font-size: 2.1rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            background: linear-gradient(90deg, #00e5ff 0%, #3b82f6 50%, #6366f1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0 0 4px 0;
        }
        .benchmark-subtitle {
            color: #94a3b8;
            font-size: 1.0rem;
            font-weight: 400;
            margin: 0;
        }
        .overview-card {
            background-color: #10192e;
            border: 1px solid #1e2d4d;
            border-radius: 10px;
            padding: 14px 18px;
            margin-bottom: 12px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
        }
        .overview-card-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748b;
            margin-bottom: 3px;
        }
        .overview-card-value {
            font-size: 1.3rem;
            font-weight: 700;
            color: #00e5ff;
        }
        .status-badge-pass {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 700;
            background-color: rgba(16, 185, 129, 0.15);
            color: #10b981;
            border: 1px solid #10b981;
        }
        .status-badge-warning {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 700;
            background-color: rgba(245, 158, 11, 0.15);
            color: #f59e0b;
            border: 1px solid #f59e0b;
        }
        .status-badge-fail {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 700;
            background-color: rgba(239, 68, 68, 0.15);
            color: #ef4444;
            border: 1px solid #ef4444;
        }
        .model-pill {
            display: inline-block;
            background-color: #1e293b;
            color: #38bdf8;
            border: 1px solid #0284c7;
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 0.8rem;
            font-family: monospace;
            margin: 2px 4px 2px 0;
        }
        .chunk-box {
            background-color: #090e17;
            border: 1px solid #1e293b;
            border-radius: 8px;
            padding: 16px;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.88rem;
            line-height: 1.5;
            color: #cbd5e1;
            white-space: pre-wrap;
            max-height: 400px;
            overflow-y: auto;
        }
        .citation-badge {
            display: inline-block;
            background-color: rgba(56, 189, 248, 0.2);
            color: #38bdf8;
            border: 1px solid #0284c7;
            padding: 1px 6px;
            border-radius: 4px;
            font-size: 0.82rem;
            font-weight: 600;
            font-family: monospace;
            margin: 0 2px;
        }
        .answer-box {
            background-color: #0d1527;
            border: 1px solid #1e293b;
            border-left: 4px solid #00e5ff;
            border-radius: 10px;
            padding: 20px;
            font-size: 1.05rem;
            line-height: 1.6;
            color: #f1f5f9;
            margin-bottom: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 2.5 User Authentication & Session Verification
    current_user = auth_manager.authenticate_streamlit()
    if current_user is None:
        return

    # 3. Application Title Header
    st.markdown(
        f"""
        <div class="benchmark-header">
            <h1 class="benchmark-title">GRAPH RAG BENCHMARK</h1>
            <p class="benchmark-subtitle">
                Retrieval-Augmented Generation &amp; Knowledge Graph Evaluation Platform
                &bull; <span style="color: #00e5ff; font-weight: 600;">v{__version__}</span>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Initialize Session State
    if "qwen_output" not in st.session_state:
        st.session_state["qwen_output"] = None
    if "embed_output" not in st.session_state:
        st.session_state["embed_output"] = None
    if "ingestion_batch_results" not in st.session_state:
        st.session_state["ingestion_batch_results"] = []
    if "selected_chunk_doc_id" not in st.session_state:
        st.session_state["selected_chunk_doc_id"] = None
    if "semantic_search_response" not in st.session_state:
        st.session_state["semantic_search_response"] = None
    if "graph_search_response" not in st.session_state:
        st.session_state["graph_search_response"] = None
    if "hybrid_search_response" not in st.session_state:
        st.session_state["hybrid_search_response"] = None
    if "grounded_rag_answer" not in st.session_state:
        st.session_state["grounded_rag_answer"] = None

    # Sidebar: Quick Status & Info
    with st.sidebar:
        auth_manager.render_user_badge(in_sidebar=True)
        st.markdown("---")
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
        st.markdown("### 📁 Storage & Engine")
        docs_count = len(repo.list_documents())
        st.caption(f"**Ingested Documents:** `{docs_count}`")
        st.caption(f"**Uploads Dir:** `{settings.upload_dir}`")
        st.caption(f"**Processed Dir:** `{settings.processed_dir}`")

        st.markdown("---")
        st.markdown("### 🧬 Vector Database (Chroma)")
        try:
            vec_count = vector_store.count()
            st.caption(f"**Indexed Vectors:** `{vec_count}`")
            st.caption(f"**Collection:** `{vector_store.collection_name}`")
            st.caption("**Metric:** `Cosine Distance`")
        except Exception:
            st.caption("**Vector DB:** `Unavailable`")

        st.markdown("---")
        st.markdown("### 🕸️ Knowledge Graph")
        try:
            st.caption(f"**Entities (Nodes):** `{graph_store.node_count}`")
            st.caption(f"**Triples (Edges):** `{graph_store.edge_count}`")
            st.caption(f"**Model:** `{settings.graph_extraction_model}`")
        except Exception:
            st.caption("**Knowledge Graph:** `Unavailable`")

        st.markdown("---")
        if st.button("🔄 Refresh Data & Checks", use_container_width=True):
            st.rerun()

    # Top-Level Tabs
    (
        tab_graph,
        tab_search,
        tab_ingest,
        tab_health,
        tab_retrieval,
        tab_hybrid,
        tab_chat,
        tab_benchmark,
        tab_ops,
    ) = st.tabs(
        [
            "🕸️ Knowledge Graph Explorer",
            "🔍 Embeddings & Vector Search",
            "📄 Document Ingestion & Preprocessing",
            "⚡ System Health & Diagnostics",
            "🧭 Graph Retrieval Lab",
            "⚖️ Hybrid Retrieval Lab",
            "💬 Grounded RAG Chat",
            "📊 Comparative Benchmark Lab",
            "🛡️ Production Operations",
        ]
    )

    # =========================================================================
    # TAB 1: KNOWLEDGE GRAPH EXPLORER (PHASE 4)
    # =========================================================================
    with tab_graph:
        kg_status = graph_builder.get_status()
        kg_stats = graph_store.get_statistics()

        # SECTION A: Knowledge Graph Overview & Metrics
        st.markdown("### A. Knowledge Graph Architecture & Metrics")

        stat_k1, stat_k2, stat_k3, stat_k4, stat_k5 = st.columns(5)
        with stat_k1:
            st.markdown(
                f"""
                <div class="overview-card">
                    <div class="overview-card-label">Total Entities (Nodes)</div>
                    <div class="overview-card-value">{kg_stats["node_count"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with stat_k2:
            st.markdown(
                f"""
                <div class="overview-card">
                    <div class="overview-card-label">Relationships (Edges)</div>
                    <div class="overview-card-value">{kg_stats["edge_count"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with stat_k3:
            st.markdown(
                f"""
                <div class="overview-card">
                    <div class="overview-card-label">Entity Categories</div>
                    <div class="overview-card-value">{len(kg_stats["entity_type_distribution"])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with stat_k4:
            st.markdown(
                f"""
                <div class="overview-card">
                    <div class="overview-card-label">Components / Density</div>
                    <div class="overview-card-value" style="font-size:1.1rem;">
                        {kg_stats["weakly_connected_components"]} <span style="font-size:0.75rem; color:#64748b;">(d: {kg_stats["density"]:.3f})</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with stat_k5:
            st.markdown(
                f"""
                <div class="overview-card">
                    <div class="overview-card-label">Processed Chunks</div>
                    <div class="overview-card-value" style="font-size:1.1rem;">
                        {kg_status["processed_chunks"]}/{kg_status["total_repository_chunks"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.caption(
            f"📁 Canonical JSON: `{settings.graph_path}` &bull; "
            f"GraphML: `{settings.graph_export_path}` &bull; "
            f"Extraction Model: `{kg_status['model']}` &bull; "
            f"Min Confidence: `{settings.graph_min_confidence:.2f}` &bull; "
            f"Last Sync: `{kg_status['last_updated'] or 'Never'}`"
        )

        st.markdown("---")

        # SECTION B: Extraction Pipeline Controller
        st.markdown("### B. Extraction Pipeline Controller")
        b_col1, b_col2 = st.columns([1.2, 0.8])

        with b_col1:
            if kg_status["unprocessed_chunks"] > 0:
                st.warning(
                    f"⚠️ **{kg_status['unprocessed_chunks']} repository chunk(s)** are awaiting extraction. "
                    "Extract triples to populate the Knowledge Graph."
                )
            elif kg_status["total_repository_chunks"] == 0:
                st.info(
                    "ℹ️ No documents ingested in repository. Upload documents in Tab 3 to extract knowledge graphs."
                )
            else:
                st.success(
                    f"✅ Knowledge Graph is fully synchronized across all {kg_status['total_repository_chunks']} canonical chunks."
                )

            if kg_status["failed_chunks"] > 0:
                st.error(
                    f"⚠️ **{kg_status['failed_chunks']} chunk(s)** encountered errors during previous extraction attempts."
                )

        with b_col2:
            st.markdown(
                """
                <div style="background:#10192e; border:1px solid #1e2d4d; border-radius:8px; padding:12px; font-size:0.85rem; color:#94a3b8;">
                    <b style="color:#38bdf8;">Deterministic Graph Construction:</b>
                    <ul style="margin:4px 0 0 -15px;">
                        <li>Model: <code>qwen3:8b</code> (structured JSON, CoT bypassed)</li>
                        <li>Type-safe entity resolution with strict taxonomy</li>
                        <li>Verbatim chunk quote verification</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        btn_c1, btn_c2, btn_c3 = st.columns([1.2, 1.2, 1.6])
        with btn_c1:
            extract_clicked = st.button(
                "⚡ Extract & Build Knowledge Graph",
                type="primary",
                use_container_width=True,
                disabled=(kg_status["total_repository_chunks"] == 0),
            )
        with btn_c2:
            force_rebuild = st.checkbox("Force full rebuild (re-extract all)", value=False)
        with btn_c3:
            retry_clicked = st.button(
                f"🔁 Retry Failed Chunks ({kg_status['failed_chunks']})",
                disabled=(kg_status["failed_chunks"] == 0),
                use_container_width=True,
            )

        if extract_clicked:
            p_bar = st.progress(0, text="Initializing extraction pipeline...")

            def on_progress(curr: int, tot: int, msg: str) -> None:
                pct = int((curr / max(tot, 1)) * 100)
                p_bar.progress(min(pct, 100), text=f"[{curr}/{tot}] {msg}")

            with st.spinner("Executing grounded LLM extraction with Qwen3:8b..."):
                res = graph_builder.build_all(
                    force_rebuild=force_rebuild,
                    progress_callback=on_progress,
                )
                p_bar.progress(100, text="Extraction complete!")
                st.success(
                    f"Successfully processed {res['chunks_successful']} chunk(s). "
                    f"Graph now contains **{res['total_entities']} entities** and "
                    f"**{res['total_relationships']} relationships**."
                )
                st.rerun()

        if retry_clicked:
            p_bar = st.progress(0, text="Retrying failed extractions...")

            def on_retry_prog(curr: int, tot: int, msg: str) -> None:
                pct = int((curr / max(tot, 1)) * 100)
                p_bar.progress(min(pct, 100), text=f"[{curr}/{tot}] {msg}")

            with st.spinner("Retrying failed extractions..."):
                res = graph_builder.retry_failed(progress_callback=on_retry_prog)
                p_bar.progress(100, text="Retry run complete!")
                st.success(
                    f"Retried {res['retried']} chunks: {res['successful']} succeeded, {res['failed']} failed."
                )
                st.rerun()

        st.markdown("---")

        # SECTION C: Interactive Knowledge Graph Visualization (Plotly)
        st.markdown("### C. Interactive Knowledge Graph Visualization")

        all_entities = graph_store.get_entities()
        all_relationships = graph_store.get_relationships()

        if not all_entities:
            st.info(
                "Knowledge Graph is currently empty. Run extraction above to construct and visualize the graph."
            )
        else:
            vis_c1, vis_c2, vis_c3 = st.columns([1.2, 1.2, 1.0])
            with vis_c1:
                available_types = sorted(kg_stats["entity_type_distribution"].keys())
                selected_types = st.multiselect(
                    "Filter by Entity Category",
                    options=available_types,
                    default=available_types,
                    key="filter_entity_types",
                )
            with vis_c2:
                available_docs = sorted(
                    {e.source_filename for e in all_relationships if e.source_filename}
                )
                selected_docs = st.multiselect(
                    "Filter by Source Document",
                    options=available_docs,
                    default=available_docs,
                    key="filter_source_docs",
                )
            with vis_c3:
                max_nodes = st.slider(
                    "Max Visible Nodes",
                    min_value=5,
                    max_value=max(len(all_entities), 10),
                    value=min(40, len(all_entities)),
                    step=5,
                    key="max_visible_nodes",
                )

            filtered_entities = [e for e in all_entities if e.entity_type in selected_types]
            if selected_docs and available_docs:
                filtered_entities = [
                    e
                    for e in filtered_entities
                    if any(
                        doc in selected_docs
                        for doc in [
                            d.filename
                            for d in repo.list_documents()
                            if d.document_id in e.source_document_ids
                        ]
                    )
                    or not e.source_document_ids
                ]

            filtered_entities.sort(key=lambda e: e.mention_count, reverse=True)
            display_entities = filtered_entities[:max_nodes]
            display_node_ids = {e.entity_id for e in display_entities}

            display_edges = [
                r
                for r in all_relationships
                if r.subject_entity_id in display_node_ids
                and r.object_entity_id in display_node_ids
            ]
            if selected_docs and available_docs:
                display_edges = [
                    r
                    for r in display_edges
                    if r.source_filename in selected_docs or not r.source_filename
                ]

            sub_g = nx.MultiDiGraph()
            for ent in display_entities:
                sub_g.add_node(ent.entity_id, **ent.model_dump())
            for rel in display_edges:
                sub_g.add_edge(
                    rel.subject_entity_id,
                    rel.object_entity_id,
                    key=rel.relationship_id,
                    **rel.model_dump(),
                )

            if sub_g.number_of_nodes() == 0:
                st.warning(
                    "No nodes match the selected filter criteria. Expand filter selections above."
                )
            else:
                if sub_g.number_of_nodes() <= 3:
                    pos = nx.circular_layout(sub_g)
                else:
                    pos = nx.spring_layout(sub_g, seed=42, k=0.7, iterations=60)

                edge_x = []
                edge_y = []
                edge_mid_x = []
                edge_mid_y = []
                edge_hover_texts = []
                edge_labels = []

                for u, v, k, data in sub_g.edges(keys=True, data=True):
                    x0, y0 = pos[u]
                    x1, y1 = pos[v]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])

                    mx = (x0 + x1) / 2.0
                    my = (y0 + y1) / 2.0
                    edge_mid_x.append(mx)
                    edge_mid_y.append(my)
                    pred = data.get("predicate", "RELATED_TO")
                    quote = data.get("evidence_quote", "")
                    u_name = sub_g.nodes[u].get("canonical_name", u)
                    v_name = sub_g.nodes[v].get("canonical_name", v)
                    edge_labels.append(pred)
                    edge_hover_texts.append(
                        f"<b>{u_name}</b> &rarr; <b>[{pred}]</b> &rarr; <b>{v_name}</b><br>"
                        f'<i>Quote:</i> "{quote[:150]}..."'
                    )

                edge_trace = go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    line={"width": 1.5, "color": "#475569"},
                    hoverinfo="none",
                    mode="lines",
                )

                edge_label_trace = go.Scatter(
                    x=edge_mid_x,
                    y=edge_mid_y,
                    mode="markers+text",
                    text=edge_labels,
                    textposition="top center",
                    textfont={"size": 9, "color": "#94a3b8"},
                    hoverinfo="text",
                    hovertext=edge_hover_texts,
                    marker={"size": 5, "color": "#38bdf8", "opacity": 0.8},
                )

                node_traces = []
                for cat in available_types:
                    cat_entities = [e for e in display_entities if e.entity_type == cat]
                    if not cat_entities:
                        continue
                    cat_x = [pos[e.entity_id][0] for e in cat_entities]
                    cat_y = [pos[e.entity_id][1] for e in cat_entities]
                    cat_sizes = [max(14, min(36, 16 + e.mention_count * 3)) for e in cat_entities]
                    cat_texts = [e.canonical_name for e in cat_entities]
                    cat_hovers = [
                        f"<b>{e.canonical_name}</b><br>"
                        f"Category: <code>{e.entity_type}</code><br>"
                        f"Mentions: {e.mention_count}<br>"
                        f"Aliases: {', '.join(e.aliases) if e.aliases else 'None'}<br>"
                        f"<i>{e.description[:180]}</i>"
                        for e in cat_entities
                    ]
                    color = ENTITY_TYPE_COLORS.get(cat, "#94a3b8")

                    node_traces.append(
                        go.Scatter(
                            x=cat_x,
                            y=cat_y,
                            mode="markers+text",
                            name=cat,
                            text=cat_texts,
                            textposition="bottom center",
                            textfont={"size": 11, "color": "#f1f5f9"},
                            hoverinfo="text",
                            hovertext=cat_hovers,
                            marker={
                                "size": cat_sizes,
                                "color": color,
                                "line": {"width": 1.5, "color": "#0f172a"},
                                "opacity": 0.95,
                            },
                        )
                    )

                fig = go.Figure(
                    data=[edge_trace, edge_label_trace, *node_traces],
                    layout=go.Layout(
                        showlegend=True,
                        legend={
                            "orientation": "h",
                            "yanchor": "bottom",
                            "y": 1.02,
                            "xanchor": "right",
                            "x": 1,
                            "font": {"color": "#cbd5e1", "size": 10},
                        },
                        hovermode="closest",
                        margin={"b": 20, "l": 20, "r": 20, "t": 40},
                        plot_bgcolor="#0b0f19",
                        paper_bgcolor="#0b0f19",
                        xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
                        yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
                        height=580,
                    ),
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # SECTION D: Entity Explorer & Disambiguation
        st.markdown("### D. Entity Explorer & Disambiguation")
        if all_entities:
            entity_names = {
                f"{e.canonical_name} [{e.entity_type}]": e.entity_id for e in all_entities
            }
            selected_ent_label = st.selectbox(
                "Select Entity to Inspect Details and Provenance:",
                options=list(entity_names.keys()),
                key="select_entity_inspect",
            )
            selected_eid = entity_names[selected_ent_label]
            target_ent = graph_store.get_entity(selected_eid)

            if target_ent:
                ent_d1, ent_d2 = st.columns([1.4, 0.6])
                with ent_d1:
                    st.markdown(
                        f"""
                        <div style="background:#10192e; border:1px solid #1e2d4d; border-radius:10px; padding:16px; margin-bottom:12px;">
                            <h4 style="margin:0 0 8px 0; color:#38bdf8;">{target_ent.canonical_name}</h4>
                            <p style="margin:0 0 8px 0; color:#cbd5e1; font-size:0.95rem;">{target_ent.description or "No grounded description available."}</p>
                            <span style="font-size:0.8rem; color:#94a3b8;">Deterministic ID: <code>{target_ent.entity_id}</code></span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with ent_d2:
                    st.markdown(
                        f"""
                        <div style="background:#10192e; border:1px solid #1e2d4d; border-radius:10px; padding:16px; margin-bottom:12px;">
                            <div style="font-size:0.8rem; color:#94a3b8;">CATEGORY</div>
                            <div style="font-size:1.1rem; font-weight:700; color:#a855f7;">{target_ent.entity_type}</div>
                            <div style="font-size:0.8rem; color:#94a3b8; margin-top:8px;">TOTAL MENTIONS</div>
                            <div style="font-size:1.2rem; font-weight:700; color:#10b981;">{target_ent.mention_count}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("**Aliases & Surface Forms:**")
                if target_ent.aliases:
                    pills = " ".join(
                        f"<span class='model-pill'>{a}</span>" for a in target_ent.aliases
                    )
                    st.markdown(pills, unsafe_allow_html=True)
                else:
                    st.caption("No registered aliases.")

                neighbors = graph_store.get_neighbors(target_ent.entity_id, direction="both")
                st.markdown(f"**Connected Relationships ({len(neighbors)}):**")
                if neighbors:
                    rel_rows = []
                    for item in neighbors:
                        rel = item["relationship"]
                        nbr = item["neighbor"]
                        direction = item["direction"]
                        if direction == "outgoing":
                            triplet_str = f"**{target_ent.canonical_name}** &rarr; `{rel.predicate}` &rarr; **{nbr.canonical_name}**"
                        else:
                            triplet_str = f"**{nbr.canonical_name}** &rarr; `{rel.predicate}` &rarr; **{target_ent.canonical_name}**"
                        rel_rows.append(
                            {
                                "Direction": "Outgoing" if direction == "outgoing" else "Incoming",
                                "Assertion": triplet_str,
                                "Evidence Quote": f'"{rel.evidence_quote}"',
                                "Confidence": f"{rel.confidence:.2f}",
                                "Source File": rel.source_filename,
                            }
                        )
                    st.table(rel_rows)
                else:
                    st.caption("This entity currently has no connected relationships.")

        st.markdown("---")

        # SECTION E: Triples & Relationship Explorer
        st.markdown("### E. Asserted Relationship Triples")
        if all_relationships:
            pred_options = ["ALL"] + sorted(kg_stats["predicate_distribution"].keys())
            selected_pred = st.selectbox(
                "Filter by Predicate:", options=pred_options, key="select_pred_filter"
            )

            filtered_rels = all_relationships
            if selected_pred != "ALL":
                filtered_rels = [r for r in filtered_rels if r.predicate == selected_pred]

            st.caption(
                f"Showing {len(filtered_rels)} of {len(all_relationships)} asserted relationship triples."
            )

            table_data = []
            for r in filtered_rels:
                sub = graph_store.get_entity(r.subject_entity_id)
                obj = graph_store.get_entity(r.object_entity_id)
                table_data.append(
                    {
                        "Subject": sub.canonical_name if sub else r.subject_entity_id,
                        "Predicate": r.predicate,
                        "Object": obj.canonical_name if obj else r.object_entity_id,
                        "Evidence Quote": f'"{r.evidence_quote[:120]}..."',
                        "Confidence": f"{r.confidence:.2f}",
                        "Source Doc": r.source_filename,
                        "Chunk ID": r.source_chunk_id,
                    }
                )
            st.dataframe(table_data, use_container_width=True, hide_index=True)
        else:
            st.info("No relationships currently asserted in the Knowledge Graph.")

        st.markdown("---")

        # SECTION F: Graph Management & Export
        st.markdown("### F. Graph Serialization & Export")
        f_col1, f_col2, f_col3 = st.columns(3)

        with f_col1:
            if Path(settings.graph_path).exists():
                with open(settings.graph_path, "r", encoding="utf-8") as f:
                    json_bytes = f.read().encode("utf-8")
                st.download_button(
                    "📥 Download Canonical JSON (knowledge_graph.json)",
                    data=json_bytes,
                    file_name="knowledge_graph.json",
                    mime="application/json",
                    use_container_width=True,
                )
            else:
                st.button(
                    "📥 Download JSON (Not Available)", disabled=True, use_container_width=True
                )

        with f_col2:
            if Path(settings.graph_export_path).exists():
                with open(settings.graph_export_path, "rb") as f:
                    graphml_bytes = f.read()
                st.download_button(
                    "📥 Download GraphML (knowledge_graph.graphml)",
                    data=graphml_bytes,
                    file_name="knowledge_graph.graphml",
                    mime="application/xml",
                    use_container_width=True,
                )
            else:
                st.button(
                    "📥 Download GraphML (Not Available)", disabled=True, use_container_width=True
                )

        with f_col3:
            confirm_reset = st.checkbox("Confirm Graph Reset", key="chk_confirm_kg_reset")
            if st.button(
                "🗑️ Reset Knowledge Graph",
                type="secondary",
                disabled=not confirm_reset,
                use_container_width=True,
            ):
                graph_store.reset()
                resolver.clear()
                graph_builder.build_all(force_rebuild=True)
                st.warning("Knowledge Graph has been reset.")
                st.rerun()

    # =========================================================================
    # TAB 2: EMBEDDINGS & CHROMADB VECTOR SEARCH (PHASE 3)
    # =========================================================================
    with tab_search:
        # SECTION A: Vector Database Overview & Statistics
        st.markdown("### A. Vector Database Overview")
        stat = indexing_pipeline.get_indexing_status()

        stat_c1, stat_c2, stat_c3, stat_c4 = st.columns(4)
        with stat_c1:
            st.markdown(
                f"""
                <div class="overview-card">
                    <div class="overview-card-label">Active Collection</div>
                    <div class="overview-card-value" style="font-size:1.1rem; word-break:break-all;">{stat["collection_name"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with stat_c2:
            st.markdown(
                f"""
                <div class="overview-card">
                    <div class="overview-card-label">Indexed Vectors</div>
                    <div class="overview-card-value">{stat["indexed_vector_count"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with stat_c3:
            st.markdown(
                """
                <div class="overview-card">
                    <div class="overview-card-label">Embedding Dimensions</div>
                    <div class="overview-card-value">768 <span style="font-size:0.8rem; color:#64748b;">(nomic)</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with stat_c4:
            st.markdown(
                """
                <div class="overview-card">
                    <div class="overview-card-label">Distance Metric</div>
                    <div class="overview-card-value" style="font-size:1.1rem;">Cosine (HNSW)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.caption(
            f"📁 Persistent ChromaDB Storage: `{stat['persist_directory']}` &bull; "
            f"Embedding Model: `{stat['configured_model']}` &bull; "
            f"Canonical Chunks: `{stat['total_canonical_chunks']}` &bull; "
            f"Pending Chunks: `{stat['pending_chunk_count']}`"
        )

        st.markdown("---")

        # SECTION B: Embedding Generation & Indexing
        st.markdown("### B. Embedding Generation & Pipeline Indexing")
        idx_col1, idx_col2 = st.columns([1.2, 0.8])

        with idx_col1:
            if stat["pending_chunk_count"] > 0:
                st.warning(
                    f"⚠️ **{stat['pending_chunk_count']} canonical chunk(s)** are currently unindexed in ChromaDB. "
                    "Generate embeddings below to include them in semantic search."
                )
            else:
                st.success(
                    "✅ **Vector index is fully synchronized** with canonical document chunks."
                )

            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                index_new_btn = st.button(
                    "⚡ Index New Chunks",
                    type="primary",
                    disabled=stat["pending_chunk_count"] == 0,
                    use_container_width=True,
                    help="Generates vector embeddings for any newly ingested document chunks not yet in ChromaDB.",
                )
            with btn_col2:
                force_reindex_btn = st.button(
                    "🔄 Force Re-index All",
                    type="secondary",
                    use_container_width=True,
                    help="Resets ChromaDB collection and recalculates embeddings for all canonical chunks.",
                )

            if index_new_btn or force_reindex_btn:
                force_flag = bool(force_reindex_btn)
                progress_bar = st.progress(0.0, text="Initializing indexing pipeline...")

                def update_progress(ratio: float, msg: str) -> None:
                    progress_bar.progress(min(1.0, max(0.0, ratio)), text=msg)

                try:
                    p_stats = indexing_pipeline.index_all(
                        force_reindex=force_flag,
                        progress_callback=update_progress,
                    )
                    if p_stats.is_success:
                        st.success(
                            f"🎉 Successfully indexed **{p_stats.newly_indexed_chunks} chunks** in "
                            f"**{p_stats.duration_ms:.1f}ms** (model: `{p_stats.model}`, dims: {p_stats.dimensions})."
                        )
                        st.rerun()
                    else:
                        st.error(
                            f"❌ Indexing encountered {p_stats.failed_chunks} errors: {', '.join(p_stats.errors)}"
                        )
                except Exception as exc:
                    st.error(f"❌ Indexing pipeline error: {exc}")
                    logger.exception("Indexing pipeline error")

        with idx_col2:
            st.markdown(
                """
                <div style="background-color: #10192e; border: 1px solid #1e293b; border-radius: 8px; padding: 14px; font-size: 0.88rem;">
                    <div style="font-weight: 600; color: #38bdf8; margin-bottom: 6px;">ℹ️ Incremental Indexing Engine</div>
                    <ul style="margin: 0; padding-left: 18px; color: #94a3b8; line-height: 1.6;">
                        <li>Chunks are cached and verified against ChromaDB by deterministic <code>chunk_id</code>.</li>
                        <li>Only new or unindexed chunks make network requests to Ollama <code>/api/embed</code>.</li>
                        <li>Canonical text and metadata are never altered during indexing.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # SECTION C: Semantic Search Interface
        st.markdown("### C. Semantic Search Retrieval")
        search_query = st.text_input(
            "Natural Language Search Query",
            placeholder="e.g. How do knowledge graphs enhance retrieval in RAG systems?",
            key="input_semantic_search_query",
        )

        filter_c1, filter_c2, filter_c3, filter_c4 = st.columns([1.0, 1.2, 1.0, 0.8])
        with filter_c1:
            top_k_val = st.slider(
                "Top-K Chunks",
                min_value=1,
                max_value=20,
                value=settings.retrieval_top_k,
                step=1,
            )
        with filter_c2:
            all_ingested = repo.list_documents()
            doc_options = ["All Documents"] + [
                f"{d.filename} ({d.document_id[:12]})" for d in all_ingested
            ]
            selected_doc_opt = st.selectbox("Filter by Document", doc_options)
            filter_doc_id = None
            if selected_doc_opt != "All Documents":
                match_doc = next(
                    (
                        d
                        for d in all_ingested
                        if d.filename in selected_doc_opt and d.document_id[:12] in selected_doc_opt
                    ),
                    None,
                )
                if match_doc:
                    filter_doc_id = match_doc.document_id
        with filter_c3:
            file_type_opts = ["All Types", "PDF", "DOCX", "TXT", "MD"]
            selected_type_opt = st.selectbox("Filter by Format", file_type_opts)
            filter_file_type = (
                None if selected_type_opt == "All Types" else selected_type_opt.lower()
            )

        with filter_c4:
            st.write("")
            st.write("")
            execute_search = st.button("🔍 Search", type="primary", use_container_width=True)

        if execute_search and search_query:
            if not server_live:
                st.error("Cannot perform semantic search: Ollama server is offline.")
            elif stat["indexed_vector_count"] == 0:
                st.warning(
                    "ChromaDB vector store is empty. Please index your documents in Section B before searching."
                )
            else:
                with st.spinner("Searching semantic vectors in ChromaDB..."):
                    try:
                        s_res = search_engine.search(
                            query=search_query,
                            top_k=top_k_val,
                            document_id=filter_doc_id,
                            file_type=filter_file_type,
                        )
                        st.session_state["semantic_search_response"] = s_res
                    except Exception as exc:
                        logger.exception("Semantic search failed")
                        st.error(f"Search failed: {exc}")

        # Display Search Results
        if st.session_state.get("semantic_search_response"):
            search_resp: SearchResponse = st.session_state["semantic_search_response"]
            st.markdown(
                f"""
                <div style="background-color: #101c36; border: 1px solid #1e3a8a; border-radius: 8px; padding: 10px 16px; margin: 12px 0 16px 0;">
                    <span style="color: #60a5fa; font-weight: 600;">Results for:</span>
                    <span style="color: #ffffff; font-style: italic;">"{search_resp.query}"</span> &bull;
                    <span style="color: #38bdf8;">Retrieved: <strong>{search_resp.total_found}</strong> chunk(s)</span> &bull;
                    <span style="color: #94a3b8;">Latency: <strong>{search_resp.duration_ms:.1f}ms</strong></span> &bull;
                    <span style="color: #94a3b8;">Model: <code>{search_resp.model_used}</code></span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if not search_resp.items:
                st.info("No matching chunks found for the given query and filters.")
            else:
                for item in search_resp.items:
                    norm_score_pct = item.normalized_score * 100
                    badge_color = (
                        "#10b981"
                        if item.cosine_similarity >= 0.7
                        else ("#3b82f6" if item.cosine_similarity >= 0.4 else "#f59e0b")
                    )
                    src_file = item.metadata.get("source_filename", "Unknown")
                    page_num = item.metadata.get("page_number", "N/A")
                    f_type = item.metadata.get("file_type", "").upper()
                    w_cnt = item.metadata.get("word_count", "N/A")
                    start_ch = item.metadata.get("start_char", 0)
                    end_ch = item.metadata.get("end_char", 0)

                    with st.container():
                        st.markdown(
                            f"""
                            <div style="background-color: #0d1527; border: 1px solid #1e293b; border-left: 4px solid {badge_color}; border-radius: 8px; padding: 14px 18px; margin-bottom: 12px;">
                                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px; flex-wrap: wrap; gap: 8px;">
                                    <div>
                                        <span style="font-weight: 700; color: #38bdf8; font-size: 1.05rem;">Rank #{item.rank}</span>
                                        <span style="margin-left: 10px; color: #e2e8f0; font-weight: 600;">📄 {src_file}</span>
                                        <span style="margin-left: 6px; font-size: 0.8rem; color: #64748b;">({f_type})</span>
                                        <span style="margin-left: 8px; font-size: 0.85rem; color: #94a3b8;">Page: {page_num}</span>
                                    </div>
                                    <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                                        <span style="background-color: rgba(56, 189, 248, 0.12); color: #38bdf8; border: 1px solid #0284c7; padding: 2px 8px; border-radius: 6px; font-size: 0.78rem; font-family: monospace;">
                                            Raw Cosine Sim: <strong>{item.cosine_similarity:+.4f}</strong>
                                        </span>
                                        <span style="background-color: rgba(16, 185, 129, 0.12); color: #10b981; border: 1px solid #10b981; padding: 2px 8px; border-radius: 6px; font-size: 0.78rem; font-family: monospace;">
                                            Norm Score: <strong>{item.normalized_score:.4f}</strong> ({norm_score_pct:.1f}%)
                                        </span>
                                        <span style="background-color: rgba(100, 116, 139, 0.15); color: #94a3b8; border: 1px solid #475569; padding: 2px 8px; border-radius: 6px; font-size: 0.78rem; font-family: monospace;">
                                            Distance: <strong>{item.distance:.4f}</strong>
                                        </span>
                                    </div>
                                </div>
                                <div class="chunk-box" style="margin-top: 8px; max-height: 220px;">{item.text}</div>
                                <div style="margin-top: 8px; font-size: 0.78rem; color: #64748b; display: flex; flex-wrap: wrap; gap: 12px;">
                                    <span><strong>Document ID:</strong> <code>{item.document_id}</code></span>
                                    <span><strong>Chunk ID:</strong> <code>{item.chunk_id}</code></span>
                                    <span><strong>Words:</strong> {w_cnt}</span>
                                    <span><strong>Offsets:</strong> {start_ch}..{end_ch}</span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

        st.markdown("---")

        # SECTION D: Vector Database Management
        st.markdown("### D. Vector Database Management")
        with st.expander("⚠️ Danger Zone: Clear ChromaDB Vector Index"):
            st.markdown(
                """
                **Caution:** Clearing the vector index deletes all vectors stored in ChromaDB.
                - Phase 2 canonical data (`data/uploads/` and `data/processed/`) **will NOT be touched**.
                - You can re-index all chunks at any time using Section B above.
                """
            )
            confirm_clear = st.checkbox(
                "I understand and confirm that this will erase all indexed vectors from ChromaDB."
            )
            if st.button("🗑️ Erase Vector Index", type="secondary", disabled=not confirm_clear):
                try:
                    vector_store.reset_collection()
                    st.success("ChromaDB vector collection has been reset successfully.")
                    if "semantic_search_response" in st.session_state:
                        st.session_state["semantic_search_response"] = None
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to reset collection: {exc}")

    # =========================================================================
    # TAB 1: DOCUMENT INGESTION & PREPROCESSING ENGINE (PHASE 2)
    # =========================================================================
    with tab_ingest:
        # SECTION A & B: Multi-File Upload & Chunking Configuration
        st.markdown("### A. Document Upload & Configuration")
        cfg_col1, cfg_col2 = st.columns([1.2, 1.0])

        with cfg_col1:
            uploaded_files = st.file_uploader(
                "Upload Documents (PDF, DOCX, TXT, MD)",
                type=["pdf", "docx", "txt", "md"],
                accept_multiple_files=True,
                help=f"Supported formats: PDF, DOCX, TXT, MD. Maximum size per file: {settings.max_upload_size_mb} MB.",
            )
            if uploaded_files:
                st.caption(f"Selected **{len(uploaded_files)}** file(s) for ingestion.")

        with cfg_col2:
            st.markdown("**Chunking Configuration (Sliding-Window Word Chunker)**")
            chunk_size = st.slider(
                "Chunk Size (words)",
                min_value=50,
                max_value=2000,
                value=settings.default_chunk_size,
                step=25,
                help="Target word count per chunk. Recommended: 300 - 500 words.",
            )
            chunk_overlap = st.slider(
                "Chunk Overlap (words)",
                min_value=0,
                max_value=min(chunk_size - 1, 500),
                value=min(settings.default_chunk_overlap, chunk_size - 1),
                step=10,
                help="Word overlap between consecutive chunks to maintain semantic continuity.",
            )
            stride = chunk_size - chunk_overlap
            st.caption(
                f"ℹ️ Effective stride: **{stride} words** per chunk window. "
                f"Overlap ratio: **{(chunk_overlap / chunk_size) * 100:.1f}%**"
            )

        # SECTION C: Ingestion Execution
        st.markdown("---")
        st.markdown("### B. Ingestion Execution")
        can_ingest = has_permission(current_user.role, Permission.DOCUMENTS_UPLOAD)
        if not can_ingest:
            st.info("🔒 Document ingestion is restricted to RESEARCHER and ADMIN roles.")
        proc_col1, _proc_col2 = st.columns([1, 3])
        with proc_col1:
            process_btn = st.button(
                "🚀 Process & Ingest Documents",
                type="primary",
                disabled=not uploaded_files or not can_ingest,
                use_container_width=True,
            )

        if process_btn and uploaded_files:
            results = []
            progress_bar = st.progress(0, text="Starting document ingestion...")
            total_files = len(uploaded_files)

            for idx, uploaded_file in enumerate(uploaded_files):
                file_bytes = uploaded_file.getvalue()
                filename = uploaded_file.name
                progress_bar.progress(
                    (idx) / total_files,
                    text=f"Processing {filename} ({idx + 1}/{total_files})...",
                )

                res = processor.process_file(
                    file_content=file_bytes,
                    filename=filename,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                results.append(res)

            progress_bar.progress(1.0, text="Document ingestion batch complete!")
            st.session_state["ingestion_batch_results"] = results
            st.rerun()

        # SECTION D: Ingestion Results
        batch_results = st.session_state["ingestion_batch_results"]
        if batch_results:
            st.markdown("### C. Batch Processing Summary")
            total_batch_docs = len(batch_results)
            new_docs = sum(1 for r in batch_results if not r.duplicate_detected and r.success)
            dup_docs = sum(1 for r in batch_results if r.duplicate_detected)
            total_batch_chunks = sum(len(r.chunks) for r in batch_results)
            failed_docs = sum(1 for r in batch_results if not r.success)

            b_c1, b_c2, b_c3, b_c4 = st.columns(4)
            with b_c1:
                st.markdown(
                    f"""
                    <div class="overview-card">
                        <div class="overview-card-label">Files Processed</div>
                        <div class="overview-card-value">{total_batch_docs}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with b_c2:
                st.markdown(
                    f"""
                    <div class="overview-card">
                        <div class="overview-card-label">New Documents</div>
                        <div class="overview-card-value" style="color:#10b981;">{new_docs}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with b_c3:
                st.markdown(
                    f"""
                    <div class="overview-card">
                        <div class="overview-card-label">Duplicates Skipped</div>
                        <div class="overview-card-value" style="color:#f59e0b;">{dup_docs}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with b_c4:
                st.markdown(
                    f"""
                    <div class="overview-card">
                        <div class="overview-card-label">Total Chunks Created</div>
                        <div class="overview-card-value" style="color:#38bdf8;">{total_batch_chunks}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if failed_docs > 0:
                st.error(f"⚠️ {failed_docs} document(s) failed validation or extraction.")

            # Detailed table of batch results
            st.markdown("**Batch Items:**")
            for r in batch_results:
                badge = (
                    '<span class="status-badge-warning">DUPLICATE</span>'
                    if r.duplicate_detected
                    else (
                        '<span class="status-badge-pass">INGESTED</span>'
                        if r.success
                        else '<span class="status-badge-fail">FAILED</span>'
                    )
                )
                meta = r.metadata
                with st.expander(
                    f"{meta.filename} — {meta.file_format.upper()} ({meta.total_chunks} chunks, {meta.total_words} words)",
                    expanded=False,
                ):
                    st.markdown(f"**Status:** {badge}", unsafe_allow_html=True)
                    st.write(f"**Document ID:** `{r.document_id}`")
                    st.write(f"**Source File:** `{meta.filename}` ({meta.file_size_bytes} bytes)")
                    st.write(f"**SHA256:** `{meta.sha256}`")
                    if meta.page_count:
                        st.write(f"**Pages:** {meta.page_count}")
                    if r.warnings:
                        st.warning(f"Warnings: {', '.join(r.warnings)}")
                    if r.errors:
                        st.error(f"Errors: {', '.join(r.errors)}")

        # SECTION E: Chunk Explorer
        st.markdown("---")
        st.markdown("### D. Chunk Explorer")
        all_docs = repo.list_documents()

        if not all_docs:
            st.info("No documents have been ingested yet. Upload and process documents above.")
        else:
            doc_options = {
                f"{d.filename} (ID: {d.doc_id[:12]}..., {d.total_chunks} chunks)": d.doc_id
                for d in all_docs
            }
            selected_label = st.selectbox("Select Document to Inspect:", list(doc_options.keys()))
            selected_doc_id = doc_options[selected_label]
            doc_chunks = repo.get_chunks(selected_doc_id)

            if not doc_chunks:
                st.warning("No chunks found for this document.")
            else:
                st.caption(f"Document contains **{len(doc_chunks)}** chunk(s).")
                if len(doc_chunks) <= 1:
                    chunk_idx = 0
                else:
                    chunk_idx = st.slider(
                        "Select Chunk Index",
                        min_value=0,
                        max_value=len(doc_chunks) - 1,
                        value=0,
                        step=1,
                        format="Chunk %d",
                    )
                chunk = doc_chunks[chunk_idx]

                c_info1, c_info2, c_info3, c_info4 = st.columns(4)
                with c_info1:
                    st.metric("Chunk Index", f"{chunk.chunk_index + 1} of {len(doc_chunks)}")
                with c_info2:
                    st.metric("Word Count", chunk.word_count)
                with c_info3:
                    st.metric("Character Span", f"{chunk.start_char} - {chunk.end_char}")
                with c_info4:
                    page_disp = str(chunk.page_number) if chunk.page_number is not None else "N/A"
                    st.metric("Source Page", page_disp)

                st.markdown(f"**Chunk ID:** `{chunk.chunk_id}`")
                st.markdown("**Chunk Content:**")
                st.markdown(f'<div class="chunk-box">{chunk.text}</div>', unsafe_allow_html=True)

        # SECTION F: Document Library
        st.markdown("---")
        st.markdown("### E. Document Library")
        library_docs = repo.list_documents()

        if not library_docs:
            st.caption("Library is currently empty.")
        else:
            st.write(f"Total documents registered: **{len(library_docs)}**")
            for doc in library_docs:
                col_lib1, col_lib2 = st.columns([4, 1])
                with (
                    col_lib1,
                    st.expander(
                        f"📄 {doc.filename} — {doc.file_format.upper()} | {doc.total_words} words | {doc.total_chunks} chunks"
                    ),
                ):
                    st.json(doc.model_dump())
                with col_lib2:
                    can_del = has_permission(current_user.role, Permission.DOCUMENTS_DELETE)
                    del_key = f"del_confirm_{doc.doc_id}"
                    confirm = st.checkbox("Confirm", key=del_key, disabled=not can_del)
                    if st.button(
                        "🗑️ Delete", key=f"btn_del_{doc.doc_id}", disabled=not confirm or not can_del
                    ):
                        if repo.delete_document(doc.doc_id):
                            user_audit.log_event(
                                user_id=current_user.user_id,
                                email=current_user.email,
                                role=current_user.role.value,
                                action="DOCUMENT_DELETION",
                                status="SUCCESS",
                                details={"doc_id": doc.doc_id, "filename": doc.filename},
                            )
                            st.success(f"Deleted document `{doc.doc_id}`")
                            st.rerun()
                        else:
                            st.error("Failed to delete document.")

        # SECTION G: Export
        st.markdown("---")
        st.markdown("### F. Export Processed Documents & Chunks")
        if not library_docs:
            st.caption("No processed documents available for export.")
        else:
            exp_col1, exp_col2 = st.columns([1.5, 2.5])
            with exp_col1:
                export_doc_choice = st.selectbox(
                    "Scope:",
                    options=["ALL"] + [d.doc_id for d in library_docs],
                    format_func=lambda x: (
                        "All Documents"
                        if x == "ALL"
                        else next(
                            (
                                f"{d.filename} ({d.doc_id[:8]}...)"
                                for d in library_docs
                                if d.doc_id == x
                            ),
                            x,
                        )
                    ),
                )
                target_doc_id = None if export_doc_choice == "ALL" else export_doc_choice

            with exp_col2:
                st.write("**Download Formats:**")
                b_c1, b_c2, b_c3 = st.columns(3)
                with b_c1:
                    json_data = repo.export_chunks(target_doc_id, format="json")
                    st.download_button(
                        label="⬇️ Export JSON",
                        data=json_data,
                        file_name=f"chunks_export_{export_doc_choice.lower()}.json",
                        mime="application/json",
                        use_container_width=True,
                    )
                with b_c2:
                    jsonl_data = repo.export_chunks(target_doc_id, format="jsonl")
                    st.download_button(
                        label="⬇️ Export JSONL",
                        data=jsonl_data,
                        file_name=f"chunks_export_{export_doc_choice.lower()}.jsonl",
                        mime="application/x-ndjson",
                        use_container_width=True,
                    )
                with b_c3:
                    csv_data = repo.export_chunks(target_doc_id, format="csv")
                    st.download_button(
                        label="⬇️ Export CSV",
                        data=csv_data,
                        file_name=f"chunks_export_{export_doc_choice.lower()}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

    # =========================================================================
    # TAB 2: SYSTEM HEALTH & FOUNDATION DIAGNOSTICS (PHASE 1)
    # =========================================================================
    with tab_health:
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
                    <div class="overview-card-label">Python Runtime</div>
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
                    <div class="overview-card-value" style="font-size:1.1rem; color:#38bdf8;">{venv_status}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                """
                <div class="overview-card">
                    <div class="overview-card-label">Application Status</div>
                    <div class="overview-card-value" style="color:#10b981;">OPERATIONAL</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # SECTION B: OLLAMA INTEGRATION STATUS
        st.markdown("---")
        st.markdown("### B. Ollama Integration")
        ollama_col1, ollama_col2 = st.columns(2)

        with ollama_col1:
            st.markdown("**Server Connection**")
            if server_live:
                st.markdown(
                    f"""
                    <div style="background-color:#0d1f18; border:1px solid #059669; padding:12px; border-radius:8px;">
                        <span style="color:#10b981; font-weight:700;">CONNECTED</span> &mdash; Ollama is responding normally at <code>{settings.ollama_base_url}</code>.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div style="background-color:#2a0f12; border:1px solid #dc2626; padding:12px; border-radius:8px;">
                        <span style="color:#ef4444; font-weight:700;">UNREACHABLE</span> &mdash; Ollama is offline at <code>{settings.ollama_base_url}</code>.<br/>
                        <small style="color:#94a3b8;">Start Ollama with <code>ollama serve</code> in a terminal.</small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.write(f"**Configured LLM Model:** `{settings.ollama_llm_model}`")
            st.write(f"**Configured Embedding Model:** `{settings.ollama_embed_model}`")

        with ollama_col2:
            st.markdown("**Installed Models on Ollama Server**")
            if server_live:
                try:
                    installed = client.list_installed_models()
                    if installed:
                        pills_html = "".join(
                            f'<span class="model-pill">{m}</span>' for m in installed
                        )
                        st.markdown(f"<div>{pills_html}</div>", unsafe_allow_html=True)

                        has_llm = any(
                            settings.ollama_llm_model.lower() in m.lower() for m in installed
                        )
                        has_embed = any(
                            settings.ollama_embed_model.lower() in m.lower() for m in installed
                        )

                        if not has_llm:
                            st.warning(
                                f"⚠️ LLM `{settings.ollama_llm_model}` is not installed. Run `ollama pull {settings.ollama_llm_model}`"
                            )
                        if not has_embed:
                            st.warning(
                                f"⚠️ Embed model `{settings.ollama_embed_model}` is not installed. Run `ollama pull {settings.ollama_embed_model}`"
                            )
                    else:
                        st.info("No models currently installed. Pull models via `ollama pull`.")
                except Exception as exc:
                    st.error(f"Failed to query installed models: {exc}")
            else:
                st.caption("Connect Ollama to view installed model tags.")

        # SECTION C: SYSTEM HEALTH
        st.markdown("---")
        st.markdown("### C. System Health")
        report = checker.run_all_checks()

        for chk in report.checks:
            badge = (
                '<span class="status-badge-pass">PASS</span>'
                if chk.status == HealthStatus.PASS
                else (
                    '<span class="status-badge-warning">WARNING</span>'
                    if chk.status == HealthStatus.WARNING
                    else '<span class="status-badge-fail">FAIL</span>'
                )
            )

            with st.container():
                h_col1, h_col2 = st.columns([1, 4])
                with h_col1:
                    st.markdown(f"**{chk.name}**<br/>{badge}", unsafe_allow_html=True)
                with h_col2:
                    st.write(chk.message)
                    if chk.details and chk.status != HealthStatus.PASS:
                        with st.expander("Diagnostic Details"):
                            st.json(chk.details)
                st.markdown(
                    "<hr style='margin: 8px 0; border: none; border-top: 1px solid #1e293b;'/>",
                    unsafe_allow_html=True,
                )

        # SECTION D: MODEL TESTING & DIAGNOSTICS
        st.markdown("### D. Model Testing & Diagnostics")
        test_col1, test_col2 = st.columns(2)

        with test_col1:
            st.markdown(
                f"""
                <div class="section-card">
                    <h4 style="margin-top:0; color:#38bdf8;">🧠 Qwen Generation Test</h4>
                    <p style="color:#94a3b8; font-size:0.9rem;">Target Model: <code>{settings.ollama_llm_model}</code></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            qwen_prompt = st.text_area(
                "Prompt",
                value="Briefly explain the primary architectural advantage of Graph RAG over Vector RAG in two sentences.",
                key="input_qwen_prompt",
                height=100,
            )

            if st.button("⚡ Test Qwen Generation", type="secondary", use_container_width=True):
                if not server_live:
                    st.error("Cannot execute test: Ollama server is offline.")
                else:
                    with st.spinner(f"Querying {settings.ollama_llm_model}..."):
                        try:
                            resp = client.generate(qwen_prompt)
                            st.session_state["qwen_output"] = {
                                "response": resp.response,
                                "duration_ns": resp.total_duration,
                                "tokens": resp.eval_count,
                                "model": resp.model,
                            }
                        except OllamaError as exc:
                            logger.exception("Qwen generation test error")
                            st.error(f"Generation failed: {exc}")

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
                <div class="section-card">
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

            if st.button(
                "⚡ Test Embedding Generation", type="secondary", use_container_width=True
            ):
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

    # =========================================================================
    # TAB 5: GRAPH RETRIEVAL LAB (PHASE 5)
    # =========================================================================
    with tab_retrieval:
        st.markdown("## 🧭 Graph Retrieval & Multi-Hop Reasoning Lab")
        st.caption(
            "Explore multi-hop entity traversal, evidence chain extraction, and deterministic path ranking "
            "over the validated Knowledge Graph."
        )

        # ---------------------------------------------------------------------
        # SECTION A: GRAPH QUERY INTERFACE
        # ---------------------------------------------------------------------
        st.markdown("### A. Graph Query Interface")

        q_col1, q_col2 = st.columns([3, 1])
        with q_col1:
            graph_query = st.text_input(
                "Natural-Language Query",
                value="How does Graph RAG perform compared to Vector RAG on multi-hop questions?",
                key="graph_query_input",
                placeholder="Enter query or entity to explore (e.g. Graph RAG vs Vector RAG)...",
            )
        with q_col2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            search_btn = st.button(
                "🧭 Search Knowledge Graph",
                type="primary",
                use_container_width=True,
                key="btn_search_graph",
            )

        with st.expander("⚙️ Advanced Retrieval Parameters", expanded=True):
            cfg_col1, cfg_col2, cfg_col3 = st.columns(3)
            with cfg_col1:
                hops_choice = st.radio(
                    "Maximum Hops",
                    options=[1, 2, 3],
                    index=1,
                    horizontal=True,
                    help="Maximum relationship edges to traverse from seed entities (1=Immediate neighbors, 2=Two-hop reasoning, 3=Three-hop bridge).",
                    key="graph_max_hops_input",
                )
            with cfg_col2:
                dir_label_map = {
                    "both": "Bidirectional (Both)",
                    "out": "Outgoing (Subject → Object)",
                    "in": "Incoming (Object ← Subject)",
                }
                dir_choice = st.selectbox(
                    "Traversal Direction",
                    options=["both", "out", "in"],
                    format_func=lambda x: dir_label_map.get(x, x),
                    index=0,
                    help="Direction of relationship edges to traverse.",
                    key="graph_dir_input",
                )
            with cfg_col3:
                top_k_paths = st.slider(
                    "Top-K Ranked Paths",
                    min_value=1,
                    max_value=25,
                    value=10,
                    help="Maximum number of ranked paths to return.",
                    key="graph_top_k_input",
                )

            flt_col1, flt_col2 = st.columns(2)
            with flt_col1:
                all_preds = {r.predicate for r in graph_store.get_relationships()}
                selected_predicates = st.multiselect(
                    "Filter by Predicates (Optional)",
                    options=sorted(all_preds),
                    default=None,
                    placeholder="Select predicates to restrict traversal (leave empty for all)",
                    key="graph_predicate_filter_input",
                )
            with flt_col2:
                all_doc_ids = set()
                for rel in graph_store.get_relationships():
                    if rel.source_document_id:
                        all_doc_ids.add(rel.source_document_id)
                doc_options = ["None (All Documents)"] + sorted(all_doc_ids)
                selected_doc = st.selectbox(
                    "Filter by Source Document (Optional)",
                    options=doc_options,
                    index=0,
                    help="Restrict paths to relationships originating from a specific document.",
                    key="graph_doc_filter_input",
                )

        doc_filter_val = None if selected_doc == "None (All Documents)" else selected_doc
        pred_filter_val = selected_predicates if selected_predicates else None

        # Execute Search Action
        if search_btn:
            if not graph_query.strip():
                st.warning("Please enter a valid query before searching.")
            elif graph_store.node_count == 0:
                st.error(
                    "Knowledge Graph is empty. Ingest documents and build the graph in Tab 1 first."
                )
            else:
                with st.spinner(
                    "Extracting query entities, traversing paths, and ranking evidence..."
                ):
                    resp = graph_search_engine.search_graph(
                        query=graph_query.strip(),
                        max_hops=hops_choice,
                        direction=dir_choice,
                        top_k=top_k_paths,
                        predicates=pred_filter_val,
                        doc_id_filter=doc_filter_val,
                    )
                    st.session_state["graph_search_response"] = resp

        # Render Results from Session State
        cached_resp: GraphSearchResponse | None = st.session_state.get("graph_search_response")

        if cached_resp is not None:
            st.markdown("---")

            # -----------------------------------------------------------------
            # SECTION E: GRAPH STATISTICS / METRICS
            # -----------------------------------------------------------------
            st.markdown("### E. Graph Retrieval Performance & Metrics")
            seed_count = len(cached_resp.linked_entities)
            path_count = len(cached_resp.results)
            configured_hops = cached_resp.statistics.get("max_hops_configured", hops_choice)
            evid_count = cached_resp.statistics.get(
                "evidence_quotes_count",
                len(cached_resp.context_evidence.evidence_quotes)
                if cached_resp.context_evidence
                else 0,
            )

            m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
            with m_col1:
                st.markdown(
                    f"""
                    <div class="overview-card">
                        <div class="overview-card-label">Seed Entities</div>
                        <div class="overview-card-value">{seed_count}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m_col2:
                st.markdown(
                    f"""
                    <div class="overview-card">
                        <div class="overview-card-label">Paths Retrieved</div>
                        <div class="overview-card-value">{path_count}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m_col3:
                st.markdown(
                    f"""
                    <div class="overview-card">
                        <div class="overview-card-label">Max Traversal Hops</div>
                        <div class="overview-card-value">{configured_hops}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m_col4:
                st.markdown(
                    f"""
                    <div class="overview-card">
                        <div class="overview-card-label">Total Latency</div>
                        <div class="overview-card-value">{cached_resp.latency_ms:.1f}ms</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m_col5:
                st.markdown(
                    f"""
                    <div class="overview-card">
                        <div class="overview-card-label">Evidence Records</div>
                        <div class="overview-card-value">{evid_count}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Status Banner
            if cached_resp.status == GraphSearchStatus.SUCCESS:
                st.success(
                    f"✅ Graph retrieval completed successfully: found {path_count} "
                    f"reasoning path(s) across {seed_count} seed entity(ies) in {cached_resp.latency_ms:.1f}ms."
                )
            elif cached_resp.status == GraphSearchStatus.NO_MATCHING_ENTITIES:
                st.warning(
                    f"⚠️ No matching entities found in the Knowledge Graph for: '{cached_resp.query}'. "
                    "Try using canonical names (e.g., 'Graph RAG', 'Vector RAG', 'Hybrid RAG')."
                )
            elif cached_resp.status == GraphSearchStatus.NO_GRAPH_RESULTS:
                st.info(
                    f"ℹ️ Seed entities were detected ({seed_count}), but no connected paths "
                    f"matched the filters (hops={configured_hops}, predicates={pred_filter_val})."
                )
            elif cached_resp.status == GraphSearchStatus.GRAPH_UNAVAILABLE:
                st.error("⚠️ The Knowledge Graph is empty or unavailable.")
            else:
                st.error(
                    f"⚠️ Search ended with status: {cached_resp.status.value}. Message: {cached_resp.message}"
                )

            # -----------------------------------------------------------------
            # SECTION B: ENTITY MATCHING
            # -----------------------------------------------------------------
            st.markdown("### B. Entity Extraction & Graph Linking")
            b_col1, b_col2 = st.columns(2)

            with b_col1:
                st.markdown("#### 1. Extracted Query Entities")
                if cached_resp.extraction_result and cached_resp.extraction_result.entities:
                    for ent in cached_resp.extraction_result.entities:
                        method_badge = (
                            '<span class="status-badge-pass">Deterministic Scan</span>'
                            if ent.extraction_method != ExtractionMethod.LLM_EXTRACTION
                            else '<span class="status-badge-warning">Qwen Fallback</span>'
                        )
                        st.markdown(
                            f"""
                            <div style="background:#10192e; border:1px solid #1e2d4d; border-radius:8px; padding:10px 14px; margin-bottom:8px;">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <strong style="color:#00e5ff; font-size:1.05rem;">{ent.canonical_name}</strong>
                                    {method_badge}
                                </div>
                                <div style="color:#94a3b8; font-size:0.85rem; margin-top:4px;">
                                    Mention: <code>"{ent.mention_text}"</code> &bull; Type: <code>{ent.entity_type}</code> &bull; Confidence: <code>{ent.confidence:.2f}</code>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("No entities detected in query.")

            with b_col2:
                st.markdown("#### 2. Linked Graph Seed Nodes")
                if cached_resp.linked_entities:
                    for linked in cached_resp.linked_entities:
                        m_type = linked.match_type.value
                        badge_color = "#10b981" if m_type == "exact" else "#38bdf8"
                        st.markdown(
                            f"""
                            <div style="background:#10192e; border:1px solid #1e2d4d; border-radius:8px; padding:10px 14px; margin-bottom:8px;">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <strong style="color:#38bdf8; font-size:1.05rem;">{linked.canonical_name}</strong>
                                    <span style="background:rgba(2,132,199,0.15); border:1px solid {badge_color}; color:{badge_color}; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:700; text-transform:uppercase;">
                                        {m_type} match
                                    </span>
                                </div>
                                <div style="color:#94a3b8; font-size:0.85rem; margin-top:4px;">
                                    Node ID: <code>{linked.entity_id}</code> &bull; Type: <code>{linked.entity_type}</code>
                                </div>
                                <div style="color:#64748b; font-size:0.8rem; margin-top:2px;">
                                    Matched via: {linked.match_explanation}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("No seed entities could be linked to the graph.")

            # -----------------------------------------------------------------
            # SECTION C: RETRIEVED GRAPH PATHS
            # -----------------------------------------------------------------
            st.markdown("### C. Retrieved Reasoning Paths (Multi-Hop Traversal)")
            if cached_resp.results:
                for res in cached_resp.results:
                    path = res.path
                    with st.expander(
                        f"Rank #{res.rank} | Score: {res.score:.3f} | {len(path.edges)}-Hop Path from [{res.seed_entity_name}]",
                        expanded=(res.rank <= 3),
                    ):
                        # Visual Path Chain representation
                        steps_html = []
                        for node in path.nodes:
                            steps_html.append(
                                f'<span style="background:#1e293b; color:#00e5ff; padding:4px 10px; border-radius:6px; font-weight:600; border:1px solid #0284c7;">'
                                f'{node.canonical_name} <span style="color:#94a3b8; font-size:0.75rem;">({node.entity_type})</span></span>'
                            )

                        edge_arrows = []
                        for edge in path.edges:
                            arrow_sym = "→" if edge.direction == "outgoing" else "←"
                            edge_arrows.append(
                                f'<span style="color:#f59e0b; font-family:monospace; font-size:0.85rem; padding:0 6px;">'
                                f"--[{edge.predicate} {arrow_sym}]--</span>"
                            )

                        # Alternate nodes and edges
                        rendered_chain = []
                        for i in range(len(path.nodes)):
                            rendered_chain.append(steps_html[i])
                            if i < len(edge_arrows):
                                rendered_chain.append(edge_arrows[i])

                        st.markdown(
                            f"""
                            <div style="background:#090e17; border:1px solid #1e293b; border-radius:8px; padding:14px; margin-bottom:12px; line-height:2.2;">
                                {"".join(rendered_chain)}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # Score Breakdown and Path Details
                        sc1, sc2, sc3, sc4 = st.columns(4)
                        sc1.metric("Final Path Score", f"{res.score:.3f}")
                        sc2.metric(
                            "Seed Weight", f"{path.score_breakdown.get('seed_score', 1.0):.2f}"
                        )
                        sc3.metric(
                            "Length Discount",
                            f"x{path.score_breakdown.get('length_penalty', 1.0):.2f}",
                        )
                        sc4.metric(
                            "Evidence Grounding",
                            f"{path.score_breakdown.get('evidence_score', 1.0):.2f}",
                        )

                        if res.score_explanation:
                            st.caption(f"**Explanation:** {res.score_explanation}")

                        # Triples Summary in tabular form
                        st.markdown("**Path Triples:**")
                        triples_data = [
                            {
                                "Hop": e.hop_index + 1 if hasattr(e, "hop_index") else h_i + 1,
                                "Subject ID": e.subject_entity_id,
                                "Predicate": e.predicate,
                                "Object ID": e.object_entity_id,
                                "Direction": e.direction,
                                "Confidence": f"{e.confidence:.2f}",
                            }
                            for h_i, e in enumerate(path.edges)
                        ]
                        st.dataframe(triples_data, use_container_width=True)
            else:
                st.info("No reasoning paths to display.")

            # -----------------------------------------------------------------
            # SECTION D: EVIDENCE INSPECTOR
            # -----------------------------------------------------------------
            st.markdown("### D. Verbatim Source Evidence Inspector")
            # Collect all distinct edges with evidence across retrieved results
            collected_edges: list[Any] = []
            seen_rel_ids: set[str] = set()
            for r in cached_resp.results:
                for edge in r.path.edges:
                    if edge.relationship_id not in seen_rel_ids:
                        seen_rel_ids.add(edge.relationship_id)
                        collected_edges.append(edge)

            if collected_edges:
                st.caption(
                    "Every relationship in the retrieved paths is strictly grounded by verbatim text extracts "
                    "from canonical Phase 2 chunks."
                )
                for ev_idx, edge in enumerate(collected_edges, start=1):
                    with st.expander(
                        f"Evidence #{ev_idx}: [{edge.subject_entity_id}] --[{edge.predicate}]--> [{edge.object_entity_id}] (Confidence: {edge.confidence:.2f})",
                        expanded=False,
                    ):
                        st.markdown(
                            f"""
                            <div style="background:#090e17; border-left:4px solid #00e5ff; padding:12px 16px; border-radius:0 8px 8px 0; margin-bottom:10px;">
                                <div style="color:#94a3b8; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:4px;">Verbatim Source Evidence Quote</div>
                                <div style="color:#f8fafc; font-style:italic; font-size:0.95rem; line-height:1.5;">"{edge.evidence_quote}"</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        meta_col1, meta_col2, meta_col3 = st.columns(3)
                        meta_col1.markdown(f"**Source File:** `{edge.source_filename}`")
                        meta_col2.markdown(f"**Chunk ID:** `{edge.source_chunk_id}`")
                        meta_col3.markdown(f"**Relationship ID:** `{edge.relationship_id}`")
                        if edge.source_document_id:
                            st.caption(f"**Document ID:** `{edge.source_document_id}`")
            else:
                st.info("No grounded evidence records available.")

            # Structured Context Preview
            with st.expander("📝 Formatted Graph RAG Prompt Context", expanded=False):
                st.caption(
                    "This structured text block is generated for injection into downstream LLM synthesis:"
                )
                st.code(cached_resp.structured_context, language="markdown")

            # -----------------------------------------------------------------
            # SECTION F: EXPORT
            # -----------------------------------------------------------------
            st.markdown("### F. Export Retrieval Results")
            export_payload = cached_resp.to_export_dict()
            export_json = json.dumps(export_payload, indent=2)

            st.download_button(
                label="📥 Download Retrieval Results (JSON)",
                data=export_json,
                file_name=f"graph_retrieval_{int(time.time())}.json",
                mime="application/json",
                use_container_width=True,
                key="btn_download_retrieval_json",
            )

    # =========================================================================
    # TAB 6: HYBRID RETRIEVAL LAB (PHASE 6)
    # =========================================================================
    with tab_hybrid:
        st.markdown("### ⚖️ Hybrid Retrieval & Reciprocal Rank Fusion Lab")
        st.caption(
            "Combines Dense Vector Semantic Search (ChromaDB) and Multi-Hop Knowledge Graph Retrieval "
            "(NetworkX) into a unified reciprocal-rank-fused evidence stream with full provenance."
        )

        # -----------------------------------------------------------------
        # SECTION A: HYBRID QUERY INTERFACE
        # -----------------------------------------------------------------
        st.markdown("#### A. Query & Fusion Configuration")

        col_q1, col_q2 = st.columns([4, 1])
        with col_q1:
            hybrid_query = st.text_input(
                "Enter Research / Benchmark Query:",
                value="What technologies are used in Graph RAG?",
                key="hybrid_search_query_input",
                placeholder="e.g. How does Hybrid RAG combine vector and graph retrieval?",
            )
        with col_q2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            run_hybrid_search = st.button(
                "⚖️ Execute Hybrid Search",
                type="primary",
                use_container_width=True,
                key="btn_run_hybrid_search",
            )

        with st.expander("⚙️ Advanced Retrieval, RRF & Filtering Parameters", expanded=False):
            p_col1, p_col2, p_col3 = st.columns(3)
            with p_col1:
                st.markdown("**Modality Top-K Limits**")
                v_top_k = st.slider(
                    "Vector Top-K:",
                    min_value=1,
                    max_value=20,
                    value=settings.hybrid_vector_top_k,
                    key="hybrid_v_top_k",
                )
                g_top_k = st.slider(
                    "Graph Top-K:",
                    min_value=1,
                    max_value=20,
                    value=settings.hybrid_graph_top_k,
                    key="hybrid_g_top_k",
                )
                f_top_k = st.slider(
                    "Final Fused Top-K:",
                    min_value=1,
                    max_value=20,
                    value=settings.hybrid_top_k,
                    key="hybrid_f_top_k",
                )
            with p_col2:
                st.markdown("**Reciprocal Rank Fusion Weights**")
                rrf_k_param = st.slider(
                    "RRF Constant (k):",
                    min_value=1,
                    max_value=100,
                    value=settings.hybrid_rrf_k,
                    help="Smoothing constant added to denominator. Standard default is 60.",
                    key="hybrid_rrf_k_param",
                )
                v_weight = st.slider(
                    "Vector Weight (w_v):",
                    min_value=0.0,
                    max_value=3.0,
                    value=float(settings.hybrid_vector_weight),
                    step=0.1,
                    key="hybrid_v_weight",
                )
                g_weight = st.slider(
                    "Graph Weight (w_g):",
                    min_value=0.0,
                    max_value=3.0,
                    value=float(settings.hybrid_graph_weight),
                    step=0.1,
                    key="hybrid_g_weight",
                )
            with p_col3:
                st.markdown("**Graph Traversal & Metadata Filters**")
                hybrid_hops = st.slider(
                    "Max Traversal Hops:",
                    min_value=1,
                    max_value=3,
                    value=settings.graph_max_hops,
                    key="hybrid_hops",
                )
                hybrid_dir = st.selectbox(
                    "Edge Direction:",
                    options=["BOTH", "OUTGOING", "INCOMING"],
                    index=0,
                    key="hybrid_dir",
                )
                all_docs = repo.list_documents()
                doc_options = [d.document_id for d in all_docs]
                doc_filter_selected = st.multiselect(
                    "Filter by Document ID:",
                    options=doc_options,
                    default=[],
                    key="hybrid_doc_filter",
                )
                file_type_selected = st.multiselect(
                    "Filter by File Type:",
                    options=["md", "txt", "pdf", "docx"],
                    default=[],
                    key="hybrid_type_filter",
                )

        if run_hybrid_search:
            if not hybrid_query.strip():
                st.warning("Please enter a non-empty query string.")
            elif v_weight <= 0.0 and g_weight <= 0.0:
                st.error(
                    "At least one modality weight (Vector Weight or Graph Weight) must be strictly greater than 0."
                )
            else:
                with st.spinner("Executing Parallel Vector & Graph Retrieval with RRF Fusion..."):
                    h_resp = hybrid_retriever.search(
                        query=hybrid_query.strip(),
                        vector_top_k=v_top_k,
                        graph_top_k=g_top_k,
                        final_top_k=f_top_k,
                        rrf_k=rrf_k_param,
                        vector_weight=v_weight,
                        graph_weight=g_weight,
                        max_hops=hybrid_hops,
                        direction=hybrid_dir,
                        document_ids=doc_filter_selected if doc_filter_selected else None,
                        file_types=file_type_selected if file_type_selected else None,
                    )
                    st.session_state["hybrid_search_response"] = h_resp

        cached_hybrid: HybridSearchResponse | None = st.session_state.get("hybrid_search_response")

        if cached_hybrid is not None:
            st.markdown("---")

            # -----------------------------------------------------------------
            # SECTION B: RETRIEVAL & FUSION STATUS
            # -----------------------------------------------------------------
            st.markdown("#### B. Retrieval & Fusion Status")

            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            with stat_col1:
                v_badge = (
                    "status-badge-pass"
                    if cached_hybrid.vector_status == "SUCCESS"
                    else "status-badge-fail"
                )
                st.markdown(
                    f"""
                    <div class="overview-card">
                        <div class="overview-card-label">Vector Engine Status</div>
                        <div style="margin-top:4px;"><span class="{v_badge}">{cached_hybrid.vector_status}</span></div>
                        <div style="font-size:0.85rem; color:#94a3b8; margin-top:6px;">Candidates: <b>{cached_hybrid.vector_candidates_count}</b></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with stat_col2:
                g_badge = (
                    "status-badge-pass"
                    if cached_hybrid.graph_status == "SUCCESS"
                    else "status-badge-fail"
                )
                st.markdown(
                    f"""
                    <div class="overview-card">
                        <div class="overview-card-label">Graph Engine Status</div>
                        <div style="margin-top:4px;"><span class="{g_badge}">{cached_hybrid.graph_status}</span></div>
                        <div style="font-size:0.85rem; color:#94a3b8; margin-top:6px;">Candidates: <b>{cached_hybrid.graph_candidates_count}</b></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with stat_col3:
                h_badge = (
                    "status-badge-pass"
                    if cached_hybrid.status == HybridSearchStatus.SUCCESS
                    else (
                        "status-badge-warning"
                        if cached_hybrid.status == HybridSearchStatus.PARTIAL_SUCCESS
                        else "status-badge-fail"
                    )
                )
                st.markdown(
                    f"""
                    <div class="overview-card">
                        <div class="overview-card-label">Overall Fusion Status</div>
                        <div style="margin-top:4px;"><span class="{h_badge}">{cached_hybrid.status.value}</span></div>
                        <div style="font-size:0.85rem; color:#94a3b8; margin-top:6px;">Fused Candidates: <b>{cached_hybrid.total_candidates}</b></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with stat_col4:
                st.markdown(
                    f"""
                    <div class="overview-card">
                        <div class="overview-card-label">Total Execution Latency</div>
                        <div class="overview-card-value">{cached_hybrid.total_latency_ms:.1f} <span style="font-size:0.9rem; font-weight:normal; color:#94a3b8;">ms</span></div>
                        <div style="font-size:0.85rem; color:#94a3b8; margin-top:6px;">Fusion: <b>{cached_hybrid.fusion_latency_ms:.2f} ms</b></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Warnings / Notices
            if cached_hybrid.warnings:
                for w in cached_hybrid.warnings:
                    st.warning(f"⚠️ {w}")

            # -----------------------------------------------------------------
            # SECTION C: RANKED FUSION RESULTS
            # -----------------------------------------------------------------
            st.markdown("#### C. Reciprocal Rank Fusion Results")
            st.caption(
                "Final fused ranking combining vector semantic proximity and multi-hop graph connectivity. "
                "Scores reflect RRF reciprocal rank summation, not uncalibrated metric probabilities."
            )

            if cached_hybrid.fused_candidates:
                # Prepare tabular view
                table_rows = []
                for fc in cached_hybrid.fused_candidates:
                    if fc.source == RetrievalSource.VECTOR_AND_GRAPH:
                        src_str = "VECTOR + GRAPH"
                    elif fc.source == RetrievalSource.VECTOR:
                        src_str = "VECTOR ONLY"
                    else:
                        src_str = "GRAPH ONLY"

                    v_rank_disp = str(fc.vector_rank) if fc.vector_rank is not None else "—"
                    g_rank_disp = str(fc.graph_rank) if fc.graph_rank is not None else "—"
                    table_rows.append(
                        {
                            "Final Rank": fc.final_rank,
                            "Source": src_str,
                            "Chunk ID": fc.chunk_id,
                            "Source Filename": fc.source_filename,
                            "Vector Rank (r_v)": v_rank_disp,
                            "Graph Rank (r_g)": g_rank_disp,
                            "Vector Contrib": f"{fc.vector_rrf_contribution:.6f}",
                            "Graph Contrib": f"{fc.graph_rrf_contribution:.6f}",
                            "RRF Score": f"{fc.rrf_score:.6f}",
                        }
                    )
                st.dataframe(table_rows, use_container_width=True)
            else:
                st.info("No candidates met the retrieval and fusion criteria.")

            # -----------------------------------------------------------------
            # SECTION D: EVIDENCE & CHUNK INSPECTOR
            # -----------------------------------------------------------------
            st.markdown("#### D. Verbatim Chunk & Provenance Inspector")

            if cached_hybrid.fused_candidates:
                candidate_opts = [
                    f"Rank #{c.final_rank}: {c.chunk_id} ({c.source.value}) - RRF: {c.rrf_score:.5f}"
                    for c in cached_hybrid.fused_candidates
                ]
                selected_cand_label = st.selectbox(
                    "Select Candidate Chunk to Inspect:",
                    options=candidate_opts,
                    key="hybrid_cand_select",
                )
                selected_idx = candidate_opts.index(selected_cand_label)
                sel_cand = cached_hybrid.fused_candidates[selected_idx]

                c_info1, c_info2 = st.columns([3, 2])
                with c_info1:
                    st.markdown("**Verbatim Canonical Chunk Text:**")
                    st.markdown(
                        f'<div class="chunk-box">{sel_cand.chunk_text}</div>',
                        unsafe_allow_html=True,
                    )
                with c_info2:
                    st.markdown("**Provenance & Diagnostics:**")
                    st.markdown(f"**Chunk ID:** `{sel_cand.chunk_id}`")
                    st.markdown(f"**Document ID:** `{sel_cand.document_id}`")
                    st.markdown(f"**Source File:** `{sel_cand.source_filename}`")
                    if sel_cand.source_page is not None:
                        st.markdown(f"**Source Page:** `{sel_cand.source_page}`")
                    st.markdown(f"**Modality Source:** `{sel_cand.source.value}`")

                    # Modality specific metrics
                    if sel_cand.vector_rank is not None:
                        st.markdown("---")
                        st.markdown("**Vector Metrics (Diagnostic Only):**")
                        if sel_cand.cosine_distance is not None:
                            st.caption(f"Cosine Distance: `{sel_cand.cosine_distance:.4f}`")
                        if sel_cand.cosine_similarity is not None:
                            st.caption(
                                f"Cosine Similarity (1 - d): `{sel_cand.cosine_similarity:.4f}`"
                            )

                    if sel_cand.graph_rank is not None:
                        st.markdown("---")
                        st.markdown("**Graph Traversal Provenance:**")
                        if sel_cand.min_hops is not None:
                            st.caption(f"Min Hops to Seed: `{sel_cand.min_hops}`")
                        if sel_cand.matched_entities:
                            st.caption(
                                f"Matched Entities: `{', '.join(sel_cand.matched_entities)}`"
                            )

                # Supporting Graph Paths and Evidence Quotes
                if sel_cand.supporting_paths or sel_cand.evidence_quotes:
                    with st.expander("🕸️ Supporting Graph Paths & Evidence Quotes", expanded=False):
                        if sel_cand.evidence_quotes:
                            st.markdown("**Verbatim Evidence Quotes Grounding Relationships:**")
                            for q in sel_cand.evidence_quotes:
                                st.markdown(f'- *"{q}"*')
                        if sel_cand.supporting_paths:
                            st.markdown("**Graph Paths Traversing to this Chunk:**")
                            for p_i, p_info in enumerate(sel_cand.supporting_paths, start=1):
                                st.markdown(
                                    f"**Path #{p_i}:** `{p_info.get('path_str', str(p_info))}`"
                                )

            # -----------------------------------------------------------------
            # SECTION E: RANK FUSION EXPLANATION
            # -----------------------------------------------------------------
            st.markdown("#### E. Reciprocal Rank Fusion Formula & Attribution Breakdown")
            st.markdown(
                """
                Reciprocal Rank Fusion integrates candidates using the weighted formulation:
                $$\\text{RRF}(d) = \\frac{w_v}{k + r_v} + \\frac{w_g}{k + r_g}$$
                where $k$ is the smoothing constant, $r_v$ is the vector rank, and $r_g$ is the graph rank.
                Candidates absent from a modality receive zero contribution for that modality.
                """
            )

            if cached_hybrid.fused_candidates:
                with st.expander("🔍 Detailed Attribution Breakdown per Candidate", expanded=False):
                    for fc in cached_hybrid.fused_candidates:
                        k_val = cached_hybrid.fusion_params.get("rrf_k", 60)
                        wv = cached_hybrid.fusion_params.get("vector_weight", 1.0)
                        wg = cached_hybrid.fusion_params.get("graph_weight", 1.0)

                        v_term_str = (
                            f"\\frac{{{wv}}}{{{k_val} + {fc.vector_rank}}} = {fc.vector_rrf_contribution:.6f}"
                            if fc.vector_rank
                            else "0.0"
                        )
                        g_term_str = (
                            f"\\frac{{{wg}}}{{{k_val} + {fc.graph_rank}}} = {fc.graph_rrf_contribution:.6f}"
                            if fc.graph_rank
                            else "0.0"
                        )

                        st.markdown(
                            f"**Rank #{fc.final_rank} (`{fc.chunk_id}`):** "
                            f"${v_term_str} + {g_term_str} = \\mathbf{{{fc.rrf_score:.6f}}}$"
                        )

            # -----------------------------------------------------------------
            # SECTION F: PERFORMANCE DIAGNOSTICS
            # -----------------------------------------------------------------
            st.markdown("#### F. Performance Diagnostics & Latency Breakdown")
            lat_col1, lat_col2, lat_col3, lat_col4 = st.columns(4)
            with lat_col1:
                st.metric("Vector Search Latency", f"{cached_hybrid.vector_latency_ms:.1f} ms")
            with lat_col2:
                st.metric("Graph Search Latency", f"{cached_hybrid.graph_latency_ms:.1f} ms")
            with lat_col3:
                st.metric("RRF Fusion Latency", f"{cached_hybrid.fusion_latency_ms:.2f} ms")
            with lat_col4:
                st.metric("Total End-to-End Latency", f"{cached_hybrid.total_latency_ms:.1f} ms")

            # Structured Prompt Context Preview
            with st.expander(
                "📝 Formatted Hybrid Context for Downstream LLM Synthesis", expanded=False
            ):
                st.caption(
                    "This structured Markdown context block contains fused evidence candidates ready "
                    "for Phase 7 LLM response generation:"
                )
                st.code(cached_hybrid.structured_context, language="markdown")

            # -----------------------------------------------------------------
            # SECTION G: EXPORT
            # -----------------------------------------------------------------
            st.markdown("#### G. Export Retrieval Results")
            export_payload = cached_hybrid.model_dump(mode="json")
            export_json = json.dumps(export_payload, indent=2)

            # Build CSV string
            csv_buf = io.StringIO()
            csv_writer = csv.writer(csv_buf)
            csv_writer.writerow(
                [
                    "final_rank",
                    "source",
                    "chunk_id",
                    "document_id",
                    "source_filename",
                    "vector_rank",
                    "graph_rank",
                    "vector_rrf_contribution",
                    "graph_rrf_contribution",
                    "rrf_score",
                    "cosine_similarity",
                    "min_hops",
                ]
            )
            for c in cached_hybrid.fused_candidates:
                csv_writer.writerow(
                    [
                        c.final_rank,
                        c.source.value,
                        c.chunk_id,
                        c.document_id,
                        c.source_filename,
                        c.vector_rank if c.vector_rank is not None else "",
                        c.graph_rank if c.graph_rank is not None else "",
                        f"{c.vector_rrf_contribution:.6f}",
                        f"{c.graph_rrf_contribution:.6f}",
                        f"{c.rrf_score:.6f}",
                        f"{c.cosine_similarity:.4f}" if c.cosine_similarity is not None else "",
                        c.min_hops if c.min_hops is not None else "",
                    ]
                )
            csv_data = csv_buf.getvalue()

            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button(
                    label="📥 Download Hybrid Results (JSON)",
                    data=export_json,
                    file_name=f"hybrid_retrieval_{int(time.time())}.json",
                    mime="application/json",
                    use_container_width=True,
                    key="btn_download_hybrid_json",
                )
            with dl_col2:
                st.download_button(
                    label="📥 Download Hybrid Results (CSV)",
                    data=csv_data,
                    file_name=f"hybrid_retrieval_{int(time.time())}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="btn_download_hybrid_csv",
                )

    # =========================================================================
    # TAB 7: GROUNDED RAG CHAT (PHASE 7)
    # =========================================================================
    with tab_chat:
        st.markdown("## 💬 Grounded Question Answering & Citation Lab")
        st.markdown(
            "End-to-end question answering pipeline powered by multi-modal retrieval "
            "(Hybrid, Vector, or Knowledge Graph), strictly grounded LLM generation with Qwen3, "
            "citation provenance verification, safety guardrails, and claim-level faithfulness checking."
        )

        rag_pipeline = RAGPipeline(
            hybrid_retriever=hybrid_retriever,
            vector_search=search_engine,
            graph_retriever=graph_search_engine.retriever,
            ollama_client=client,
        )

        # ---------------------------------------------------------------------
        # SECTION A: Query Formulation & Retrieval Parameters
        # ---------------------------------------------------------------------
        st.markdown("### A. Query Formulation & Retrieval Parameters")

        col_q1, col_q2 = st.columns([3, 1])
        with col_q1:
            default_query = "What are the core components, vector embeddings, and evaluation metrics of Graph RAG?"
            user_query = st.text_area(
                "Natural-Language Question",
                value=st.session_state.get("rag_last_query", default_query),
                placeholder="Enter a domain question grounded in the ingested documents...",
                height=110,
                key="rag_query_input",
            )
        with col_q2:
            st.markdown("**Sample Benchmark Queries:**")
            if st.button("Query 1: Graph RAG Architecture", use_container_width=True):
                user_query = "What are the core components and architecture of Graph RAG?"
                st.session_state["rag_last_query"] = user_query
                st.rerun()
            if st.button("Query 2: Embedding Models", use_container_width=True):
                user_query = (
                    "Which vector embedding models and dimensions are used for semantic retrieval?"
                )
                st.session_state["rag_last_query"] = user_query
                st.rerun()
            if st.button("Query 3: Unanswerable Test", use_container_width=True):
                user_query = "What is the average rainfall in Tokyo during October 2024?"
                st.session_state["rag_last_query"] = user_query
                st.rerun()

        param_c1, param_c2, param_c3 = st.columns(3)
        with param_c1:
            retrieval_mode_label = st.selectbox(
                "Retrieval Strategy",
                options=[
                    "Hybrid (Vector + Graph RRF)",
                    "Vector Search Only",
                    "Knowledge Graph Only",
                ],
                index=0,
                key="rag_retrieval_mode",
            )
            mode_map = {
                "Hybrid (Vector + Graph RRF)": "hybrid",
                "Vector Search Only": "vector",
                "Knowledge Graph Only": "graph",
            }
            selected_mode = mode_map[retrieval_mode_label]

        with param_c2:
            rag_top_k = st.slider(
                "Evidence Top-K Candidates",
                min_value=1,
                max_value=15,
                value=settings.rag_top_k,
                step=1,
                key="rag_top_k_slider",
            )
            rag_temp = st.slider(
                "LLM Temperature",
                min_value=0.0,
                max_value=1.0,
                value=float(settings.rag_temperature),
                step=0.05,
                key="rag_temp_slider",
            )

        with param_c3:
            rag_max_ctx = st.number_input(
                "Max Context Token Budget",
                min_value=500,
                max_value=8000,
                value=settings.rag_max_context_tokens,
                step=500,
                key="rag_max_ctx_input",
            )
            col_cb1, col_cb2 = st.columns(2)
            with col_cb1:
                require_cits = st.checkbox("Require Citations", value=True, key="rag_cb_cits")
            with col_cb2:
                enable_faith = st.checkbox("Check Faithfulness", value=True, key="rag_cb_faith")

        btn_c1, btn_c2, _ = st.columns([2, 1, 3])
        with btn_c1:
            run_rag_btn = st.button(
                "🚀 Ask Grounded Question", type="primary", use_container_width=True
            )
        with btn_c2:
            clear_rag_btn = st.button("🗑️ Clear Answer", use_container_width=True)

        if clear_rag_btn:
            st.session_state["grounded_rag_answer"] = None
            st.rerun()

        if run_rag_btn:
            if not user_query or not user_query.strip():
                st.warning("Please enter a valid question.")
            else:
                with st.spinner(
                    "Executing retrieval, safety guardrails & generating grounded answer..."
                ):
                    ans_result = rag_pipeline.answer_question(
                        request=user_query.strip(),
                        retrieval_mode=selected_mode,  # type: ignore[arg-type]
                        top_k=rag_top_k,
                        temperature=rag_temp,
                        max_context_tokens=rag_max_ctx,
                        require_citations=require_cits,
                        enable_faithfulness_check=enable_faith,
                    )
                    user_audit.log_event(
                        user_id=current_user.user_id,
                        email=current_user.email,
                        role=current_user.role.value,
                        action="QUERY_EXECUTION",
                        status="SUCCESS",
                        details={
                            "retrieval_mode": selected_mode,
                            "top_k": rag_top_k,
                            "answer_status": ans_result.status.value,
                            "evidence_count": len(ans_result.evidence_items),
                            "citations_count": len(ans_result.citations),
                            "query_length": len(user_query.strip()),
                        },
                    )
                    st.session_state["grounded_rag_answer"] = ans_result
                    st.session_state["rag_last_query"] = user_query.strip()
                    st.rerun()

        # ---------------------------------------------------------------------
        # SECTIONS B - G: Answer Output, Footnotes, Claims, Diagnostics
        # ---------------------------------------------------------------------
        cached_answer: GroundedAnswer | None = st.session_state.get("grounded_rag_answer")

        if cached_answer is None:
            st.info(
                "💡 Formulate a question above and click **Ask Grounded Question** to generate an evidence-backed answer."
            )
        else:
            ans: GroundedAnswer = cached_answer

            st.markdown("---")
            # SECTION B: Grounded Answer & Attribution
            st.markdown("### B. Grounded Answer & Attribution")

            # Status Badge and Latency Metrics
            st_b1, st_b2, st_b3, st_b4, st_b5 = st.columns(5)
            with st_b1:
                if ans.status == AnswerStatus.SUCCESS:
                    st.markdown(
                        '<span class="status-badge-ok">✅ SUCCESS</span>', unsafe_allow_html=True
                    )
                elif ans.status == AnswerStatus.INSUFFICIENT_EVIDENCE:
                    st.markdown(
                        '<span class="status-badge-warning">⚠️ INSUFFICIENT EVIDENCE</span>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<span class="status-badge-fail">❌ {ans.status.value.upper()}</span>',
                        unsafe_allow_html=True,
                    )
            with st_b2:
                st.caption(f"**Retrieval Latency:** `{ans.latency_retrieval_ms:.1f} ms`")
            with st_b3:
                if ans.generation_invoked:
                    gen_str = (
                        f"`{ans.latency_generation_ms / 1000.0:.2f} s` ({ans.latency_generation_ms:.1f} ms)"
                        if ans.latency_generation_ms >= 1000
                        else f"`{ans.latency_generation_ms:.1f} ms`"
                    )
                else:
                    gen_str = "`0.0 ms (Skipped)`"
                st.caption(f"**Generation Latency:** {gen_str}")
            with st_b4:
                tot_str = (
                    f"`{ans.latency_total_ms / 1000.0:.2f} s` ({ans.latency_total_ms:.1f} ms)"
                    if ans.latency_total_ms >= 1000
                    else f"`{ans.latency_total_ms:.1f} ms`"
                )
                st.caption(f"**Total Latency:** {tot_str}")
            with st_b5:
                gen_invoked_str = "Yes" if ans.generation_invoked else "No (Short-Circuited)"
                st.caption(f"**LLM Gen:** `{gen_invoked_str}`")
                st.caption(
                    f"**Model Calls:** `{ans.total_model_invocations}` (Ext: {ans.query_extraction_model_calls}, Gen: {ans.answer_generation_model_calls})"
                )

            # Display Answer Card
            formatted_answer = ans.answer
            # Highlight citation markers [S1], [S2]
            import re as _re

            highlighted_html = _re.sub(
                r"\[(S\d+)\]",
                r'<span class="citation-badge">[\1]</span>',
                formatted_answer,
            )
            st.markdown(
                f"""
                <div class="answer-box">
                    {highlighted_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

            if ans.is_grounded:
                st.success(
                    "✅ **Grounding Certified**: Statements are attributed to verified source chunks with strict provenance."
                )
            elif ans.status == AnswerStatus.INSUFFICIENT_EVIDENCE:
                st.warning(
                    "⚠️ **Safety Guardrail Activated**: No sufficient evidence exists in the indexed repository to substantiate an answer."
                )

            # SECTION C: Interactive Citation Footnotes & Evidence Inspector
            st.markdown("### C. Citation Footnotes & Evidence Provenance")

            cit_tab1, cit_tab2 = st.tabs(
                [
                    f"📑 Citations Cited ({len(ans.citations)})",
                    f"📦 All Packaged Evidence Items ({len(ans.evidence_items)})",
                ]
            )

            with cit_tab1:
                if not ans.citations:
                    st.caption("No citation markers were generated in the response.")
                else:
                    for cit in ans.citations:
                        valid_icon = "✅" if cit.is_valid else "❌"
                        header = f"{valid_icon} [{cit.evidence_id}] {cit.document_title or 'Unknown Source'} — {'Valid Provenance' if cit.is_valid else 'Hallucinated Citation'}"
                        with st.expander(header, expanded=cit.is_valid):
                            c_col1, c_col2 = st.columns(2)
                            with c_col1:
                                st.markdown(f"**Marker:** `{cit.citation_marker}`")
                                st.markdown(f"**Document ID:** `{cit.document_id or 'N/A'}`")
                            with c_col2:
                                st.markdown(f"**Chunk ID:** `{cit.chunk_id or 'N/A'}`")
                                st.markdown(
                                    f"**Status:** `{'VALID' if cit.is_valid else 'INVALID'}`"
                                )
                            if cit.snippet:
                                st.markdown("**Verbatim Excerpt:**")
                                st.markdown(f"> *{cit.snippet}*")
                            if cit.validation_error:
                                st.error(cit.validation_error)

            with cit_tab2:
                if not ans.evidence_items:
                    st.caption("No evidence items were retrieved for this query.")
                else:
                    for item in ans.evidence_items:
                        with st.expander(
                            f"[{item.evidence_id}] {item.document_title} — Modality: {item.source_modality.upper()} | Score: {item.retrieval_score:.4f} | Tokens: ~{item.token_count}",
                            expanded=False,
                        ):
                            st.caption(
                                f"**Chunk ID:** `{item.chunk_id}` | **Document ID:** `{item.document_id}`"
                            )
                            st.markdown(f"```text\n{item.content}\n```")
                            if item.metadata:
                                st.json(item.metadata)

            # SECTION D: Claim-to-Evidence Grounding Matrix
            st.markdown("### D. Claim-to-Evidence Grounding Matrix")

            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                faith_val = (
                    f"{ans.faithfulness_score * 100:.1f}%"
                    if ans.faithfulness_score is not None
                    else "N/A (No claims)"
                )
                st.markdown(
                    f"""
                    <div class="overview-card">
                        <div class="overview-card-label">Faithfulness Score</div>
                        <div class="overview-card-value">{faith_val}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m_col2:
                cit_prec_val = (
                    f"{ans.citation_reference_precision * 100:.1f}%"
                    if ans.citation_reference_precision is not None
                    else "N/A (No citations)"
                )
                st.markdown(
                    f"""
                    <div class="overview-card">
                        <div class="overview-card-label">Citation Reference Precision</div>
                        <div class="overview-card-value">{cit_prec_val}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m_col3:
                st.markdown(
                    f"""
                    <div class="overview-card">
                        <div class="overview-card-label">Extracted Claims</div>
                        <div class="overview-card-value">{len(ans.claims)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if ans.claims:
                for claim in ans.claims:
                    if claim.verdict == "SUPPORTED":
                        verdict_badge = '<span class="status-badge-ok">🟢 SUPPORTED</span>'
                    elif claim.verdict == "UNCERTAIN":
                        verdict_badge = '<span class="status-badge-warning">🟡 UNCERTAIN</span>'
                    else:
                        verdict_badge = '<span class="status-badge-fail">🔴 UNSUPPORTED</span>'

                    with st.expander(
                        f"Claim {claim.claim_id}: {claim.claim_text[:80]}...", expanded=True
                    ):
                        st.markdown(f"**Verdict:** {verdict_badge}", unsafe_allow_html=True)
                        st.markdown(f"**Claim Statement:** {claim.claim_text}")
                        st.markdown(
                            f"**Cited Sources:** `{', '.join(claim.cited_sources) if claim.cited_sources else 'None'}`"
                        )
                        st.markdown(f"**Verification Reasoning:** {claim.reasoning}")
                        st.caption(f"Grounding Confidence Score: `{claim.grounding_score:.2f}`")
            else:
                st.caption(
                    "No atomic factual claims evaluated (e.g., negative response or no declarative assertions)."
                )

            # SECTION E: Prompt & Context Inspector (Collapsible)
            with st.expander("🔍 E. Prompt & Delimited Context Inspector", expanded=False):
                st.markdown("#### System Grounding Prompt")
                from graphrag_bench.prompt_builder import GROUNDED_QA_SYSTEM_PROMPT

                st.code(GROUNDED_QA_SYSTEM_PROMPT, language="text")

                st.markdown("#### Evidence Delimited Context Block")
                from graphrag_bench.evidence_preparer import EvidencePreparer

                st.code(
                    EvidencePreparer.format_evidence_context(ans.evidence_items), language="text"
                )

                st.markdown("#### Full Answer Object Metadata")
                st.json(
                    {
                        "model_used": ans.model_used,
                        "retrieval_mode": ans.retrieval_mode,
                        "total_model_invocations": ans.total_model_invocations,
                        "query_extraction_model_calls": ans.query_extraction_model_calls,
                        "answer_generation_model_calls": ans.answer_generation_model_calls,
                        "prompt_tokens_estimated": ans.prompt_tokens_estimated,
                        "answer_tokens_estimated": ans.answer_tokens_estimated,
                        "latency_retrieval_ms": ans.latency_retrieval_ms,
                        "latency_generation_ms": ans.latency_generation_ms,
                        "latency_total_ms": ans.latency_total_ms,
                        "latency_breakdown": ans.latency_breakdown,
                        "created_at": ans.created_at.isoformat(),
                    }
                )

            # SECTION F: Safety Guardrails & Diagnostic Logs
            st.markdown("### F. Safety Guardrails & Diagnostic Logs")
            if ans.warnings:
                for w in ans.warnings:
                    st.warning(f"⚠️ {w}")
            if ans.errors:
                for e in ans.errors:
                    st.error(f"❌ {e}")
            if not ans.warnings and not ans.errors:
                st.success(
                    "✅ **Zero Guardrail Violations**: All prompt injection checks, citation verifications, and safety bounds passed cleanly."
                )

            # SECTION G: Export Grounded QA Artifacts
            st.markdown("### G. Export Grounded QA Artifacts")

            export_json = ans.model_dump_json(indent=2)

            csv_buf = io.StringIO()
            csv_writer = csv.writer(csv_buf)
            csv_writer.writerow(
                [
                    "evidence_id",
                    "chunk_id",
                    "document_id",
                    "document_title",
                    "source_modality",
                    "retrieval_score",
                    "token_count",
                    "is_cited",
                ]
            )
            cited_ids = {c.evidence_id for c in ans.citations if c.is_valid}
            for item in ans.evidence_items:
                csv_writer.writerow(
                    [
                        item.evidence_id,
                        item.chunk_id,
                        item.document_id,
                        item.document_title,
                        item.source_modality,
                        f"{item.retrieval_score:.6f}",
                        item.token_count,
                        item.evidence_id in cited_ids,
                    ]
                )
            csv_data = csv_buf.getvalue()

            dl_c1, dl_c2 = st.columns(2)
            with dl_c1:
                st.download_button(
                    label="📥 Download Grounded Answer (JSON)",
                    data=export_json,
                    file_name=f"grounded_rag_answer_{int(time.time())}.json",
                    mime="application/json",
                    use_container_width=True,
                    key="btn_download_rag_json",
                )
            with dl_c2:
                st.download_button(
                    label="📥 Download Evidence Provenance (CSV)",
                    data=csv_data,
                    file_name=f"grounded_rag_evidence_{int(time.time())}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="btn_download_rag_csv",
                )

    # =========================================================================
    # TAB 8: COMPARATIVE BENCHMARK LAB (PHASE 8)
    # =========================================================================
    with tab_benchmark:
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #0d1e38 0%, #162a52 100%);
                        border: 1px solid #1e3a6a; border-radius: 12px; padding: 20px 24px; margin-bottom: 24px;">
                <h2 style="color: #38bdf8; margin: 0 0 6px 0; font-size: 1.6rem; font-weight: 700;">
                    📊 Phase 8: Formal Comparative Benchmark Evaluation Lab
                </h2>
                <p style="color: #94a3b8; margin: 0; font-size: 0.95rem;">
                    Empirical, scientifically reproducible evaluation contrasting <strong>Vector RAG</strong>,
                    <strong>Graph RAG</strong>, and <strong>Hybrid RAG</strong> on a common 22-document evaluation corpus
                    with manually validated 4-point graded relevance judgments.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        bm_storage = BenchmarkStorage()
        bm_runner = BenchmarkRunner(storage=bm_storage)

        # ---------------------------------------------------------------------
        # SECTION A: BENCHMARK DATASET OVERVIEW & CORPUS HEALTH
        # ---------------------------------------------------------------------
        st.markdown("### A. Benchmark Corpus Architecture & Dataset Health")

        b_c1, b_c2, b_c3, b_c4, b_c5, b_c6 = st.columns(6)
        with b_c1:
            st.markdown(
                """
                <div class="overview-card">
                    <div class="overview-card-label">Corpus Documents</div>
                    <div class="overview-card-value">22</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with b_c2:
            st.markdown(
                """
                <div class="overview-card">
                    <div class="overview-card-label">Canonical Chunks</div>
                    <div class="overview-card-value">110</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with b_c3:
            st.markdown(
                """
                <div class="overview-card">
                    <div class="overview-card-label">Embeddings (Chroma)</div>
                    <div class="overview-card-value">768-d</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with b_c4:
            st.markdown(
                f"""
                <div class="overview-card">
                    <div class="overview-card-label">KG Nodes / Edges</div>
                    <div class="overview-card-value" style="font-size:1.15rem;">
                        {bm_runner.graph_store.node_count} / {bm_runner.graph_store.edge_count}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with b_c5:
            st.markdown(
                f"""
                <div class="overview-card">
                    <div class="overview-card-label">Benchmark Queries</div>
                    <div class="overview-card-value">{len(bm_runner.queries)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with b_c6:
            st.markdown(
                """
                <div class="overview-card">
                    <div class="overview-card-label">Relevance Scale</div>
                    <div class="overview-card-value" style="font-size:1.1rem; color:#10b981;">0 to 3 Graded</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander(
            "📖 Benchmark Dataset Specifications & Corpus Documents (22 Source Documents)"
        ):
            st.markdown(
                """
                **Evaluation Corpus Characteristics:**
                - **22 Documents:** 110 sections covering Vector Embeddings, Similarity Metrics, ANN Indexing, Knowledge Graph Modeling, BFS/DFS Traversal, RRF Fusion, Grounded LLM Generation, Information Retrieval Metrics, and Failure Modes.
                - **60 Queries:** 15 Single-Hop (25%), 15 Multi-Hop (25%), 10 Entity-Relationship (16.7%), 10 Comparative Synthesis (16.7%), 10 Unanswerable Corpus-Absent Rejections (16.7%).
                - **Splits:** Dev (15 queries), Validation (15 queries), Test (30 queries).
                - **Graded Scale:** Grade 3 (Direct Answer), Grade 2 (Relevant Supporting), Grade 1 (Related / Partial Context), Grade 0 (Irrelevant).
                """
            )
            try:
                _raw_queries, raw_docs = bm_storage.load_dataset()
                doc_table = [
                    {
                        "Doc ID": d.document_id,
                        "Title": d.title,
                        "Chunks": d.chunk_count,
                        "File Type": d.file_type.upper(),
                        "Author / Source": d.author_or_publisher,
                        "SHA-256 (Prefix)": d.sha256_hash[:16] + "..." if d.sha256_hash else "",
                    }
                    for d in (raw_docs or [])
                ]
                if doc_table:
                    st.dataframe(doc_table, use_container_width=True)
            except Exception as exc:
                st.info(f"Corpus documents loaded: {exc}")

        # ---------------------------------------------------------------------
        # SECTION B: EXPERIMENT CONFIGURATION & EVALUATION CONTROLS
        # ---------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### B. Experiment Configuration & Execution Controls")

        cfg_col1, cfg_col2, cfg_col3 = st.columns(3)
        with cfg_col1:
            split_option = st.selectbox(
                "Benchmark Split to Evaluate",
                [
                    "All Queries (60 queries)",
                    "Test Split (30 queries)",
                    "Validation Split (15 queries)",
                    "Dev Split (15 queries)",
                ],
                index=0,
                key="bm_split_select",
            )
        with cfg_col2:
            strat_options = st.multiselect(
                "Retrieval Strategies to Benchmark",
                ["vector", "graph", "hybrid"],
                default=["vector", "graph", "hybrid"],
                format_func=lambda x: f"{x.upper()} RAG",
                key="bm_strat_select",
            )
        with cfg_col3:
            k_options = st.multiselect(
                "Evaluation Cutoffs (K)",
                [1, 3, 5, 10],
                default=[1, 3, 5, 10],
                format_func=lambda x: f"K={x}",
                key="bm_k_select",
            )

        active_queries = bm_runner.queries
        if "Test" in split_option:
            active_queries = [q for q in bm_runner.queries if q.split == BenchmarkSplit.TEST]
        elif "Validation" in split_option:
            active_queries = [q for q in bm_runner.queries if q.split == BenchmarkSplit.VALIDATION]
        elif "Dev" in split_option:
            active_queries = [q for q in bm_runner.queries if q.split == BenchmarkSplit.DEV]

        historical_runs = bm_storage.list_historical_runs()
        run_ids = [r["run_id"] for r in historical_runs]

        if "active_benchmark_run" not in st.session_state:
            st.session_state["active_benchmark_run"] = run_ids[0] if run_ids else None

        hist_col1, hist_col2 = st.columns([3, 1])
        with hist_col1:
            if run_ids:
                selected_run_id = st.selectbox(
                    "Select Historical Benchmark Run to Inspect",
                    run_ids,
                    index=run_ids.index(st.session_state["active_benchmark_run"])
                    if st.session_state["active_benchmark_run"] in run_ids
                    else 0,
                    key="bm_historical_run_select",
                )
                st.session_state["active_benchmark_run"] = selected_run_id
            else:
                st.info(
                    "No historical runs found. Execute Experiment A or B below to create a benchmark run."
                )
        with hist_col2:
            if st.button("🔄 Refresh Runs", use_container_width=True, key="btn_refresh_bm_runs"):
                st.rerun()

        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            run_exp_a = st.button(
                "▶️ Run Retrieval Benchmark (Experiment A)",
                type="primary",
                use_container_width=True,
                key="btn_run_exp_a",
            )
        with btn_c2:
            run_exp_b = st.button(
                "▶️ Run Grounded Generation Benchmark (Experiment B)",
                use_container_width=True,
                key="btn_run_exp_b",
            )

        if run_exp_a:
            if not active_queries:
                st.error("No queries available in selected split.")
            elif not strat_options:
                st.error("Select at least one retrieval strategy.")
            else:
                new_run_id = f"run_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
                st.session_state["active_benchmark_run"] = new_run_id
                progress_bar = st.progress(0, text="Initializing Experiment A...")
                status_text = st.empty()

                def ui_callback(cur, total, msg):
                    pct = min(1.0, max(0.0, cur / float(total)))
                    progress_bar.progress(pct, text=f"[{cur}/{total}] {msg}")
                    status_text.caption(f"Executing: `{msg}`")

                strat_enums = [RetrievalStrategy(s) for s in strat_options]
                target_k = sorted(k_options) if k_options else [1, 3, 5, 10]

                t_a0 = time.perf_counter()
                _, results_a = bm_runner.run_experiment_a(
                    queries=active_queries,
                    strategies=strat_enums,
                    k_values=target_k,
                    run_id=new_run_id,
                    resume=True,
                    progress_callback=ui_callback,
                )
                bm_runner.compile_run_manifest(
                    run_id=new_run_id,
                    retrieval_results=results_a,
                    k_values=target_k,
                )
                t_elapsed = time.perf_counter() - t_a0
                progress_bar.progress(1.0, text="Experiment A Completed!")
                st.success(
                    f"Experiment A completed in {t_elapsed:.2f}s! Evaluated {len(results_a)} query runs."
                )
                time.sleep(1)
                st.rerun()

        if run_exp_b:
            sample_b = []
            seen_c = set()
            for q in active_queries:
                if q.category not in seen_c:
                    sample_b.append(q)
                    seen_c.add(q.category)
                if len(sample_b) == 5:
                    break
            if not sample_b:
                sample_b = active_queries[:5]

            active_run = (
                st.session_state.get("active_benchmark_run")
                or f"run_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
            )
            st.session_state["active_benchmark_run"] = active_run
            progress_bar = st.progress(0, text="Initializing Experiment B...")
            status_text = st.empty()

            def ui_callback_b(cur, total, msg):
                pct = min(1.0, max(0.0, cur / float(total)))
                progress_bar.progress(pct, text=f"[{cur}/{total}] {msg}")
                status_text.caption(f"Generating: `{msg}`")

            strat_enums = [RetrievalStrategy(s) for s in strat_options]
            t_b0 = time.perf_counter()
            _, results_b = bm_runner.run_experiment_b(
                queries=sample_b,
                strategies=strat_enums,
                top_k=5,
                run_id=active_run,
                resume=True,
                progress_callback=ui_callback_b,
            )
            existing_retrieval = bm_storage.load_run_results(active_run, "retrieval_results.jsonl")
            if existing_retrieval:
                bm_runner.compile_run_manifest(
                    run_id=active_run,
                    retrieval_results=existing_retrieval,
                    answer_results=results_b,
                )
            t_elapsed = time.perf_counter() - t_b0
            progress_bar.progress(1.0, text="Experiment B Completed!")
            st.success(
                f"Experiment B completed in {t_elapsed:.2f}s! Generated {len(results_b)} answers."
            )
            time.sleep(1)
            st.rerun()

        current_run = st.session_state.get("active_benchmark_run")
        manifest_data = bm_storage.load_run_manifest(current_run) if current_run else None
        retrieval_records = (
            bm_storage.load_run_results(current_run, "retrieval_results.jsonl")
            if current_run
            else []
        )
        answer_records = (
            bm_storage.load_run_results(current_run, "answer_results.jsonl") if current_run else []
        )

        if current_run and retrieval_records:
            by_strat: dict[str, list[BenchmarkQueryResult]] = {}
            for r in retrieval_records:
                s_key = r.strategy.value if hasattr(r.strategy, "value") else str(r.strategy)
                by_strat.setdefault(s_key, []).append(r)

            strat_summaries = manifest_data.strategy_summaries if manifest_data else {}
            if not strat_summaries and by_strat:
                for s_key, r_list in by_strat.items():
                    strat_summaries[s_key] = aggregate_strategy_metrics(r_list)

            # -----------------------------------------------------------------
            # SECTION C: RETRIEVAL BENCHMARK RESULTS (EXPERIMENT A)
            # -----------------------------------------------------------------
            st.markdown("---")
            st.markdown(f"### C. Retrieval Comparative Performance (Run: `{current_run}`)")

            summary_rows = []
            for s_key in ["vector", "graph", "hybrid"]:
                if s_key in strat_summaries:
                    s_sum = strat_summaries[s_key]
                    summary_rows.append(
                        {
                            "Retrieval Strategy": s_key.upper() + " RAG",
                            "P@5": f"{s_sum.mean_precision_at_k.get(5, 0.0):.4f}",
                            "R@5": f"{s_sum.mean_recall_at_k.get(5, 0.0):.4f}",
                            "HitRate@5": f"{s_sum.mean_hit_rate_at_k.get(5, 0.0):.4f}",
                            "MRR": f"{s_sum.mean_mrr:.4f}",
                            "nDCG@5": f"{s_sum.mean_ndcg_at_k.get(5, 0.0):.4f}",
                            "nDCG@10": f"{s_sum.mean_ndcg_at_k.get(10, 0.0):.4f}",
                            "MAP@10": f"{s_sum.mean_map_at_k.get(10, 0.0):.4f}",
                            "Mean Latency": f"{s_sum.mean_total_latency_ms:.2f} ms",
                        }
                    )

            if summary_rows:
                st.table(summary_rows)

            k_vals_plot = [1, 3, 5, 10]
            fig_ndcg = go.Figure()
            colors_map = {"vector": "#3b82f6", "graph": "#a855f7", "hybrid": "#10b981"}

            for s_key in ["vector", "graph", "hybrid"]:
                if s_key in strat_summaries:
                    s_sum = strat_summaries[s_key]
                    y_vals = [s_sum.mean_ndcg_at_k.get(k, 0.0) for k in k_vals_plot]
                    fig_ndcg.add_trace(
                        go.Bar(
                            x=[f"K={k}" for k in k_vals_plot],
                            y=y_vals,
                            name=f"{s_key.upper()} RAG",
                            marker_color=colors_map.get(s_key, "#64748b"),
                        )
                    )

            fig_ndcg.update_layout(
                title="nDCG@K Across Evaluation Cutoffs (K=1, 3, 5, 10)",
                barmode="group",
                paper_bgcolor="#0b0f19",
                plot_bgcolor="#10192e",
                font={"color": "#e2e8f0"},
                yaxis={"title": "nDCG Score (0.0 to 1.0)", "range": [0, 1.05]},
                margin={"l": 40, "r": 20, "t": 50, "b": 40},
            )

            cat_keys = [c.value for c in BenchmarkQueryCategory]
            fig_cat = go.Figure()

            for s_key in ["vector", "graph", "hybrid"]:
                if s_key in by_strat:
                    records_for_strat = by_strat[s_key]
                    cat_scores = []
                    for c_val in cat_keys:
                        matching = [
                            r.metrics_by_k[10].ndcg
                            for r in records_for_strat
                            if (
                                r.category.value
                                if hasattr(r.category, "value")
                                else str(r.category)
                            )
                            == c_val
                            and 10 in r.metrics_by_k
                        ]
                        mean_cat = sum(matching) / float(len(matching)) if matching else 0.0
                        cat_scores.append(round(mean_cat, 4))

                    fig_cat.add_trace(
                        go.Bar(
                            x=[c.replace("_", " ").title() for c in cat_keys],
                            y=cat_scores,
                            name=f"{s_key.upper()} RAG",
                            marker_color=colors_map.get(s_key, "#64748b"),
                        )
                    )

            fig_cat.update_layout(
                title="Retrieval Effectiveness by Query Category (nDCG@10)",
                barmode="group",
                paper_bgcolor="#0b0f19",
                plot_bgcolor="#10192e",
                font={"color": "#e2e8f0"},
                yaxis={"title": "nDCG@10", "range": [0, 1.05]},
                margin={"l": 40, "r": 20, "t": 50, "b": 40},
            )

            ch_col1, ch_col2 = st.columns(2)
            with ch_col1:
                st.plotly_chart(fig_ndcg, use_container_width=True)
            with ch_col2:
                st.plotly_chart(fig_cat, use_container_width=True)

            # -----------------------------------------------------------------
            # SECTION D: ANSWER QUALITY & GROUNDED GENERATION (EXPERIMENT B)
            # -----------------------------------------------------------------
            st.markdown("---")
            st.markdown("### D. Grounded Answer Quality & Provenance (Experiment B)")

            if answer_records:
                ans_by_strat: dict[str, list[BenchmarkQueryResult]] = {}
                for r in answer_records:
                    sk = r.strategy.value if hasattr(r.strategy, "value") else str(r.strategy)
                    ans_by_strat.setdefault(sk, []).append(r)

                ans_summary_rows = []
                for sk in ["vector", "graph", "hybrid"]:
                    if sk in ans_by_strat:
                        r_list = ans_by_strat[sk]
                        n_ans = len(r_list)
                        faith_vals = [
                            r.answer_quality.faithfulness_score
                            for r in r_list
                            if r.answer_quality and r.answer_quality.faithfulness_score is not None
                        ]
                        cite_vals = [
                            r.answer_quality.citation_reference_precision
                            for r in r_list
                            if r.answer_quality
                            and r.answer_quality.citation_reference_precision is not None
                        ]
                        cov_vals = [
                            r.answer_quality.citation_coverage for r in r_list if r.answer_quality
                        ]
                        ref_vals = [
                            r.answer_quality.refusal_accuracy
                            for r in r_list
                            if r.answer_quality and r.answer_quality.refusal_accuracy is not None
                        ]

                        ans_summary_rows.append(
                            {
                                "Strategy": sk.upper() + " RAG",
                                "Answers Generated": n_ans,
                                "Mean Faithfulness": f"{sum(faith_vals) / len(faith_vals):.4f}"
                                if faith_vals
                                else "N/A",
                                "Citation Precision": f"{sum(cite_vals) / len(cite_vals):.4f}"
                                if cite_vals
                                else "N/A",
                                "Citation Coverage": f"{sum(cov_vals) / len(cov_vals):.4f}"
                                if cov_vals
                                else "N/A",
                                "Refusal Accuracy": f"{sum(ref_vals) / len(ref_vals):.4f}"
                                if ref_vals
                                else "N/A",
                            }
                        )

                if ans_summary_rows:
                    st.table(ans_summary_rows)
            else:
                st.info(
                    "Experiment B (Grounded Answer Generation) has not been run for this active run yet. "
                    "Click '▶️ Run Grounded Generation Benchmark (Experiment B)' above to evaluate answer quality, "
                    "faithfulness, and citation validity."
                )

            # -----------------------------------------------------------------
            # SECTION E: LATENCY & TELEMETRY BREAKDOWN
            # -----------------------------------------------------------------
            st.markdown("---")
            st.markdown("### E. Latency & Telemetry Analysis")

            lat_rows = []
            for s_key in ["vector", "graph", "hybrid"]:
                if s_key in by_strat:
                    lat_vals = [r.latency.total_latency_ms for r in by_strat[s_key]]
                    lat_stat = compute_latency_summary(lat_vals)
                    lat_rows.append(
                        {
                            "Modality": s_key.upper() + " RAG",
                            "Mean (ms)": f"{lat_stat['mean_ms']:.2f}",
                            "p50 / Median (ms)": f"{lat_stat['median_ms']:.2f}",
                            "p95 (ms)": f"{lat_stat['p95_ms']:.2f}",
                            "Min (ms)": f"{lat_stat['min_ms']:.2f}",
                            "Max (ms)": f"{lat_stat['max_ms']:.2f}",
                            "Std Dev": f"{lat_stat['std_dev_ms']:.2f}",
                        }
                    )

            if lat_rows:
                st.table(lat_rows)

            # -----------------------------------------------------------------
            # SECTION F: STATISTICAL SIGNIFICANCE & HYPOTHESIS TESTING
            # -----------------------------------------------------------------
            st.markdown("---")
            st.markdown("### F. Paired Statistical Contrasts & Significance")

            paired_comparisons = manifest_data.paired_comparisons if manifest_data else []
            if not paired_comparisons and "vector" in by_strat and "hybrid" in by_strat:
                paired_comparisons = []
                c1 = compare_strategies_paired(by_strat["vector"], by_strat["hybrid"], k=10)
                paired_comparisons.append(c1)
                if "graph" in by_strat:
                    c2 = compare_strategies_paired(by_strat["graph"], by_strat["hybrid"], k=10)
                    paired_comparisons.append(c2)

            comp_table = []
            for comp in paired_comparisons:
                s_a = comp.strategy_a.value.upper()
                s_b = comp.strategy_b.value.upper()
                p_disp = (
                    f"{comp.wilcoxon_p_value:.4f}" if comp.wilcoxon_p_value is not None else "N/A"
                )
                stat_badge = (
                    "✅ Significant (p<0.05)"
                    if comp.is_statistically_significant
                    else "⚖️ Not Significant"
                )
                comp_table.append(
                    {
                        "Contrast": f"{s_b} vs {s_a}",
                        "Metric": comp.metric_name,
                        "Mean Diff (Δ)": f"{comp.mean_diff:+.4f}",
                        "95% Bootstrap CI": f"[{comp.bootstrap_ci_95[0]:.4f}, {comp.bootstrap_ci_95[1]:.4f}]",
                        "Wilcoxon p-val": p_disp,
                        "Wins / Losses / Ties": f"{comp.win_count_b} / {comp.win_count_a} / {comp.tie_count}",
                        "Verdict": stat_badge,
                    }
                )

            if comp_table:
                st.table(comp_table)

            st.caption(
                "📌 **Methodology Note:** Paired contrasts evaluate paired differences across identical queries. "
                "Confidence intervals use 2,000 non-parametric bootstrap resamples (α=0.05). Wilcoxon signed-rank "
                "tests evaluate median rank shifts without normality assumptions. For sample sizes N < 30, interpret "
                "p-values as exploratory."
            )

            # -----------------------------------------------------------------
            # SECTION G: GRANULAR QUERY INSPECTOR & MODALITY ATTRIBUTION
            # -----------------------------------------------------------------
            st.markdown("---")
            st.markdown("### G. Granular Query Inspection & Modality Attribution")

            all_qids = sorted({r.query_id for r in retrieval_records})
            selected_qid = st.selectbox(
                "Select Query to Inspect",
                all_qids,
                key="bm_inspect_qid_select",
            )

            selected_q_obj = bm_runner.queries_by_id.get(selected_qid)
            if selected_q_obj:
                st.markdown(f"**Question:** `{selected_q_obj.question}`")
                q_inf1, q_inf2, q_inf3 = st.columns(3)
                with q_inf1:
                    st.caption(f"**Category:** `{selected_q_obj.category.value}`")
                with q_inf2:
                    st.caption(f"**Split:** `{selected_q_obj.split.value}`")
                with q_inf3:
                    st.caption(f"**Answerable:** `{selected_q_obj.answerable}`")

                if selected_q_obj.expected_answer:
                    st.info(f"**Ground-Truth Expected Answer:** {selected_q_obj.expected_answer}")

                q_tab_vec, q_tab_graph, q_tab_hybrid = st.tabs(
                    ["Vector Results", "Graph Results", "Hybrid Results"]
                )

                def render_retrieved_items_table(strat_key: str):
                    matching = [
                        r
                        for r in retrieval_records
                        if r.query_id == selected_qid
                        and (r.strategy.value if hasattr(r.strategy, "value") else str(r.strategy))
                        == strat_key
                    ]
                    if not matching:
                        st.write("No results recorded for this strategy.")
                        return
                    res = matching[0]
                    t_rows = []
                    for it in res.retrieved_items:
                        rel_badge = "🟢 Relevant" if it.is_relevant else "⚪ Non-relevant"
                        t_rows.append(
                            {
                                "Rank": it.rank,
                                "Chunk ID": it.chunk_id,
                                "Modality": it.modality,
                                "Score": f"{it.score:.4f}",
                                "Gold Grade": it.gold_grade,
                                "Relevance": rel_badge,
                                "Snippet": it.evidence_text[:120] + "..."
                                if it.evidence_text
                                else "",
                            }
                        )
                    if t_rows:
                        st.dataframe(t_rows, use_container_width=True)
                    else:
                        st.caption("Zero items retrieved.")

                with q_tab_vec:
                    render_retrieved_items_table("vector")
                with q_tab_graph:
                    render_retrieved_items_table("graph")
                with q_tab_hybrid:
                    render_retrieved_items_table("hybrid")

            # -----------------------------------------------------------------
            # SECTION I: ARTIFACT EXPORT & DOWNLOADS
            # -----------------------------------------------------------------
            st.markdown("---")
            st.markdown("### I. Benchmark Artifact Exports & Reports")

            run_dir = bm_storage.results_dir / current_run
            man_path = run_dir / "run_manifest.json"
            csv_path = run_dir / "results.csv"

            exp_c1, exp_c2, exp_c3 = st.columns(3)
            with exp_c1:
                if man_path.exists():
                    st.download_button(
                        "📥 Download Run Manifest (JSON)",
                        data=man_path.read_text(encoding="utf-8"),
                        file_name=f"{current_run}_manifest.json",
                        mime="application/json",
                        use_container_width=True,
                        key="btn_dl_manifest",
                    )
            with exp_c2:
                if csv_path.exists():
                    st.download_button(
                        "📥 Download Evaluation Results (CSV)",
                        data=csv_path.read_text(encoding="utf-8"),
                        file_name=f"{current_run}_results.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="btn_dl_csv",
                    )
            with exp_c3:
                res_path = run_dir / "retrieval_results.jsonl"
                if res_path.exists():
                    st.download_button(
                        "📥 Download Raw JSONL Logs",
                        data=res_path.read_text(encoding="utf-8"),
                        file_name=f"{current_run}_retrieval.jsonl",
                        mime="application/jsonl",
                        use_container_width=True,
                        key="btn_dl_jsonl",
                    )

    # =========================================================================
    # TAB 9: PRODUCTION OPERATIONS & OBSERVABILITY (PHASE 9)
    # =========================================================================
    with tab_ops:
        st.markdown("## 🛡️ Production Operations & Site Reliability Dashboard")
        st.markdown(
            "Centralized observability, real-time telemetry, automated backups, and deployment health monitoring."
        )

        telemetry = TelemetryCollector.get_instance()
        concurrency = ConcurrencyLimiter.get_instance(settings)
        backup_mgr = BackupManager(settings=settings)
        health_monitor = ProductionHealthMonitor(settings=settings)

        # ---------------------------------------------------------------------
        # SECTION A: APPLICATION STATUS
        # ---------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### A. Application Status & Runtime Environment")

        uptime_secs = int(telemetry.uptime_seconds)
        uptime_str = f"{uptime_secs // 3600}h {(uptime_secs % 3600) // 60}m {uptime_secs % 60}s"

        col_a1, col_a2, col_a3, col_a4, col_a5 = st.columns(5)
        with col_a1:
            st.metric("System State", "OPERATIONAL", delta="Healthy")
        with col_a2:
            st.metric("App Version", f"v{__version__}")
        with col_a3:
            st.metric("Environment", settings.app_env.upper())
        with col_a4:
            st.metric("Python Runtime", sys.version.split()[0])
        with col_a5:
            st.metric("Uptime", uptime_str)

        st.caption(
            f"Process PID: `{os.getpid()}` | Host Interface: `{settings.app_host}:{settings.app_port}` | "
            f"Log Verbosity: `{settings.log_level}` | Telemetry Enabled: `{settings.enable_telemetry}`"
        )

        # ---------------------------------------------------------------------
        # SECTION B: DEPENDENCY HEALTH PROBES
        # ---------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### B. Production Dependency Health Probes")

        if "prod_health_report" not in st.session_state:
            st.session_state["prod_health_report"] = None

        run_health_col, status_col = st.columns([1, 3])
        with run_health_col:
            if st.button(
                "🔄 Run 13-Point Health Check", key="btn_run_prod_health", use_container_width=True
            ):
                with st.spinner("Evaluating production subsystems..."):
                    st.session_state["prod_health_report"] = health_monitor.run_all_checks()

        # Display cached or freshly generated report
        report = st.session_state["prod_health_report"]
        if report:
            state_color = {
                ProductionHealthState.HEALTHY: "🟢",
                ProductionHealthState.DEGRADED: "🟠",
                ProductionHealthState.UNAVAILABLE: "🔴",
            }.get(report.overall_state, "⚪")

            with status_col:
                st.markdown(
                    f"**Aggregated Status:** {state_color} **{report.overall_state.value}** "
                    f"— *{report.summary}* (Evaluated at `{report.timestamp}`)"
                )

            # Health Grid
            grid_c1, grid_c2 = st.columns(2)
            for i, check in enumerate(report.checks):
                target_col = grid_c1 if i % 2 == 0 else grid_c2
                badge = {
                    ProductionHealthState.HEALTHY: "🟢 HEALTHY",
                    ProductionHealthState.DEGRADED: "🟠 DEGRADED",
                    ProductionHealthState.UNAVAILABLE: "🔴 UNAVAILABLE",
                }.get(check.state, "⚪ UNKNOWN")

                with target_col:
                    st.markdown(
                        f"""
                        <div style="border: 1px solid #334155; border-radius: 6px; padding: 10px; margin-bottom: 8px; background-color: #0f172a;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <strong>{check.name}</strong>
                                <span style="font-size: 0.85em; font-weight: bold;">{badge}</span>
                            </div>
                            <div style="font-size: 0.88em; color: #94a3b8; margin-top: 4px;">{check.message}</div>
                            <div style="font-size: 0.75em; color: #64748b; margin-top: 2px;">Latency: {check.duration_ms:.2f} ms</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            st.info(
                "Click 'Run 13-Point Health Check' to inspect live status of all production dependencies."
            )

        # ---------------------------------------------------------------------
        # SECTION C: PERFORMANCE MONITORING & TELEMETRY
        # ---------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### C. Real-Time Performance & Execution Telemetry")

        perf_c1, perf_c2, perf_c3, perf_c4 = st.columns(4)
        with perf_c1:
            st.metric(
                "Active Generations",
                f"{concurrency.active_generations} / {settings.max_concurrent_generations}",
            )
        with perf_c2:
            st.metric(
                "Active Benchmarks",
                f"{concurrency.active_benchmarks} / {settings.max_concurrent_benchmarks}",
            )
        with perf_c3:
            total_events = len(telemetry._history)
            st.metric("Recorded Operations", total_events)
        with perf_c4:
            st.metric("Ollama Generation Timeout", f"{settings.rag_generation_timeout}s")

        agg_metrics = telemetry.get_aggregated_metrics()
        if agg_metrics:
            st.markdown("#### Aggregated Operation Latencies")
            metric_rows = []
            for ev_name, ev_stats in agg_metrics.items():
                err_rate = (
                    f"{(ev_stats['error_count'] / ev_stats['count']) * 100:.1f}%"
                    if ev_stats["count"] > 0
                    else "0.0%"
                )
                metric_rows.append(
                    {
                        "Operation Event": ev_name,
                        "Count": ev_stats["count"],
                        "Success": ev_stats["success_count"],
                        "Avg (ms)": ev_stats["avg_duration_ms"],
                        "P95 (ms)": ev_stats["p95_duration_ms"],
                        "Min (ms)": ev_stats["min_duration_ms"],
                        "Max (ms)": ev_stats["max_duration_ms"],
                        "Error Rate": err_rate,
                    }
                )
            st.dataframe(metric_rows, use_container_width=True)

        recent_events = telemetry.get_recent_events(limit=15)
        if recent_events:
            with st.expander("🕒 Recent Telemetry Stream (Last 15 Events)", expanded=False):
                st.dataframe(recent_events, use_container_width=True)

        # ---------------------------------------------------------------------
        # SECTION D: OPERATIONAL LOGS & ERROR AUDIT
        # ---------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### D. Operational Logs & Error Audit")

        log_path = Path(settings.log_file).resolve()
        log_f1, log_f2, log_f3 = st.columns([1, 1, 2])
        with log_f1:
            selected_sev = st.selectbox(
                "Filter Severity",
                ["ALL", "INFO", "WARNING", "ERROR"],
                key="prod_log_sev",
            )
        with log_f2:
            num_lines = st.slider("Tail Lines", 20, 300, 100, step=20, key="prod_log_lines")

        if log_path.exists():
            with open(log_path, encoding="utf-8", errors="replace") as lf:
                lines = lf.readlines()

            if selected_sev != "ALL":
                lines = [line for line in lines if f"| {selected_sev}" in line]

            tail_lines = "".join(lines[-num_lines:])
            st.code(tail_lines or "No log entries matching the selected filter.", language="text")

            with log_f3:
                st.download_button(
                    "📥 Download Complete Operational Log",
                    data=log_path.read_text(encoding="utf-8", errors="replace"),
                    file_name="graphrag_production.log",
                    mime="text/plain",
                    use_container_width=True,
                    key="btn_dl_prod_log",
                )
        else:
            st.info(f"Log file not yet generated at `{settings.log_file}`.")

        # ---------------------------------------------------------------------
        # SECTION E: BACKUP & RECOVERY MANAGEMENT
        # ---------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### E. Cryptographic Backup & Disaster Recovery")

        bk_c1, bk_c2 = st.columns(2)
        with bk_c1:
            st.markdown("#### Create On-Demand Backup")
            bk_cat = st.selectbox(
                "Backup Scope",
                [BackupCategory.FULL, BackupCategory.CANONICAL, BackupCategory.BENCHMARK],
                format_func=lambda x: (
                    f"{x.value.upper()} (All Data & Config)"
                    if x == BackupCategory.FULL
                    else f"{x.value.upper()} Scope"
                ),
                key="bk_create_scope",
            )
            bk_desc = st.text_input(
                "Backup Note / Label",
                placeholder="e.g. Pre-deployment baseline",
                key="bk_create_note",
            )

            can_create_bk = has_permission(current_user.role, Permission.OPERATIONS_BACKUP)
            if not can_create_bk:
                st.info("🔒 Backup creation requires RESEARCHER or ADMIN role.")
            if st.button(
                "💾 Create Backup Snapshot",
                key="btn_exec_backup",
                disabled=not can_create_bk,
                use_container_width=True,
            ):
                with st.spinner("Generating cryptographically verified backup archive..."):
                    bk_res = backup_mgr.create_backup(category=bk_cat, description=bk_desc)
                    if bk_res.success and bk_res.manifest:
                        user_audit.log_event(
                            user_id=current_user.user_id,
                            email=current_user.email,
                            role=current_user.role.value,
                            action="BACKUP_CREATED",
                            status="SUCCESS",
                            details={"backup_id": bk_res.backup_id, "category": bk_cat.value},
                        )
                        st.success(
                            f"✅ Backup created successfully: `{bk_res.backup_id}` "
                            f"({bk_res.manifest.total_files} files, {bk_res.manifest.total_bytes:,} bytes, {bk_res.duration_ms:.2f} ms)"
                        )
                    else:
                        user_audit.log_event(
                            user_id=current_user.user_id,
                            email=current_user.email,
                            role=current_user.role.value,
                            action="BACKUP_CREATED",
                            status="FAILURE",
                            details={"error": bk_res.error},
                        )
                        st.error(f"❌ Backup failed: {bk_res.error}")

        with bk_c2:
            st.markdown("#### Disaster Recovery & Restore")
            available_backups = backup_mgr.list_backups()
            if available_backups:
                selected_bk_id = st.selectbox(
                    "Select Target Backup",
                    [b.backup_id for b in available_backups],
                    key="bk_restore_target",
                )

                op_col1, op_col2 = st.columns(2)
                with op_col1:
                    if st.button(
                        "🔍 Verify Integrity", key="btn_verify_bk", use_container_width=True
                    ):
                        valid, errs = backup_mgr.verify_backup_integrity(selected_bk_id)
                        if valid:
                            st.success(
                                f"✅ SHA-256 and TarSlip checks passed for `{selected_bk_id}`."
                            )
                        else:
                            st.error(
                                "❌ Verification failed:\n" + "\n".join(f"- {e}" for e in errs)
                            )

                with op_col2:
                    if st.button("👁️ Dry-Run Preview", key="btn_prev_bk", use_container_width=True):
                        prev = backup_mgr.preview_restore(selected_bk_id)
                        st.info(
                            f"Archive contains {prev.total_files} files ({prev.total_bytes:,} bytes). "
                            f"Integrity valid: {prev.valid_integrity}."
                        )

                can_restore = has_permission(current_user.role, Permission.OPERATIONS_RESTORE)
                if not can_restore:
                    st.warning("🔒 Disaster recovery restore is restricted to the ADMIN role.")
                confirm_restore = st.checkbox(
                    "⚠️ I understand that restoring will overwrite active operational data.",
                    key="chk_confirm_restore",
                    disabled=not can_restore,
                )
                if st.button(
                    "⚡ Restore From Backup",
                    disabled=not confirm_restore or not can_restore,
                    key="btn_exec_restore",
                    use_container_width=True,
                ):
                    with st.spinner(f"Restoring from `{selected_bk_id}`..."):
                        rest_res = backup_mgr.restore_backup(selected_bk_id, confirm=True)
                        if rest_res.success:
                            user_audit.log_event(
                                user_id=current_user.user_id,
                                email=current_user.email,
                                role=current_user.role.value,
                                action="BACKUP_RESTORED",
                                status="SUCCESS",
                                details={
                                    "backup_id": selected_bk_id,
                                    "restored_files": rest_res.restored_files_count,
                                },
                            )
                            st.success(
                                f"✅ Restored {rest_res.restored_files_count} files in {rest_res.duration_ms:.2f} ms! "
                                f"(Safety snapshot: `{rest_res.pre_restore_snapshot_id}`)"
                            )
                        else:
                            user_audit.log_event(
                                user_id=current_user.user_id,
                                email=current_user.email,
                                role=current_user.role.value,
                                action="BACKUP_RESTORED",
                                status="FAILURE",
                                details={"backup_id": selected_bk_id, "error": rest_res.error},
                            )
                            st.error(f"❌ Restoration failed: {rest_res.error}")
            else:
                st.info("No backups currently stored in `data/backups/`.")

        # Backup History Table
        if available_backups:
            st.markdown("#### Backup Inventory & SHA-256 Hashes")
            bk_table = [
                {
                    "Backup ID": b.backup_id,
                    "Scope": b.category.value.upper(),
                    "Created At": b.timestamp,
                    "Files": b.total_files,
                    "Size (Bytes)": f"{b.total_bytes:,}",
                    "Archive SHA-256": f"{b.archive_checksum[:16]}...",
                    "Description": b.description or "—",
                }
                for b in available_backups
            ]
            st.dataframe(bk_table, use_container_width=True)

        # ---------------------------------------------------------------------
        # SECTION F: DEPLOYMENT DIAGNOSTICS & SECURITY AUDIT
        # ---------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### F. Deployment Diagnostics & Security Boundary")

        diag_c1, diag_c2 = st.columns(2)
        with diag_c1:
            st.markdown("#### Storage Directories & Write Permissions")
            paths_to_verify = [
                ("Root Data", settings.data_dir),
                ("Processed Documents", settings.processed_dir),
                ("ChromaDB Persistence", settings.chroma_path),
                ("Knowledge Graph", settings.graph_dir),
                ("Benchmark Corpus", settings.benchmark_dir),
                ("Backup Directory", settings.backup_dir),
                ("Log Destination", Path(settings.log_file).parent),
            ]

            dir_status_rows = []
            for label, p_str in paths_to_verify:
                p = Path(p_str).resolve()
                exists = p.exists()
                writable = os.access(p, os.W_OK) if exists else False
                dir_status_rows.append(
                    {
                        "Directory": label,
                        "Path": str(p),
                        "Exists": "✅ Yes" if exists else "❌ No",
                        "Writable": "✅ Yes" if writable else "❌ No",
                    }
                )
            st.dataframe(dir_status_rows, use_container_width=True)

        with diag_c2:
            st.markdown("#### Active Configuration (Secrets Redacted)")
            st.json(settings.to_safe_dict())

        # Security warnings
        if settings.app_host == "0.0.0.0" and settings.auth_mode == "none":
            st.warning(
                "⚠️ Network Exposure Warning: Application is configured to bind on `0.0.0.0` with `AUTH_MODE=none`. "
                "Ensure that access is restricted via a TLS reverse proxy (e.g. Nginx, Caddy) or VPN."
            )
        else:
            st.success(
                f"🔒 Network Security nominal: Bound to `{settings.app_host}:{settings.app_port}` "
                f"with authentication mode `{settings.auth_mode}`."
            )

        # ---------------------------------------------------------------------
        # SECTION G: VERSION & RELEASE METADATA
        # ---------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### G. Version & Release Metadata")

        rel_c1, rel_c2 = st.columns([2, 1])
        with rel_c1:
            st.markdown(
                f"""
                - **Target Release:** `v{__version__}` (Phase 10 — Real-User Pilot Deployment, Identity Management & Product Launch)
                - **Architecture Profile:** Single-Host Pilot Architecture with TLS Reverse Proxy & Native OIDC
                - **GPU Acceleration:** Local NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)
                - **Identity & Governance:** Role-Based Access Control (Admin, Researcher, Viewer), Allowlist User Registry, Attributable JSONL Audit Logging.
                - **Key Capabilities:** Vector RAG, Graph RAG, Hybrid RAG with RRF, Grounded QA with Qwen3:8b, Insufficient-Evidence Guardrails, 60-Query Comparative Benchmark Lab, Cryptographic Backup & Restore, Multi-User Session Isolation.
                """
            )
        with rel_c2:
            st.markdown("#### Pilot Documentation Runbooks")
            st.markdown(
                """
                - [Phase 10 User Guide](file:///c:/VS%20CODE/Graph%20RAG%20Benchmark/docs/PHASE10_USER_GUIDE.md)
                - [Phase 10 Admin Guide](file:///c:/VS%20CODE/Graph%20RAG%20Benchmark/docs/PHASE10_ADMIN_GUIDE.md)
                - [Authentication Architecture](file:///c:/VS%20CODE/Graph%20RAG%20Benchmark/docs/PHASE10_AUTHENTICATION_ARCHITECTURE.md)
                - [Authorization Matrix](file:///c:/VS%20CODE/Graph%20RAG%20Benchmark/docs/PHASE10_AUTHORIZATION_MATRIX.md)
                - [Data & Privacy Policy](file:///c:/VS%20CODE/Graph%20RAG%20Benchmark/docs/PHASE10_PRIVACY_AND_DATA_POLICY.md)
                - [Audit Logging Reference](file:///c:/VS%20CODE/Graph%20RAG%20Benchmark/docs/PHASE10_AUDIT_LOGGING_REFERENCE.md)
                """
            )

        # ---------------------------------------------------------------------
        # SECTION H: USER MANAGEMENT & IDENTITY GOVERNANCE (PHASE 10)
        # ---------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### H. 👥 User Management & Identity Governance")

        can_manage_users = has_permission(current_user.role, Permission.USERS_MANAGE)

        if not can_manage_users:
            st.info(
                "🔒 **Access Restricted**: User account administration and identity governance require "
                "administrator privileges (`ADMIN` role). Please contact an administrator to request account changes "
                "or access elevation."
            )
        else:
            usr_tab1, usr_tab2, usr_tab3 = st.tabs(
                [
                    "📋 User Directory & Access Control",
                    "➕ Authorize New User",
                    "📜 Attributable Audit Log",
                ]
            )

            with usr_tab1:
                users_list = user_registry.list_users()
                st.caption(f"Total Registered Users: **{len(users_list)}**")

                user_rows = []
                for u in users_list:

                    def _fmt_ts(val: Any) -> str:
                        if not val:
                            return "Never"
                        if hasattr(val, "strftime"):
                            return val.strftime("%Y-%m-%d %H:%M")
                        return str(val)[:16].replace("T", " ")

                    user_rows.append(
                        {
                            "User ID": u.user_id[:8] + "...",
                            "Email": u.email,
                            "Name": u.name or "N/A",
                            "Role": u.role.value,
                            "Status": u.status.value,
                            "Created": _fmt_ts(u.created_at),
                            "Last Login": _fmt_ts(u.last_login_at),
                        }
                    )
                st.dataframe(user_rows, use_container_width=True)

                st.markdown("#### Modify User Role or Status")
                mod_c1, mod_c2, mod_c3 = st.columns([2, 1.5, 1.5])
                with mod_c1:
                    target_email = st.selectbox(
                        "Select Target User",
                        [u.email for u in users_list],
                        key="sel_mod_user",
                    )
                target_user = (
                    user_registry.get_user_by_email(target_email) if target_email else None
                )

                if target_user:
                    with mod_c2:
                        new_role_str = st.selectbox(
                            "Assign Role",
                            [Role.VIEWER.value, Role.RESEARCHER.value, Role.ADMIN.value],
                            index=[
                                Role.VIEWER.value,
                                Role.RESEARCHER.value,
                                Role.ADMIN.value,
                            ].index(target_user.role.value),
                            key="sel_new_role",
                        )
                        if st.button("Update Role", key="btn_apply_role"):
                            new_r = Role(new_role_str)
                            if target_user.role != new_r:
                                user_registry.update_user_role(target_user.user_id, new_r)
                                user_audit.log_event(
                                    user_id=current_user.user_id,
                                    email=current_user.email,
                                    role=current_user.role.value,
                                    action="ROLE_CHANGED",
                                    status="SUCCESS",
                                    details={"target_email": target_email, "new_role": new_r.value},
                                )
                                st.success(f"Updated `{target_email}` role to `{new_r.value}`.")
                                st.rerun()

                    with mod_c3:
                        new_status_str = st.selectbox(
                            "Account Status",
                            [
                                UserStatus.ACTIVE.value,
                                UserStatus.SUSPENDED.value,
                                UserStatus.PENDING.value,
                            ],
                            index=[
                                UserStatus.ACTIVE.value,
                                UserStatus.SUSPENDED.value,
                                UserStatus.PENDING.value,
                            ].index(target_user.status.value),
                            key="sel_new_status",
                        )
                        if st.button("Update Status", key="btn_apply_status"):
                            new_s = UserStatus(new_status_str)
                            if target_user.status != new_s:
                                user_registry.update_user_status(target_user.user_id, new_s)
                                user_audit.log_event(
                                    user_id=current_user.user_id,
                                    email=current_user.email,
                                    role=current_user.role.value,
                                    action="USER_SUSPENDED"
                                    if new_s == UserStatus.SUSPENDED
                                    else "USER_UPDATED",
                                    status="SUCCESS",
                                    details={
                                        "target_email": target_email,
                                        "new_status": new_s.value,
                                    },
                                )
                                st.success(f"Updated `{target_email}` status to `{new_s.value}`.")
                                st.rerun()

            with usr_tab2:
                st.markdown("#### Authorize New Pilot Evaluator")
                st.caption(
                    "Pre-register users on the allowlist before their initial OIDC login. "
                    "Unregistered users are blocked at sign-in."
                )
                with st.form("form_add_user"):
                    new_email = st.text_input(
                        "User Email Address", placeholder="evaluator@organization.com"
                    )
                    new_name = st.text_input("Full Name", placeholder="Jane Doe")
                    new_role_val = st.selectbox(
                        "Initial Role",
                        [Role.RESEARCHER.value, Role.VIEWER.value, Role.ADMIN.value],
                        index=0,
                    )
                    new_notes = st.text_input("Notes / Team", placeholder="NLP Evaluation Group")
                    submit_user = st.form_submit_button("Authorize & Add User", type="primary")

                    if submit_user:
                        if not new_email or "@" not in new_email:
                            st.error("Please enter a valid email address.")
                        else:
                            try:
                                created_user = user_registry.register_user(
                                    email=new_email.strip(),
                                    name=new_name.strip() if new_name else None,
                                    role=Role(new_role_val),
                                    status=UserStatus.ACTIVE,
                                    notes=new_notes.strip() if new_notes else None,
                                )
                                user_audit.log_event(
                                    user_id=current_user.user_id,
                                    email=current_user.email,
                                    role=current_user.role.value,
                                    action="USER_CREATED",
                                    status="SUCCESS",
                                    details={
                                        "new_user_id": created_user.user_id,
                                        "email": new_email,
                                        "role": new_role_val,
                                    },
                                )
                                st.success(
                                    f"✅ Successfully authorized `{new_email}` as `{new_role_val}`."
                                )
                                st.rerun()
                            except ValueError as e:
                                st.error(f"Cannot add user: {e}")

            with usr_tab3:
                st.markdown("#### Security & User Action Audit Stream")
                audit_col1, audit_col2, audit_col3 = st.columns(3)
                with audit_col1:
                    filter_act = st.selectbox(
                        "Filter Action",
                        [
                            "ALL",
                            "AUTH_LOGIN",
                            "AUTH_LOGOUT",
                            "AUTH_FAILURE",
                            "QUERY_EXECUTION",
                            "BACKUP_CREATED",
                            "BACKUP_RESTORED",
                            "USER_CREATED",
                            "ROLE_CHANGED",
                            "PERMISSION_DENIED",
                        ],
                        index=0,
                    )
                with audit_col2:
                    filter_stat = st.selectbox(
                        "Filter Status", ["ALL", "SUCCESS", "FAILURE", "DENIED"], index=0
                    )
                with audit_col3:
                    limit_val = st.slider(
                        "Max Events", min_value=10, max_value=200, value=50, step=10
                    )

                evts = user_audit.get_recent_events(
                    limit=limit_val,
                    action=None if filter_act == "ALL" else filter_act,
                    status=None if filter_stat == "ALL" else filter_stat,
                )
                stats = user_audit.get_summary_statistics()

                stat_c1, stat_c2, stat_c3, stat_c4 = st.columns(4)
                stat_c1.metric("Total Events", stats.get("total_events", 0))
                stat_c2.metric("Successful Logins", stats.get("actions", {}).get("AUTH_LOGIN", 0))
                stat_c3.metric(
                    "Queries Executed", stats.get("actions", {}).get("QUERY_EXECUTION", 0)
                )
                stat_c4.metric("Denied Operations", stats.get("statuses", {}).get("DENIED", 0))

                if evts:
                    evt_rows = []
                    for e in evts:
                        evt_rows.append(
                            {
                                "Timestamp (UTC)": e.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                                if hasattr(e.timestamp, "strftime")
                                else str(e.timestamp)[:19].replace("T", " "),
                                "User": e.email,
                                "Role": e.role,
                                "Action": e.action,
                                "Status": e.status,
                                "Details": json.dumps(e.details) if e.details else "{}",
                            }
                        )
                    st.dataframe(evt_rows, use_container_width=True)
                else:
                    st.info("No audit events match the selected criteria.")


def main() -> None:
    """Application main entry router."""
    if is_running_in_streamlit():
        run_streamlit()
    else:
        run_cli()


if __name__ == "__main__":
    main()
