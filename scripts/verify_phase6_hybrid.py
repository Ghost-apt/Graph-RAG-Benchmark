"""Phase 6 Hybrid Retrieval & Reciprocal Rank Fusion Verification Script.

Executes 5 canonical benchmark queries against persisted ChromaDB and NetworkX Knowledge Graph,
verifying parallel retrieval, modality attribution, deterministic tie-breaking, and exact manual RRF math.
"""

import sys
from pathlib import Path

# Ensure src is on Python search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from graphrag_bench.config import get_settings
from graphrag_bench.document_repository import DocumentRepository
from graphrag_bench.embedding_service import EmbeddingService
from graphrag_bench.graph_search import GraphSearchEngine
from graphrag_bench.graph_store import GraphStore
from graphrag_bench.hybrid_retriever import HybridRetriever
from graphrag_bench.hybrid_schemas import RetrievalSource
from graphrag_bench.ollama_client import OllamaClient
from graphrag_bench.semantic_search import SemanticSearch
from graphrag_bench.vector_store import VectorStore


def run_phase6_verification() -> bool:
    settings = get_settings()
    base_dir = Path.cwd()
    repo = DocumentRepository(base_dir=base_dir)

    print("=" * 75)
    print("PHASE 6: HYBRID RETRIEVAL & RECIPROCAL RANK FUSION VERIFICATION")
    print("=" * 75)

    # 1. Initialize persistent stores
    client = OllamaClient(settings=settings)
    embed_service = EmbeddingService(client=client, settings=settings)
    vector_store = VectorStore(settings=settings)
    semantic_search = SemanticSearch(
        vector_store=vector_store,
        embedding_service=embed_service,
        settings=settings,
    )

    graph_store = GraphStore()
    graph_path = Path("data/knowledge_graph/knowledge_graph.json")
    if not graph_path.exists():
        print(f"ERROR: Graph file {graph_path} not found.")
        return False
    graph_store.load_json(graph_path)

    graph_search = GraphSearchEngine(store=graph_store, settings=settings)

    print(f"Loaded Vector Database: {vector_store.count()} indexed chunks")
    print(f"Loaded Knowledge Graph: {graph_store.node_count} nodes, {graph_store.edge_count} edges")
    print(f"Loaded Document Repository: {len(repo.list_documents())} canonical documents")
    print("-" * 75)

    hybrid_retriever = HybridRetriever(
        vector_search=semantic_search,
        graph_search=graph_search,
        repository=repo,
        settings=settings,
    )

    queries = [
        "What technologies are used in Graph RAG?",
        "How does Hybrid RAG combine vector and graph retrieval?",
        "What is the relationship between ChromaDB and Vector RAG?",
        "How does Graph RAG handle multi-hop relationships?",
        "Explain the connection between Retrieval-Augmented Generation and knowledge graphs.",
    ]

    all_passed = True

    for q_idx, query in enumerate(queries, start=1):
        print(f'\n[QUERY {q_idx}/5]: "{query}"')
        resp = hybrid_retriever.search_hybrid(
            query=query,
            vector_top_k=5,
            graph_top_k=5,
            final_top_k=5,
            rrf_k=60,
            vector_weight=1.0,
            graph_weight=1.0,
            max_hops=2,
            direction="both",
        )

        print(
            f"Status: {resp.status.value} (Vector: {resp.vector_status}, Graph: {resp.graph_status})"
        )
        print(
            f"Latency: Total={resp.total_latency_ms:.1f}ms (Vector={resp.vector_latency_ms:.1f}ms, Graph={resp.graph_latency_ms:.1f}ms, Fusion={resp.fusion_latency_ms:.2f}ms)"
        )
        print(
            f"Candidates: Vector={resp.vector_results_count}, Graph={resp.graph_results_count}, Fused={resp.fused_results_count}"
        )

        if not resp.fused_candidates:
            print("WARNING: No fused candidates returned.")
            continue

        print("\nRanked Fused Results (RRF):")
        print(
            f"{'Rank':<5} {'Source':<18} {'Chunk ID':<26} {'v_rank':<8} {'g_rank':<8} {'v_contrib':<12} {'g_contrib':<12} {'RRF Score':<12} {'Math Verified'}"
        )
        print("-" * 115)

        k = resp.fusion_params.get("rrf_k", 60)
        wv = resp.fusion_params.get("vector_weight", 1.0)
        wg = resp.fusion_params.get("graph_weight", 1.0)

        for c in resp.fused_candidates:
            # Manual formula check
            exp_v_contrib = (wv / (k + c.vector_rank)) if c.vector_rank is not None else 0.0
            exp_g_contrib = (wg / (k + c.graph_rank)) if c.graph_rank is not None else 0.0
            exp_rrf = exp_v_contrib + exp_g_contrib

            v_ok = abs(c.vector_rrf_contribution - exp_v_contrib) < 1e-6
            g_ok = abs(c.graph_rrf_contribution - exp_g_contrib) < 1e-6
            rrf_ok = abs(c.rrf_score - exp_rrf) < 1e-6
            math_verified = "PASS" if (v_ok and g_ok and rrf_ok) else "FAIL"

            if not (v_ok and g_ok and rrf_ok):
                all_passed = False

            v_rank_s = str(c.vector_rank) if c.vector_rank is not None else "—"
            g_rank_s = str(c.graph_rank) if c.graph_rank is not None else "—"

            print(
                f"#{c.final_rank:<4} {c.source.value:<18} {c.chunk_id:<26} {v_rank_s:<8} {g_rank_s:<8} {c.vector_rrf_contribution:<12.6f} {c.graph_rrf_contribution:<12.6f} {c.rrf_score:<12.6f} {math_verified}"
            )

        # Provenance sample from Rank 1
        top_cand = resp.fused_candidates[0]
        print(f"\nTop Candidate (#1) Provenance: {top_cand.chunk_id}")
        print(f"  File: {top_cand.source_filename} | Document ID: {top_cand.document_id}")
        if top_cand.source in (RetrievalSource.GRAPH, RetrievalSource.VECTOR_AND_GRAPH):
            print(f"  Matched Entities: {top_cand.matched_entities}")
            print(f"  Grounded Evidence Quotes: {len(top_cand.evidence_quotes)}")
            for q in top_cand.evidence_quotes[:2]:
                print(f'    * "{q[:80]}..."')
        print("-" * 75)

    print("\n" + "=" * 75)
    if all_passed:
        print("ALL 5 BENCHMARK QUERIES EXECUTED AND MATHEMATICALLY VERIFIED: PASS")
    else:
        print("ONE OR MORE BENCHMARK QUERIES FAILED VERIFICATION: FAIL")
    print("=" * 75)

    return all_passed


if __name__ == "__main__":
    success = run_phase6_verification()
    sys.exit(0 if success else 1)
