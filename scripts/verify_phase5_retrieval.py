"""Phase 5 Graph Retrieval Verification Script.
Executes representative multi-hop semantic queries against the persisted canonical Knowledge Graph.
"""

import sys
from pathlib import Path

# Ensure src is on Python search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from graphrag_bench.config import get_settings
from graphrag_bench.graph_search import GraphSearchEngine
from graphrag_bench.graph_store import GraphStore


def run_phase5_verification() -> None:
    settings = get_settings()
    store = GraphStore()
    graph_path = Path("data/knowledge_graph/knowledge_graph.json")
    store.load_json(graph_path)

    print("=" * 70)
    print("PHASE 5: GRAPH RETRIEVAL VERIFICATION")
    print(
        f"Loaded Knowledge Graph: {store.node_count} nodes, {store.edge_count} edges from {graph_path}"
    )
    print("=" * 70)

    engine = GraphSearchEngine(store=store, settings=settings)

    test_queries = [
        (
            "Query 1: Multi-Hop Reasoning Architecture",
            "How does Graph RAG perform compared to Vector RAG on multi-hop questions?",
            2,
            "both",
        ),
        ("Query 2: Component Fusion", "What does Hybrid RAG integrate or combine?", 2, "out"),
        (
            "Query 3: Vector Database Integration",
            "How does ChromaDB integrate with retrieval architectures?",
            2,
            "both",
        ),
        (
            "Query 4: Single-Hop Grounding",
            "What eliminates Vector RAG structural blind spots?",
            1,
            "both",
        ),
        (
            "Query 5: Unlinked Query Handling",
            "What is Quantum Computing Superconductivity?",
            2,
            "both",
        ),
    ]

    for title, query, hops, direction in test_queries:
        print(f"\n--- {title} ---")
        print(f"Query: '{query}'")
        print(f"Config: max_hops={hops}, direction={direction}")

        resp = engine.search_graph(query=query, max_hops=hops, direction=direction, top_k=5)

        print(f"Status: {resp.status.value}")
        print(f"Latency: {resp.latency_ms:.2f}ms")
        print(f"Seed Entities Linked: {len(resp.linked_entities)}")
        for le in resp.linked_entities:
            print(
                f"  * [{le.match_type.value}] {le.canonical_name} ({le.entity_type}) - ID: {le.entity_id}"
            )

        print(f"Paths Retrieved: {len(resp.results)}")
        for idx, res in enumerate(resp.results[:3], 1):
            path_str = " -> ".join([n.canonical_name for n in res.path.nodes])
            print(f"  Rank #{idx} (Score: {res.score:.3f}): {path_str}")
            print(f"    Hop count: {res.path.hop_count}")
            for edge in res.path.edges:
                print(
                    f"    - [{edge.subject_entity_id}] --{edge.predicate} ({edge.direction})--> [{edge.object_entity_id}]"
                )
                print(f'      Citation: "{edge.evidence_quote[:80]}..." ({edge.source_filename})')

        print(
            f"Evidence Quotes Count: {len(resp.context_evidence.evidence_quotes) if resp.context_evidence else 0}"
        )

    print("\n" + "=" * 70)
    print("PHASE 5 RETRIEVAL VERIFICATION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    run_phase5_verification()
