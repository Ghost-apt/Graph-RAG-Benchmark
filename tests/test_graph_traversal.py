"""Unit tests for GraphTraversalEngine."""

from graphrag_bench.graph_schemas import GraphEntity, GraphRelationship
from graphrag_bench.graph_store import GraphStore
from graphrag_bench.graph_traversal import GraphTraversalEngine


def build_linear_traversal_graph() -> GraphStore:
    """Build linear test graph: A -> B -> C -> D with cycle back from D -> A.

    Node A --[PROVIDES]--> Node B --[USES]--> Node C --[EXTRACTS]--> Node D --[FUSES]--> Node A
    """
    store = GraphStore()
    nodes = [
        ("node_a", "Vector RAG", "TECHNOLOGY"),
        ("node_b", "ChromaDB", "SOFTWARE"),
        ("node_c", "Nomic Embeddings", "METHOD"),
        ("node_d", "Hybrid RAG", "TECHNOLOGY"),
    ]
    for n_id, name, t in nodes:
        store.add_entity(
            GraphEntity(
                entity_id=n_id,
                canonical_name=name,
                entity_type=t,
                source_document_ids=["doc_1"],
                source_chunk_ids=["chunk_1"],
            )
        )

    edges = [
        ("rel_1", "node_a", "PROVIDES", "node_b", "doc_1", "chunk_1"),
        ("rel_2", "node_b", "USES", "node_c", "doc_1", "chunk_2"),
        ("rel_3", "node_c", "EXTRACTS", "node_d", "doc_2", "chunk_3"),
        ("rel_4", "node_d", "FUSES", "node_a", "doc_2", "chunk_4"),
    ]
    for r_id, s, p, o, d, c in edges:
        store.add_relationship(
            GraphRelationship(
                relationship_id=r_id,
                subject_entity_id=s,
                predicate=p,
                object_entity_id=o,
                evidence_quote=f"{s} {p} {o} quotation",
                confidence=0.95,
                source_document_id=d,
                source_chunk_id=c,
                source_filename="test.md",
            )
        )
    return store


class TestGraphTraversalEngine:
    """Test suite for GraphTraversalEngine multi-hop exploration."""

    def test_one_hop_outgoing(self) -> None:
        store = build_linear_traversal_graph()
        engine = GraphTraversalEngine(store=store)

        paths = engine.traverse(
            seed_entity_id="node_a",
            max_hops=1,
            direction="out",
        )

        assert len(paths) == 1
        path = paths[0]
        assert len(path.edges) == 1
        assert len(path.nodes) == 2
        assert path.nodes[0].entity_id == "node_a"
        assert path.nodes[1].entity_id == "node_b"
        assert path.edges[0].predicate == "PROVIDES"
        assert path.edges[0].direction == "outgoing"

    def test_two_hop_traversal(self) -> None:
        store = build_linear_traversal_graph()
        engine = GraphTraversalEngine(store=store)

        paths = engine.traverse(
            seed_entity_id="node_a",
            max_hops=2,
            direction="out",
        )

        # Should return 1-hop and 2-hop paths
        hop_counts = {len(p.edges) for p in paths}
        assert 1 in hop_counts
        assert 2 in hop_counts

        two_hop = next(p for p in paths if len(p.edges) == 2)
        assert two_hop.nodes[0].entity_id == "node_a"
        assert two_hop.nodes[1].entity_id == "node_b"
        assert two_hop.nodes[2].entity_id == "node_c"
        assert [e.predicate for e in two_hop.edges] == ["PROVIDES", "USES"]

    def test_three_hop_traversal(self) -> None:
        store = build_linear_traversal_graph()
        engine = GraphTraversalEngine(store=store)

        paths = engine.traverse(
            seed_entity_id="node_a",
            max_hops=3,
            direction="out",
        )

        three_hop = [p for p in paths if len(p.edges) == 3]
        assert len(three_hop) == 1
        assert [n.entity_id for n in three_hop[0].nodes] == ["node_a", "node_b", "node_c", "node_d"]

    def test_incoming_traversal(self) -> None:
        store = build_linear_traversal_graph()
        engine = GraphTraversalEngine(store=store)

        # Node A has incoming edge from Node D (rel_4: node_d -> node_a)
        paths = engine.traverse(
            seed_entity_id="node_a",
            max_hops=1,
            direction="in",
        )

        assert len(paths) == 1
        path = paths[0]
        assert path.nodes[0].entity_id == "node_a"
        assert path.nodes[1].entity_id == "node_d"
        assert path.edges[0].predicate == "FUSES"
        assert path.edges[0].direction == "incoming"

    def test_bidirectional_traversal(self) -> None:
        store = build_linear_traversal_graph()
        engine = GraphTraversalEngine(store=store)

        paths = engine.traverse(
            seed_entity_id="node_a",
            max_hops=1,
            direction="both",
        )

        # Node A has 1 outgoing (to B) and 1 incoming (from D)
        assert len(paths) == 2
        neighbor_ids = {p.nodes[1].entity_id for p in paths}
        assert neighbor_ids == {"node_b", "node_d"}

    def test_predicate_filtering(self) -> None:
        store = build_linear_traversal_graph()
        engine = GraphTraversalEngine(store=store)

        # Filter only "PROVIDES"
        paths = engine.traverse(
            seed_entity_id="node_a",
            max_hops=3,
            direction="out",
            predicates=["PROVIDES"],
        )

        assert len(paths) == 1
        assert paths[0].edges[0].predicate == "PROVIDES"

    def test_document_id_filtering(self) -> None:
        store = build_linear_traversal_graph()
        engine = GraphTraversalEngine(store=store)

        # Filter by doc_1: only rel_1 and rel_2 belong to doc_1
        paths = engine.traverse(
            seed_entity_id="node_a",
            max_hops=3,
            direction="out",
            doc_id_filter="doc_1",
        )

        # Hop 3 (rel_3) belongs to doc_2 and should be excluded
        for p in paths:
            assert all(e.source_document_id == "doc_1" for e in p.edges)

    def test_cycle_prevention(self) -> None:
        store = build_linear_traversal_graph()
        engine = GraphTraversalEngine(store=store)

        # Node A -> B -> C -> D -> A creates a potential 4-hop cycle.
        # Traversal up to 3 hops should never visit node A again in the same path.
        paths = engine.traverse(
            seed_entity_id="node_a",
            max_hops=3,
            direction="out",
        )

        for p in paths:
            node_ids = [n.entity_id for n in p.nodes]
            assert len(node_ids) == len(set(node_ids)), f"Cycle detected in path: {node_ids}"

    def test_empty_or_nonexistent_seed(self) -> None:
        store = build_linear_traversal_graph()
        engine = GraphTraversalEngine(store=store)

        paths = engine.traverse(seed_entity_id="nonexistent_node", max_hops=2)
        assert len(paths) == 0
