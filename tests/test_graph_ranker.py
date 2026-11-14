"""Unit tests for GraphRanker module."""

from graphrag_bench.entity_linker import MatchType
from graphrag_bench.graph_ranker import GraphRanker
from graphrag_bench.graph_retrieval_schemas import (
    GraphPath,
    LinkedEntity,
    PathEdge,
    PathNode,
)


def make_test_path(
    path_id: str,
    seed_id: str,
    seed_name: str,
    hop_count: int,
    confidence: float = 1.0,
) -> GraphPath:
    """Helper to synthesize a GraphPath with hop_count hops."""
    nodes = [
        PathNode(
            entity_id=seed_id,
            canonical_name=seed_name,
            entity_type="TECHNOLOGY",
            hop_index=0,
        )
    ]
    edges = []
    curr_id = seed_id
    for h in range(1, hop_count + 1):
        next_id = f"{seed_id}_hop{h}"
        next_name = f"Node Hop {h}"
        nodes.append(
            PathNode(
                entity_id=next_id,
                canonical_name=next_name,
                entity_type="CONCEPT",
                hop_index=h,
            )
        )
        edges.append(
            PathEdge(
                relationship_id=f"rel_{path_id}_{h}",
                subject_entity_id=curr_id,
                predicate="USES",
                object_entity_id=next_id,
                direction="outgoing",
                evidence_quote=f"Quote for hop {h}",
                confidence=confidence,
                source_document_id="doc_1",
                source_chunk_id="chunk_1",
                source_filename="bench.md",
            )
        )
        curr_id = next_id

    return GraphPath(
        path_id=path_id,
        seed_entity_id=seed_id,
        seed_entity_name=seed_name,
        nodes=nodes,
        edges=edges,
        hop_count=hop_count,
        source_document_ids=["doc_1"],
        source_chunk_ids=["chunk_1"],
    )


class TestGraphRanker:
    """Test suite for deterministic path scoring and ranking."""

    def test_rank_paths_deterministic_scores(self) -> None:
        ranker = GraphRanker()
        linked = [
            LinkedEntity(
                query_mention="Vector RAG",
                entity_id="node_vrag",
                canonical_name="Vector RAG",
                entity_type="TECHNOLOGY",
                match_type=MatchType.EXACT,
            )
        ]
        p1 = make_test_path("p1", "node_vrag", "Vector RAG", hop_count=1)
        p2 = make_test_path("p2", "node_vrag", "Vector RAG", hop_count=2)

        ranked_1 = ranker.rank_paths([p1, p2], linked_entities=linked, query="Vector RAG")
        ranked_2 = ranker.rank_paths([p1, p2], linked_entities=linked, query="Vector RAG")

        assert len(ranked_1) == len(ranked_2) == 2
        for r1, r2 in zip(ranked_1, ranked_2, strict=False):
            assert r1.score == r2.score
            assert r1.path_id == r2.path_id

    def test_shorter_path_preferred_by_hop_economy(self) -> None:
        ranker = GraphRanker()
        linked = [
            LinkedEntity(
                query_mention="Vector RAG",
                entity_id="node_vrag",
                canonical_name="Vector RAG",
                entity_type="TECHNOLOGY",
                match_type=MatchType.EXACT,
            )
        ]
        p_1hop = make_test_path("p_1hop", "node_vrag", "Vector RAG", hop_count=1)
        p_3hop = make_test_path("p_3hop", "node_vrag", "Vector RAG", hop_count=3)

        ranked = ranker.rank_paths([p_3hop, p_1hop], linked_entities=linked, query="Vector RAG")
        # 1-hop path has higher length discount than 3-hop
        assert ranked[0].path_id == "p_1hop"
        assert ranked[0].score > ranked[1].score

    def test_exact_match_seed_outranks_fuzzy_seed(self) -> None:
        ranker = GraphRanker()
        linked = [
            LinkedEntity(
                query_mention="Vector RAG",
                entity_id="node_vrag",
                canonical_name="Vector RAG",
                entity_type="TECHNOLOGY",
                match_type=MatchType.EXACT,
            ),
            LinkedEntity(
                query_mention="Rag",
                entity_id="node_rag_survey",
                canonical_name="RAG Survey",
                entity_type="DATASET",
                match_type=MatchType.FUZZY,
            ),
        ]
        p_exact = make_test_path("p_exact", "node_vrag", "Vector RAG", hop_count=1)
        p_fuzzy = make_test_path("p_fuzzy", "node_rag_survey", "RAG Survey", hop_count=1)

        ranked = ranker.rank_paths(
            [p_fuzzy, p_exact], linked_entities=linked, query="Vector RAG performance"
        )
        assert ranked[0].path_id == "p_exact"
        assert ranked[0].score > ranked[1].score

    def test_higher_confidence_edges_yield_higher_score(self) -> None:
        ranker = GraphRanker()
        linked = [
            LinkedEntity(
                query_mention="Vector RAG",
                entity_id="node_vrag",
                canonical_name="Vector RAG",
                entity_type="TECHNOLOGY",
                match_type=MatchType.EXACT,
            )
        ]
        p_high_conf = make_test_path(
            "p_high", "node_vrag", "Vector RAG", hop_count=2, confidence=1.0
        )
        p_low_conf = make_test_path("p_low", "node_vrag", "Vector RAG", hop_count=2, confidence=0.4)

        ranked = ranker.rank_paths(
            [p_low_conf, p_high_conf], linked_entities=linked, query="Vector RAG"
        )
        assert ranked[0].path_id == "p_high"
        assert ranked[0].score > ranked[1].score

    def test_score_breakdown_transparency(self) -> None:
        ranker = GraphRanker()
        linked = [
            LinkedEntity(
                query_mention="Vector RAG",
                entity_id="node_vrag",
                canonical_name="Vector RAG",
                entity_type="TECHNOLOGY",
                match_type=MatchType.EXACT,
            )
        ]
        p = make_test_path("p1", "node_vrag", "Vector RAG", hop_count=1)
        ranked = ranker.rank_paths([p], linked_entities=linked, query="Vector RAG")

        assert len(ranked) == 1
        res = ranked[0]
        assert "seed_precision" in res.score_breakdown
        assert "hop_economy" in res.score_breakdown
        assert "evidence_support" in res.score_breakdown
        assert res.score > 0
