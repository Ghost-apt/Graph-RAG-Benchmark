"""Unit tests for ReciprocalRankFusion module."""

import pytest

from graphrag_bench.hybrid_schemas import (
    GraphCandidate,
    RetrievalSource,
    VectorCandidate,
)
from graphrag_bench.reciprocal_rank_fusion import ReciprocalRankFusion


def make_vector_candidate(chunk_id: str, rank: int) -> VectorCandidate:
    """Helper to construct a mock VectorCandidate."""
    return VectorCandidate(
        chunk_id=chunk_id,
        document_id=f"doc_{chunk_id}",
        source_filename=f"{chunk_id}.md",
        file_type="md",
        chunk_text=f"Text for {chunk_id}",
        vector_rank=rank,
        cosine_distance=0.2,
        cosine_similarity=0.8,
        normalized_score=0.9,
    )


def make_graph_candidate(chunk_id: str, rank: int) -> GraphCandidate:
    """Helper to construct a mock GraphCandidate."""
    return GraphCandidate(
        chunk_id=chunk_id,
        document_id=f"doc_{chunk_id}",
        source_filename=f"{chunk_id}.md",
        chunk_text=f"Text for {chunk_id}",
        graph_rank=rank,
        matched_entities=["EntityA"],
        supporting_paths=[],
        evidence_quotes=[f"Quote for {chunk_id}"],
        best_path_score=0.9,
        hop_count=1,
    )


class TestReciprocalRankFusion:
    """Test suite for deterministic RRF fusion and tie-breaking."""

    def test_parameter_validation_k_less_than_one(self) -> None:
        """Verify k < 1 raises ValueError."""
        with pytest.raises(ValueError, match="RRF constant k must be >= 1"):
            ReciprocalRankFusion(k=0)

    def test_parameter_validation_negative_weight(self) -> None:
        """Verify negative weights raise ValueError."""
        with pytest.raises(ValueError, match="Modality weights must be non-negative"):
            ReciprocalRankFusion(vector_weight=-0.5)

    def test_parameter_validation_both_weights_zero(self) -> None:
        """Verify both weights zero raises ValueError."""
        with pytest.raises(ValueError, match="At least one modality weight"):
            ReciprocalRankFusion(vector_weight=0.0, graph_weight=0.0)

    def test_standard_rrf_manual_calculation(self) -> None:
        """Verify standard RRF math: 1/(k + r_v) + 1/(k + r_g)."""
        rrf = ReciprocalRankFusion(k=60, vector_weight=1.0, graph_weight=1.0)

        # chunk_a: vector rank 1, graph rank 2 (Dual modality)
        # chunk_b: vector rank 2 only (Vector only)
        # chunk_c: graph rank 1 only (Graph only)
        v_list = [make_vector_candidate("chunk_a", 1), make_vector_candidate("chunk_b", 2)]
        g_list = [make_graph_candidate("chunk_c", 1), make_graph_candidate("chunk_a", 2)]

        fused = rrf.fuse(v_list, g_list)

        assert len(fused) == 3

        # chunk_a expected: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.01639344 + 0.01612903 = 0.03252247
        cand_a = next(c for c in fused if c.chunk_id == "chunk_a")
        assert cand_a.source == RetrievalSource.VECTOR_AND_GRAPH
        assert cand_a.vector_rank == 1
        assert cand_a.graph_rank == 2
        assert cand_a.vector_rrf_contribution == pytest.approx(1.0 / 61, abs=1e-6)
        assert cand_a.graph_rrf_contribution == pytest.approx(1.0 / 62, abs=1e-6)
        assert cand_a.rrf_score == pytest.approx(1.0 / 61 + 1.0 / 62, abs=1e-6)
        assert cand_a.final_rank == 1

        # chunk_c expected: 1/(60+1) = 1/61 = 0.01639344
        cand_c = next(c for c in fused if c.chunk_id == "chunk_c")
        assert cand_c.source == RetrievalSource.GRAPH
        assert cand_c.vector_rank is None
        assert cand_c.graph_rank == 1
        assert cand_c.vector_rrf_contribution == 0.0
        assert cand_c.graph_rrf_contribution == pytest.approx(1.0 / 61, abs=1e-6)
        assert cand_c.rrf_score == pytest.approx(1.0 / 61, abs=1e-6)
        assert cand_c.final_rank == 2

        # chunk_b expected: 1/(60+2) = 1/62 = 0.01612903
        cand_b = next(c for c in fused if c.chunk_id == "chunk_b")
        assert cand_b.source == RetrievalSource.VECTOR
        assert cand_b.vector_rank == 2
        assert cand_b.graph_rank is None
        assert cand_b.vector_rrf_contribution == pytest.approx(1.0 / 62, abs=1e-6)
        assert cand_b.graph_rrf_contribution == 0.0
        assert cand_b.rrf_score == pytest.approx(1.0 / 62, abs=1e-6)
        assert cand_b.final_rank == 3

    def test_weighted_rrf_manual_calculation(self) -> None:
        """Verify weighted RRF: w_v/(k + r_v) + w_g/(k + r_g)."""
        rrf = ReciprocalRankFusion(k=60, vector_weight=2.0, graph_weight=0.5)

        v_list = [make_vector_candidate("chunk_v", 1)]
        g_list = [make_graph_candidate("chunk_g", 1)]

        fused = rrf.fuse(v_list, g_list)

        # chunk_v: 2.0 / 61 = 0.03278688
        # chunk_g: 0.5 / 61 = 0.00819672
        assert fused[0].chunk_id == "chunk_v"
        assert fused[0].rrf_score == pytest.approx(2.0 / 61, abs=1e-6)
        assert fused[1].chunk_id == "chunk_g"
        assert fused[1].rrf_score == pytest.approx(0.5 / 61, abs=1e-6)

    def test_tie_breaking_dual_modality_beats_single(self) -> None:
        """When RRF scores tie, VECTOR_AND_GRAPH should rank above single modality."""
        # Construct exact score tie:
        # Candidate 1 (dual): r_v=62, r_g=62 with w_v=1.0, w_g=1.0 -> 1/122 + 1/122 = 2/122 = 1/61
        # Candidate 2 (single): r_v=1 with w_v=1.0, w_g=1.0 -> 1/61
        # Scores are identical (1/61). Candidate 1 (dual) should win tie-break over Candidate 2 (single).
        rrf = ReciprocalRankFusion(k=60, vector_weight=1.0, graph_weight=1.0)

        v_list = [
            make_vector_candidate("chunk_single", 1),
            make_vector_candidate("chunk_dual", 62),
        ]
        g_list = [
            make_graph_candidate("chunk_dual", 62),
        ]
        fused = rrf.fuse(v_list, g_list)
        assert fused[0].chunk_id == "chunk_dual"
        assert fused[1].chunk_id == "chunk_single"

        # Now test exact score tie with same single modality:
        # candidate X: vector-only rank 1, score = 1/61
        # candidate Y: graph-only rank 1, score = 1/61
        # Both have same score and single-modality priority, min_rank is 1 for both.
        # Lexicographical tie break: "chunk_a" before "chunk_b"
        v_list_tie = [make_vector_candidate("chunk_b", 1)]
        g_list_tie = [make_graph_candidate("chunk_a", 1)]
        fused_tie = rrf.fuse(v_list_tie, g_list_tie)
        assert fused_tie[0].chunk_id == "chunk_a"
        assert fused_tie[1].chunk_id == "chunk_b"

    def test_fuse_empty_inputs(self) -> None:
        """Verify empty candidate lists return empty result without crashing."""
        rrf = ReciprocalRankFusion()
        assert rrf.fuse([], []) == []

    def test_min_rrf_score_filter(self) -> None:
        """Verify min_rrf_score filters out lower scoring candidates."""
        rrf = ReciprocalRankFusion(k=60, min_rrf_score=0.0163)  # > 1/62 (0.016129)
        v_list = [make_vector_candidate("chunk_1", 1), make_vector_candidate("chunk_2", 2)]
        fused = rrf.fuse(v_list, [])
        assert len(fused) == 1
        assert fused[0].chunk_id == "chunk_1"
