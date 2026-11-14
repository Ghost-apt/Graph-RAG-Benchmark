"""Reciprocal Rank Fusion (RRF) Engine for Phase 6 Hybrid Retrieval.

Implements standard and weighted Reciprocal Rank Fusion over normalized VectorCandidate
and GraphCandidate ranked lists. Operates strictly on ordinal ranks without mixing
incompatible metric distance or heuristic score scales.
"""

from typing import Any

from graphrag_bench.hybrid_schemas import (
    FusionCandidate,
    GraphCandidate,
    RetrievalSource,
    VectorCandidate,
)
from graphrag_bench.logger import get_logger

logger = get_logger("reciprocal_rank_fusion")


class ReciprocalRankFusion:
    """Computes deterministic Reciprocal Rank Fusion across retrieval modalities."""

    def __init__(
        self,
        k: int = 60,
        vector_weight: float = 1.0,
        graph_weight: float = 1.0,
        min_rrf_score: float = 0.0,
    ) -> None:
        """Initialize Reciprocal Rank Fusion parameters.

        Args:
            k: Smoothing constant added to denominator (must be >= 1, standard default 60).
            vector_weight: Multiplier weight for vector rank contribution (>= 0.0).
            graph_weight: Multiplier weight for graph rank contribution (>= 0.0).
            min_rrf_score: Minimum RRF score required for a candidate to be included.

        Raises:
            ValueError: If k < 1, weights are negative, or both weights are zero.
        """
        if k < 1:
            raise ValueError(f"RRF constant k must be >= 1, got: {k}")
        if vector_weight < 0.0 or graph_weight < 0.0:
            raise ValueError(
                f"Modality weights must be non-negative, got vector_weight={vector_weight}, "
                f"graph_weight={graph_weight}"
            )
        if vector_weight == 0.0 and graph_weight == 0.0:
            raise ValueError("At least one modality weight must be strictly positive.")
        if min_rrf_score < 0.0:
            raise ValueError(f"min_rrf_score cannot be negative, got: {min_rrf_score}")

        self.k = k
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight
        self.min_rrf_score = min_rrf_score

    def fuse(
        self,
        vector_candidates: list[VectorCandidate],
        graph_candidates: list[GraphCandidate],
        top_k: int | None = None,
        k: int | None = None,
        vector_weight: float | None = None,
        graph_weight: float | None = None,
    ) -> list[FusionCandidate]:
        """Perform Reciprocal Rank Fusion over vector and graph candidates.

        Args:
            vector_candidates: Ranked list of VectorCandidate objects.
            graph_candidates: Ranked list of GraphCandidate objects.
            top_k: Optional limit on the number of returned fused candidates.
            k: Optional override for RRF smoothing constant.
            vector_weight: Optional override for vector modality weight.
            graph_weight: Optional override for graph modality weight.

        Returns:
            List of FusionCandidate objects ordered by fused_rank (1-based).
        """
        effective_k = k if k is not None else self.k
        eff_v_weight = vector_weight if vector_weight is not None else self.vector_weight
        eff_g_weight = graph_weight if graph_weight is not None else self.graph_weight

        if effective_k < 1:
            raise ValueError(f"RRF constant k must be >= 1, got: {effective_k}")
        if eff_v_weight < 0.0 or eff_g_weight < 0.0:
            raise ValueError("Modality weights must be non-negative.")
        if eff_v_weight == 0.0 and eff_g_weight == 0.0:
            raise ValueError("At least one modality weight must be positive.")

        # Map candidates by chunk_id
        # chunk_id -> { "vector": VectorCandidate, "graph": GraphCandidate }
        all_chunk_ids: list[str] = []
        vec_map: dict[str, VectorCandidate] = {}
        graph_map: dict[str, GraphCandidate] = {}

        for vc in vector_candidates:
            if vc.chunk_id not in vec_map:
                vec_map[vc.chunk_id] = vc
                if vc.chunk_id not in all_chunk_ids:
                    all_chunk_ids.append(vc.chunk_id)

        for gc in graph_candidates:
            if gc.chunk_id not in graph_map:
                graph_map[gc.chunk_id] = gc
                if gc.chunk_id not in all_chunk_ids:
                    all_chunk_ids.append(gc.chunk_id)

        raw_fused: list[dict[str, Any]] = []

        for cid in all_chunk_ids:
            v_cand = vec_map.get(cid)
            g_cand = graph_map.get(cid)

            v_rank = v_cand.vector_rank if v_cand else None
            g_rank = gc.graph_rank if (gc := g_cand) else None

            # Calculate individual contributions
            v_contrib = (eff_v_weight / (effective_k + v_rank)) if v_rank is not None else 0.0
            g_contrib = (eff_g_weight / (effective_k + g_rank)) if g_rank is not None else 0.0
            total_rrf = v_contrib + g_contrib

            if total_rrf < self.min_rrf_score:
                continue

            # Determine attribution
            if v_cand is not None and g_cand is not None:
                source = RetrievalSource.VECTOR_AND_GRAPH
            elif v_cand is not None:
                source = RetrievalSource.VECTOR
            else:
                source = RetrievalSource.GRAPH

            # Consolidate metadata & citations
            doc_id = v_cand.document_id if v_cand else (g_cand.document_id if g_cand else "")
            filename = (
                v_cand.source_filename if v_cand else (g_cand.source_filename if g_cand else "")
            )
            text = v_cand.chunk_text if v_cand else (g_cand.chunk_text if g_cand else "")

            # Diagnostic metadata
            meta: dict[str, Any] = {}
            if v_cand:
                meta.update(
                    {
                        "cosine_distance": v_cand.cosine_distance,
                        "cosine_similarity": v_cand.cosine_similarity,
                        "normalized_score": v_cand.normalized_score,
                        "file_type": v_cand.file_type,
                        "source_page": v_cand.source_page,
                    }
                )
            if g_cand:
                meta.update(
                    {
                        "graph_score": g_cand.graph_score,
                        "hop_count": g_cand.hop_count,
                        "matched_entities": g_cand.matched_entities,
                        "relationship_ids": g_cand.relationship_ids,
                    }
                )

            # Consolidate evidence quotes
            quotes = list(dict.fromkeys(g_cand.evidence_quotes)) if g_cand else []
            paths = g_cand.supporting_paths if g_cand else []

            raw_fused.append(
                {
                    "chunk_id": cid,
                    "document_id": doc_id,
                    "source_filename": filename,
                    "chunk_text": text,
                    "vector_rank": v_rank,
                    "graph_rank": g_rank,
                    "vector_contribution": v_contrib,
                    "graph_contribution": g_contrib,
                    "rrf_score": total_rrf,
                    "retrieval_sources": source,
                    "supporting_graph_paths": paths,
                    "evidence_quotes": quotes,
                    "source_metadata": meta,
                    "both_modalities": (v_cand is not None and g_cand is not None),
                    "best_rank": min(
                        v_rank if v_rank is not None else 999999,
                        g_rank if g_rank is not None else 999999,
                    ),
                }
            )

        # Deterministic multi-tier sort:
        # 1. rrf_score descending (-rrf_score)
        # 2. presence in both modalities (True first -> -1, False -> 0)
        # 3. best individual rank ascending
        # 4. canonical chunk_id lexicographical ascending
        sorted_fused = sorted(
            raw_fused,
            key=lambda x: (
                -round(x["rrf_score"], 8),
                -1 if x["both_modalities"] else 0,
                x["best_rank"],
                x["chunk_id"],
            ),
        )

        # Limit to top_k if requested
        if top_k is not None and top_k > 0:
            sorted_fused = sorted_fused[:top_k]

        # Construct final FusionCandidate objects with 1-based fused_rank
        fused_candidates: list[FusionCandidate] = []
        for idx, item in enumerate(sorted_fused, start=1):
            candidate = FusionCandidate(
                chunk_id=item["chunk_id"],
                document_id=item["document_id"],
                source_filename=item["source_filename"],
                chunk_text=item["chunk_text"],
                vector_rank=item["vector_rank"],
                graph_rank=item["graph_rank"],
                vector_contribution=round(item["vector_contribution"], 6),
                graph_contribution=round(item["graph_contribution"], 6),
                rrf_score=round(item["rrf_score"], 6),
                fused_rank=idx,
                retrieval_sources=item["retrieval_sources"],
                supporting_graph_paths=item["supporting_graph_paths"],
                evidence_quotes=item["evidence_quotes"],
                source_metadata=item["source_metadata"],
            )
            fused_candidates.append(candidate)

        return fused_candidates
