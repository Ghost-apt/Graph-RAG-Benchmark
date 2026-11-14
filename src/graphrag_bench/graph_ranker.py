"""
Graph Ranker Module.
Implements transparent, deterministic scoring and ranking for retrieved graph paths.
Scores paths using seed match precision, path hop economy, query term coverage,
and source evidence strength.
"""

from graphrag_bench.graph_retrieval_schemas import GraphPath, LinkedEntity
from graphrag_bench.logger import get_logger

logger = get_logger("graph_ranker")


class GraphRanker:
    """Transparent deterministic ranker for multi-hop graph retrieval paths."""

    def __init__(
        self,
        weight_seed_match: float = 0.35,
        weight_path_length: float = 0.30,
        weight_query_coverage: float = 0.20,
        weight_evidence: float = 0.15,
    ) -> None:
        """Initialize ranker with configurable component weights.

        Args:
            weight_seed_match: Weight assigned to seed entity linking precision (0.0 to 1.0).
            weight_path_length: Weight favoring compact paths (shorter hops).
            weight_query_coverage: Weight rewarding nodes that match terms in the query.
            weight_evidence: Weight reflecting source evidence quotes and confidence.
        """
        total = weight_seed_match + weight_path_length + weight_query_coverage + weight_evidence
        if total <= 0:
            raise ValueError("Total ranker weights must be strictly positive.")

        # Normalize weights so they sum to 1.0
        self.w_seed = weight_seed_match / total
        self.w_length = weight_path_length / total
        self.w_coverage = weight_query_coverage / total
        self.w_evidence = weight_evidence / total

    def score_path(
        self,
        path: GraphPath,
        linked_entities_map: dict[str, LinkedEntity],
        query_terms: set[str],
    ) -> tuple[float, dict[str, float], str]:
        """Compute transparent heuristic score for a single graph path.

        Returns:
            Tuple of (composite_score, score_breakdown_dict, score_explanation).
        """
        # 1. Seed match quality score
        linked_info = linked_entities_map.get(path.seed_entity_id)
        if linked_info:
            match_type = str(linked_info.match_type).lower()
            if "exact" in match_type:
                seed_score = 1.0
            elif "alias" in match_type:
                seed_score = 0.85
            elif "normalized" in match_type:
                seed_score = 0.70
            else:
                seed_score = 0.50

            if linked_info.is_ambiguous:
                seed_score *= 0.60  # Ambiguity penalty
        else:
            seed_score = 0.50

        # 2. Path length / hop economy score (shorter paths favored)
        # 1 hop -> 1.0, 2 hops -> 0.75, 3 hops -> 0.50
        hops = max(1, path.hop_count)
        if hops == 1:
            length_score = 1.0
        elif hops == 2:
            length_score = 0.75
        elif hops == 3:
            length_score = 0.50
        else:
            length_score = max(0.20, 1.0 / (hops**0.7))

        # 3. Query term coverage across path nodes and predicates
        path_tokens: set[str] = set()
        for node in path.nodes:
            path_tokens.update(node.canonical_name.lower().split())
        for edge in path.edges:
            path_tokens.update(edge.predicate.lower().replace("_", " ").split())

        if query_terms:
            overlap = query_terms.intersection(path_tokens)
            coverage_score = min(1.0, len(overlap) / max(1, len(query_terms)))
        else:
            coverage_score = 0.50

        # 4. Evidence quality & confidence score
        if path.edges:
            conf_sum = sum(e.confidence for e in path.edges)
            avg_conf = conf_sum / len(path.edges)
            has_quotes = sum(1 for e in path.edges if len(e.evidence_quote.strip()) > 10)
            quote_ratio = has_quotes / len(path.edges)
            evidence_score = 0.5 * avg_conf + 0.5 * quote_ratio
        else:
            evidence_score = 0.50

        # Composite score calculation
        composite = (
            self.w_seed * seed_score
            + self.w_length * length_score
            + self.w_coverage * coverage_score
            + self.w_evidence * evidence_score
        )
        final_score = round(composite, 4)

        breakdown = {
            "seed_precision": round(seed_score, 4),
            "hop_economy": round(length_score, 4),
            "query_coverage": round(coverage_score, 4),
            "evidence_support": round(evidence_score, 4),
            "composite_score": final_score,
        }

        explanation = (
            f"Score: {final_score:.4f} (Seed: {seed_score:.2f} × {self.w_seed:.2f} + "
            f"Hops ({hops}): {length_score:.2f} × {self.w_length:.2f} + "
            f"Coverage: {coverage_score:.2f} × {self.w_coverage:.2f} + "
            f"Evidence: {evidence_score:.2f} × {self.w_evidence:.2f})"
        )

        return final_score, breakdown, explanation

    def rank_paths(
        self,
        paths: list[GraphPath],
        linked_entities: list[LinkedEntity],
        query: str = "",
    ) -> list[GraphPath]:
        """Score and sort graph retrieval paths in descending order of relevance.

        Deterministic tie-breaking: higher composite score -> fewer hops -> path_id.
        """
        if not paths:
            return []

        linked_map = {e.entity_id: e for e in linked_entities}
        query_terms = {t.lower() for t in query.split() if len(t) > 2 and t.isalnum()}

        for path in paths:
            score, breakdown, _ = self.score_path(path, linked_map, query_terms)
            path.score = score
            path.score_breakdown = breakdown

        # Sort descending by score, ascending by hop count, ascending by path_id
        ranked = sorted(
            paths,
            key=lambda p: (-p.score, p.hop_count, p.path_id),
        )
        return ranked
