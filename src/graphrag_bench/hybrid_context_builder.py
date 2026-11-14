"""Hybrid context builder for Phase 6.

Formats Reciprocal Rank Fused (RRF) candidates into a structured, provenance-rich
prompt context suitable for downstream LLM synthesis (Phase 7).
Preserves verbatim chunk content, origin file citations, and relational graph paths.
"""

from graphrag_bench.hybrid_schemas import FusionCandidate, RetrievalSource
from graphrag_bench.logger import get_logger

logger = get_logger("hybrid_context_builder")


class HybridContextBuilder:
    """Constructs structured evidence context from hybrid retrieval candidates."""

    def build_context(
        self,
        fused_candidates: list[FusionCandidate],
        max_context_chunks: int = 5,
    ) -> str:
        """Format top fused candidates into a structured Markdown evidence block.

        Args:
            fused_candidates: Ranked list of FusionCandidate objects.
            max_context_chunks: Maximum number of top chunks to include in the context.

        Returns:
            Formatted Markdown string ready for downstream RAG prompt injection.
        """
        if not fused_candidates:
            return "No relevant hybrid retrieval context found for this query."

        top_candidates = fused_candidates[:max_context_chunks]

        lines: list[str] = []
        lines.append("### HYBRID RETRIEVAL EVIDENCE CONTEXT (RRF FUSED)\n")
        lines.append(
            f"Retrieved {len(fused_candidates)} fused candidates (showing top {len(top_candidates)}):"
        )

        # Count modality distribution
        source_counts: dict[str, int] = {}
        for c in top_candidates:
            src_val = c.retrieval_sources.value
            source_counts[src_val] = source_counts.get(src_val, 0) + 1
        summary_parts = [f"{src}: {cnt}" for src, cnt in source_counts.items()]
        lines.append(f"Modalities breakdown: {', '.join(summary_parts)}\n")

        for cand in top_candidates:
            lines.append("---")
            lines.append(
                f"#### [Evidence #{cand.fused_rank}] Chunk ID: `{cand.chunk_id}` | Source: **{cand.retrieval_sources.value}**"
            )

            # Metadata header
            meta = cand.source_metadata
            doc_info = f"Document: `{cand.source_filename}` ({cand.document_id[:16]}...)"
            if meta.get("source_page") is not None:
                doc_info += f" | Page: {meta['source_page']}"
            lines.append(f"*{doc_info}*")

            # Score attribution details
            rank_details: list[str] = [f"RRF Score: `{cand.rrf_score:.6f}`"]
            if cand.vector_rank is not None:
                rank_details.append(
                    f"Vector Rank #{cand.vector_rank} (contrib: {cand.vector_contribution:.6f})"
                )
                if "cosine_similarity" in meta:
                    rank_details.append(f"Cosine Sim: {meta['cosine_similarity']:.4f}")
            if cand.graph_rank is not None:
                rank_details.append(
                    f"Graph Rank #{cand.graph_rank} (contrib: {cand.graph_contribution:.6f})"
                )
                if "graph_score" in meta:
                    rank_details.append(f"Graph Score: {meta['graph_score']:.3f}")
            lines.append(f"**Ranking Details:** {', '.join(rank_details)}")

            # Supporting Graph Relationships (if present)
            if cand.retrieval_sources in (RetrievalSource.GRAPH, RetrievalSource.VECTOR_AND_GRAPH):
                if cand.supporting_graph_paths:
                    lines.append("\n**Supporting Graph Paths:**")
                    for p_idx, p in enumerate(cand.supporting_graph_paths[:2], start=1):
                        nodes = p.get("nodes", [])
                        node_str = " -> ".join(n.get("canonical_name", "") for n in nodes)
                        lines.append(f"  {p_idx}. {node_str} ({p.get('hop_count', 1)} hops)")

                if cand.evidence_quotes:
                    lines.append("\n**Grounded Knowledge Graph Quotes:**")
                    for q in cand.evidence_quotes[:2]:
                        lines.append(f'  - "{q}"')

            # Canonical Chunk Content
            lines.append("\n**Verbatim Chunk Content:**")
            lines.append(f"> {cand.chunk_text.strip()}")
            lines.append("")

        return "\n".join(lines)
