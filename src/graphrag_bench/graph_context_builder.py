"""
Graph Context Builder Module.
Constructs structured, traceable evidence context from retrieved graph paths.
Formats relational triples, entity descriptions, and verbatim evidence citations
for downstream presentation and Phase 7 prompt synthesis.
"""

from typing import Any

from graphrag_bench.graph_retrieval_schemas import (
    GraphContextEvidence,
    GraphRetrievalResult,
)
from graphrag_bench.graph_store import GraphStore
from graphrag_bench.logger import get_logger

logger = get_logger("graph_context_builder")


class GraphContextBuilder:
    """Builds structured evidence context from retrieved graph paths."""

    def __init__(self, store: GraphStore | None = None) -> None:
        """Initialize context builder with optional GraphStore for entity metadata lookups."""
        self.store = store

    def build_context(
        self,
        results: list[GraphRetrievalResult],
        max_context_paths: int = 10,
    ) -> GraphContextEvidence:
        """Convert top retrieved paths into formatted traceable context.

        Args:
            results: Ordered list of GraphRetrievalResult items from retriever.
            max_context_paths: Maximum number of top paths to incorporate into context.

        Returns:
            GraphContextEvidence containing structured triples, descriptions, citations, and markdown text.
        """
        if not results:
            return GraphContextEvidence(
                triples=[],
                entity_descriptions={},
                evidence_quotes=[],
                source_citations=[],
                formatted_context="No relevant knowledge graph context retrieved for this query.",
            )

        top_results = results[:max_context_paths]

        seen_triples: set[str] = set()
        unique_triples: list[str] = []
        entity_descriptions: dict[str, str] = {}
        unique_quotes: list[str] = []
        seen_quotes: set[str] = set()
        citations: list[dict[str, Any]] = []
        seen_edges: set[str] = set()

        for res in top_results:
            path = res.path
            for node in path.nodes:
                if node.canonical_name not in entity_descriptions:
                    ent = self.store.get_entity(node.entity_id) if self.store else None
                    desc = ent.description if ent and ent.description else "Document concept"
                    entity_descriptions[node.canonical_name] = f"[{node.entity_type}] {desc}"

            for edge in path.edges:
                if edge.relationship_id in seen_edges:
                    continue
                seen_edges.add(edge.relationship_id)

                sub_name = (
                    self.store.get_entity(edge.subject_entity_id).canonical_name
                    if self.store and self.store.get_entity(edge.subject_entity_id)
                    else edge.subject_entity_id
                )
                obj_name = (
                    self.store.get_entity(edge.object_entity_id).canonical_name
                    if self.store and self.store.get_entity(edge.object_entity_id)
                    else edge.object_entity_id
                )

                triple_str = f"({sub_name}) --[{edge.predicate}]--> ({obj_name})"
                if triple_str not in seen_triples:
                    seen_triples.add(triple_str)
                    unique_triples.append(triple_str)

                quote = edge.evidence_quote.strip()
                if quote and quote not in seen_quotes:
                    seen_quotes.add(quote)
                    unique_quotes.append(quote)
                    citations.append(
                        {
                            "relationship_id": edge.relationship_id,
                            "subject": sub_name,
                            "predicate": edge.predicate,
                            "object": obj_name,
                            "evidence_quote": quote,
                            "source_filename": edge.source_filename,
                            "source_document_id": edge.source_document_id,
                            "source_chunk_id": edge.source_chunk_id,
                            "confidence": edge.confidence,
                        }
                    )

        # Build human-readable markdown format
        lines: list[str] = []
        lines.append("### Knowledge Graph Relational Context\n")

        lines.append("#### Asserted Relational Triples:")
        for idx, t in enumerate(unique_triples, 1):
            lines.append(f"{idx}. {t}")

        lines.append("\n#### Grounded Evidence Quotations:")
        for idx, cit in enumerate(citations, 1):
            src_info = f"{cit['source_filename']} (Chunk: {cit['source_chunk_id'][:16]}...)"
            lines.append(f'{idx}. "{cit["evidence_quote"]}"')
            lines.append(f"   *Source: {src_info} | Confidence: {cit['confidence']:.2f}*")

        lines.append("\n#### Entity Glossary:")
        for name, desc in sorted(entity_descriptions.items()):
            lines.append(f"- **{name}**: {desc}")

        formatted = "\n".join(lines)

        return GraphContextEvidence(
            triples=unique_triples,
            entity_descriptions=entity_descriptions,
            evidence_quotes=unique_quotes,
            source_citations=citations,
            formatted_context=formatted,
        )
