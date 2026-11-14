"""
Graph Search Orchestrator Module.
Exposes the primary high-level search interface search_graph() coordinating
query entity extraction, entity linking, traversal, ranking, and structured context assembly.
"""

import time
from pathlib import Path
from typing import Literal

from graphrag_bench.config import Settings, get_settings
from graphrag_bench.entity_linker import EntityLinker
from graphrag_bench.graph_context_builder import GraphContextBuilder
from graphrag_bench.graph_ranker import GraphRanker
from graphrag_bench.graph_retrieval_schemas import (
    GraphSearchResponse,
    GraphSearchStatus,
)
from graphrag_bench.graph_retriever import GraphRetriever
from graphrag_bench.graph_store import GraphStore
from graphrag_bench.graph_traversal import GraphTraversalEngine
from graphrag_bench.logger import get_logger
from graphrag_bench.query_entity_extractor import QueryEntityExtractor

logger = get_logger("graph_search")


class GraphSearchEngine:
    """Unified coordinator for knowledge graph retrieval operations."""

    def __init__(
        self,
        store: GraphStore | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize graph search engine.

        Loads persisted canonical graph if store is not explicitly provided.
        """
        self.settings = settings or get_settings()
        if store is not None:
            self.store = store
        else:
            self.store = GraphStore()
            graph_path = Path(self.settings.graph_path)
            if graph_path.exists():
                try:
                    self.store.load_json(graph_path)
                    logger.info(
                        "GraphSearchEngine loaded %d nodes, %d edges from %s",
                        self.store.node_count,
                        self.store.edge_count,
                        graph_path,
                    )
                except Exception as exc:
                    logger.error("Failed to load canonical graph at %s: %s", graph_path, exc)

        # Wire up subcomponents
        self.extractor = QueryEntityExtractor(store=self.store, settings=self.settings)
        self.linker = EntityLinker(store=self.store)
        self.traversal = GraphTraversalEngine(store=self.store)
        self.ranker = GraphRanker()
        self.retriever = GraphRetriever(
            store=self.store,
            settings=self.settings,
            extractor=self.extractor,
            linker=self.linker,
            traversal=self.traversal,
            ranker=self.ranker,
        )
        self.context_builder = GraphContextBuilder(store=self.store)

    def search_graph(
        self,
        query: str,
        max_hops: int | None = None,
        direction: Literal["both", "out", "in"] = "both",
        top_k: int = 10,
        predicates: list[str] | set[str] | None = None,
        doc_id_filter: str | None = None,
        force_llm_extractor: bool = False,
    ) -> GraphSearchResponse:
        """Execute complete graph retrieval pipeline for a natural language query.

        Args:
            query: User's search query or question.
            max_hops: Maximum traversal depth (1 to 3).
            direction: Traversal direction ('both', 'out', 'in').
            top_k: Maximum number of ranked results to return.
            predicates: Optional filter restricting relationships to specific predicates.
            doc_id_filter: Optional source document ID filter.
            force_llm_extractor: Whether to force LLM extraction instead of fast dictionary matching.

        Returns:
            Typed GraphSearchResponse with results, structured context, status, and statistics.
        """
        t0 = time.perf_counter()

        # 1. Query validation
        clean_query = query.strip() if query else ""
        if not clean_query:
            return GraphSearchResponse(
                query=query or "",
                status=GraphSearchStatus.INVALID_QUERY,
                message="Query string cannot be empty.",
                latency_ms=0.0,
            )

        # 2. Graph availability check
        if self.store.node_count == 0 or self.store.edge_count == 0:
            return GraphSearchResponse(
                query=clean_query,
                status=GraphSearchStatus.GRAPH_UNAVAILABLE,
                message="Knowledge Graph is empty or unavailable.",
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )

        try:
            # 3. Query Entity Extraction
            extraction_res = self.extractor.extract_query_entities(
                clean_query, force_llm=force_llm_extractor
            )
            if not extraction_res.entities:
                return GraphSearchResponse(
                    query=clean_query,
                    status=GraphSearchStatus.NO_MATCHING_ENTITIES,
                    message=f"No recognizable entities found in query '{clean_query}'.",
                    extraction_result=extraction_res,
                    statistics={
                        "entities_extracted": 0,
                        "extraction_duration_ms": extraction_res.duration_ms,
                        "llm_invoked": extraction_res.llm_invoked,
                    },
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                )

            # 4. Entity Linking
            linking_res = self.linker.link_entities(extraction_res.entities, query=clean_query)
            if not linking_res.linked_entities:
                return GraphSearchResponse(
                    query=clean_query,
                    status=GraphSearchStatus.NO_MATCHING_ENTITIES,
                    message=f"Extracted mentions ({', '.join(e.mention_text for e in extraction_res.entities)}) could not be mapped to any graph nodes.",
                    extraction_result=extraction_res,
                    statistics={
                        "entities_extracted": len(extraction_res.entities),
                        "entities_linked": 0,
                        "unlinked_mentions": linking_res.unlinked_mentions,
                    },
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                )

            # 5. Graph Retrieval & Ranking
            effective_hops = max_hops or getattr(self.settings, "graph_max_hops", 2)
            results = self.retriever.retrieve(
                query=clean_query,
                linked_entities=linking_res.linked_entities,
                max_hops=effective_hops,
                direction=direction,
                predicates=predicates,
                top_k=top_k,
                doc_id_filter=doc_id_filter,
            )

            if not results:
                status = (
                    GraphSearchStatus.AMBIGUOUS_ENTITY
                    if linking_res.has_ambiguity
                    else GraphSearchStatus.NO_GRAPH_RESULTS
                )
                msg = (
                    "Ambiguous entity candidates found but no connecting paths satisfied constraints."
                    if linking_res.has_ambiguity
                    else "No relational paths found matching the specified hops, direction, and predicate filters."
                )
                return GraphSearchResponse(
                    query=clean_query,
                    status=status,
                    message=msg,
                    extraction_result=extraction_res,
                    linked_entities=linking_res.linked_entities,
                    statistics={
                        "entities_extracted": len(extraction_res.entities),
                        "entities_linked": len(linking_res.linked_entities),
                        "has_ambiguity": linking_res.has_ambiguity,
                        "paths_retrieved": 0,
                    },
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                )

            # 6. Structured Context Construction
            context_evidence = self.context_builder.build_context(results)

            dt_ms = (time.perf_counter() - t0) * 1000.0
            stats = {
                "entities_extracted": len(extraction_res.entities),
                "entities_linked": len(linking_res.linked_entities),
                "has_ambiguity": linking_res.has_ambiguity,
                "paths_retrieved": len(results),
                "max_hops_configured": effective_hops,
                "direction": direction,
                "evidence_quotes_count": len(context_evidence.evidence_quotes),
                "extraction_duration_ms": extraction_res.duration_ms,
                "llm_invoked": extraction_res.llm_invoked,
            }

            return GraphSearchResponse(
                query=clean_query,
                status=GraphSearchStatus.SUCCESS,
                message=f"Retrieved {len(results)} ranked graph paths across {len(linking_res.linked_entities)} seed entities.",
                extraction_result=extraction_res,
                linked_entities=linking_res.linked_entities,
                results=results,
                structured_context=context_evidence.formatted_context,
                context_evidence=context_evidence,
                statistics=stats,
                latency_ms=round(dt_ms, 2),
            )

        except Exception as exc:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            logger.error("Internal error during graph search for '%s': %s", clean_query, exc)
            return GraphSearchResponse(
                query=clean_query,
                status=GraphSearchStatus.INTERNAL_ERROR,
                message=f"Internal graph search error: {exc}",
                latency_ms=round(dt_ms, 2),
            )


# Global singleton instance helper
_search_engine_instance: GraphSearchEngine | None = None


def get_graph_search_engine() -> GraphSearchEngine:
    """Return singleton GraphSearchEngine instance."""
    global _search_engine_instance
    if _search_engine_instance is None:
        _search_engine_instance = GraphSearchEngine()
    return _search_engine_instance


def search_graph(
    query: str,
    max_hops: int | None = None,
    direction: Literal["both", "out", "in"] = "both",
    top_k: int = 10,
    predicates: list[str] | set[str] | None = None,
    doc_id_filter: str | None = None,
    store: GraphStore | None = None,
) -> GraphSearchResponse:
    """Convenience top-level entry point for executing graph search."""
    if store is not None:
        engine = GraphSearchEngine(store=store)
    else:
        engine = get_graph_search_engine()

    return engine.search_graph(
        query=query,
        max_hops=max_hops,
        direction=direction,
        top_k=top_k,
        predicates=predicates,
        doc_id_filter=doc_id_filter,
    )
