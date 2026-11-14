"""
Graph Retriever Module.
Coordinates seed node resolution, bounded multi-hop traversal, document filtering,
and evidence assembly into ranked GraphRetrievalResult items.
"""

from typing import Literal

from graphrag_bench.config import Settings, get_settings
from graphrag_bench.entity_linker import EntityLinker
from graphrag_bench.graph_ranker import GraphRanker
from graphrag_bench.graph_retrieval_schemas import (
    GraphPath,
    GraphRetrievalResult,
    LinkedEntity,
)
from graphrag_bench.graph_store import GraphStore
from graphrag_bench.graph_traversal import GraphTraversalEngine
from graphrag_bench.logger import get_logger
from graphrag_bench.query_entity_extractor import QueryEntityExtractor

logger = get_logger("graph_retriever")


class GraphRetriever:
    """Retrieves structured graph paths and grounded evidence from user queries."""

    def __init__(
        self,
        store: GraphStore,
        settings: Settings | None = None,
        extractor: QueryEntityExtractor | None = None,
        linker: EntityLinker | None = None,
        traversal: GraphTraversalEngine | None = None,
        ranker: GraphRanker | None = None,
    ) -> None:
        """Initialize graph retriever.

        Args:
            store: Active GraphStore instance.
            settings: Optional application settings.
            extractor: Query entity extractor.
            linker: Entity linker instance.
            traversal: Graph traversal engine.
            ranker: Graph ranker instance.
        """
        self.store = store
        self.settings = settings or get_settings()
        self.extractor = extractor or QueryEntityExtractor(store=self.store, settings=self.settings)
        self.linker = linker or EntityLinker(store=self.store)
        self.traversal = traversal or GraphTraversalEngine(store=self.store)
        self.ranker = ranker or GraphRanker()

    def retrieve(
        self,
        query: str,
        linked_entities: list[LinkedEntity] | None = None,
        max_hops: int | None = None,
        direction: Literal["both", "out", "in"] | None = None,
        predicates: list[str] | set[str] | None = None,
        top_k: int | None = None,
        doc_id_filter: str | None = None,
    ) -> list[GraphRetrievalResult]:
        """Execute graph retrieval from query or pre-resolved seed entities.

        Args:
            query: User's search query.
            linked_entities: Optional pre-linked entities (bypasses extraction if provided).
            max_hops: Depth cutoff (defaults to settings.graph_max_hops, max 3).
            direction: Traversal direction ('both', 'out', 'in').
            predicates: Optional allowed predicate set.
            top_k: Maximum ranked paths to return (defaults to settings.graph_max_results).
            doc_id_filter: Optional source document ID filter.

        Returns:
            Ranked list of GraphRetrievalResult items with validated evidence quotes.
        """
        hops = max_hops if max_hops is not None else getattr(self.settings, "graph_max_hops", 2)
        hops = max(1, min(3, hops))

        trav_dir = (
            direction
            if direction is not None
            else getattr(self.settings, "graph_traversal_direction", "both")
        )
        k_results = top_k if top_k is not None else getattr(self.settings, "graph_max_results", 10)
        max_paths = getattr(self.settings, "graph_max_paths", 50)
        max_degree = getattr(self.settings, "graph_max_degree", 50)

        # 1. Resolve Linked Entities if not supplied
        active_linked: list[LinkedEntity] = []
        if linked_entities is not None:
            active_linked = linked_entities
        else:
            ext_res = self.extractor.extract_query_entities(query)
            link_res = self.linker.link_entities(ext_res.entities, query=query)
            active_linked = link_res.linked_entities

        if not active_linked:
            logger.info("No linked entities identified for query: '%s'", query)
            return []

        # 2. Traverse from each linked seed entity
        all_candidate_paths: list[GraphPath] = []
        seen_path_fingerprints: set[tuple] = set()

        for l_ent in active_linked:
            seed_id = l_ent.entity_id
            paths = self.traversal.traverse_from_seed(
                seed_entity_id=seed_id,
                max_hops=hops,
                direction=trav_dir,
                predicates=predicates,
                doc_id_filter=doc_id_filter,
                max_paths=max_paths,
                max_degree=max_degree,
            )

            for p in paths:
                # Optional Document ID filtering
                if doc_id_filter and doc_id_filter not in p.source_document_ids:
                    continue

                # Deduplicate identical node-edge sequences
                fingerprint = tuple(
                    (e.subject_entity_id, e.predicate, e.object_entity_id) for e in p.edges
                )
                if fingerprint not in seen_path_fingerprints:
                    seen_path_fingerprints.add(fingerprint)
                    all_candidate_paths.append(p)

        if not all_candidate_paths:
            logger.info("No traversal paths discovered for query: '%s'", query)
            return []

        # 3. Deterministically rank candidate paths
        ranked_paths = self.ranker.rank_paths(
            paths=all_candidate_paths,
            linked_entities=active_linked,
            query=query,
        )

        # 4. Assemble top-k GraphRetrievalResult items with source citations
        results: list[GraphRetrievalResult] = []
        for idx, path in enumerate(ranked_paths[:k_results], start=1):
            quotes = [e.evidence_quote for e in path.edges if e.evidence_quote]
            filenames = list(
                dict.fromkeys(e.source_filename for e in path.edges if e.source_filename)
            )
            _, _, explanation = self.ranker.score_path(
                path,
                {e.entity_id: e for e in active_linked},
                {t.lower() for t in query.split() if len(t) > 2},
            )

            results.append(
                GraphRetrievalResult(
                    result_id=f"res_{idx:03d}",
                    rank=idx,
                    seed_entity_name=path.seed_entity_name,
                    path=path,
                    score=path.score,
                    score_explanation=explanation,
                    source_document_ids=path.source_document_ids,
                    source_chunk_ids=path.source_chunk_ids,
                    source_filenames=filenames,
                    evidence_quotes=quotes,
                )
            )

        return results
