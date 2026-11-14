"""Result normalizer for Phase 6 Hybrid Retrieval.

Converts disparate Vector Retrieval outputs (SearchResultItem) and Graph Retrieval outputs
(GraphRetrievalResult / GraphPath) into standardized, canonical chunk-level candidates.
Enforces canonical Phase 2 chunk ID validation, document repository verification,
and intra-modality deduplication.
"""

from typing import Any

from graphrag_bench.document_repository import DocumentRepository
from graphrag_bench.graph_retrieval_schemas import GraphSearchResponse
from graphrag_bench.hybrid_schemas import GraphCandidate, VectorCandidate
from graphrag_bench.logger import get_logger
from graphrag_bench.schemas import DocumentChunk
from graphrag_bench.semantic_search import SearchResponse

logger = get_logger("result_normalizer")


class ResultNormalizer:
    """Normalizes vector and graph retrieval outputs into verified canonical chunk candidates."""

    def __init__(self, repository: DocumentRepository | None = None) -> None:
        """Initialize result normalizer.

        Args:
            repository: Optional DocumentRepository instance to verify chunk existence and text.
        """
        self.repository = repository
        self._chunk_cache: dict[str, DocumentChunk] = {}
        if self.repository is not None:
            self._warm_cache()

    def _warm_cache(self) -> None:
        """Load all canonical chunks into memory for sub-millisecond lookup."""
        if self.repository is None:
            return
        try:
            chunks = self.repository._get_all_chunks()
            for c in chunks:
                self._chunk_cache[c.chunk_id] = c
            logger.debug(
                "ResultNormalizer warmed cache with %d canonical chunks", len(self._chunk_cache)
            )
        except Exception as exc:
            logger.warning("Failed to warm chunk cache from repository: %s", exc)

    def get_canonical_chunk(self, chunk_id: str) -> DocumentChunk | None:
        """Look up canonical chunk by ID from cache or repository."""
        if chunk_id in self._chunk_cache:
            return self._chunk_cache[chunk_id]
        if self.repository is not None:
            # Try reloading cache if missing
            self._warm_cache()
            return self._chunk_cache.get(chunk_id)
        return None

    def normalize_vector_results(
        self,
        search_response: SearchResponse,
    ) -> tuple[list[VectorCandidate], list[str]]:
        """Normalize vector search response into validated VectorCandidate list.

        Args:
            search_response: Output from SemanticSearch.search().

        Returns:
            Tuple of (list of VectorCandidate, list of warning messages).
        """
        candidates: list[VectorCandidate] = []
        warnings: list[str] = []
        seen_chunks: set[str] = set()

        for idx, item in enumerate(search_response.items, start=1):
            chunk_id = item.chunk_id.strip() if item.chunk_id else ""
            if not chunk_id:
                msg = f"Vector search result at index {idx} lacks a valid chunk_id; rejected from fusion."
                logger.warning(msg)
                warnings.append(msg)
                continue

            if chunk_id in seen_chunks:
                # Discard intra-modality duplicate if returned by vector search
                continue
            seen_chunks.add(chunk_id)

            # Resolve canonical text from repository if possible
            canonical_chunk = self.get_canonical_chunk(chunk_id)
            chunk_text = canonical_chunk.text if canonical_chunk else item.text
            doc_id = canonical_chunk.document_id if canonical_chunk else item.document_id
            filename = (
                canonical_chunk.source_filename
                if canonical_chunk
                else str(item.metadata.get("source_filename", ""))
            )
            file_type = (
                canonical_chunk.file_type
                if canonical_chunk
                else str(item.metadata.get("file_type", "txt"))
            )
            page_num = (
                canonical_chunk.page_number if canonical_chunk else item.metadata.get("page_number")
            )

            # Preserve explicit cosine distance, raw cosine similarity, and normalized score
            dist = float(item.distance)
            cos_sim = float(item.cosine_similarity)
            norm_score = float(item.normalized_score)

            candidate = VectorCandidate(
                chunk_id=chunk_id,
                document_id=doc_id,
                source_filename=filename,
                file_type=file_type,
                chunk_text=chunk_text,
                vector_rank=item.rank if item.rank > 0 else idx,
                cosine_distance=dist,
                cosine_similarity=cos_sim,
                normalized_score=norm_score,
                source_page=page_num,
                metadata=item.metadata,
            )
            candidates.append(candidate)

        return candidates, warnings

    def normalize_graph_results(
        self,
        graph_response: GraphSearchResponse,
    ) -> tuple[list[GraphCandidate], list[str]]:
        """Normalize graph search response into chunk-level GraphCandidate list.

        Aggregates multiple paths referencing the same canonical chunk, retains all
        supporting paths and evidence quotes, assigns the best path rank, and verifies
        canonical repository existence.

        Args:
            graph_response: Output from GraphSearchEngine.search_graph().

        Returns:
            Tuple of (list of GraphCandidate, list of warning messages).
        """
        warnings: list[str] = []
        # Intermediate aggregation map: chunk_id -> dict of properties
        chunk_aggregates: dict[str, dict[str, Any]] = {}

        for res in graph_response.results:
            path_obj = res.path
            path_rank = res.rank
            path_score = res.score
            path_dict = path_obj.model_dump(mode="json")
            path_entities = [n.canonical_name for n in path_obj.nodes]

            # Collect unique chunk IDs referenced in this specific path's edges
            path_chunk_ids: list[str] = []
            for edge in path_obj.edges:
                c_id = edge.source_chunk_id.strip() if edge.source_chunk_id else ""
                if c_id and c_id not in path_chunk_ids:
                    path_chunk_ids.append(c_id)

            for c_id in path_chunk_ids:
                # 1. Verify chunk existence in canonical repository
                canonical_chunk = self.get_canonical_chunk(c_id)
                if canonical_chunk is None and self.repository is not None:
                    msg = (
                        f"Rejected unverified graph chunk ID '{c_id}' referenced in "
                        f"path '{path_obj.path_id}': not found in canonical document repository."
                    )
                    logger.warning(msg)
                    if msg not in warnings:
                        warnings.append(msg)
                    continue

                # 2. Extract edge-level quotations for this specific chunk
                chunk_quotes = [
                    edge.evidence_quote.strip()
                    for edge in path_obj.edges
                    if edge.source_chunk_id == c_id and edge.evidence_quote.strip()
                ]

                # 3. Aggregate or initialize chunk entry
                if c_id not in chunk_aggregates:
                    doc_id = canonical_chunk.document_id if canonical_chunk else ""
                    filename = canonical_chunk.source_filename if canonical_chunk else ""
                    text = canonical_chunk.text if canonical_chunk else ""
                    # Fallback to edge metadata if repository wasn't provided
                    if not doc_id:
                        edge_with_doc = next(
                            (e for e in path_obj.edges if e.source_chunk_id == c_id), None
                        )
                        if edge_with_doc:
                            doc_id = edge_with_doc.source_document_id
                            filename = edge_with_doc.source_filename

                    chunk_aggregates[c_id] = {
                        "chunk_id": c_id,
                        "document_id": doc_id,
                        "source_filename": filename,
                        "chunk_text": text,
                        "best_path_rank": path_rank,
                        "best_path_score": path_score,
                        "min_hops": path_obj.hop_count,
                        "matched_entities": set(path_entities),
                        "supporting_paths": [path_dict],
                        "relationship_ids": {
                            e.relationship_id for e in path_obj.edges if e.source_chunk_id == c_id
                        },
                        "evidence_quotes": set(chunk_quotes),
                    }
                else:
                    agg = chunk_aggregates[c_id]
                    # Update best rank & score if this path is higher ranked
                    agg["best_path_rank"] = min(agg["best_path_rank"], path_rank)
                    agg["best_path_score"] = max(agg["best_path_score"], path_score)
                    agg["min_hops"] = min(agg["min_hops"], path_obj.hop_count)
                    agg["matched_entities"].update(path_entities)
                    agg["supporting_paths"].append(path_dict)
                    agg["relationship_ids"].update(
                        e.relationship_id for e in path_obj.edges if e.source_chunk_id == c_id
                    )
                    agg["evidence_quotes"].update(chunk_quotes)

        # 4. Sort aggregated chunks by best_path_rank (ascending), then best_path_score (descending)
        sorted_chunks = sorted(
            chunk_aggregates.values(),
            key=lambda item: (item["best_path_rank"], -item["best_path_score"], item["chunk_id"]),
        )

        # 5. Build final GraphCandidate models with sequential 1-based ranks
        candidates: list[GraphCandidate] = []
        for idx, item in enumerate(sorted_chunks, start=1):
            candidate = GraphCandidate(
                chunk_id=item["chunk_id"],
                document_id=item["document_id"],
                source_filename=item["source_filename"],
                chunk_text=item["chunk_text"],
                graph_rank=idx,
                matched_entities=sorted(item["matched_entities"]),
                supporting_paths=item["supporting_paths"],
                relationship_ids=sorted(item["relationship_ids"]),
                hop_count=item["min_hops"],
                evidence_quotes=sorted(item["evidence_quotes"]),
                graph_score=item["best_path_score"],
            )
            candidates.append(candidate)

        return candidates, warnings
