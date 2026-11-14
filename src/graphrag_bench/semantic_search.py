"""Semantic search retrieval service for Graph RAG Benchmark.

Implements nearest-neighbor dense vector retrieval:
- Natural-language query validation
- Query embedding generation via EmbeddingService
- Cosine similarity ranking against ChromaDB VectorStore
- Top-K result filtering (by document ID, by file type)
- Distinct evaluation of raw cosine similarity (1 - distance) and normalized score (1 - distance/2)
- Rich citation metadata and citation offsets
"""

import time
from dataclasses import dataclass
from typing import Any

from graphrag_bench.config import Settings, get_settings
from graphrag_bench.embedding_service import EmbeddingError, EmbeddingService, EmptyTextError
from graphrag_bench.logger import get_logger
from graphrag_bench.vector_store import SearchResultItem, VectorStore, VectorStoreError

logger = get_logger("semantic_search")


class SearchError(Exception):
    """Base exception for semantic search query failures."""


@dataclass
class SearchResponse:
    """Complete container for semantic search query output."""

    query: str
    top_k: int
    total_found: int
    duration_ms: float
    items: list[SearchResultItem]
    model_used: str
    applied_filter: dict[str, Any] | None = None


class SemanticSearch:
    """Orchestrates natural language vector retrieval against ChromaDB."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embedding_service: EmbeddingService | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize semantic search engine.

        Args:
            vector_store: Target ChromaDB vector store.
            embedding_service: Embedding generation service.
            settings: Optional application settings.
        """
        self.settings = settings or get_settings()
        self.vector_store = vector_store or VectorStore(settings=self.settings)
        self.embedding_service = embedding_service or EmbeddingService(settings=self.settings)
        self.default_top_k = self.settings.retrieval_top_k

    def search(
        self,
        query: str,
        top_k: int | None = None,
        document_id: str | None = None,
        file_type: str | None = None,
    ) -> SearchResponse:
        """Execute semantic retrieval for a natural-language query.

        Args:
            query: User's search query string.
            top_k: Number of most similar chunks to return (default settings.retrieval_top_k).
            document_id: Optional filter to restrict results to a specific document.
            file_type: Optional filter to restrict results to a specific file format ('pdf', 'docx', 'txt', 'md').

        Returns:
            SearchResponse containing ranked SearchResultItem objects and metadata.

        Raises:
            EmptyTextError: If query is blank or empty.
            SearchError: If embedding generation or ChromaDB retrieval fails.
        """
        if not query or not query.strip():
            raise EmptyTextError("Search query cannot be empty or whitespace-only.")

        clean_query = query.strip()
        k = max(1, top_k or self.default_top_k)

        start_time = time.perf_counter()
        logger.debug("Executing semantic search for query: '%s' (top_k=%d)", clean_query, k)

        # Handle empty vector database gracefully
        if self.vector_store.count() == 0:
            logger.warning("Search executed on empty vector database.")
            return SearchResponse(
                query=clean_query,
                top_k=k,
                total_found=0,
                duration_ms=(time.perf_counter() - start_time) * 1000.0,
                items=[],
                model_used=self.embedding_service.default_model,
            )

        # 1. Compute query vector embedding
        try:
            query_vector = self.embedding_service.embed_text(clean_query)
        except EmbeddingError as exc:
            logger.error("Failed to generate query embedding: %s", exc)
            raise SearchError(f"Query embedding generation failed: {exc}") from exc

        # 2. Build where filter clause if filtering is specified
        where_filter: dict[str, Any] = {}
        if document_id:
            where_filter["document_id"] = document_id
        if file_type:
            where_filter["file_type"] = file_type.lower()

        # ChromaDB requires combining multiple filter clauses with $and
        final_where: dict[str, Any] | None = None
        if len(where_filter) == 1:
            final_where = where_filter
        elif len(where_filter) > 1:
            final_where = {"$and": [{k: v} for k, v in where_filter.items()]}

        # 3. Query vector store
        try:
            items = self.vector_store.similarity_search(
                query_embedding=query_vector,
                top_k=k,
                where_filter=final_where,
            )
        except VectorStoreError as exc:
            logger.error("ChromaDB vector search failed: %s", exc)
            raise SearchError(f"Vector search execution failed: {exc}") from exc

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            "Semantic search found %d items in %.1fms for query '%s'",
            len(items),
            duration_ms,
            clean_query,
        )

        return SearchResponse(
            query=clean_query,
            top_k=k,
            total_found=len(items),
            duration_ms=duration_ms,
            items=items,
            model_used=self.embedding_service.default_model,
            applied_filter=final_where,
        )
