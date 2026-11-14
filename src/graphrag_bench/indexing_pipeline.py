"""Embedding indexing pipeline for Graph RAG Benchmark.

Orchestrates transforming Phase 2 persisted chunks into vector embeddings and storing
them in ChromaDB with:
- Canonical single source of truth (DocumentRepository)
- Incremental indexing (skips already-indexed chunk IDs)
- Controlled batching and progress tracking
- Model compatibility checks (detects model/dimension changes to prevent index corruption)
- Full re-indexing support for individual documents or entire collection
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from graphrag_bench.document_repository import DocumentRepository
from graphrag_bench.embedding_service import EmbeddingService
from graphrag_bench.logger import get_logger
from graphrag_bench.schemas import DocumentChunk
from graphrag_bench.vector_store import VectorRecord, VectorStore

logger = get_logger("indexing_pipeline")


class IndexingError(Exception):
    """Base exception for indexing pipeline failures."""


class ModelCompatibilityError(IndexingError):
    """Raised when an indexing request attempts to use a different model without re-indexing."""


@dataclass
class IndexingStats:
    """Detailed metrics resulting from an indexing pipeline run."""

    total_chunks_considered: int = 0
    newly_indexed_chunks: int = 0
    already_indexed_chunks: int = 0
    failed_chunks: int = 0
    model: str = ""
    dimensions: int = 0
    duration_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def is_success(self) -> bool:
        """True if no chunk indexing operations failed."""
        return self.failed_chunks == 0 and len(self.errors) == 0


class IndexingPipeline:
    """Manages the indexing of DocumentChunks into ChromaDB."""

    def __init__(
        self,
        repository: DocumentRepository | None = None,
        embedding_service: EmbeddingService | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        """Initialize indexing pipeline with storage, embedding, and vector services.

        Args:
            repository: Source DocumentRepository (defaults to standard project repository).
            embedding_service: Embedding generation service.
            vector_store: Target ChromaDB persistent vector store.
        """
        self.repository = repository or DocumentRepository()
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or VectorStore()

    def get_indexing_status(self) -> dict:
        """Inspect current alignment between Phase 2 chunks and Phase 3 vectors."""
        all_docs = self.repository.list_documents()
        total_chunks = sum(d.chunk_count for d in all_docs)
        indexed_ids = set(self.vector_store.get_all_indexed_chunk_ids())

        all_canonical_chunk_ids = set()
        for d in all_docs:
            for c in self.repository.get_chunks(d.document_id):
                all_canonical_chunk_ids.add(c.chunk_id)

        pending_ids = all_canonical_chunk_ids - indexed_ids

        return {
            "total_documents": len(all_docs),
            "total_canonical_chunks": total_chunks,
            "indexed_vector_count": self.vector_store.count(),
            "pending_chunk_count": len(pending_ids),
            "collection_name": self.vector_store.collection_name,
            "persist_directory": self.vector_store.persist_directory,
            "configured_model": self.embedding_service.default_model,
        }

    def index_all(
        self,
        force_reindex: bool = False,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> IndexingStats:
        """Index all unindexed canonical document chunks from DocumentRepository.

        Args:
            force_reindex: If True, resets the Chroma collection and re-indexes all chunks from scratch.
            progress_callback: Optional callable accepting (progress_ratio: float, status_text: str).

        Returns:
            IndexingStats summarizing execution results.
        """
        if force_reindex:
            logger.warning("force_reindex=True requested; resetting ChromaDB collection...")
            self.vector_store.reset_collection()

        all_docs = self.repository.list_documents()
        all_chunks: list[DocumentChunk] = []
        for d in all_docs:
            all_chunks.extend(self.repository.get_chunks(d.document_id))

        return self._index_chunks_internal(all_chunks, progress_callback=progress_callback)

    def index_document(
        self,
        document_id: str,
        force_reindex: bool = False,
    ) -> IndexingStats:
        """Index chunks belonging to a specific document ID.

        Args:
            document_id: Unique document identifier to index.
            force_reindex: If True, deletes existing vectors for this document before re-indexing.

        Returns:
            IndexingStats summarizing execution results.
        """
        if force_reindex:
            self.vector_store.delete_by_document_id(document_id)

        doc_chunks = self.repository.get_chunks(document_id)
        return self._index_chunks_internal(doc_chunks)

    def _index_chunks_internal(
        self,
        chunks: list[DocumentChunk],
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> IndexingStats:
        """Execute core batch embedding and vector insertion."""
        start_time = time.perf_counter()
        stats = IndexingStats(
            total_chunks_considered=len(chunks),
            model=self.embedding_service.default_model,
        )

        if not chunks:
            stats.duration_ms = (time.perf_counter() - start_time) * 1000.0
            return stats

        # 1. Identify already-indexed chunks to avoid duplicate embedding calls
        already_indexed = set(self.vector_store.get_all_indexed_chunk_ids())
        to_index: list[DocumentChunk] = [c for c in chunks if c.chunk_id not in already_indexed]
        stats.already_indexed_chunks = len(chunks) - len(to_index)

        if not to_index:
            logger.info("All %d chunks already indexed in ChromaDB; skipping.", len(chunks))
            if progress_callback:
                progress_callback(1.0, "All chunks are already up to date.")
            stats.duration_ms = (time.perf_counter() - start_time) * 1000.0
            return stats

        batch_size = self.embedding_service.default_batch_size
        total_batches = (len(to_index) + batch_size - 1) // batch_size

        logger.info(
            "Starting indexing for %d chunks in %d batches (model: '%s')",
            len(to_index),
            total_batches,
            stats.model,
        )

        for batch_idx in range(total_batches):
            start_i = batch_idx * batch_size
            end_i = min(start_i + batch_size, len(to_index))
            batch_chunks = to_index[start_i:end_i]

            status_msg = (
                f"Embedding batch {batch_idx + 1}/{total_batches} ({len(batch_chunks)} chunks)..."
            )
            if progress_callback:
                progress_callback(batch_idx / total_batches, status_msg)

            try:
                # Generate embeddings for batch
                texts = [c.text for c in batch_chunks]
                embed_res = self.embedding_service.embed_batch(texts, model=stats.model)
                stats.dimensions = embed_res.dimensions

                # Build VectorRecord models
                records: list[VectorRecord] = []
                for c, vec in zip(batch_chunks, embed_res.vectors, strict=True):
                    meta = {
                        "document_id": c.document_id,
                        "chunk_id": c.chunk_id,
                        "chunk_index": c.chunk_index,
                        "source_filename": c.source_filename,
                        "file_type": c.file_type,
                        "word_count": c.word_count,
                        "char_count": c.char_count,
                        "start_char": c.start_char,
                        "end_char": c.end_char,
                    }
                    if c.page_number is not None:
                        meta["page_number"] = c.page_number

                    records.append(
                        VectorRecord(
                            chunk_id=c.chunk_id,
                            document_id=c.document_id,
                            text=c.text,
                            embedding=vec,
                            metadata=meta,
                        )
                    )

                self.vector_store.add_records(records)
                stats.newly_indexed_chunks += len(records)

            except Exception as exc:
                err_msg = f"Failed batch {batch_idx + 1}: {exc}"
                logger.error(err_msg)
                stats.errors.append(err_msg)
                stats.failed_chunks += len(batch_chunks)

        if progress_callback:
            progress_callback(1.0, "Indexing completed.")

        stats.duration_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            "Indexing finished in %.1fms. New: %d, Cached: %d, Failed: %d",
            stats.duration_ms,
            stats.newly_indexed_chunks,
            stats.already_indexed_chunks,
            stats.failed_chunks,
        )
        return stats
