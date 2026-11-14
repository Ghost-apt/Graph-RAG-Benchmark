"""ChromaDB persistent vector store interface for Graph RAG Benchmark.

Manages persistent on-disk vector storage for document chunk embeddings using:
- ChromaDB PersistentClient
- Cosine distance metric ("hnsw:space": "cosine")
- Strict metadata sanitization to supported primitive types
- Preservation of Phase 2 chunk_id, document_id, and citation offsets
- Nearest-neighbor similarity search with optional metadata filtering
"""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from graphrag_bench.config import Settings, get_settings
from graphrag_bench.logger import get_logger

logger = get_logger("vector_store")


class VectorStoreError(Exception):
    """Base exception for ChromaDB vector store operations."""


@dataclass
class VectorRecord:
    """Represents a vector record to be stored or retrieved from ChromaDB."""

    chunk_id: str
    document_id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any]


@dataclass
class SearchResultItem:
    """A single retrieved neighbor chunk from semantic similarity search."""

    rank: int
    chunk_id: str
    document_id: str
    text: str
    distance: float
    cosine_similarity: float = 0.0  # Raw cosine similarity: 1.0 - distance in [-1.0, 1.0]
    normalized_score: float = 0.0  # Normalized score: 1.0 - (distance / 2.0) in [0.0, 1.0]
    metadata: dict[str, Any] = field(default_factory=dict)
    similarity_score: float = 0.0  # Backward-compatible alias for normalized_score

    def __post_init__(self) -> None:
        """Ensure cosine_similarity, normalized_score, and similarity_score are consistent."""
        if self.cosine_similarity == 0.0 and self.normalized_score == 0.0:
            if self.similarity_score != 0.0:
                self.normalized_score = self.similarity_score
                self.cosine_similarity = round(max(-1.0, min(1.0, 1.0 - self.distance)), 4)
            else:
                self.cosine_similarity = round(max(-1.0, min(1.0, 1.0 - self.distance)), 4)
                self.normalized_score = round(max(0.0, min(1.0, 1.0 - (self.distance / 2.0))), 4)
                self.similarity_score = self.normalized_score
        elif self.similarity_score == 0.0:
            self.similarity_score = self.normalized_score


class VectorStore:
    """Persistent ChromaDB vector database manager with cosine distance."""

    def __init__(
        self,
        persist_directory: Path | str | None = None,
        collection_name: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize ChromaDB persistent client and collection.

        Args:
            persist_directory: Optional directory path for SQLite and index files.
            collection_name: Optional collection name override.
            settings: Optional application settings.
        """
        self.settings = settings or get_settings()
        self.persist_directory = str(persist_directory or self.settings.chroma_path)
        self.collection_name = collection_name or self.settings.chroma_collection_name

        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        logger.debug(
            "Initializing ChromaDB client at '%s' (collection: '%s')",
            self.persist_directory,
            self.collection_name,
        )

        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(
                    is_persistent=True,
                    anonymized_telemetry=False,
                ),
            )
            # Create or get collection with cosine similarity space
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            logger.error("Failed to initialize ChromaDB PersistentClient: %s", exc)
            raise VectorStoreError(f"ChromaDB initialization failure: {exc}") from exc

    def count(self) -> int:
        """Return total number of vectors in the active collection."""
        try:
            return self.collection.count()
        except Exception as exc:
            logger.error("Failed to count vectors: %s", exc)
            raise VectorStoreError(f"Error querying collection count: {exc}") from exc

    def add_records(self, records: list[VectorRecord]) -> int:
        """Add new vector records to the collection.

        Args:
            records: List of VectorRecord instances to insert.

        Returns:
            Number of successfully inserted records.
        """
        if not records:
            return 0

        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for r in records:
            ids.append(r.chunk_id)
            embeddings.append(r.embedding)
            documents.append(r.text)
            clean_meta = self._sanitize_metadata(r.metadata, r.document_id, r.chunk_id)
            metadatas.append(clean_meta)

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,  # type: ignore[arg-type]
                documents=documents,
                metadatas=metadatas,  # type: ignore[arg-type]
            )
            logger.info(
                "Added %d vector records to collection '%s'", len(ids), self.collection_name
            )
            return len(ids)
        except Exception as exc:
            logger.error("Failed to add vector records to ChromaDB: %s", exc)
            raise VectorStoreError(f"Error adding vector records: {exc}") from exc

    def upsert_records(self, records: list[VectorRecord]) -> int:
        """Upsert (insert or update) vector records in the collection."""
        if not records:
            return 0

        ids = [r.chunk_id for r in records]
        embeddings = [r.embedding for r in records]
        documents = [r.text for r in records]
        metadatas = [
            self._sanitize_metadata(r.metadata, r.document_id, r.chunk_id) for r in records
        ]

        try:
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,  # type: ignore[arg-type]
                documents=documents,
                metadatas=metadatas,  # type: ignore[arg-type]
            )
            logger.info(
                "Upserted %d vector records in collection '%s'", len(ids), self.collection_name
            )
            return len(ids)
        except Exception as exc:
            logger.error("Failed to upsert records in ChromaDB: %s", exc)
            raise VectorStoreError(f"Error upserting vector records: {exc}") from exc

    def get_by_id(self, chunk_id: str) -> VectorRecord | None:
        """Retrieve a vector record and its metadata by chunk ID."""
        try:
            res = self.collection.get(
                ids=[chunk_id],
                include=["embeddings", "documents", "metadatas"],  # type: ignore[list-item]
            )
            if not res["ids"] or not res["ids"][0]:
                return None

            cid = res["ids"][0]
            emb = res["embeddings"][0] if res.get("embeddings") is not None else []
            doc = res["documents"][0] if res.get("documents") is not None else ""
            meta = res["metadatas"][0] if res.get("metadatas") is not None else {}

            return VectorRecord(
                chunk_id=cid,
                document_id=str(meta.get("document_id", "")),
                text=doc,
                embedding=list(emb),
                metadata=dict(meta),
            )
        except Exception as exc:
            logger.error("Error retrieving chunk ID '%s': %s", chunk_id, exc)
            raise VectorStoreError(f"Failed to get record '{chunk_id}': {exc}") from exc

    def delete_by_ids(self, chunk_ids: list[str]) -> int:
        """Delete vector records by their chunk IDs."""
        if not chunk_ids:
            return 0

        try:
            self.collection.delete(ids=chunk_ids)
            logger.info("Deleted %d vector records from ChromaDB", len(chunk_ids))
            return len(chunk_ids)
        except Exception as exc:
            logger.error("Failed to delete chunk IDs %s: %s", chunk_ids, exc)
            raise VectorStoreError(f"Error deleting records: {exc}") from exc

    def delete_by_document_id(self, document_id: str) -> int:
        """Delete all chunk vectors originating from a specific parent document ID."""
        try:
            records = self.collection.get(
                where={"document_id": document_id},
                include=[],
            )
            ids_to_del = records.get("ids", [])
            if ids_to_del:
                self.collection.delete(ids=ids_to_del)
                logger.info("Deleted %d vectors for document_id '%s'", len(ids_to_del), document_id)
                return len(ids_to_del)
            return 0
        except Exception as exc:
            logger.error("Failed to delete vectors for document_id '%s': %s", document_id, exc)
            raise VectorStoreError(f"Error deleting vectors by document_id: {exc}") from exc

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where_filter: dict[str, Any] | None = None,
    ) -> list[SearchResultItem]:
        """Perform nearest-neighbor cosine similarity search.

        Args:
            query_embedding: Vector representation of the search query.
            top_k: Maximum number of closest chunks to retrieve.
            where_filter: Optional metadata filter dict (e.g., {"file_type": "pdf"}).

        Returns:
            List of SearchResultItem ordered from most similar to least similar.
        """
        if self.count() == 0:
            return []

        k = max(1, min(top_k, self.count()))

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            kwargs["where"] = where_filter

        try:
            res = self.collection.query(**kwargs)
        except Exception as exc:
            logger.error("ChromaDB query failure: %s", exc)
            raise VectorStoreError(f"Similarity search failed: {exc}") from exc

        results: list[SearchResultItem] = []
        if not res.get("ids") or not res["ids"][0]:
            return results

        chunk_ids = res["ids"][0]
        distances = res["distances"][0] if res.get("distances") is not None else []
        documents = res["documents"][0] if res.get("documents") is not None else []
        metadatas = res["metadatas"][0] if res.get("metadatas") is not None else []

        for idx, cid in enumerate(chunk_ids):
            raw_dist = distances[idx] if idx < len(distances) else 0.0
            doc = str(documents[idx]) if idx < len(documents) else ""
            meta = dict(metadatas[idx]) if idx < len(metadatas) else {}

            # Handle missing or invalid distance values safely
            try:
                dist = float(raw_dist)
                if math.isnan(dist) or math.isinf(dist):
                    dist = 1.0  # Neutral fallback (orthogonal)
            except (TypeError, ValueError):
                dist = 1.0

            # Cosine distance in ChromaDB ("hnsw:space": "cosine") is defined as:
            # distance = 1.0 - cosine_similarity, ranging in [0.0, 2.0]
            # 1. Raw cosine similarity: 1.0 - distance in [-1.0, 1.0]
            raw_cosine_sim = max(-1.0, min(1.0, 1.0 - dist))

            # 2. Normalized cosine score: 1.0 - (distance / 2.0) in [0.0, 1.0]
            normalized_score = max(0.0, min(1.0, 1.0 - (dist / 2.0)))

            results.append(
                SearchResultItem(
                    rank=idx + 1,
                    chunk_id=cid,
                    document_id=str(meta.get("document_id", "")),
                    text=doc,
                    distance=round(dist, 4),
                    cosine_similarity=round(raw_cosine_sim, 4),
                    normalized_score=round(normalized_score, 4),
                    metadata=meta,
                    similarity_score=round(normalized_score, 4),
                )
            )

        return results

    def get_all_indexed_chunk_ids(self) -> list[str]:
        """Return a complete list of all chunk IDs currently stored in the collection."""
        try:
            res = self.collection.get(include=[])
            return res.get("ids", [])
        except Exception as exc:
            logger.error("Error retrieving indexed chunk IDs: %s", exc)
            return []

    def reset_collection(self) -> None:
        """Safely delete and recreate the active collection (leaves Phase 2 data untouched)."""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("ChromaDB collection '%s' reset successfully", self.collection_name)
        except Exception as exc:
            logger.error("Failed to reset collection '%s': %s", self.collection_name, exc)
            raise VectorStoreError(f"Error resetting collection: {exc}") from exc

    def _sanitize_metadata(
        self,
        metadata: dict[str, Any],
        document_id: str,
        chunk_id: str,
    ) -> dict[str, str | int | float | bool]:
        """Convert metadata values into primitive types supported by ChromaDB."""
        clean: dict[str, str | int | float | bool] = {
            "document_id": document_id,
            "chunk_id": chunk_id,
        }

        for k, v in metadata.items():
            if v is None:
                continue  # Omit None fields to satisfy ChromaDB typing
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
            else:
                clean[k] = str(v)

        return clean
