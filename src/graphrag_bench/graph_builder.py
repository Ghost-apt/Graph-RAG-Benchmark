"""Knowledge Graph Construction Pipeline.

Coordinates:
- DocumentRepository: Ingestion chunk access.
- GraphExtractionService: LLM-driven structured extraction from text chunks.
- EntityResolver: Deduplication, disambiguation, alias indexing, and canonical resolution.
- GraphStore: NetworkX MultiDiGraph storage, statistics, and serialization.
- Manifest tracking: Resilient incremental processing and failure recovery.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from graphrag_bench.config import get_settings
from graphrag_bench.document_repository import DocumentRepository
from graphrag_bench.entity_resolver import EntityResolver
from graphrag_bench.graph_extraction_service import GraphExtractionService
from graphrag_bench.graph_schemas import GraphRelationship
from graphrag_bench.graph_store import GraphStore
from graphrag_bench.schemas import DocumentChunk

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Orchestrates end-to-end Knowledge Graph construction from document chunks."""

    def __init__(
        self,
        doc_repo: DocumentRepository | None = None,
        extraction_service: GraphExtractionService | None = None,
        resolver: EntityResolver | None = None,
        store: GraphStore | None = None,
        graph_dir: Path | str | None = None,
    ) -> None:
        settings = get_settings()
        self.doc_repo = doc_repo or DocumentRepository()
        self.extraction_service = extraction_service or GraphExtractionService(
            model_name=settings.graph_extraction_model,
            min_confidence=settings.graph_min_confidence,
        )
        self.resolver = resolver or EntityResolver()
        self.store = store or GraphStore()

        self.graph_dir = Path(graph_dir or settings.graph_dir)
        self.graph_dir.mkdir(parents=True, exist_ok=True)

        self.graph_path = self.graph_dir / "knowledge_graph.json"
        self.graphml_path = self.graph_dir / "knowledge_graph.graphml"
        self.manifest_path = self.graph_dir / "manifest.json"
        self.extractions_path = self.graph_dir / "extractions.jsonl"

        self._manifest: dict[str, Any] = self._load_or_create_manifest()

        # Load existing graph into store and resolver if present on disk
        if self.graph_path.exists():
            try:
                self.store.load_json(self.graph_path)
                self.resolver.load_entities(self.store.get_entities())
                logger.info(
                    "Initialized GraphBuilder with %d nodes, %d edges from %s",
                    self.store.node_count,
                    self.store.edge_count,
                    self.graph_path,
                )
            except Exception as exc:
                logger.error("Failed to load existing graph from %s: %s", self.graph_path, exc)

    def _load_or_create_manifest(self) -> dict[str, Any]:
        """Load manifest from disk or initialize empty manifest structure."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        data.setdefault("processed_chunk_ids", [])
                        data.setdefault("failed_chunk_ids", {})
                        data.setdefault("document_chunk_mapping", {})
                        return data
            except Exception as exc:
                logger.warning(
                    "Could not read graph manifest %s: %s. Rebuilding.", self.manifest_path, exc
                )

        now = datetime.now(UTC).isoformat()
        return {
            "schema_version": "1.0.0",
            "model": self.extraction_service.model_name,
            "created_at": now,
            "updated_at": now,
            "processed_chunk_ids": [],
            "failed_chunk_ids": {},
            "document_chunk_mapping": {},
            "total_entities": 0,
            "total_relationships": 0,
            "total_chunks_processed": 0,
        }

    def _save_manifest(self) -> None:
        """Atomically persist manifest file."""
        self._manifest["updated_at"] = datetime.now(UTC).isoformat()
        self._manifest["total_entities"] = self.store.node_count
        self._manifest["total_relationships"] = self.store.edge_count
        self._manifest["total_chunks_processed"] = len(self._manifest["processed_chunk_ids"])

        temp_manifest = self.manifest_path.with_suffix(".tmp")
        with open(temp_manifest, "w", encoding="utf-8") as f:
            json.dump(self._manifest, f, indent=2, ensure_ascii=False)
        temp_manifest.replace(self.manifest_path)

    def _process_chunk(self, chunk: DocumentChunk, doc_filename: str = "") -> tuple[int, int]:
        """Extract and resolve entities and relationships for a single chunk."""
        extraction = self.extraction_service.extract_from_chunk(chunk=chunk)

        # 1. Resolve extracted entities
        for ent in extraction.entities:
            resolved_entity = self.resolver.resolve_entity(
                mention_text=ent.mention_text,
                canonical_name=ent.canonical_name,
                entity_type=ent.entity_type,
                aliases=ent.aliases,
                doc_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                description=ent.description,
            )
            self.store.add_entity(resolved_entity)

        # 2. Resolve relationships and ground edges
        added_relationships_count = 0
        for rel in extraction.relationships:
            sub = self.resolver.find_entity(rel.subject)
            if not sub:
                sub = self.resolver.resolve_entity(
                    mention_text=rel.subject,
                    canonical_name=rel.subject,
                    entity_type="OTHER",
                    doc_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                )
                self.store.add_entity(sub)

            obj = self.resolver.find_entity(rel.object)
            if not obj:
                obj = self.resolver.resolve_entity(
                    mention_text=rel.object,
                    canonical_name=rel.object,
                    entity_type="OTHER",
                    doc_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                )
                self.store.add_entity(obj)

            rel_digest = hashlib.sha256(
                f"{sub.entity_id}:{rel.predicate}:{obj.entity_id}:{chunk.chunk_id}".encode()
            ).hexdigest()[:16]
            relationship_id = f"rel_{rel_digest}"

            graph_relationship = GraphRelationship(
                relationship_id=relationship_id,
                subject_entity_id=sub.entity_id,
                predicate=rel.predicate,
                object_entity_id=obj.entity_id,
                evidence_quote=rel.evidence_quote,
                confidence=rel.confidence,
                source_document_id=chunk.document_id,
                source_chunk_id=chunk.chunk_id,
                source_filename=doc_filename,
                extraction_model=self.extraction_service.model_name,
            )
            self.store.add_relationship(graph_relationship)
            added_relationships_count += 1

        # 3. Log extraction to extractions.jsonl
        log_entry = {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "document_filename": doc_filename,
            "extracted_at": datetime.now(UTC).isoformat(),
            "entities_count": len(extraction.entities),
            "relationships_count": len(extraction.relationships),
            "extraction_duration_ms": extraction.extraction_duration_ms,
            "entities": [e.model_dump() for e in extraction.entities],
            "relationships": [r.model_dump() for r in extraction.relationships],
        }
        with open(self.extractions_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        # 4. Update manifest tracking
        processed_set = set(self._manifest["processed_chunk_ids"])
        if chunk.chunk_id not in processed_set:
            self._manifest["processed_chunk_ids"].append(chunk.chunk_id)

        # Clear from failed chunks if it was previously failed
        self._manifest["failed_chunk_ids"].pop(chunk.chunk_id, None)

        # Update document-chunk mapping
        mapping = self._manifest["document_chunk_mapping"]
        if chunk.document_id not in mapping:
            mapping[chunk.document_id] = []
        if chunk.chunk_id not in mapping[chunk.document_id]:
            mapping[chunk.document_id].append(chunk.chunk_id)

        return len(extraction.entities), added_relationships_count

    def build_all(
        self,
        force_rebuild: bool = False,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> dict[str, Any]:
        """Build or incrementally update Knowledge Graph from all repository documents.

        Args:
            force_rebuild: If True, resets graph, clears extractions and manifest, and rebuilds.
            progress_callback: Optional callback(current, total, status_message).

        Returns:
            Dictionary summarizing construction metrics.
        """
        if force_rebuild:
            logger.info("Force rebuild requested. Resetting Knowledge Graph state.")
            self.store.reset()
            self.resolver.clear()
            self._manifest = self._load_or_create_manifest()
            self._manifest["processed_chunk_ids"] = []
            self._manifest["failed_chunk_ids"] = {}
            self._manifest["document_chunk_mapping"] = {}
            if self.extractions_path.exists():
                self.extractions_path.unlink()

        documents = self.doc_repo.list_documents()
        processed_chunks_set = set(self._manifest["processed_chunk_ids"])

        # Gather target chunks to process
        chunks_to_process: list[tuple[DocumentChunk, str]] = []
        for doc in documents:
            chunks = self.doc_repo.get_chunks(doc.document_id)
            for ch in chunks:
                if force_rebuild or ch.chunk_id not in processed_chunks_set:
                    chunks_to_process.append((ch, doc.filename))

        total_to_process = len(chunks_to_process)
        logger.info(
            "Graph construction started: %d chunks to process across %d documents",
            total_to_process,
            len(documents),
        )

        successful_chunks = 0
        failed_chunks = 0

        for idx, (chunk, filename) in enumerate(chunks_to_process, start=1):
            if progress_callback:
                progress_callback(
                    idx,
                    total_to_process,
                    f"Processing chunk {idx}/{total_to_process} ({filename})...",
                )

            try:
                self._process_chunk(chunk, doc_filename=filename)
                successful_chunks += 1
            except Exception as exc:
                failed_chunks += 1
                err_msg = str(exc)
                logger.error("Error extracting from chunk %s: %s", chunk.chunk_id, err_msg)
                self._manifest["failed_chunk_ids"][chunk.chunk_id] = err_msg

        # Persist updated graph and metadata
        self.store.save_json(self.graph_path)
        self.store.export_graphml(self.graphml_path)
        self._save_manifest()

        return {
            "total_documents": len(documents),
            "chunks_inspected": total_to_process,
            "chunks_successful": successful_chunks,
            "chunks_failed": failed_chunks,
            "total_processed_chunks": len(self._manifest["processed_chunk_ids"]),
            "total_entities": self.store.node_count,
            "total_relationships": self.store.edge_count,
            "graph_json_path": str(self.graph_path),
            "graphml_path": str(self.graphml_path),
        }

    def build_document(self, doc_id: str, force_rebuild: bool = False) -> dict[str, Any]:
        """Build or update graph for a specific document."""
        doc = self.doc_repo.get_document(doc_id)
        if not doc:
            raise ValueError(f"Document ID {doc_id} not found in repository.")

        if force_rebuild:
            self.remove_document(doc_id)

        chunks = self.doc_repo.get_chunks(doc_id)
        processed_set = set(self._manifest["processed_chunk_ids"])
        target_chunks = [c for c in chunks if force_rebuild or c.chunk_id not in processed_set]

        successful = 0
        failed = 0
        for ch in target_chunks:
            try:
                self._process_chunk(ch, doc_filename=doc.filename)
                successful += 1
            except Exception as exc:
                failed += 1
                self._manifest["failed_chunk_ids"][ch.chunk_id] = str(exc)

        self.store.save_json(self.graph_path)
        self.store.export_graphml(self.graphml_path)
        self._save_manifest()

        return {
            "document_id": doc_id,
            "filename": doc.filename,
            "chunks_processed": successful,
            "chunks_failed": failed,
            "total_entities": self.store.node_count,
            "total_relationships": self.store.edge_count,
        }

    def retry_failed(
        self,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> dict[str, Any]:
        """Retry extraction for all currently failed chunks."""
        failed_map = dict(self._manifest.get("failed_chunk_ids", {}))
        if not failed_map:
            return {"retried": 0, "successful": 0, "failed": 0}

        # Index all repository chunks
        all_chunks: dict[str, tuple[DocumentChunk, str]] = {}
        for doc in self.doc_repo.list_documents():
            for c in self.doc_repo.get_chunks(doc.document_id):
                all_chunks[c.chunk_id] = (c, doc.filename)

        total_failed = len(failed_map)
        successful = 0
        remaining_failed = 0

        for idx, (chunk_id, _prev_err) in enumerate(failed_map.items(), start=1):
            if chunk_id not in all_chunks:
                # Chunk no longer in repository, prune from failed
                self._manifest["failed_chunk_ids"].pop(chunk_id, None)
                continue

            chunk, filename = all_chunks[chunk_id]
            if progress_callback:
                progress_callback(idx, total_failed, f"Retrying chunk {chunk_id}...")

            try:
                self._process_chunk(chunk, doc_filename=filename)
                successful += 1
            except Exception as exc:
                remaining_failed += 1
                self._manifest["failed_chunk_ids"][chunk_id] = str(exc)

        self.store.save_json(self.graph_path)
        self.store.export_graphml(self.graphml_path)
        self._save_manifest()

        return {
            "retried": total_failed,
            "successful": successful,
            "failed": remaining_failed,
            "total_entities": self.store.node_count,
            "total_relationships": self.store.edge_count,
        }

    def remove_document(self, doc_id: str) -> tuple[int, int]:
        """Remove a document from the Knowledge Graph and purge orphan entities.

        Returns:
            Tuple of (edges_removed, nodes_removed).
        """
        # 1. Remove from graph store and entity resolver
        doc_chunk_ids = self._manifest["document_chunk_mapping"].get(doc_id, [])
        edges_removed, nodes_removed = self.store.remove_document(doc_id)
        self.resolver.remove_document(doc_id, chunk_ids=doc_chunk_ids)

        # 2. Update manifest
        chunk_ids_to_remove = set(doc_chunk_ids)
        self._manifest["processed_chunk_ids"] = [
            cid for cid in self._manifest["processed_chunk_ids"] if cid not in chunk_ids_to_remove
        ]
        for cid in chunk_ids_to_remove:
            self._manifest["failed_chunk_ids"].pop(cid, None)
        self._manifest["document_chunk_mapping"].pop(doc_id, None)

        self.store.save_json(self.graph_path)
        self.store.export_graphml(self.graphml_path)
        self._save_manifest()

        logger.info(
            "Document %s removed from GraphBuilder: %d edges removed, %d nodes purged",
            doc_id,
            edges_removed,
            nodes_removed,
        )
        return edges_removed, nodes_removed

    def get_status(self) -> dict[str, Any]:
        """Return diagnostic health and synchronization status of the Knowledge Graph."""
        all_docs = self.doc_repo.list_documents()
        total_repo_chunks = sum(len(self.doc_repo.get_chunks(d.document_id)) for d in all_docs)
        processed_chunks_count = len(self._manifest.get("processed_chunk_ids", []))
        failed_chunks_count = len(self._manifest.get("failed_chunk_ids", {}))
        unprocessed_chunks_count = max(0, total_repo_chunks - processed_chunks_count)

        return {
            "total_documents": len(all_docs),
            "total_repository_chunks": total_repo_chunks,
            "processed_chunks": processed_chunks_count,
            "unprocessed_chunks": unprocessed_chunks_count,
            "failed_chunks": failed_chunks_count,
            "node_count": self.store.node_count,
            "edge_count": self.store.edge_count,
            "graph_file_exists": self.graph_path.exists(),
            "graphml_file_exists": self.graphml_path.exists(),
            "last_updated": self._manifest.get("updated_at"),
            "model": self.extraction_service.model_name,
        }
