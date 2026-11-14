"""Persistent document and chunk storage repository for Graph RAG Benchmark.

Manages local file persistence:
- Original uploaded raw files in data/uploads/
- Structured document metadata in data/processed/documents.json
- Granular chunks in data/processed/chunks.jsonl
- Master index in data/processed/manifest.json

Provides:
- SHA-256 duplicate detection
- Atomic metadata writes with corruption recovery
- Chunk retrieval and document deletion
- Multi-format export utilities (JSON, JSONL, CSV)
"""

import csv
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from graphrag_bench.logger import get_logger
from graphrag_bench.schemas import DocumentChunk, DocumentMetadata

logger = get_logger("document_repository")


class RepositoryError(Exception):
    """Base exception for document repository storage errors."""


class DocumentRepository:
    """Local JSON/JSONL document repository with integrity checks."""

    def __init__(
        self,
        base_dir: Path | None = None,
        uploads_dir: Path | None = None,
        processed_dir: Path | None = None,
    ) -> None:
        """Initialize repository storage directories.

        Args:
            base_dir: Root project directory (defaults to current working directory).
            uploads_dir: Target directory for original uploaded documents.
            processed_dir: Target directory for processed metadata and chunks.
        """
        root = base_dir or Path.cwd()
        self.uploads_dir = uploads_dir or (root / "data" / "uploads")
        self.processed_dir = processed_dir or (root / "data" / "processed")

        # Ensure directory structure exists
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.docs_file = self.processed_dir / "documents.json"
        self.chunks_file = self.processed_dir / "chunks.jsonl"
        self.manifest_file = self.processed_dir / "manifest.json"

        # Initialize storage files if not present or recover if corrupt
        self._initialize_or_recover()

    def _initialize_or_recover(self) -> None:
        """Initialize empty storage files if missing, or rebuild corrupt manifest."""
        if not self.docs_file.exists():
            self._write_json_atomic(self.docs_file, {})

        if not self.chunks_file.exists():
            self.chunks_file.touch()

        if not self.manifest_file.exists():
            self.rebuild_manifest()
        else:
            try:
                with open(self.manifest_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if not isinstance(data, dict) or "document_ids" not in data:
                        raise ValueError("Invalid manifest structure")
            except Exception as exc:
                logger.warning("Corrupt manifest detected (%s); rebuilding from storage...", exc)
                self.rebuild_manifest()

    def _write_json_atomic(self, file_path: Path, data: dict) -> None:
        """Atomically write JSON data to disk via temporary file swap."""
        temp_file = file_path.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        temp_file.replace(file_path)

    def rebuild_manifest(self) -> None:
        """Rebuild manifest.json from documents.json and chunks.jsonl."""
        docs = self._load_documents_dict()
        doc_ids = list(docs.keys())
        sha256_index = {
            doc_data.get("sha256", doc_data.get("sha256_hash", "")): doc_id
            for doc_id, doc_data in docs.items()
        }

        total_chunks = 0
        if self.chunks_file.exists():
            with open(self.chunks_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        total_chunks += 1

        manifest = {
            "document_count": len(doc_ids),
            "total_chunks": total_chunks,
            "document_ids": doc_ids,
            "sha256_index": sha256_index,
            "last_updated": datetime.now(UTC).isoformat(),
        }
        self._write_json_atomic(self.manifest_file, manifest)
        logger.info("Manifest rebuilt: %d documents, %d total chunks", len(doc_ids), total_chunks)

    def _load_documents_dict(self) -> dict[str, dict]:
        """Read documents.json returning dict of doc_id -> metadata_dict."""
        if not self.docs_file.exists():
            return {}
        try:
            with open(self.docs_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return {
                        item["document_id"]: item
                        for item in data
                        if isinstance(item, dict) and "document_id" in item
                    }
                return data
        except Exception as exc:
            logger.error("Failed to read documents.json: %s", exc)
            return {}

    def get_document_by_sha256(self, sha256_hash: str) -> DocumentMetadata | None:
        """Check if a document with identical SHA-256 hash already exists."""
        docs = self._load_documents_dict()
        for doc_data in docs.values():
            if doc_data.get("sha256") == sha256_hash:
                return DocumentMetadata.model_validate(doc_data)
        return None

    def is_duplicate(self, sha256_hash: str) -> tuple[bool, DocumentMetadata | None]:
        """Query whether duplicate exists by SHA256 hash.

        Returns:
            Tuple of (is_duplicate: bool, existing_metadata: Optional[DocumentMetadata]).
        """
        doc = self.get_document_by_sha256(sha256_hash)
        return (doc is not None, doc)

    def get_document(self, document_id: str) -> DocumentMetadata | None:
        """Retrieve document metadata by document ID."""
        docs = self._load_documents_dict()
        if document_id in docs:
            return DocumentMetadata.model_validate(docs[document_id])
        return None

    def list_documents(self) -> list[DocumentMetadata]:
        """List all ingested documents ordered by ingestion date descending."""
        docs = self._load_documents_dict()
        doc_list = [DocumentMetadata.model_validate(d) for d in docs.values()]
        doc_list.sort(key=lambda d: d.ingested_at, reverse=True)
        return doc_list

    def save_document(
        self,
        *args: Any,
        file_bytes: bytes | None = None,
        metadata: DocumentMetadata | None = None,
        chunks: list[DocumentChunk] | None = None,
        raw_content: bytes | None = None,
        **kwargs: Any,
    ) -> bool:
        """Persist original file, document metadata, and generated chunks.

        Flexible parameter handling supporting positional or keyword invocations.
        """
        # Resolve positional args if provided
        if len(args) == 3 and isinstance(args[0], (bytes, bytearray)):
            file_bytes = bytes(args[0])
            metadata = args[1]
            chunks = args[2]
        elif len(args) == 2 and isinstance(args[0], DocumentMetadata):
            metadata = args[0]
            chunks = args[1]

        resolved_bytes = file_bytes or raw_content

        if metadata is None:
            raise RepositoryError("metadata parameter is required to save document.")
        if chunks is None:
            chunks = []

        document_id = metadata.document_id

        # 1. Save raw file to uploads directory if provided
        if resolved_bytes:
            save_path = self.uploads_dir / metadata.filename
            if save_path.exists():
                stem = save_path.stem
                suffix = save_path.suffix
                save_path = self.uploads_dir / f"{stem}_{document_id[:8]}{suffix}"
            save_path.write_bytes(resolved_bytes)
            logger.debug("Saved uploaded file to %s", save_path)

        # 2. Update documents.json
        docs = self._load_documents_dict()
        docs[document_id] = metadata.model_dump(mode="json")
        self._write_json_atomic(self.docs_file, docs)

        # 3. Append chunks to chunks.jsonl
        if chunks:
            with open(self.chunks_file, "a", encoding="utf-8") as f:
                f.writelines(chunk.model_dump_json() + "\n" for chunk in chunks)

        # 4. Update manifest
        self.rebuild_manifest()
        logger.info(
            "Document '%s' (ID: %s) saved with %d chunks",
            metadata.filename,
            document_id,
            len(chunks),
        )
        return True

    def get_chunks(self, document_id: str) -> list[DocumentChunk]:
        """Retrieve all chunks for a given document ID ordered by chunk_index."""
        chunks: list[DocumentChunk] = []
        if not self.chunks_file.exists():
            return chunks

        with open(self.chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if data.get("document_id") == document_id:
                        chunks.append(DocumentChunk.model_validate(data))
                except Exception as exc:
                    logger.warning("Error parsing chunk line: %s", exc)

        chunks.sort(key=lambda c: c.chunk_index)
        return chunks

    def delete_document(self, document_id: str) -> bool:
        """Delete document metadata, chunks, and uploaded file."""
        docs = self._load_documents_dict()
        if document_id not in docs:
            logger.warning("Cannot delete non-existent document ID '%s'", document_id)
            return False

        doc_meta = DocumentMetadata.model_validate(docs[document_id])

        # 1. Remove from documents.json
        del docs[document_id]
        self._write_json_atomic(self.docs_file, docs)

        # 2. Remove uploaded file
        raw_path = self.uploads_dir / doc_meta.filename
        if raw_path.exists():
            try:
                raw_path.unlink()
            except OSError as exc:
                logger.warning("Failed to remove uploaded file '%s': %s", raw_path, exc)

        # 3. Filter out chunks from chunks.jsonl
        if self.chunks_file.exists():
            temp_chunks = self.chunks_file.with_suffix(".tmp")
            with (
                open(self.chunks_file, "r", encoding="utf-8") as src,
                open(temp_chunks, "w", encoding="utf-8") as dst,
            ):
                for line in src:
                    if not line.strip():
                        continue
                    try:
                        chunk_dict = json.loads(line)
                        if chunk_dict.get("document_id") != document_id:
                            dst.write(line)
                    except json.JSONDecodeError:
                        continue
            temp_chunks.replace(self.chunks_file)

        # 4. Rebuild manifest
        self.rebuild_manifest()
        logger.info("Document '%s' (ID: %s) deleted successfully", doc_meta.filename, document_id)
        return True

    def export_chunks(
        self,
        document_id: str | None = None,
        doc_id: str | None = None,
        format: str = "json",
    ) -> str:
        """Export chunks in specified format ('json', 'jsonl', 'csv')."""
        target_id = document_id or doc_id
        fmt = format.lower()
        if fmt == "json":
            return self.export_chunks_json(target_id)
        elif fmt == "jsonl":
            return self.export_chunks_jsonl(target_id)
        elif fmt == "csv":
            return self.export_chunks_csv(target_id)
        else:
            raise ValueError(
                f"Unsupported export format '{format}'. Choose from 'json', 'jsonl', 'csv'."
            )

    def export_chunks_json(self, document_id: str | None = None) -> str:
        """Export chunks as a formatted JSON array string."""
        chunks = self.get_chunks(document_id) if document_id else self._get_all_chunks()
        return json.dumps([c.model_dump(mode="json") for c in chunks], indent=2)

    def export_chunks_jsonl(self, document_id: str | None = None) -> str:
        """Export chunks as a JSON Lines (JSONL) string."""
        chunks = self.get_chunks(document_id) if document_id else self._get_all_chunks()
        return "\n".join(c.model_dump_json() for c in chunks)

    def export_chunks_csv(self, document_id: str | None = None) -> str:
        """Export chunks as a CSV formatted string with standard citation headers."""
        chunks = self.get_chunks(document_id) if document_id else self._get_all_chunks()
        output = io.StringIO()
        fieldnames = [
            "chunk_id",
            "document_id",
            "chunk_index",
            "source_filename",
            "file_type",
            "page_number",
            "word_count",
            "char_count",
            "start_char",
            "end_char",
            "text",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for c in chunks:
            row = c.model_dump(mode="json")
            writer.writerow(row)
        return output.getvalue()

    def _get_all_chunks(self) -> list[DocumentChunk]:
        """Retrieve all chunks across all documents."""
        all_chunks: list[DocumentChunk] = []
        if not self.chunks_file.exists():
            return all_chunks

        with open(self.chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        all_chunks.append(DocumentChunk.model_validate_json(line))
                    except Exception:  # noqa: S112
                        continue
        return all_chunks
