"""Typed Pydantic data schemas for Document Ingestion and Preprocessing.

Defines validated data models for:
- DocumentMetadata: High-level document attributes, hashes, timestamps, and statistics.
- DocumentChunk: Granular chunk with text content, word counts, character offsets, and source page.
- DocumentIngestionResult: Ingestion outcome with metadata, chunks, warnings, and duplicate indicators.
"""

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

DocumentFileType = Literal["pdf", "docx", "txt", "md"]
ProcessingStatus = Literal["completed", "failed", "duplicate", "warning"]


class DocumentMetadata(BaseModel):
    """Metadata representing an ingested source document."""

    document_id: str = Field(
        description="Unique deterministic document identifier (e.g. doc_<sha256[:16]>)"
    )
    filename: str = Field(description="Sanitized original document filename")
    file_type: DocumentFileType = Field(
        description="Document format extension (pdf, docx, txt, md)"
    )
    mime_type: str = Field(
        default="application/octet-stream",
        description="MIME type classification of the document",
    )
    file_size_bytes: int = Field(ge=0, description="Size of the uploaded document in bytes")
    sha256: str = Field(
        min_length=64,
        max_length=64,
        description="Hexadecimal SHA-256 hash of original file",
    )
    ingested_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timezone-aware UTC timestamp when document was ingested",
    )
    total_pages: int | None = Field(
        default=None,
        ge=1,
        description="Total page count (available for PDF, null for formats without reliable pages)",
    )
    word_count: int = Field(
        default=0,
        ge=0,
        description="Total word count extracted from the document",
    )
    char_count: int = Field(
        default=0,
        ge=0,
        description="Total character count extracted from the document",
    )
    chunk_count: int = Field(
        default=0,
        ge=0,
        description="Number of text chunks generated from the document",
    )
    processing_status: ProcessingStatus = Field(
        default="completed",
        description="Status of the ingestion pipeline execution",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal warnings recorded during ingestion (e.g. scanned pages, fallback encodings)",
    )
    extra_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional structured metadata attributes",
    )

    @model_validator(mode="before")
    @classmethod
    def preprocess_metadata_inputs(cls, data: Any) -> Any:
        """Map legacy / convenience alias field names to standard attributes."""
        if isinstance(data, dict):
            data = dict(data)
            if "doc_id" in data and "document_id" not in data:
                data["document_id"] = data["doc_id"]
            if "file_format" in data and "file_type" not in data:
                data["file_type"] = data["file_format"]
            if "total_words" in data and "word_count" not in data:
                data["word_count"] = data["total_words"]
            if "total_chunks" in data and "chunk_count" not in data:
                data["chunk_count"] = data["total_chunks"]
            if "page_count" in data and "total_pages" not in data:
                data["total_pages"] = data["page_count"]
            if "mime_type" not in data:
                ft = data.get("file_type", "txt")
                mime_map = {
                    "pdf": "application/pdf",
                    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "txt": "text/plain",
                    "md": "text/markdown",
                }
                data["mime_type"] = mime_map.get(ft, "application/octet-stream")
            if "char_count" not in data:
                data["char_count"] = int(data.get("word_count", 0)) * 5
            if "word_count" not in data:
                data["word_count"] = 0
            if "chunk_count" not in data:
                data["chunk_count"] = 0
        return data

    @field_validator("ingested_at", mode="before")
    @classmethod
    def ensure_utc_timestamp(cls, v: Any) -> datetime:
        """Ensure timestamps are timezone-aware UTC datetime objects."""
        if isinstance(v, str):
            dt = datetime.fromisoformat(v)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=UTC)
            return dt.astimezone(UTC)
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=UTC)
            return v.astimezone(UTC)
        return datetime.now(UTC)

    @property
    def doc_id(self) -> str:
        """Convenience alias for document_id."""
        return self.document_id

    @property
    def file_format(self) -> str:
        """Convenience alias for file_type."""
        return self.file_type

    @property
    def page_count(self) -> int | None:
        """Convenience alias for total_pages."""
        return self.total_pages

    @property
    def total_words(self) -> int:
        """Convenience alias for word_count."""
        return self.word_count

    @property
    def total_chunks(self) -> int:
        """Convenience alias for chunk_count."""
        return self.chunk_count

    @property
    def created_at(self) -> datetime:
        """Convenience alias for ingested_at."""
        return self.ingested_at


class DocumentChunk(BaseModel):
    """A granular chunk of extracted text prepared for retrieval and embedding."""

    chunk_id: str = Field(
        description="Unique deterministic chunk identifier (e.g. <doc_id>_c0001_<content_hash>)"
    )
    document_id: str = Field(description="Foreign key identifier of parent document")
    chunk_index: int = Field(ge=0, description="0-indexed position of chunk within document")
    text: str = Field(min_length=1, description="Normalized extracted chunk text")
    word_count: int = Field(ge=1, description="Total word count in this chunk")
    char_count: int = Field(ge=1, description="Total character count in this chunk")
    source_filename: str = Field(default="unknown", description="Name of source file for citations")
    file_type: DocumentFileType = Field(default="txt", description="Source file format")
    page_number: int | None = Field(
        default=None,
        ge=1,
        description="1-based page number where chunk originates (PDFs only; null for unpaged docs)",
    )
    start_char: int = Field(ge=0, description="Starting character index in normalized full text")
    end_char: int = Field(ge=0, description="Ending character index in normalized full text")

    @model_validator(mode="before")
    @classmethod
    def preprocess_chunk_inputs(cls, data: Any) -> Any:
        """Map legacy / convenience alias field names to standard attributes."""
        if isinstance(data, dict):
            data = dict(data)
            if "content" in data and "text" not in data:
                data["text"] = data["content"]
            if "doc_id" in data and "document_id" not in data:
                data["document_id"] = data["doc_id"]
            if "source_filename" not in data:
                data["source_filename"] = data.get("document_title", "unknown")
            if "file_type" not in data:
                data["file_type"] = "txt"
            if "start_char" not in data:
                data["start_char"] = 0
            if "end_char" not in data:
                data["end_char"] = len(data.get("text", data.get("content", "")))
            if "char_count" not in data:
                data["char_count"] = max(1, len(data.get("text", data.get("content", ""))))
            if "word_count" not in data:
                data["word_count"] = max(1, len(data.get("text", data.get("content", "")).split()))
        return data

    @property
    def doc_id(self) -> str:
        """Convenience alias for document_id."""
        return self.document_id


class DocumentIngestionResult(BaseModel):
    """Comprehensive result object returned by the document ingestion pipeline."""

    success: bool = Field(description="True if document was successfully processed and stored")
    document_metadata: DocumentMetadata | None = Field(
        default=None,
        description="Metadata of the ingested document (or existing document if duplicate)",
    )
    chunks: list[DocumentChunk] = Field(
        default_factory=list,
        description="List of generated text chunks",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Warnings encountered during validation or extraction",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Error messages encountered if ingestion failed",
    )
    duplicate_detected: bool = Field(
        default=False,
        description="True if an identical document (matching SHA-256) already exists in storage",
    )

    @property
    def metadata(self) -> DocumentMetadata | None:
        """Convenience alias for document_metadata."""
        return self.document_metadata

    @property
    def document_id(self) -> str:
        """Convenience accessor for parent document identifier."""
        return self.document_metadata.document_id if self.document_metadata else ""
