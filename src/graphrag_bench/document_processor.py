"""Central document processing pipeline for Graph RAG Benchmark.

Orchestrates end-to-end ingestion:
1. File validation & path traversal protection
2. SHA-256 cryptographic hashing
3. Duplicate document detection
4. Multi-format text extraction (PDF, DOCX, TXT, MD)
5. Text normalization and cleaning
6. Word-based chunking with exact offsets
7. Metadata generation and persistent storage
8. Clean rollback on failures
"""

import hashlib

from graphrag_bench.chunking import (
    DEFAULT_CHUNK_OVERLAP_WORDS,
    DEFAULT_CHUNK_SIZE_WORDS,
    TextChunker,
)
from graphrag_bench.document_loader import DocumentLoader, ExtractedDocument
from graphrag_bench.document_repository import DocumentRepository
from graphrag_bench.document_validator import DocumentValidator
from graphrag_bench.logger import get_logger
from graphrag_bench.schemas import DocumentIngestionResult, DocumentMetadata
from graphrag_bench.text_cleaner import TextCleaner

logger = get_logger("document_processor")


class DocumentProcessor:
    """Orchestrates end-to-end ingestion and preprocessing of uploaded documents."""

    def __init__(
        self,
        repository: DocumentRepository | None = None,
        validator: DocumentValidator | None = None,
        loader: DocumentLoader | None = None,
    ) -> None:
        """Initialize document processor with dependencies.

        Args:
            repository: Optional DocumentRepository instance (defaults to standard storage).
            validator: Optional DocumentValidator instance.
            loader: Optional DocumentLoader instance.
        """
        self.repository = repository or DocumentRepository()
        self.validator = validator or DocumentValidator()
        self.loader = loader or DocumentLoader()

    def process_file(
        self,
        file_content: bytes,
        filename: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE_WORDS,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP_WORDS,
    ) -> DocumentIngestionResult:
        """Execute document ingestion pipeline (alias for process_document)."""
        return self.process_document(
            file_bytes=file_content,
            filename=filename,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def process_document(
        self,
        file_bytes: bytes,
        filename: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE_WORDS,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP_WORDS,
    ) -> DocumentIngestionResult:
        """Execute complete document ingestion pipeline.

        Args:
            file_bytes: Raw binary content of uploaded document.
            filename: Original document filename.
            chunk_size: Target word count per chunk.
            chunk_overlap: Overlap word count between adjacent chunks.

        Returns:
            DocumentIngestionResult with processing metadata, chunks, warnings, and errors.
        """
        warnings: list[str] = []

        logger.info("Beginning ingestion for document: '%s'", filename)

        # 1. Validation (size, extension, signature, traversal)
        val_res = self.validator.validate(file_bytes, filename)
        warnings.extend(val_res.warnings)

        if not val_res.is_valid:
            logger.error("Validation failed for '%s': %s", filename, val_res.errors)
            return DocumentIngestionResult(
                success=False,
                errors=val_res.errors,
                warnings=warnings,
            )

        file_type = val_res.file_type or "txt"
        safe_filename = val_res.sanitized_filename
        mime_type = val_res.mime_type

        # 2. Compute SHA-256 hash
        sha256_hash = hashlib.sha256(file_bytes).hexdigest()
        document_id = f"doc_{sha256_hash[:16]}"

        # 3. Duplicate Detection
        existing_doc = self.repository.get_document_by_sha256(sha256_hash)
        if existing_doc is not None:
            dup_msg = (
                f"Duplicate document detected! Identical file already exists in repository "
                f"with Document ID: '{existing_doc.document_id}' (Filename: '{existing_doc.filename}')."
            )
            logger.warning(dup_msg)
            warnings.append(dup_msg)
            existing_chunks = self.repository.get_chunks(existing_doc.document_id)
            return DocumentIngestionResult(
                success=True,
                document_metadata=existing_doc,
                chunks=existing_chunks,
                warnings=warnings,
                duplicate_detected=True,
            )

        # 4. Text Extraction
        try:
            extracted_raw = self.loader.load(file_bytes, safe_filename, file_type=file_type)
            warnings.extend(extracted_raw.warnings)
        except Exception as exc:
            err_msg = f"Extraction failed for '{safe_filename}': {exc}"
            logger.error(err_msg)
            return DocumentIngestionResult(
                success=False,
                errors=[err_msg],
                warnings=warnings,
            )

        # 5. Text Cleaning
        # Clean both full text and individual pages
        cleaned_full_text = TextCleaner.clean(extracted_raw.full_text)

        cleaned_pages = []
        for p in extracted_raw.pages:
            cleaned_page_text = TextCleaner.clean(p.text)
            p.text = cleaned_page_text
            cleaned_pages.append(p)

        cleaned_doc = ExtractedDocument(
            filename=safe_filename,
            file_type=file_type,
            pages=cleaned_pages,
            full_text=cleaned_full_text,
            total_pages=extracted_raw.total_pages,
            warnings=warnings,
        )

        # 6. Validate Extracted Content
        if not cleaned_full_text.strip():
            msg = (
                f"Document '{safe_filename}' contains no extractable digital text after processing. "
                f"If this is a scanned document or image-only PDF, text extraction requires OCR (future phase)."
            )
            logger.warning(msg)
            warnings.append(msg)
            return DocumentIngestionResult(
                success=False,
                errors=[msg],
                warnings=warnings,
            )

        # 7. Apply Chunking
        try:
            chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = chunker.chunk_document(cleaned_doc, document_id)
        except Exception as exc:
            err_msg = f"Chunking failed for '{safe_filename}': {exc}"
            logger.error(err_msg)
            return DocumentIngestionResult(
                success=False,
                errors=[err_msg],
                warnings=warnings,
            )

        # 8. Create DocumentMetadata
        metadata = DocumentMetadata(
            document_id=document_id,
            filename=safe_filename,
            file_type=file_type,  # type: ignore[arg-type]
            mime_type=mime_type,
            file_size_bytes=len(file_bytes),
            sha256=sha256_hash,
            total_pages=cleaned_doc.total_pages,
            word_count=cleaned_doc.word_count,
            char_count=cleaned_doc.char_count,
            chunk_count=len(chunks),
            processing_status="completed" if not warnings else "warning",
            warnings=warnings,
        )

        # 9. Store in DocumentRepository
        try:
            self.repository.save_document(file_bytes, metadata, chunks)
        except Exception as exc:
            err_msg = f"Failed to persist document '{safe_filename}': {exc}"
            logger.error(err_msg)
            return DocumentIngestionResult(
                success=False,
                errors=[err_msg],
                warnings=warnings,
            )

        logger.info(
            "Document '%s' successfully ingested (ID: %s, Chunks: %d, Words: %d)",
            safe_filename,
            document_id,
            len(chunks),
            metadata.word_count,
        )

        return DocumentIngestionResult(
            success=True,
            document_metadata=metadata,
            chunks=chunks,
            warnings=warnings,
            duplicate_detected=False,
        )
