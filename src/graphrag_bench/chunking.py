"""Word-based text chunking engine for Graph RAG Benchmark.

Implements sliding-window word chunking with:
- Configurable chunk size and overlap
- Exact character offset tracking (start_char, end_char)
- PDF page reference mapping
- Unique deterministic chunk identifier generation
- Preservation of original formatting and punctuation
"""

import hashlib
import re
from typing import Any

from graphrag_bench.document_loader import ExtractedDocument, ExtractedPage
from graphrag_bench.logger import get_logger
from graphrag_bench.schemas import DocumentChunk

logger = get_logger("chunking")

DEFAULT_CHUNK_SIZE_WORDS = 400
DEFAULT_CHUNK_OVERLAP_WORDS = 60


class ChunkingError(ValueError):
    """Raised when chunking configuration or execution encounters an invalid state."""


def validate_chunk_parameters(chunk_size: int, chunk_overlap: int) -> None:
    """Validate that chunk size and overlap parameters are mathematically sound.

    Args:
        chunk_size: Target word count per chunk.
        chunk_overlap: Overlap word count between consecutive chunks.

    Raises:
        ChunkingError: If parameters are invalid.
    """
    if chunk_size <= 0:
        raise ChunkingError(f"chunk_size must be positive (> 0), got {chunk_size}")
    if chunk_overlap < 0:
        raise ChunkingError(f"chunk_overlap cannot be negative (>= 0), got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ChunkingError(
            f"chunk_overlap must be strictly less than chunk_size (got overlap={chunk_overlap}, size={chunk_size})"
        )


class TextChunker:
    """Sliding-window word chunker with character offset tracking and page mapping."""

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE_WORDS,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP_WORDS,
    ) -> None:
        """Initialize chunker with validated parameters.

        Args:
            chunk_size: Number of words per chunk (default 400).
            chunk_overlap: Number of overlapping words between adjacent chunks (default 60).
        """
        validate_chunk_parameters(chunk_size, chunk_overlap)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(
        self,
        extracted_doc: ExtractedDocument,
        document_id: str,
    ) -> list[DocumentChunk]:
        """Divide an extracted document into ordered, metadata-enriched DocumentChunk models.

        Args:
            extracted_doc: ExtractedDocument instance containing full text and page structures.
            document_id: Unique document identifier.

        Returns:
            List of ordered DocumentChunk instances.
        """
        full_text = extracted_doc.full_text.strip()
        if not full_text:
            return []

        # Find all word tokens and their character boundaries in full_text
        word_spans: list[tuple[int, int]] = [
            (m.start(), m.end()) for m in re.finditer(r"\S+", full_text)
        ]
        total_words = len(word_spans)

        if total_words == 0:
            return []

        # Build page offset mapping for PDF page resolution
        page_boundaries = self._build_page_boundaries(extracted_doc)

        step = self.chunk_size - self.chunk_overlap
        chunks: list[DocumentChunk] = []
        chunk_idx = 0
        start_word_idx = 0

        while start_word_idx < total_words:
            end_word_idx = min(start_word_idx + self.chunk_size, total_words)

            start_char = word_spans[start_word_idx][0]
            end_char = word_spans[end_word_idx - 1][1]
            chunk_text = full_text[start_char:end_char]
            word_count = end_word_idx - start_word_idx
            char_count = len(chunk_text)

            # Determine page number based on chunk midpoint
            mid_char = (start_char + end_char) // 2
            page_num = self._resolve_page_number(mid_char, page_boundaries)

            # Generate unique deterministic chunk ID
            # format: doc_<hash[:16]>_c<index:04d>_<content_hash[:8]>
            content_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()[:8]
            chunk_id = f"{document_id}_c{chunk_idx:04d}_{content_hash}"

            chunk = DocumentChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                chunk_index=chunk_idx,
                text=chunk_text,
                word_count=word_count,
                char_count=char_count,
                source_filename=extracted_doc.filename,
                file_type=extracted_doc.file_type,  # type: ignore[arg-type]
                page_number=page_num,
                start_char=start_char,
                end_char=end_char,
            )
            chunks.append(chunk)
            chunk_idx += 1

            if end_word_idx >= total_words:
                break

            start_word_idx += step

        logger.debug(
            "Document '%s' (%s words) divided into %d chunks (size=%d, overlap=%d)",
            extracted_doc.filename,
            total_words,
            len(chunks),
            self.chunk_size,
            self.chunk_overlap,
        )
        return chunks

    def chunk_text(
        self,
        text: str,
        doc_id: str,
        pages: list[Any] | None = None,
        filename: str = "document.txt",
        file_type: str = "txt",
    ) -> list[DocumentChunk]:
        """Divide raw text string into chunks."""
        extracted_pages: list[ExtractedPage] = []
        if pages:
            for p in pages:
                if isinstance(p, ExtractedPage):
                    extracted_pages.append(p)
                elif isinstance(p, dict):
                    extracted_pages.append(ExtractedPage(**p))
        doc = ExtractedDocument(
            filename=filename,
            file_type=file_type,
            pages=extracted_pages,
            full_text=text,
        )
        return self.chunk_document(doc, document_id=doc_id)

    def _build_page_boundaries(
        self, extracted_doc: ExtractedDocument
    ) -> list[tuple[int, int, int]]:
        """Compute (start_char, end_char, page_number) tuples for multi-page documents."""
        boundaries: list[tuple[int, int, int]] = []
        if not extracted_doc.pages:
            return boundaries

        current_char = 0
        full_text = extracted_doc.full_text

        for p in extracted_doc.pages:
            if not p.text.strip():
                continue
            # Locate page text within full_text starting from current_char
            pos = full_text.find(p.text, current_char)
            if pos != -1:
                start_c = pos
                end_c = pos + len(p.text)
                if p.page_number is not None:
                    boundaries.append((start_c, end_c, p.page_number))
                current_char = end_c

        return boundaries

    def _resolve_page_number(
        self,
        char_pos: int,
        page_boundaries: list[tuple[int, int, int]],
    ) -> int | None:
        """Resolve character offset to a 1-based page number."""
        if not page_boundaries:
            return None

        for start_c, end_c, page_num in page_boundaries:
            if start_c <= char_pos <= end_c:
                return page_num

        # If slightly outside, snap to nearest boundary
        if char_pos < page_boundaries[0][0]:
            return page_boundaries[0][2]
        return page_boundaries[-1][2]
