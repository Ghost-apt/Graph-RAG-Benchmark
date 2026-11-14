"""Document content extraction module for Graph RAG Benchmark.

Implements unified, memory-safe extraction across:
- PDF (pypdf page-by-page extraction with page number preservation and scanned PDF detection)
- DOCX (python-docx paragraph and table extraction)
- TXT (UTF-8, UTF-8 BOM, and robust legacy encoding fallbacks)
- Markdown (structure-preserving plain text extraction)
"""

import io
from dataclasses import dataclass, field

import docx
import pypdf

from graphrag_bench.logger import get_logger

logger = get_logger("document_loader")


class DocumentExtractionError(Exception):
    """Raised when document extraction fails critically."""


@dataclass
class ExtractedPage:
    """Represents text extracted from a distinct document page or unit."""

    page_number: int | None
    text: str
    char_count: int = 0
    word_count: int = 0
    char_start: int = 0
    char_end: int = 0

    def __post_init__(self) -> None:
        self.char_count = len(self.text)
        self.word_count = len(self.text.split()) if self.text.strip() else 0
        if self.char_end == 0 and self.char_count > 0:
            self.char_end = self.char_start + self.char_count


@dataclass
class ExtractedDocument:
    """Unified container for all extracted document content and metadata."""

    filename: str
    file_type: str
    pages: list[ExtractedPage] = field(default_factory=list)
    full_text: str = ""
    total_pages: int | None = None
    word_count: int = 0
    char_count: int = 0
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.full_text and self.pages:
            self.full_text = "\n\n".join(p.text for p in self.pages if p.text.strip())
        self.char_count = len(self.full_text)
        self.word_count = len(self.full_text.split()) if self.full_text.strip() else 0
        if self.total_pages is None:
            self.total_pages = len(self.pages) if self.pages else 1

    @property
    def raw_text(self) -> str:
        """Alias for full_text."""
        return self.full_text

    @property
    def file_format(self) -> str:
        """Alias for file_type."""
        return self.file_type

    @property
    def page_count(self) -> int | None:
        """Alias for total_pages."""
        return self.total_pages

    @property
    def scanned_pdf(self) -> bool:
        """True if document is a PDF with zero extractable text."""
        return self.file_type == "pdf" and any("scanned" in w.lower() for w in self.warnings)


class DocumentLoader:
    """Unified document loader orchestrating format-specific text extractors."""

    def load(
        self,
        file_bytes: bytes,
        filename: str,
        file_type: str | None = None,
    ) -> ExtractedDocument:
        """Extract text and page structures from raw file bytes.

        Args:
            file_bytes: Raw binary content of the document.
            filename: Original document filename.
            file_type: Optional explicit file type ('pdf', 'docx', 'txt', 'md').

        Returns:
            ExtractedDocument containing ordered pages, full text, and warnings.

        Raises:
            DocumentExtractionError: If document parsing fails completely.
        """
        fmt = (file_type or filename.split(".")[-1]).lower()
        if fmt == "markdown":
            fmt = "md"

        logger.debug("Loading document '%s' with extractor for format '%s'", filename, fmt)

        if fmt == "pdf":
            return self._extract_pdf(file_bytes, filename)
        elif fmt == "docx":
            return self._extract_docx(file_bytes, filename)
        elif fmt == "txt":
            return self._extract_txt(file_bytes, filename)
        elif fmt == "md":
            return self._extract_markdown(file_bytes, filename)
        else:
            raise DocumentExtractionError(
                f"Unsupported document format '{fmt}' for file '{filename}'."
            )

    def load_pdf(self, file_bytes: bytes, filename: str) -> ExtractedDocument:
        """Extract text from PDF file."""
        return self._extract_pdf(file_bytes, filename)

    def load_docx(self, file_bytes: bytes, filename: str) -> ExtractedDocument:
        """Extract text from DOCX file."""
        return self._extract_docx(file_bytes, filename)

    def load_txt(self, file_bytes: bytes, filename: str) -> ExtractedDocument:
        """Extract text from plain text file."""
        return self._extract_txt(file_bytes, filename)

    def load_md(self, file_bytes: bytes, filename: str) -> ExtractedDocument:
        """Extract text from Markdown file."""
        return self._extract_markdown(file_bytes, filename)

    def _extract_pdf(self, file_bytes: bytes, filename: str) -> ExtractedDocument:
        """Extract text page-by-page from PDF files using pypdf."""
        warnings: list[str] = []
        pages: list[ExtractedPage] = []
        stream = io.BytesIO(file_bytes)

        try:
            reader = pypdf.PdfReader(stream)
            if reader.is_encrypted:
                raise DocumentExtractionError(
                    f"Cannot extract text from encrypted PDF: '{filename}'."
                )

            total_pages = len(reader.pages)
            if total_pages == 0:
                raise DocumentExtractionError(f"PDF '{filename}' contains 0 pages.")

            total_chars = 0
            current_offset = 0
            for idx, page in enumerate(reader.pages, start=1):
                try:
                    page_text = page.extract_text() or ""
                except Exception as exc:
                    warnings.append(f"Failed to extract text from page {idx}: {exc}")
                    page_text = ""

                cleaned_page_text = page_text.strip()
                p_len = len(cleaned_page_text)
                total_chars += p_len
                pages.append(
                    ExtractedPage(
                        page_number=idx,
                        text=cleaned_page_text,
                        char_start=current_offset,
                        char_end=current_offset + p_len,
                    )
                )
                current_offset += p_len + 2  # account for "\n\n" separator

            # Check for scanned PDF (no extractable text)
            if total_chars == 0:
                warning_msg = (
                    f"Scanned PDF detected for '{filename}': no digital text could be extracted. "
                    f"Optical Character Recognition (OCR) is not currently implemented."
                )
                logger.warning(warning_msg)
                warnings.append(warning_msg)

            full_text = "\n\n".join(p.text for p in pages if p.text)

            return ExtractedDocument(
                filename=filename,
                file_type="pdf",
                pages=pages,
                full_text=full_text,
                total_pages=total_pages,
                warnings=warnings,
            )
        except Exception as exc:
            if isinstance(exc, DocumentExtractionError):
                raise
            msg = f"Failed to extract PDF '{filename}': {exc}"
            logger.error(msg)
            raise DocumentExtractionError(msg) from exc

    def _extract_docx(self, file_bytes: bytes, filename: str) -> ExtractedDocument:
        """Extract paragraphs and table text from DOCX files using python-docx."""
        warnings: list[str] = []
        text_blocks: list[str] = []
        stream = io.BytesIO(file_bytes)

        try:
            doc = docx.Document(stream)

            # Extract paragraphs in document order
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    text_blocks.append(text)

            # Extract table cell contents
            for table in doc.tables:
                for row in table.rows:
                    row_texts = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if row_texts:
                        text_blocks.append("\t".join(row_texts))

            full_text = "\n\n".join(text_blocks)
            if not full_text.strip():
                warnings.append(
                    f"DOCX document '{filename}' is empty or contains no readable text."
                )

            page = ExtractedPage(
                page_number=1,
                text=full_text,
                char_start=0,
                char_end=len(full_text),
            )

            return ExtractedDocument(
                filename=filename,
                file_type="docx",
                pages=[page],
                full_text=full_text,
                total_pages=1,
                warnings=warnings,
            )
        except Exception as exc:
            msg = f"Failed to extract DOCX '{filename}': {exc}"
            logger.error(msg)
            raise DocumentExtractionError(msg) from exc

    def _extract_txt(self, file_bytes: bytes, filename: str) -> ExtractedDocument:
        """Extract text from plain text files with encoding fallbacks."""
        warnings: list[str] = []
        decoded_text: str | None = None

        # Prioritize UTF-8 with BOM, then standard UTF-8, then legacy encodings
        for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
            try:
                decoded_text = file_bytes.decode(enc)
                if enc not in ("utf-8-sig", "utf-8"):
                    warnings.append(
                        f"Text file '{filename}' decoded using fallback encoding '{enc}'. UTF-8 is recommended."
                    )
                break
            except UnicodeDecodeError:
                continue

        if decoded_text is None:
            raise DocumentExtractionError(
                f"Failed to decode text file '{filename}': incompatible binary or unreadable encoding."
            )

        if not decoded_text.strip():
            warnings.append(f"Text file '{filename}' contains no readable text.")

        page = ExtractedPage(
            page_number=1,
            text=decoded_text.strip(),
            char_start=0,
            char_end=len(decoded_text.strip()),
        )

        return ExtractedDocument(
            filename=filename,
            file_type="txt",
            pages=[page],
            full_text=decoded_text.strip(),
            total_pages=1,
            warnings=warnings,
        )

    def _extract_markdown(self, file_bytes: bytes, filename: str) -> ExtractedDocument:
        """Extract Markdown text while preserving structural syntax (headings, lists, code)."""
        txt_doc = self._extract_txt(file_bytes, filename)
        return ExtractedDocument(
            filename=filename,
            file_type="md",
            pages=txt_doc.pages,
            full_text=txt_doc.full_text,
            total_pages=1,
            warnings=txt_doc.warnings,
        )
