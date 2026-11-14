"""Unit tests for DocumentLoader."""

import io

import docx
import pypdf
import pytest

from graphrag_bench.document_loader import (
    DocumentExtractionError,
    DocumentLoader,
    ExtractedDocument,
)


def create_sample_pdf(pages_text: list[str]) -> bytes:
    """Create a multi-page PDF in memory with specified text per page."""
    # Note: pypdf can create pages and annotations, but standard text extraction looks for content streams.
    # We can create a simple PDF with reportlab if available or pypdf with content streams.
    # Alternatively, use pypdf PageObject or write standard PDF syntax.
    # A standard raw PDF stream with Text objects:
    writer = pypdf.PdfWriter()
    for text in pages_text:
        # Create a page and add text stream if possible, or add blank page
        writer.add_blank_page(width=612, height=792)

    # Let's write valid PDF structure
    stream = io.BytesIO()
    writer.write(stream)
    return stream.getvalue()


def create_sample_docx(paragraphs: list[str], table_data: list[list[str]] | None = None) -> bytes:
    """Create a DOCX file with paragraphs and optional table."""
    doc = docx.Document()
    for p in paragraphs:
        doc.add_paragraph(p)
    if table_data:
        table = doc.add_table(rows=len(table_data), cols=len(table_data[0]))
        for r_idx, row in enumerate(table_data):
            for c_idx, cell_value in enumerate(row):
                table.cell(r_idx, c_idx).text = cell_value
    stream = io.BytesIO()
    doc.save(stream)
    return stream.getvalue()


class TestDocumentLoader:
    """Tests for DocumentLoader extraction routines."""

    @pytest.fixture
    def loader(self) -> DocumentLoader:
        return DocumentLoader()

    def test_load_txt_utf8(self, loader: DocumentLoader) -> None:
        content = "Graph RAG Benchmarking\nComprehensive multi-hop retrieval."
        raw_bytes = content.encode("utf-8")
        doc = loader.load(raw_bytes, "test.txt")

        assert isinstance(doc, ExtractedDocument)
        assert doc.filename == "test.txt"
        assert doc.file_format == "txt"
        assert "Comprehensive multi-hop retrieval" in doc.raw_text
        assert doc.page_count == 1
        assert len(doc.pages) == 1
        assert doc.pages[0].page_number == 1

    def test_load_txt_with_bom(self, loader: DocumentLoader) -> None:
        content = "BOM prefixed text file testing."
        raw_bytes = content.encode("utf-8-sig")
        doc = loader.load(raw_bytes, "bom.txt")
        assert doc.raw_text.startswith("BOM prefixed")

    def test_load_txt_latin1_fallback(self, loader: DocumentLoader) -> None:
        content = "Caf\xe9 au lait et cr\xe8me br\xfbl\xe9e"
        raw_bytes = content.encode("latin-1")
        doc = loader.load(raw_bytes, "french.txt")
        assert "Café" in doc.raw_text
        assert any("fallback" in w.lower() for w in doc.warnings)

    def test_load_md(self, loader: DocumentLoader) -> None:
        markdown_text = (
            "# Benchmark Report\n\n"
            "## Section 1: Vector RAG\n"
            "Vector RAG performs cosine similarity matching.\n\n"
            "## Section 2: Graph RAG\n"
            "Graph RAG performs multi-hop entity traversal.\n"
        )
        doc = loader.load(markdown_text.encode("utf-8"), "report.md")
        assert doc.file_format == "md"
        assert "## Section 2: Graph RAG" in doc.raw_text
        assert doc.page_count == 1

    def test_load_docx_with_paragraphs_and_table(self, loader: DocumentLoader) -> None:
        paragraphs = [
            "Introduction to Knowledge Graphs in RAG systems.",
            "Entities are extracted and linked via semantic edges.",
        ]
        table_data = [
            ["Model", "F1 Score", "Latency"],
            ["Vector RAG", "0.78", "120ms"],
            ["Graph RAG", "0.89", "250ms"],
        ]
        docx_bytes = create_sample_docx(paragraphs, table_data)
        doc = loader.load(docx_bytes, "knowledge_graph.docx")

        assert doc.file_format == "docx"
        assert "Introduction to Knowledge Graphs" in doc.raw_text
        assert "Vector RAG" in doc.raw_text
        assert "0.89" in doc.raw_text
        assert doc.page_count == 1

    def test_load_empty_or_scanned_pdf(self, loader: DocumentLoader) -> None:
        # Blank PDF contains 0 extractable text characters
        pdf_bytes = create_sample_pdf([""])
        doc = loader.load(pdf_bytes, "scanned.pdf")
        assert doc.file_format == "pdf"
        assert doc.scanned_pdf is True
        assert len(doc.warnings) > 0
        assert any("scanned" in w.lower() or "ocr" in w.lower() for w in doc.warnings)

    def test_load_unsupported_extension_raises_error(self, loader: DocumentLoader) -> None:
        with pytest.raises(DocumentExtractionError) as exc_info:
            loader.load(b"some content", "data.xyz")
        assert "unsupported" in str(exc_info.value).lower()

    def test_load_corrupted_stream_raises_error(self, loader: DocumentLoader) -> None:
        with pytest.raises(DocumentExtractionError):
            loader.load(b"PK\x03\x04corrupted zip payload", "bad.docx")
