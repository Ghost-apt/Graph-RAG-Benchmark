"""Unit tests for DocumentProcessor end-to-end pipeline."""

import io
from pathlib import Path

import docx
import pytest

from graphrag_bench.document_processor import DocumentProcessor
from graphrag_bench.document_repository import DocumentRepository


@pytest.fixture
def temp_processor(tmp_path: Path) -> DocumentProcessor:
    repo = DocumentRepository(base_dir=tmp_path)
    return DocumentProcessor(repository=repo)


def create_test_docx(text: str) -> bytes:
    doc = docx.Document()
    doc.add_paragraph(text)
    stream = io.BytesIO()
    doc.save(stream)
    return stream.getvalue()


class TestDocumentProcessor:
    """Tests for the end-to-end ingestion pipeline."""

    def test_process_txt_document(self, temp_processor: DocumentProcessor) -> None:
        content = (
            b"Knowledge graphs capture structured relational information across domain entities. "
            b"Retrieval-Augmented Generation leverages these entities to enhance contextual prompts."
        )

        result = temp_processor.process_file(
            content, "kg_overview.txt", chunk_size=20, chunk_overlap=5
        )

        assert result.success is True
        assert result.duplicate_detected is False
        assert result.document_id.startswith("doc_")
        assert result.metadata.filename == "kg_overview.txt"
        assert result.metadata.file_format == "txt"
        assert len(result.chunks) > 0
        assert result.metadata.total_chunks == len(result.chunks)
        assert result.metadata.total_words > 0

    def test_process_duplicate_document(self, temp_processor: DocumentProcessor) -> None:
        content = b"Identical content for duplicate verification testing."

        # First ingestion
        res1 = temp_processor.process_file(content, "original.txt")
        assert res1.success is True
        assert res1.duplicate_detected is False

        # Second ingestion with same content (even with different filename)
        res2 = temp_processor.process_file(content, "copy.txt")
        assert res2.success is True
        assert res2.duplicate_detected is True
        assert res2.document_id == res1.document_id
        assert len(res2.chunks) == len(res1.chunks)
        assert any("duplicate" in w.lower() for w in res2.warnings)

    def test_process_markdown_document(self, temp_processor: DocumentProcessor) -> None:
        md_content = (
            b"# Evaluation Matrix\n\n"
            b"## Vector Search\n"
            b"Vector search relies on dense embedding representations.\n\n"
            b"## Graph Search\n"
            b"Graph search traverses relational edges to answer complex queries."
        )

        result = temp_processor.process_file(md_content, "eval.md", chunk_size=15, chunk_overlap=3)
        assert result.success is True
        assert result.metadata.file_format == "md"
        assert len(result.chunks) >= 2

    def test_process_docx_document(self, temp_processor: DocumentProcessor) -> None:
        docx_bytes = create_test_docx(
            "This is a Microsoft Word DOCX test document for Phase 2 ingestion verification."
        )
        result = temp_processor.process_file(
            docx_bytes, "test.docx", chunk_size=10, chunk_overlap=2
        )
        assert result.success is True
        assert result.metadata.file_format == "docx"
        assert len(result.chunks) > 0

    def test_process_invalid_file_rejected(self, temp_processor: DocumentProcessor) -> None:
        result = temp_processor.process_file(b"bad content", "malicious.exe")
        assert result.success is False
        assert len(result.errors) > 0
        assert len(result.chunks) == 0

    def test_process_empty_file_rejected(self, temp_processor: DocumentProcessor) -> None:
        result = temp_processor.process_file(b"", "empty.txt")
        assert result.success is False
        assert len(result.errors) > 0
