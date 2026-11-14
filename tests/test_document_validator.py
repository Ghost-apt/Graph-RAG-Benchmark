"""Unit tests for DocumentValidator and sanitize_filename."""

import io
import zipfile

import pypdf
import pytest

from graphrag_bench.document_validator import (
    DEFAULT_MAX_FILE_SIZE_BYTES,
    DocumentValidator,
    sanitize_filename,
)


def create_minimal_pdf(content: str = "Test PDF Document", encrypted: bool = False) -> bytes:
    """Helper to generate a minimal in-memory PDF."""
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=72 * 8.5, height=72 * 11)
    # Note: Blank page is fine for structure validation
    if encrypted:
        writer.encrypt("secret_pass")
    stream = io.BytesIO()
    writer.write(stream)
    return stream.getvalue()


def create_minimal_docx(text: str = "Test DOCX Document") -> bytes:
    """Helper to generate a minimal valid DOCX file."""
    import docx

    doc = docx.Document()
    doc.add_paragraph(text)
    stream = io.BytesIO()
    doc.save(stream)
    return stream.getvalue()


class TestSanitizeFilename:
    """Tests for sanitize_filename utility."""

    def test_clean_filename(self) -> None:
        assert sanitize_filename("research_paper.pdf") == "research_paper.pdf"

    def test_path_traversal_removal(self) -> None:
        assert sanitize_filename("../../etc/passwd.txt") == "passwd.txt"
        assert sanitize_filename("..\\..\\windows\\system32\\cmd.exe.docx") == "cmd.exe.docx"

    def test_null_bytes_and_control_chars(self) -> None:
        assert sanitize_filename("malicious\x00file.pdf") == "maliciousfile.pdf"
        assert sanitize_filename("test\x1f\x08doc.md") == "testdoc.md"

    def test_windows_forbidden_chars(self) -> None:
        sanitized = sanitize_filename('report:part<1>?"*.txt')
        assert ":" not in sanitized
        assert "<" not in sanitized
        assert ">" not in sanitized
        assert '"' not in sanitized
        assert "?" not in sanitized
        assert "*" not in sanitized

    def test_empty_or_whitespace_filename(self) -> None:
        assert sanitize_filename("") == "unnamed_document.txt"
        assert sanitize_filename("   ...   ") == "unnamed_document.txt"


class TestDocumentValidator:
    """Tests for DocumentValidator class."""

    @pytest.fixture
    def validator(self) -> DocumentValidator:
        return DocumentValidator(max_file_size_bytes=DEFAULT_MAX_FILE_SIZE_BYTES)

    def test_validate_valid_txt(self, validator: DocumentValidator) -> None:
        content = b"Graph RAG Benchmark sample text content for validation testing."
        result = validator.validate(content, "sample.txt")
        assert result.is_valid is True
        assert result.file_type == "txt"
        assert result.sanitized_filename == "sample.txt"
        assert result.file_size_bytes == len(content)
        assert len(result.errors) == 0

    def test_validate_valid_md(self, validator: DocumentValidator) -> None:
        content = b"# Markdown Document\n\nTesting markdown file validation."
        result = validator.validate(content, "doc.md")
        assert result.is_valid is True
        assert result.file_type == "md"
        assert len(result.errors) == 0

    def test_validate_valid_pdf(self, validator: DocumentValidator) -> None:
        content = create_minimal_pdf()
        result = validator.validate(content, "report.pdf")
        assert result.is_valid is True
        assert result.file_type == "pdf"
        assert result.mime_type == "application/pdf"
        assert len(result.errors) == 0

    def test_validate_valid_docx(self, validator: DocumentValidator) -> None:
        content = create_minimal_docx()
        result = validator.validate(content, "report.docx")
        assert result.is_valid is True
        assert result.file_type == "docx"
        assert len(result.errors) == 0

    def test_empty_file_rejected(self, validator: DocumentValidator) -> None:
        result = validator.validate(b"", "empty.txt")
        assert result.is_valid is False
        assert any("empty" in e.lower() for e in result.errors)

    def test_oversized_file_rejected(self) -> None:
        small_validator = DocumentValidator(max_file_size_bytes=100)
        oversized_content = b"x" * 150
        result = small_validator.validate(oversized_content, "big.txt")
        assert result.is_valid is False
        assert any("exceeds" in e.lower() for e in result.errors)

    def test_unsupported_extension_rejected(self, validator: DocumentValidator) -> None:
        result = validator.validate(b"binary content", "malware.exe")
        assert result.is_valid is False
        assert any("unsupported" in e.lower() for e in result.errors)

    def test_corrupted_pdf_rejected(self, validator: DocumentValidator) -> None:
        # File with .pdf extension but random bytes
        fake_pdf = b"%PDF-1.4 garbage corrupted byte stream"
        result = validator.validate(fake_pdf, "corrupted.pdf")
        assert result.is_valid is False
        assert any("pdf" in e.lower() for e in result.errors)

    def test_encrypted_pdf_rejected(self, validator: DocumentValidator) -> None:
        encrypted_pdf = create_minimal_pdf(encrypted=True)
        result = validator.validate(encrypted_pdf, "protected.pdf")
        assert result.is_valid is False
        assert any("password" in e.lower() or "encrypted" in e.lower() for e in result.errors)

    def test_corrupted_docx_rejected(self, validator: DocumentValidator) -> None:
        # File with .docx extension but not a zip file
        result = validator.validate(b"Not a zip file", "fake.docx")
        assert result.is_valid is False
        assert any("docx" in e.lower() or "zip" in e.lower() for e in result.errors)

    def test_docx_missing_document_xml_rejected(self, validator: DocumentValidator) -> None:
        # A valid zip file but lacking word/document.xml
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as zf:
            zf.writestr("test.txt", "hello")
        result = validator.validate(stream.getvalue(), "invalid.docx")
        assert result.is_valid is False
        assert any("word/document.xml" in e.lower() for e in result.errors)

    def test_binary_in_text_file_rejected(self, validator: DocumentValidator) -> None:
        # Text file packed with null bytes
        binary_data = b"Some header text\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00" * 20
        result = validator.validate(binary_data, "masquerading.txt")
        assert result.is_valid is False
        assert any("binary" in e.lower() for e in result.errors)

    def test_fallback_encoding_warning(self, validator: DocumentValidator) -> None:
        # Text encoded in latin-1 with accented characters not valid UTF-8
        latin1_content = "Caf\xe9 and na\xefve and \xbfque?".encode("latin-1")
        result = validator.validate(latin1_content, "latin1.txt")
        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert any("fallback" in w.lower() for w in result.warnings)
