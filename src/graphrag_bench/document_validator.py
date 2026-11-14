"""Document validation module for Graph RAG Benchmark.

Performs multi-layer validation on uploaded files:
1. Extension & MIME classification
2. File size enforcement (configurable upload limits)
3. Path traversal & filename sanitization
4. Magic byte signature verification
5. Format-specific structural integrity checks (PDF, DOCX, TXT, MD)
6. Security checks (encrypted PDFs, binary masquerading)
"""

import io
import os
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import docx
import pypdf

from graphrag_bench.logger import get_logger

logger = get_logger("document_validator")

# Supported file extensions mapped to canonical types
SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".txt": "txt",
    ".md": "md",
    ".markdown": "md",
}

# Standard MIME type mappings
MIME_MAPPINGS = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "txt": "text/plain",
    "md": "text/markdown",
}

DEFAULT_MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 Megabytes


class ValidationError(Exception):
    """Raised when document validation fails."""


@dataclass
class ValidationResult:
    """Outcome of document validation."""

    is_valid: bool
    file_type: str | None = None
    sanitized_filename: str = ""
    file_size_bytes: int = 0
    mime_type: str = "application/octet-stream"
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent directory traversal and filesystem attacks.

    Strips directory separators, relative path markers (..), null bytes,
    and non-standard control characters, returning a safe basename.
    """
    if not filename:
        return "unnamed_document.txt"

    # Extract basename only
    base = os.path.basename(filename).strip()

    # Remove null characters and control codes
    base = re.sub(r"[\x00-\x1f\x7f]", "", base)

    # Remove path traversal characters (/, \, ..)
    base = re.sub(r"[/\\]+", "", base)
    base = base.replace("..", "")

    # Replace forbidden Windows/Unix filename characters (<, >, :, ", |, ?, *)
    base = re.sub(r'[<>:"/\\|?*]', "_", base)

    # Trim leading/trailing dots and spaces
    base = base.strip(". ")

    if not base:
        return "unnamed_document.txt"

    return base


class DocumentValidator:
    """Validates document size, extensions, signatures, and format integrity."""

    def __init__(self, max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES) -> None:
        """Initialize validator with upload limits.

        Args:
            max_file_size_bytes: Maximum allowed document size in bytes (default 25 MB).
        """
        self.max_file_size_bytes = max_file_size_bytes

    def validate(self, file_bytes: bytes, raw_filename: str) -> ValidationResult:
        """Run complete validation suite against document bytes and filename.

        Args:
            file_bytes: In-memory raw content of the document.
            raw_filename: Original filename from upload.

        Returns:
            ValidationResult with status, sanitized filename, and diagnostic errors/warnings.
        """
        errors: list[str] = []
        warnings: list[str] = []

        sanitized_name = sanitize_filename(raw_filename)
        file_size = len(file_bytes)

        # 1. Check empty content
        if file_size == 0:
            errors.append("File is empty (0 bytes).")
            return ValidationResult(
                is_valid=False,
                sanitized_filename=sanitized_name,
                file_size_bytes=0,
                errors=errors,
            )

        # 2. Check maximum size limit
        if file_size > self.max_file_size_bytes:
            limit_mb = self.max_file_size_bytes / (1024 * 1024)
            size_mb = file_size / (1024 * 1024)
            errors.append(
                f"File size ({size_mb:.2f} MB) exceeds maximum allowed limit ({limit_mb:.1f} MB)."
            )
            return ValidationResult(
                is_valid=False,
                sanitized_filename=sanitized_name,
                file_size_bytes=file_size,
                errors=errors,
            )

        # 3. Check extension
        ext = Path(sanitized_name).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            errors.append(
                f"Unsupported file extension '{ext}'. Supported formats are: {', '.join(sorted(set(SUPPORTED_EXTENSIONS.values())))}."
            )
            return ValidationResult(
                is_valid=False,
                sanitized_filename=sanitized_name,
                file_size_bytes=file_size,
                errors=errors,
            )

        file_type = SUPPORTED_EXTENSIONS[ext]
        mime_type = MIME_MAPPINGS.get(file_type, "application/octet-stream")

        # 4. Signature & structural validation per format
        _sig_valid, sig_errors, sig_warnings = self._validate_signature_and_structure(
            file_bytes, file_type
        )
        errors.extend(sig_errors)
        warnings.extend(sig_warnings)

        return ValidationResult(
            is_valid=(len(errors) == 0),
            file_type=file_type,
            sanitized_filename=sanitized_name,
            file_size_bytes=file_size,
            mime_type=mime_type,
            errors=errors,
            warnings=warnings,
        )

    def _validate_signature_and_structure(
        self, file_bytes: bytes, file_type: str
    ) -> tuple[bool, list[str], list[str]]:
        """Verify file magic bytes and attempt lightweight structural parsing."""
        errors: list[str] = []
        warnings: list[str] = []

        if file_type == "pdf":
            # PDF magic byte check: must start with %PDF-
            if not file_bytes.startswith(b"%PDF-"):
                errors.append(
                    "Invalid PDF file signature: file does not start with standard '%PDF-' header."
                )
                return False, errors, warnings

            # Structural check via pypdf
            try:
                stream = io.BytesIO(file_bytes)
                reader = pypdf.PdfReader(stream)
                if reader.is_encrypted:
                    errors.append(
                        "Encrypted/password-protected PDF detected. Document cannot be processed without decryption."
                    )
                    return False, errors, warnings
                if len(reader.pages) == 0:
                    errors.append("PDF contains zero pages.")
                    return False, errors, warnings
            except pypdf.errors.PdfReadError as exc:
                errors.append(f"Malformed PDF document: {exc}")
                return False, errors, warnings
            except Exception as exc:
                errors.append(f"Error parsing PDF structure: {exc}")
                return False, errors, warnings

        elif file_type == "docx":
            # DOCX is a zip archive starting with PK\x03\x04
            if not file_bytes.startswith(b"PK\x03\x04"):
                errors.append(
                    "Invalid DOCX file signature: file is not a valid ZIP/OpenXML archive."
                )
                return False, errors, warnings

            # Check required internal OpenXML XML structures
            try:
                stream = io.BytesIO(file_bytes)
                with zipfile.ZipFile(stream, "r") as zf:
                    namelist = zf.namelist()
                    if "word/document.xml" not in namelist:
                        errors.append(
                            "Malformed DOCX: archive missing primary 'word/document.xml' part."
                        )
                        return False, errors, warnings

                # Test parsing with python-docx
                stream.seek(0)
                doc = docx.Document(stream)
                if doc is None:
                    errors.append("python-docx failed to open document.")
                    return False, errors, warnings
            except zipfile.BadZipFile as exc:
                errors.append(f"Corrupted DOCX archive: {exc}")
                return False, errors, warnings
            except Exception as exc:
                errors.append(f"Error validating DOCX document: {exc}")
                return False, errors, warnings

        elif file_type in ("txt", "md"):
            # Text / Markdown validation: ensure content is decodable text, not binary executable
            # Reject if file contains high proportion of null bytes
            null_count = file_bytes.count(b"\x00")
            if null_count > 0 and (null_count / len(file_bytes)) > 0.01:
                errors.append("Binary data detected in text file (excessive null bytes).")
                return False, errors, warnings

            # Test text decodability (UTF-8, UTF-8 BOM, latin-1)
            decoded = False
            for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
                try:
                    file_bytes.decode(enc)
                    decoded = True
                    if enc not in ("utf-8-sig", "utf-8"):
                        warnings.append(
                            f"Text file is encoded with fallback encoding '{enc}'. UTF-8 is recommended."
                        )
                    break
                except UnicodeDecodeError:
                    continue

            if not decoded:
                errors.append(
                    "Unable to decode text file with UTF-8 or standard fallback encodings."
                )
                return False, errors, warnings

        return len(errors) == 0, errors, warnings
