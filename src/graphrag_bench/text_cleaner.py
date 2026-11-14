"""Text cleaning and normalization module for Graph RAG Benchmark.

Implements deterministic text sanitization:
- Unicode NFKC normalization
- Null and non-printable control character removal
- Line ending standardization (\r\n and \r -> \n)
- Horizontal whitespace normalization
- Excessive blank line collapsing (preserving paragraph breaks)
- Boundary whitespace trimming
- Preservation of citations, numbers, formulas, and Markdown syntax
"""

import re
import unicodedata


class TextCleaner:
    """Provides deterministic, meaning-preserving text cleaning and normalization."""

    @classmethod
    def clean(cls, text: str) -> str:
        """Clean and normalize extracted document text.

        Args:
            text: Raw extracted text string.

        Returns:
            Normalized, clean text ready for chunking.
        """
        if not text:
            return ""

        # 1. Unicode NFKC normalization (standardizes composite characters, ligatures)
        cleaned = unicodedata.normalize("NFKC", text)

        # 2. Remove null bytes and non-printable control characters except \t and \n
        # Control characters: 0x00-0x08, 0x0B-0x0C, 0x0E-0x1F, 0x7F
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)

        # 3. Standardize line endings to Unix style \n
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

        # 4. Remove trailing whitespace on each individual line
        cleaned = re.sub(r"[ \t]+$", "", cleaned, flags=re.MULTILINE)

        # 5. Normalize repeated horizontal spaces (2 or more spaces/tabs) to a single space
        # (while preserving intentional line breaks \n)
        cleaned = re.sub(r"[^\S\n\r]+", " ", cleaned)

        # 6. Collapse excessive consecutive blank lines (3 or more newlines become 2 newlines)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        # 7. Strip leading and trailing whitespace from the overall document
        cleaned = cleaned.strip()

        return cleaned
