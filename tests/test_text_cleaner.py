"""Unit tests for TextCleaner."""

import pytest

from graphrag_bench.text_cleaner import TextCleaner


class TestTextCleaner:
    """Tests for text normalization and sanitization."""

    @pytest.fixture
    def cleaner(self) -> TextCleaner:
        return TextCleaner()

    def test_clean_empty_text(self, cleaner: TextCleaner) -> None:
        assert cleaner.clean("") == ""
        assert cleaner.clean("    \n\n\t   ") == ""

    def test_unicode_nfkc_normalization(self, cleaner: TextCleaner) -> None:
        # Fullwidth characters and ligatures
        ligature_text = "ﬁnding ﬀ efficient ﬃ solutions"
        cleaned = cleaner.clean(ligature_text)
        assert cleaned == "finding ff efficient ffi solutions"

    def test_control_character_stripping(self, cleaner: TextCleaner) -> None:
        text_with_control = "Hello\x00\x07\x1bWorld\x0c!"
        cleaned = cleaner.clean(text_with_control)
        assert cleaned == "HelloWorld!"

    def test_newline_standardization(self, cleaner: TextCleaner) -> None:
        crlf_text = "Line 1\r\nLine 2\r\nLine 3"
        cleaned = cleaner.clean(crlf_text)
        assert cleaned == "Line 1\nLine 2\nLine 3"
        assert "\r" not in cleaned

    def test_consecutive_blank_lines_collapsed(self, cleaner: TextCleaner) -> None:
        text = "Paragraph 1\n\n\n\n\nParagraph 2"
        cleaned = cleaner.clean(text)
        assert cleaned == "Paragraph 1\n\nParagraph 2"

    def test_multiple_spaces_collapsed(self, cleaner: TextCleaner) -> None:
        text = "This   has    excessive     spacing."
        cleaned = cleaner.clean(text)
        assert cleaned == "This has excessive spacing."

    def test_preserves_citations(self, cleaner: TextCleaner) -> None:
        text = "Knowledge graph retrieval outperforms naive vector search [1, 2]. As shown in [14]."
        cleaned = cleaner.clean(text)
        assert "[1, 2]" in cleaned
        assert "[14]" in cleaned

    def test_preserves_code_and_formatting(self, cleaner: TextCleaner) -> None:
        code_block = (
            "Here is the algorithm:\n\n"
            "```python\n"
            "def retrieve(query: str):\n"
            "    return graph.search(query)\n"
            "```"
        )
        cleaned = cleaner.clean(code_block)
        assert "```python" in cleaned
        assert "def retrieve(query: str):" in cleaned

    def test_preserves_bullet_lists_and_math(self, cleaner: TextCleaner) -> None:
        text = "- Item 1: precision\n- Item 2: recall\nFormula: E = mc^2 and f(x) = sum(x_i)"
        cleaned = cleaner.clean(text)
        assert "- Item 1: precision" in cleaned
        assert "- Item 2: recall" in cleaned
        assert "E = mc^2 and f(x) = sum(x_i)" in cleaned
