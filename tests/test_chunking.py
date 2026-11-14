"""Unit tests for TextChunker."""

import pytest

from graphrag_bench.chunking import TextChunker
from graphrag_bench.document_loader import ExtractedPage


class TestTextChunker:
    """Tests for sliding-window word chunking."""

    def test_chunker_invalid_parameters(self) -> None:
        with pytest.raises(ValueError) as exc:
            TextChunker(chunk_size=0)
        assert "chunk_size must be positive" in str(exc.value)

        with pytest.raises(ValueError) as exc:
            TextChunker(chunk_size=100, chunk_overlap=-1)
        assert "chunk_overlap cannot be negative" in str(exc.value)

        with pytest.raises(ValueError) as exc:
            TextChunker(chunk_size=100, chunk_overlap=100)
        assert "chunk_overlap must be strictly less than chunk_size" in str(exc.value)

    def test_empty_text_returns_empty_list(self) -> None:
        chunker = TextChunker()
        chunks = chunker.chunk_text("", doc_id="doc_1234567890abcdef")
        assert chunks == []

    def test_short_text_single_chunk(self) -> None:
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = "This is a brief text document that contains fewer than fifty words in total."
        chunks = chunker.chunk_text(text, doc_id="doc_1234567890abcdef")

        assert len(chunks) == 1
        assert chunks[0].chunk_index == 0
        assert chunks[0].text == text
        assert chunks[0].word_count == len(text.split())
        assert chunks[0].start_char == 0
        assert chunks[0].end_char == len(text)
        assert text[chunks[0].start_char : chunks[0].end_char] == chunks[0].text

    def test_sliding_window_overlap(self) -> None:
        chunker = TextChunker(chunk_size=10, chunk_overlap=3)
        # Create a 25-word sequence
        words = [f"word{i}" for i in range(25)]
        text = " ".join(words)
        chunks = chunker.chunk_text(text, doc_id="doc_testoverlap1234")

        # Stride is 10 - 3 = 7
        # Window 0: 0..10 (words 0..9)
        # Window 1: 7..17 (words 7..16)
        # Window 2: 14..24 (words 14..23)
        # Window 3: 21..25 (words 21..24)
        assert len(chunks) == 4

        # Check overlapping words between chunk 0 and chunk 1:
        # words 7, 8, 9 should be at the end of chunk 0 and beginning of chunk 1
        assert "word7 word8 word9" in chunks[0].text
        assert chunks[1].text.startswith("word7 word8 word9")

    def test_exact_character_offsets(self) -> None:
        chunker = TextChunker(chunk_size=15, chunk_overlap=5)
        # Multiline text with varied punctuation
        text = (
            "Knowledge graph retrieval provides multi-hop reasoning capabilities.\n"
            "By mapping entities and relationships into a structured graph, questions requiring\n"
            "connecting facts across multiple documents can be answered accurately."
        )
        chunks = chunker.chunk_text(text, doc_id="doc_offsettest1234")

        assert len(chunks) > 1
        for chunk in chunks:
            # Crucial requirement: text sliced by [start_char:end_char] must equal chunk.text exactly
            sliced = text[chunk.start_char : chunk.end_char]
            assert sliced == chunk.text

    def test_chunk_id_format(self) -> None:
        chunker = TextChunker(chunk_size=20, chunk_overlap=5)
        text = "This is a test document to verify deterministic chunk ID generation format."
        chunks = chunker.chunk_text(text, doc_id="doc_abcdef1234567890")

        assert len(chunks) == 1
        cid = chunks[0].chunk_id
        # Expect format: doc_{16chars}_c0000_{8chars}
        assert cid.startswith("doc_abcdef1234567890_c0000_")
        assert len(cid.split("_")) == 4

    def test_page_number_mapping(self) -> None:
        chunker = TextChunker(chunk_size=10, chunk_overlap=2)
        # Page 1: 15 words
        p1_text = "Page one contains fifteen words of important text data for evaluating document split accuracy now."
        # Page 2: 15 words
        p2_text = "Page two continues the document discussion with fifteen additional words describing knowledge graph benchmarking methodology."
        full_text = p1_text + "\n\n" + p2_text

        pages = [
            ExtractedPage(page_number=1, text=p1_text, char_start=0, char_end=len(p1_text)),
            ExtractedPage(
                page_number=2,
                text=p2_text,
                char_start=len(p1_text) + 2,
                char_end=len(full_text),
            ),
        ]

        chunks = chunker.chunk_text(full_text, doc_id="doc_pagemapping123", pages=pages)
        assert len(chunks) >= 3

        # First chunk should be on page 1
        assert chunks[0].page_number == 1
        # Last chunk should be on page 2
        assert chunks[-1].page_number == 2

    def test_controlled_word_counts_default_configuration(self) -> None:
        """Verify chunk counts for specific controlled word counts with default settings (400 / 60)."""
        chunker = TextChunker(chunk_size=400, chunk_overlap=60)

        # 60 words -> 1 chunk
        text_60 = " ".join([f"w{i}" for i in range(60)])
        c_60 = chunker.chunk_text(text_60, doc_id="doc_60")
        assert len(c_60) == 1
        assert c_60[0].word_count == 60
        assert text_60[c_60[0].start_char : c_60[0].end_char] == c_60[0].text

        # 157 words -> 1 chunk
        text_157 = " ".join([f"w{i}" for i in range(157)])
        c_157 = chunker.chunk_text(text_157, doc_id="doc_157")
        assert len(c_157) == 1
        assert c_157[0].word_count == 157

        # 163 words -> 1 chunk
        text_163 = " ".join([f"w{i}" for i in range(163)])
        c_163 = chunker.chunk_text(text_163, doc_id="doc_163")
        assert len(c_163) == 1
        assert c_163[0].word_count == 163

        # 400 words -> exactly 1 chunk
        text_400 = " ".join([f"w{i}" for i in range(400)])
        c_400 = chunker.chunk_text(text_400, doc_id="doc_400")
        assert len(c_400) == 1
        assert c_400[0].word_count == 400

        # 401 words -> 2 chunks (window 0..400 and window 340..401)
        text_401 = " ".join([f"w{i}" for i in range(401)])
        c_401 = chunker.chunk_text(text_401, doc_id="doc_401")
        assert len(c_401) == 2
        assert c_401[0].word_count == 400
        assert c_401[1].word_count == 61
        # Verify 60-word overlap
        assert c_401[0].text.split()[-60:] == c_401[1].text.split()[:60]

        # 800 words -> with stride 340 (400 - 60):
        # Chunk 0: words 0..400 (400 words)
        # Chunk 1: words 340..740 (400 words)
        # Chunk 2: words 680..800 (120 words)
        # Guarantees all 800 words are retained without exceeding 400-word limit
        text_800 = " ".join([f"w{i}" for i in range(800)])
        c_800 = chunker.chunk_text(text_800, doc_id="doc_800")
        assert len(c_800) == 3
        assert [c.word_count for c in c_800] == [400, 400, 120]

        # 1000 words -> 3 chunks
        # Chunk 0: words 0..400 (400 words)
        # Chunk 1: words 340..740 (400 words)
        # Chunk 2: words 680..1000 (320 words)
        text_1000 = " ".join([f"w{i}" for i in range(1000)])
        c_1000 = chunker.chunk_text(text_1000, doc_id="doc_1000")
        assert len(c_1000) == 3
        assert [c.word_count for c in c_1000] == [400, 400, 320]

    def test_custom_chunk_configurations(self) -> None:
        """Verify demonstration settings: small chunk sizes produce expected multiple chunks."""
        text_163 = " ".join([f"w{i}" for i in range(163)])
        # chunk_size=80, overlap=20, stride=60 -> 3 chunks
        chunker_80_20 = TextChunker(chunk_size=80, chunk_overlap=20)
        c_80 = chunker_80_20.chunk_text(text_163, doc_id="doc_163_custom")
        assert len(c_80) == 3
        assert [c.word_count for c in c_80] == [80, 80, 43]

        text_60 = " ".join([f"w{i}" for i in range(60)])
        # chunk_size=50, overlap=10, stride=40 -> 2 chunks
        chunker_50_10 = TextChunker(chunk_size=50, chunk_overlap=10)
        c_50 = chunker_50_10.chunk_text(text_60, doc_id="doc_60_custom")
        assert len(c_50) == 2
        assert [c.word_count for c in c_50] == [50, 20]
