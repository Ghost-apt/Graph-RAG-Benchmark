"""Unit tests for Answer Generator (graphrag_bench.answer_generator)."""

from unittest.mock import MagicMock

import pytest

from graphrag_bench.answer_generator import AnswerGenerator, strip_think_tags
from graphrag_bench.answer_schemas import EvidenceItem
from graphrag_bench.ollama_client import GenerateResponse, OllamaAPIError, OllamaClient


class TestAnswerGenerator:
    """Test suite for LLM answer generation and reasoning token isolation."""

    def test_strip_think_tags(self):
        """Test separating <think> tokens from clean generated output."""
        raw = "<think>Analyzing context chunks S1 and S2...</think>Graph RAG integrates knowledge graphs [S1]."
        clean, thinking = strip_think_tags(raw)
        assert clean == "Graph RAG integrates knowledge graphs [S1]."
        assert thinking == "Analyzing context chunks S1 and S2..."

        # No think tags
        plain = "Direct answer without reasoning tokens [S1]."
        clean_plain, thinking_plain = strip_think_tags(plain)
        assert clean_plain == "Direct answer without reasoning tokens [S1]."
        assert thinking_plain is None

    def test_generate_success(self):
        """Test successful answer generation with mocked OllamaClient."""
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.default_llm = "qwen3:8b"
        mock_client.generate.return_value = GenerateResponse(
            model="qwen3:8b",
            response="Graph RAG evaluates multi-hop questions [S1].",
            done=True,
            eval_count=20,
            prompt_eval_count=150,
            total_duration_ns=500_000_000,
        )

        generator = AnswerGenerator(ollama_client=mock_client)
        item = EvidenceItem(
            evidence_id="S1",
            chunk_id="c1",
            document_id="d1",
            document_title="guide.md",
            content="Evidence text",
        )

        answer, thinking, stats = generator.generate(
            query="What does Graph RAG evaluate?",
            evidence_items=[item],
            temperature=0.1,
            max_tokens=256,
        )

        assert answer == "Graph RAG evaluates multi-hop questions [S1]."
        assert thinking is None
        assert stats["model"] == "qwen3:8b"
        assert stats["latency_ms"] > 0
        assert stats["eval_count"] == 20
        mock_client.generate.assert_called_once()

    def test_generate_ollama_error_propagates(self):
        """Test that Ollama errors raise appropriately."""
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.default_llm = "qwen3:8b"
        mock_client.generate.side_effect = OllamaAPIError("Connection refused")

        generator = AnswerGenerator(ollama_client=mock_client)
        with pytest.raises(OllamaAPIError):
            generator.generate(
                query="Test",
                evidence_items=[],
            )

    def test_generate_stream(self):
        """Test streaming generator token yield."""
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.default_llm = "qwen3:8b"
        mock_client.generate_stream.return_value = iter(["Graph ", "RAG ", "is ", "effective."])

        generator = AnswerGenerator(ollama_client=mock_client)
        tokens = list(generator.generate_stream(query="Test", evidence_items=[]))
        assert tokens == ["Graph ", "RAG ", "is ", "effective."]
