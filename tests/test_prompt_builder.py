"""Unit tests for Grounded QA prompt builder (graphrag_bench.prompt_builder)."""

from graphrag_bench.answer_schemas import EvidenceItem
from graphrag_bench.prompt_builder import (
    GROUNDED_QA_SYSTEM_PROMPT,
    PromptBuilder,
)


class TestPromptBuilder:
    """Test suite for anti-injection prompt templates and grounding instructions."""

    def test_build_qa_prompt_with_citations(self):
        """Test building QA prompt with mandatory citation markers."""
        item = EvidenceItem(
            evidence_id="S1",
            chunk_id="chunk_test",
            document_id="doc_test",
            document_title="test.txt",
            content="Qwen is a large language model family.",
            retrieval_score=0.9,
        )
        builder = PromptBuilder()
        system_p, user_p = builder.build_qa_prompt(
            query="What is Qwen?",
            evidence_items=[item],
            require_citations=True,
        )

        assert "You are an expert AI research assistant" in system_p
        assert "Citation Attribution" in system_p
        assert "<<<EVIDENCE_BLOCK_START>>>" in user_p
        assert "<<<EVIDENCE_BLOCK_END>>>" in user_p
        assert "[S1] Source Document: test.txt" in user_p
        assert "What is Qwen?" in user_p
        assert "Attribute your statements to the sources above" in user_p

    def test_build_qa_prompt_without_citations(self):
        """Test building QA prompt when citations are disabled."""
        item = EvidenceItem(
            evidence_id="S1",
            chunk_id="chunk_test",
            document_id="doc_test",
            document_title="test.txt",
            content="Some factual content.",
        )
        builder = PromptBuilder()
        _system_p, user_p = builder.build_qa_prompt(
            query="Explain facts.",
            evidence_items=[item],
            require_citations=False,
        )

        assert "Do not include citation markers." in user_p

    def test_build_combined_prompt(self):
        """Test single string combined prompt generation."""
        builder = PromptBuilder()
        combined = builder.build_combined_prompt(
            query="Test query",
            evidence_items=[],
        )
        assert GROUNDED_QA_SYSTEM_PROMPT in combined
        assert "User Question:\nTest query" in combined

    def test_build_claim_extraction_and_verification_prompts(self):
        """Test claim extraction and verification prompt generation."""
        builder = PromptBuilder()
        sys_ext, usr_ext = builder.build_claim_extraction_prompt("Graph RAG uses NetworkX [S1].")
        assert "factual claim extraction" in sys_ext
        assert "Graph RAG uses NetworkX [S1]." in usr_ext

        item = EvidenceItem(
            evidence_id="S1",
            chunk_id="c1",
            document_id="d1",
            document_title="doc.md",
            content="Graph RAG builds graphs using NetworkX.",
        )
        sys_ver, usr_ver = builder.build_claim_verification_prompt(
            claim="Graph RAG uses NetworkX.",
            evidence_items=[item],
        )
        assert "strict fact-checking" in sys_ver
        assert "Claim to Verify:" in usr_ver
        assert "[S1] Graph RAG builds graphs using NetworkX." in usr_ver
