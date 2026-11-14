"""Unit tests validating security controls against OWASP Top 10 and LLM vulnerabilities."""

from graphrag_bench.answer_guardrails import AnswerGuardrails
from graphrag_bench.answer_schemas import AnswerStatus, EvidenceItem
from graphrag_bench.config import Settings
from graphrag_bench.document_validator import DocumentValidator, sanitize_filename
from graphrag_bench.prompt_builder import PromptBuilder


class TestSecurityAudit:
    """Test suite for security boundaries, sanitization, and refusal guardrails."""

    def test_secrets_redaction(self):
        """Verify secrets masking in to_safe_dict and string representation."""
        cfg = Settings(
            auth_mode="basic",
            _env_file=None,
        )
        safe = cfg.to_safe_dict()
        assert "ollama_base_url" in safe
        # Fields containing 'auth', 'key', 'token' are masked with '******'
        assert safe["auth_mode"] == "******"

        # Representation must mask sensitive fields
        rep = repr(cfg)
        assert "auth_mode='******'" in rep

    def test_forbidden_file_extensions_rejected(self):
        """Verify DocumentValidator rejects dangerous executable extensions."""
        validator = DocumentValidator()
        forbidden_files = ["malware.exe", "script.py", "payload.sh", "macro.vbs", "exploit.bat"]
        for fname in forbidden_files:
            res = validator.validate(b"echo hello", fname)
            assert res.is_valid is False
            assert any("Unsupported file extension" in e for e in res.errors)

    def test_pdf_magic_byte_enforcement(self):
        """Verify DocumentValidator rejects files masquerading as PDF with invalid magic header."""
        validator = DocumentValidator()
        res = validator.validate(b"This is not a real PDF file header", "spoofed.pdf")
        assert res.is_valid is False
        assert any("PDF file signature" in e for e in res.errors)

    def test_filename_path_traversal_sanitization(self):
        """Verify sanitize_filename neutralizes path traversal characters in filenames."""
        clean_name = sanitize_filename("../../etc/passwd.txt")
        assert ".." not in clean_name
        assert "/" not in clean_name
        assert "\\" not in clean_name
        assert clean_name == "passwd.txt"

    def test_prompt_delimiter_boundary_integrity(self):
        """Verify PromptBuilder encapsulates retrieved context and questions in strict delimiters."""
        item = EvidenceItem(
            evidence_id="S1",
            chunk_id="c1",
            document_id="d1",
            document_title="doc.md",
            content="Graph RAG uses knowledge graphs for multi-hop retrieval.",
            retrieval_score=0.9,
        )
        builder = PromptBuilder()
        _system_p, user_p = builder.build_qa_prompt(
            query="What is Graph RAG?",
            evidence_items=[item],
            require_citations=True,
        )
        assert "<<<EVIDENCE_BLOCK_START>>>" in user_p
        assert "<<<EVIDENCE_BLOCK_END>>>" in user_p
        assert "[S1] Source Document: doc.md" in user_p
        assert "What is Graph RAG?" in user_p
        assert "Attribute your statements to the sources above" in user_p

    def test_refusal_guardrail_on_empty_evidence(self):
        """Verify AnswerGuardrails short-circuits on empty evidence."""
        guardrails = AnswerGuardrails(min_evidence_items=1)
        res = guardrails.evaluate_pre_generation(
            query="What is the speed of light?", evidence_items=[]
        )
        assert res.can_proceed is False
        assert res.status == AnswerStatus.INSUFFICIENT_EVIDENCE
        assert "insufficient information" in res.early_answer.lower()
