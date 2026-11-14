"""Unit tests for Faithfulness Checker (graphrag_bench.faithfulness_checker)."""

from unittest.mock import MagicMock

from graphrag_bench.answer_schemas import EvidenceItem
from graphrag_bench.faithfulness_checker import FaithfulnessChecker
from graphrag_bench.ollama_client import GenerateResponse, OllamaClient


class TestFaithfulnessChecker:
    """Test suite for claim extraction and grounding faithfulness verification."""

    def test_split_into_heuristic_claims(self):
        """Test sentence-level claim splitting rules."""
        checker = FaithfulnessChecker()
        text = (
            "Graph RAG utilizes NetworkX for constructing knowledge graphs [S1]. "
            "It also leverages ChromaDB for semantic vector retrieval [S2]. "
            "In conclusion, both modalities complement each other."
        )
        claims = checker.split_into_heuristic_claims(text)
        assert len(claims) >= 2
        assert any("NetworkX" in c for c in claims)
        assert any("ChromaDB" in c for c in claims)

    def test_evaluate_heuristic_grounding_supported(self):
        """Test lexical overlap grounding for directly supported claims."""
        ev = EvidenceItem(
            evidence_id="S1",
            chunk_id="c1",
            document_id="d1",
            document_title="guide.md",
            content="NetworkX is used to construct directed knowledge graphs with deterministic entity identifiers.",
        )
        checker = FaithfulnessChecker()
        claim = (
            "NetworkX constructs directed knowledge graphs with deterministic entity identifiers."
        )
        verdict, _reasoning, score = checker.evaluate_heuristic_grounding(claim, [ev])
        assert verdict == "SUPPORTED"
        assert score > 0.5

    def test_evaluate_heuristic_grounding_unsupported(self):
        """Test lexical overlap grounding for completely unsupported claims."""
        ev = EvidenceItem(
            evidence_id="S1",
            chunk_id="c1",
            document_id="d1",
            document_title="guide.md",
            content="NetworkX is used to construct directed knowledge graphs.",
        )
        checker = FaithfulnessChecker()
        claim = "Quantum computing algorithms simulate galactic dark matter distribution."
        verdict, _reasoning, score = checker.evaluate_heuristic_grounding(claim, [ev])
        assert verdict == "UNSUPPORTED"
        assert score < 0.25

    def test_check_faithfulness_heuristic_fallback(self):
        """Test end-to-end check_faithfulness in heuristic mode."""
        ev = EvidenceItem(
            evidence_id="S1",
            chunk_id="c1",
            document_id="d1",
            document_title="guide.md",
            content="Graph RAG combines vector search with knowledge graph multi-hop traversal.",
        )
        checker = FaithfulnessChecker()
        answer = "Graph RAG combines vector search with knowledge graph multi-hop traversal [S1]."
        claims, score = checker.check_faithfulness(answer, [ev], use_llm=False)
        assert len(claims) >= 1
        assert score == 1.0
        assert claims[0].verdict == "SUPPORTED"

    def test_check_faithfulness_with_mocked_llm(self):
        """Test LLM-based claim extraction and verification with mocked OllamaClient."""
        mock_client = MagicMock(spec=OllamaClient)
        # First call: extract claims JSON
        mock_client.generate.side_effect = [
            GenerateResponse(
                model="qwen3:8b",
                response='["Graph RAG combines vector and graph search."]',
                done=True,
            ),
            # Second call: verify claim JSON
            GenerateResponse(
                model="qwen3:8b",
                response='{"verdict": "SUPPORTED", "reasoning": "Explicitly stated in evidence S1.", "grounding_score": 1.0}',
                done=True,
            ),
        ]

        ev = EvidenceItem(
            evidence_id="S1",
            chunk_id="c1",
            document_id="d1",
            document_title="guide.md",
            content="Graph RAG combines vector and graph search.",
        )
        checker = FaithfulnessChecker(ollama_client=mock_client)
        claims, score = checker.check_faithfulness(
            "Graph RAG combines vector and graph search [S1].", [ev], use_llm=True
        )

        assert len(claims) == 1
        assert claims[0].verdict == "SUPPORTED"
        assert score == 1.0

    def test_empty_claims_returns_null_faithfulness(self):
        """When zero factual claims are evaluated, faithfulness_score must be None (not 0.0 or 1.0)."""
        ev = EvidenceItem(
            evidence_id="S1",
            chunk_id="c1",
            document_id="d1",
            document_title="guide.md",
            content="Some reference content.",
        )
        checker = FaithfulnessChecker()

        # Empty string
        claims_empty, score_empty = checker.check_faithfulness("", [ev], use_llm=False)
        assert claims_empty == []
        assert score_empty is None

        # Pure disclaimer with no declarative factual assertions
        disclaimer = "Based on the provided evidence, there is insufficient information to answer this question."
        claims_disc, score_disc = checker.check_faithfulness(disclaimer, [ev], use_llm=False)
        assert len(claims_disc) == 0
        assert score_disc is None

    def test_uncertain_verdict_partial_overlap(self):
        """Test that partial lexical overlap (between 25% and 50%) yields UNCERTAIN verdict."""
        ev = EvidenceItem(
            evidence_id="S1",
            chunk_id="c1",
            document_id="d1",
            document_title="guide.md",
            content="Reciprocal Rank Fusion merges ranked retrieval lists using reciprocal scores.",
        )
        checker = FaithfulnessChecker()
        # Shares 'Reciprocal' and 'Fusion', but has other unrelated terms
        claim = (
            "Reciprocal Rank Fusion optimizes neural network weights via quantum backpropagation."
        )
        verdict, reasoning, score = checker.evaluate_heuristic_grounding(claim, [ev])
        assert verdict == "UNCERTAIN"
        assert 0.25 <= score < 0.50
        assert "Partial lexical overlap" in reasoning

    def test_q1_faithfulness_demonstration_reconciliation(self):
        """Audit Q1 condition: 4 atomic claims with 3 supported and 1 unsupported yields 75% faithfulness."""
        ev1 = EvidenceItem(
            evidence_id="S1",
            chunk_id="c1",
            document_id="d1",
            document_title="guide.md",
            content=(
                "Graph RAG represents an advanced hybrid architecture integrating vector search "
                "with structural knowledge graph traversal. It constructs entity graphs using NetworkX "
                "and performs semantic retrieval with ChromaDB."
            ),
        )
        checker = FaithfulnessChecker()

        # Answer composed of 4 candidate sentences
        # 3 are grounded in ev1; 1 makes an ungrounded claim about blockchain consensus
        answer = (
            "Graph RAG represents an advanced hybrid architecture integrating vector search [S1]. "
            "It constructs entity graphs using NetworkX [S1]. "
            "It performs semantic retrieval with ChromaDB [S1]. "
            "Graph RAG also requires proof-of-stake blockchain consensus validators."
        )

        claims, faithfulness = checker.check_faithfulness(answer, [ev1], use_llm=False)
        assert len(claims) == 4
        supported = [c for c in claims if c.verdict == "SUPPORTED"]
        unsupported = [c for c in claims if c.verdict in ("UNSUPPORTED", "UNCERTAIN")]
        assert len(supported) == 3
        assert len(unsupported) == 1
        assert faithfulness == 0.75  # Exactly 75.0%
