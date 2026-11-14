"""Unit tests for GraphExtractionService."""

from unittest.mock import MagicMock

import pytest

from graphrag_bench.graph_extraction_service import GraphExtractionService
from graphrag_bench.ollama_client import OllamaClient


@pytest.fixture
def sample_chunk_text() -> str:
    return (
        "Retrieval-Augmented Generation (RAG) is an AI architecture that enhances large language "
        "models by retrieving relevant context from external knowledge bases. ChromaDB is a vector "
        "database used to store and search dense vector embeddings efficiently. NetworkX is a Python "
        "library used to construct and analyze knowledge graph structures."
    )


def test_successful_extraction_and_grounding(sample_chunk_text):
    mock_ollama = MagicMock(spec=OllamaClient)
    # LLM returns structured JSON with entities and relationships
    mock_ollama.generate.return_value = {
        "response": """```json
{
  "entities": [
    {
      "mention_text": "Retrieval-Augmented Generation",
      "canonical_name": "Retrieval-Augmented Generation",
      "entity_type": "CONCEPT",
      "description": "AI architecture that enhances LLMs",
      "aliases": ["RAG"]
    },
    {
      "mention_text": "ChromaDB",
      "canonical_name": "ChromaDB",
      "entity_type": "SOFTWARE",
      "description": "Vector database for dense embeddings",
      "aliases": ["Chroma"]
    }
  ],
  "relationships": [
    {
      "subject": "Retrieval-Augmented Generation",
      "predicate": "USES",
      "object": "ChromaDB",
      "evidence_quote": "ChromaDB is a vector database used to store and search dense vector embeddings efficiently.",
      "confidence": 0.95
    }
  ]
}
```"""
    }

    service = GraphExtractionService(ollama_client=mock_ollama, min_confidence=0.5)
    result = service.extract_from_chunk(sample_chunk_text, chunk_id="test_chunk_1")

    assert len(result.entities) == 2
    assert result.entities[0].canonical_name == "Retrieval-Augmented Generation"
    assert result.entities[0].entity_type == "CONCEPT"
    assert result.entities[1].canonical_name == "ChromaDB"

    assert len(result.relationships) == 1
    assert result.relationships[0].predicate == "USES"
    assert result.relationships[0].confidence == 0.95
    assert result.source_chunk_id == "test_chunk_1"


def test_hallucinated_evidence_quote_rejected(sample_chunk_text):
    """Verify that relationships with evidence quotes not present in the chunk are rejected."""
    mock_ollama = MagicMock(spec=OllamaClient)
    mock_ollama.generate.return_value = {
        "response": """{
  "entities": [
    {"mention_text": "RAG", "canonical_name": "RAG", "entity_type": "CONCEPT"}
  ],
  "relationships": [
    {
      "subject": "RAG",
      "predicate": "DEVELOPED_BY",
      "object": "OpenAI",
      "evidence_quote": "OpenAI invented RAG in the year 2020.",
      "confidence": 0.99
    }
  ]
}"""
    }

    service = GraphExtractionService(ollama_client=mock_ollama, min_confidence=0.5)
    result = service.extract_from_chunk(sample_chunk_text, chunk_id="test_chunk_2")

    # The hallucinated relationship must be rejected
    assert len(result.relationships) == 0
    # The entity should still be retained if grounded or valid
    assert len(result.entities) == 1


def test_confidence_threshold_filtering(sample_chunk_text):
    """Verify relationships with confidence below min_confidence are filtered out."""
    mock_ollama = MagicMock(spec=OllamaClient)
    mock_ollama.generate.return_value = {
        "response": """{
  "entities": [
    {"mention_text": "ChromaDB", "canonical_name": "ChromaDB", "entity_type": "SOFTWARE"}
  ],
  "relationships": [
    {
      "subject": "ChromaDB",
      "predicate": "RELATED_TO",
      "object": "NetworkX",
      "evidence_quote": "ChromaDB is a vector database used to store and search dense vector embeddings efficiently.",
      "confidence": 0.40
    }
  ]
}"""
    }

    service = GraphExtractionService(ollama_client=mock_ollama, min_confidence=0.70)
    result = service.extract_from_chunk(sample_chunk_text, chunk_id="test_chunk_3")

    # Confidence 0.40 < 0.70 min_confidence
    assert len(result.relationships) == 0


def test_empty_or_malformed_llm_response(sample_chunk_text):
    """Verify graceful handling when LLM response is unparseable text."""
    mock_ollama = MagicMock(spec=OllamaClient)
    mock_ollama.generate.return_value = {
        "response": "I am not able to extract any knowledge from this text."
    }

    service = GraphExtractionService(ollama_client=mock_ollama)
    result = service.extract_from_chunk(sample_chunk_text, chunk_id="test_chunk_4")

    assert len(result.entities) == 0
    assert len(result.relationships) == 0
