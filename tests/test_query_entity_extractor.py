"""Unit tests for QueryEntityExtractor."""

from unittest.mock import MagicMock

from graphrag_bench.config import Settings
from graphrag_bench.graph_retrieval_schemas import ExtractionMethod
from graphrag_bench.graph_schemas import GraphEntity
from graphrag_bench.graph_store import GraphStore
from graphrag_bench.ollama_client import OllamaAPIError, OllamaClient, OllamaTimeoutError
from graphrag_bench.query_entity_extractor import QueryEntityExtractor


def create_sample_graph_store() -> GraphStore:
    """Create a GraphStore populated with representative benchmark entities."""
    store = GraphStore()
    store.add_entity(
        GraphEntity(
            entity_id="ent_vector_rag",
            canonical_name="Vector RAG",
            entity_type="TECHNOLOGY",
            aliases=["VRAG", "Dense Vector Retrieval"],
            source_document_ids=["doc_1"],
            source_chunk_ids=["chunk_1"],
        )
    )
    store.add_entity(
        GraphEntity(
            entity_id="ent_graph_rag",
            canonical_name="Graph RAG",
            entity_type="TECHNOLOGY",
            aliases=["GRAG", "Knowledge Graph RAG"],
            source_document_ids=["doc_1"],
            source_chunk_ids=["chunk_1"],
        )
    )
    store.add_entity(
        GraphEntity(
            entity_id="ent_chromadb",
            canonical_name="ChromaDB",
            entity_type="SOFTWARE",
            aliases=["Chroma"],
            source_document_ids=["doc_1"],
            source_chunk_ids=["chunk_1"],
        )
    )
    return store


class TestQueryEntityExtractor:
    """Test suite for deterministic fast-path and LLM fallback entity extraction."""

    def test_extract_exact_match(self) -> None:
        store = create_sample_graph_store()
        extractor = QueryEntityExtractor(store=store)

        res = extractor.extract_query_entities("How does Vector RAG work?")
        assert len(res.entities) == 1
        assert res.entities[0].canonical_name == "Vector RAG"
        assert res.entities[0].extraction_method == ExtractionMethod.EXACT_MATCH
        assert res.llm_invoked is False

    def test_extract_multiple_entities(self) -> None:
        store = create_sample_graph_store()
        extractor = QueryEntityExtractor(store=store)

        res = extractor.extract_query_entities("Compare Vector RAG and Graph RAG with ChromaDB.")
        assert len(res.entities) == 3
        names = {e.canonical_name for e in res.entities}
        assert names == {"Vector RAG", "Graph RAG", "ChromaDB"}
        assert res.llm_invoked is False

    def test_extract_alias_match(self) -> None:
        store = create_sample_graph_store()
        extractor = QueryEntityExtractor(store=store)

        res = extractor.extract_query_entities("Is VRAG faster than GRAG?")
        assert len(res.entities) == 2
        names = {e.canonical_name for e in res.entities}
        assert names == {"Vector RAG", "Graph RAG"}
        methods = {e.extraction_method for e in res.entities}
        assert ExtractionMethod.ALIAS_MATCH in methods

    def test_empty_or_whitespace_query(self) -> None:
        store = create_sample_graph_store()
        extractor = QueryEntityExtractor(store=store)

        res1 = extractor.extract_query_entities("")
        assert len(res1.entities) == 0
        assert res1.llm_invoked is False

        res2 = extractor.extract_query_entities("    \n\t  ")
        assert len(res2.entities) == 0

    def test_llm_fallback_invoked_when_no_deterministic_match(self) -> None:
        store = create_sample_graph_store()
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.generate.return_value.response = (
            '{"entities": [{"mention_text": "Hybrid RAG", "canonical_name": "Hybrid RAG", '
            '"entity_type": "TECHNOLOGY", "confidence": 0.95}]}'
        )
        extractor = QueryEntityExtractor(
            store=store,
            client=mock_client,
            settings=Settings(ollama_api_base="http://127.0.0.1:11434"),
        )

        res = extractor.extract_query_entities(
            "Explain the benefits of combining them in Hybrid RAG."
        )
        assert res.llm_invoked is True
        assert len(res.entities) >= 1
        assert any(e.canonical_name == "Hybrid RAG" for e in res.entities)

    def test_llm_fallback_malformed_json_triggers_heuristic_recovery(self) -> None:
        store = create_sample_graph_store()
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.generate.return_value.response = "Sorry, I cannot return JSON right now."
        extractor = QueryEntityExtractor(
            store=store,
            client=mock_client,
            settings=Settings(),
        )

        res = extractor.extract_query_entities(
            "Explain Multimodal Embeddings in benchmark pipeline."
        )
        assert res.llm_invoked is True
        # Heuristic recovery should capture significant non-stop words
        assert len(res.entities) >= 1
        assert any("Embeddings" in e.canonical_name for e in res.entities)

    def test_llm_fallback_timeout_graceful_handling(self) -> None:
        store = create_sample_graph_store()
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.generate.side_effect = OllamaTimeoutError("Request timed out after 300s")
        extractor = QueryEntityExtractor(
            store=store,
            client=mock_client,
            settings=Settings(),
        )

        res = extractor.extract_query_entities("Unknown semantic system comparison.")
        assert res.llm_invoked is True
        # Should gracefully return heuristic terms without raising uncaught exception
        assert isinstance(res.entities, list)

    def test_llm_fallback_api_error_graceful_handling(self) -> None:
        store = create_sample_graph_store()
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.generate.side_effect = OllamaAPIError("Model unavailable")
        extractor = QueryEntityExtractor(
            store=store,
            client=mock_client,
            settings=Settings(),
        )

        res = extractor.extract_query_entities("Deep learning graph systems.")
        assert res.llm_invoked is True
        assert isinstance(res.entities, list)
