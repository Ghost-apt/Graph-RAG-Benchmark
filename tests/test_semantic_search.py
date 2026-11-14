"""Unit tests for SemanticSearch retrieval service."""

from unittest.mock import MagicMock

import pytest

from graphrag_bench.embedding_service import EmbeddingError, EmbeddingService, EmptyTextError
from graphrag_bench.semantic_search import SearchError, SearchResponse, SemanticSearch
from graphrag_bench.vector_store import SearchResultItem, VectorStore, VectorStoreError


@pytest.fixture
def mock_embedding_service() -> MagicMock:
    service = MagicMock(spec=EmbeddingService)
    service.default_model = "nomic-embed-text"
    service.embed_text.return_value = [0.1, 0.2, 0.3, 0.4]
    return service


@pytest.fixture
def mock_vector_store() -> MagicMock:
    store = MagicMock(spec=VectorStore)
    store.count.return_value = 5
    store.similarity_search.return_value = [
        SearchResultItem(
            rank=1,
            chunk_id="chunk_1",
            document_id="doc_1",
            text="First relevant chunk content",
            distance=0.1,
            similarity_score=0.95,
            metadata={"source_filename": "doc1.txt", "file_type": "txt"},
        ),
        SearchResultItem(
            rank=2,
            chunk_id="chunk_2",
            document_id="doc_2",
            text="Second relevant chunk content",
            distance=0.3,
            similarity_score=0.85,
            metadata={"source_filename": "doc2.pdf", "file_type": "pdf"},
        ),
    ]
    return store


class TestSemanticSearch:
    """Test suite for SemanticSearch query execution, filtering, and error handling."""

    def test_search_empty_query_raises_error(
        self, mock_vector_store: MagicMock, mock_embedding_service: MagicMock
    ) -> None:
        engine = SemanticSearch(
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service,
        )
        with pytest.raises(EmptyTextError):
            engine.search("")

        with pytest.raises(EmptyTextError):
            engine.search("   \n\t  ")

    def test_search_empty_database_returns_empty_response(
        self, mock_embedding_service: MagicMock
    ) -> None:
        empty_store = MagicMock(spec=VectorStore)
        empty_store.count.return_value = 0

        engine = SemanticSearch(
            vector_store=empty_store,
            embedding_service=mock_embedding_service,
        )

        res = engine.search("What is Graph RAG?", top_k=3)
        assert isinstance(res, SearchResponse)
        assert res.total_found == 0
        assert res.items == []
        assert res.query == "What is Graph RAG?"
        mock_embedding_service.embed_text.assert_not_called()

    def test_search_normal_query_success(
        self, mock_vector_store: MagicMock, mock_embedding_service: MagicMock
    ) -> None:
        engine = SemanticSearch(
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service,
        )

        res = engine.search("Explain vector embeddings", top_k=2)
        assert isinstance(res, SearchResponse)
        assert res.query == "Explain vector embeddings"
        assert res.top_k == 2
        assert res.total_found == 2
        assert len(res.items) == 2
        assert res.items[0].chunk_id == "chunk_1"
        assert res.items[0].rank == 1
        assert res.items[0].distance == 0.1
        assert res.items[0].cosine_similarity == 0.9
        assert res.items[0].normalized_score == 0.95
        assert res.model_used == "nomic-embed-text"
        assert res.applied_filter is None

        mock_embedding_service.embed_text.assert_called_once_with("Explain vector embeddings")
        mock_vector_store.similarity_search.assert_called_once_with(
            query_embedding=[0.1, 0.2, 0.3, 0.4],
            top_k=2,
            where_filter=None,
        )

    def test_search_with_document_filter(
        self, mock_vector_store: MagicMock, mock_embedding_service: MagicMock
    ) -> None:
        engine = SemanticSearch(
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service,
        )

        res = engine.search("Test query", document_id="doc_1")
        assert res.applied_filter == {"document_id": "doc_1"}
        mock_vector_store.similarity_search.assert_called_once_with(
            query_embedding=[0.1, 0.2, 0.3, 0.4],
            top_k=5,
            where_filter={"document_id": "doc_1"},
        )

    def test_search_with_file_type_filter(
        self, mock_vector_store: MagicMock, mock_embedding_service: MagicMock
    ) -> None:
        engine = SemanticSearch(
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service,
        )

        res = engine.search("Test query", file_type="PDF")
        assert res.applied_filter == {"file_type": "pdf"}
        mock_vector_store.similarity_search.assert_called_once_with(
            query_embedding=[0.1, 0.2, 0.3, 0.4],
            top_k=5,
            where_filter={"file_type": "pdf"},
        )

    def test_search_with_both_filters(
        self, mock_vector_store: MagicMock, mock_embedding_service: MagicMock
    ) -> None:
        engine = SemanticSearch(
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service,
        )

        res = engine.search("Test query", document_id="doc_1", file_type="txt")
        assert res.applied_filter is not None
        assert "$and" in res.applied_filter
        assert len(res.applied_filter["$and"]) == 2

    def test_embedding_error_raises_search_error(
        self, mock_vector_store: MagicMock, mock_embedding_service: MagicMock
    ) -> None:
        mock_embedding_service.embed_text.side_effect = EmbeddingError("Ollama connection lost")
        engine = SemanticSearch(
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service,
        )

        with pytest.raises(SearchError) as exc:
            engine.search("Failed query")
        assert "embedding generation failed" in str(exc.value).lower()

    def test_vector_store_error_raises_search_error(
        self, mock_vector_store: MagicMock, mock_embedding_service: MagicMock
    ) -> None:
        mock_vector_store.similarity_search.side_effect = VectorStoreError("Corrupted Chroma index")
        engine = SemanticSearch(
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service,
        )

        with pytest.raises(SearchError) as exc:
            engine.search("Failed vector query")
        assert "vector search execution failed" in str(exc.value).lower()
