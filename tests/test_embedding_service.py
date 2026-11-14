"""Unit tests for EmbeddingService."""

from unittest.mock import MagicMock

import pytest

from graphrag_bench.embedding_service import (
    DimensionMismatchError,
    EmbeddingError,
    EmbeddingResult,
    EmbeddingService,
    EmptyTextError,
    InvalidVectorError,
)
from graphrag_bench.ollama_client import (
    EmbeddingResponse,
    OllamaClient,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
)


@pytest.fixture
def mock_client() -> MagicMock:
    client = MagicMock(spec=OllamaClient)
    client.base_url = "http://localhost:11434"
    return client


class TestEmbeddingService:
    """Tests for EmbeddingService embedding generation and validation."""

    def test_embed_text_success(self, mock_client: MagicMock) -> None:
        mock_client.embed.return_value = EmbeddingResponse(
            model="nomic-embed-text",
            embeddings=[[0.1, 0.2, 0.3, 0.4]],
            dimensions=4,
        )
        service = EmbeddingService(client=mock_client)
        vector = service.embed_text("Test query string")

        assert isinstance(vector, list)
        assert len(vector) == 4
        assert vector == [0.1, 0.2, 0.3, 0.4]
        mock_client.embed.assert_called_once()

    def test_embed_batch_success(self, mock_client: MagicMock) -> None:
        mock_client.embed.return_value = EmbeddingResponse(
            model="nomic-embed-text",
            embeddings=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            dimensions=2,
        )
        service = EmbeddingService(client=mock_client)
        res = service.embed_batch(["text1", "text2", "text3"], batch_size=5)

        assert isinstance(res, EmbeddingResult)
        assert res.input_count == 3
        assert len(res.vectors) == 3
        assert res.dimensions == 2
        assert res.duration_ms > 0

    def test_embed_empty_text_raises_error(self, mock_client: MagicMock) -> None:
        service = EmbeddingService(client=mock_client)
        with pytest.raises(EmptyTextError):
            service.embed_text("")

        with pytest.raises(EmptyTextError):
            service.embed_text("   \n\t  ")

    def test_embed_empty_batch_raises_error(self, mock_client: MagicMock) -> None:
        service = EmbeddingService(client=mock_client)
        with pytest.raises(EmptyTextError):
            service.embed_batch([])

    def test_embed_batch_with_blank_item_raises_error(self, mock_client: MagicMock) -> None:
        service = EmbeddingService(client=mock_client)
        with pytest.raises(EmptyTextError) as exc:
            service.embed_batch(["valid text", "   ", "another valid"])
        assert "index 1" in str(exc.value)

    def test_dimension_mismatch_raises_error(self, mock_client: MagicMock) -> None:
        # Mock client returning vectors with different lengths
        mock_client.embed.return_value = EmbeddingResponse(
            model="nomic-embed-text",
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5]],  # 3 dims vs 2 dims
            dimensions=3,
        )
        service = EmbeddingService(client=mock_client)
        with pytest.raises(DimensionMismatchError) as exc:
            service.embed_batch(["doc1", "doc2"])
        assert "Inconsistent vector dimensions" in str(exc.value)

    def test_invalid_vector_nan_raises_error(self, mock_client: MagicMock) -> None:
        mock_client.embed.return_value = EmbeddingResponse(
            model="nomic-embed-text",
            embeddings=[[0.1, float("nan"), 0.3]],
            dimensions=3,
        )
        service = EmbeddingService(client=mock_client)
        with pytest.raises(InvalidVectorError):
            service.embed_text("test")

    def test_model_not_found_raises_embedding_error(self, mock_client: MagicMock) -> None:
        mock_client.embed.side_effect = OllamaModelNotFoundError("Model not found")
        service = EmbeddingService(client=mock_client)
        with pytest.raises(EmbeddingError) as exc:
            service.embed_text("test")
        assert "failed" in str(exc.value).lower()

    def test_connection_error_raises_embedding_error(self, mock_client: MagicMock) -> None:
        mock_client.embed.side_effect = OllamaConnectionError("Connection refused")
        service = EmbeddingService(client=mock_client)
        with pytest.raises(EmbeddingError):
            service.embed_text("test")

    def test_timeout_error_raises_embedding_error(self, mock_client: MagicMock) -> None:
        mock_client.embed.side_effect = OllamaTimeoutError("Timed out")
        service = EmbeddingService(client=mock_client)
        with pytest.raises(EmbeddingError):
            service.embed_text("test")
