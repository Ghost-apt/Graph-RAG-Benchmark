"""Unit tests for Ollama HTTP client (graphrag_bench.ollama_client).

All tests in this module use requests-mock to simulate Ollama HTTP responses.
A running Ollama server is NOT required.
"""

import pytest
import requests

from graphrag_bench.config import Settings
from graphrag_bench.ollama_client import (
    EmbeddingResponse,
    GenerateResponse,
    OllamaAPIError,
    OllamaClient,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
)


@pytest.fixture
def mock_settings():
    """Isolated test settings."""
    return Settings(
        ollama_base_url="http://mock-ollama:11434",
        ollama_llm_model="qwen3:8b",
        ollama_embed_model="nomic-embed-text",
        ollama_timeout=5,
        _env_file=None,
    )


@pytest.fixture
def client(mock_settings):
    """Instantiated client targeting mock URL."""
    return OllamaClient(settings=mock_settings)


class TestOllamaClientReachable:
    """Tests for server reachability checks."""

    def test_server_reachable_success(self, client, requests_mock):
        requests_mock.get("http://mock-ollama:11434/api/tags", status_code=200, json={"models": []})
        assert client.is_server_reachable() is True

    def test_server_reachable_failure_status(self, client, requests_mock):
        requests_mock.get("http://mock-ollama:11434/api/tags", status_code=500)
        assert client.is_server_reachable() is False

    def test_server_reachable_connection_error(self, client, requests_mock):
        requests_mock.get(
            "http://mock-ollama:11434/api/tags",
            exc=requests.exceptions.ConnectionError("Connection refused"),
        )
        assert client.is_server_reachable() is False


class TestOllamaClientListModels:
    """Tests for model catalog listing and presence verification."""

    def test_list_models_success(self, client, requests_mock):
        payload = {
            "models": [
                {"name": "qwen3:8b", "size": 5000000},
                {"name": "nomic-embed-text:latest", "size": 274000000},
            ]
        }
        requests_mock.get("http://mock-ollama:11434/api/tags", json=payload)
        models = client.list_models()
        assert models == ["qwen3:8b", "nomic-embed-text:latest"]

    def test_list_models_malformed_json(self, client, requests_mock):
        requests_mock.get("http://mock-ollama:11434/api/tags", json={"wrong_key": []})
        with pytest.raises(OllamaAPIError, match="missing 'models' key"):
            client.list_models()

    def test_list_models_connection_error(self, client, requests_mock):
        requests_mock.get(
            "http://mock-ollama:11434/api/tags",
            exc=requests.exceptions.ConnectionError("Failed to connect"),
        )
        with pytest.raises(OllamaConnectionError):
            client.list_models()

    def test_list_models_timeout(self, client, requests_mock):
        requests_mock.get(
            "http://mock-ollama:11434/api/tags",
            exc=requests.exceptions.Timeout("Request timed out"),
        )
        with pytest.raises(OllamaTimeoutError):
            client.list_models()

    def test_is_model_installed_exact_and_normalized(self, client, requests_mock):
        payload = {
            "models": [
                {"name": "qwen3:8b"},
                {"name": "nomic-embed-text:latest"},
            ]
        }
        requests_mock.get("http://mock-ollama:11434/api/tags", json=payload)

        # Exact match
        assert client.is_model_installed("qwen3:8b") is True
        # Tag normalization: 'nomic-embed-text' should match 'nomic-embed-text:latest'
        assert client.is_model_installed("nomic-embed-text") is True
        assert client.is_model_installed("nomic-embed-text:latest") is True
        # Missing model
        assert client.is_model_installed("nonexistent-model") is False


class TestOllamaClientGenerate:
    """Tests for text generation."""

    def test_generate_success(self, client, requests_mock):
        payload = {
            "model": "qwen3:8b",
            "response": "Graph RAG is ready.",
            "done": True,
            "total_duration": 123456789,
            "eval_count": 10,
            "prompt_eval_count": 5,
        }
        requests_mock.post("http://mock-ollama:11434/api/generate", json=payload)

        res = client.generate("Hello Qwen", model="qwen3:8b")
        assert isinstance(res, GenerateResponse)
        assert res.model == "qwen3:8b"
        assert res.response == "Graph RAG is ready."
        assert res.done is True
        assert res.eval_count == 10

    def test_generate_empty_prompt_raises_value_error(self, client):
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            client.generate("   ")

    def test_generate_model_not_found_404(self, client, requests_mock):
        requests_mock.post(
            "http://mock-ollama:11434/api/generate",
            status_code=404,
            text="model not found",
        )
        with pytest.raises(OllamaModelNotFoundError):
            client.generate("Test prompt", model="missing-model")

    def test_generate_connection_error(self, client, requests_mock):
        requests_mock.post(
            "http://mock-ollama:11434/api/generate",
            exc=requests.exceptions.ConnectionError("Refused"),
        )
        with pytest.raises(OllamaConnectionError):
            client.generate("Test prompt")

    def test_generate_timeout_error(self, client, requests_mock):
        requests_mock.post(
            "http://mock-ollama:11434/api/generate",
            exc=requests.exceptions.Timeout("Read timeout"),
        )
        with pytest.raises(OllamaTimeoutError):
            client.generate("Test prompt")


class TestOllamaClientEmbed:
    """Tests for dense vector embedding generation."""

    def test_embed_single_text_success(self, client, requests_mock):
        payload = {
            "model": "nomic-embed-text",
            "embeddings": [[0.1, 0.2, 0.3, 0.4]],
        }
        requests_mock.post("http://mock-ollama:11434/api/embed", json=payload)

        res = client.embed("Sample sentence")
        assert isinstance(res, EmbeddingResponse)
        assert res.dimensions == 4
        assert res.embeddings == [[0.1, 0.2, 0.3, 0.4]]

    def test_embed_batch_texts_success(self, client, requests_mock):
        payload = {
            "model": "nomic-embed-text",
            "embeddings": [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
            ],
        }
        requests_mock.post("http://mock-ollama:11434/api/embed", json=payload)

        res = client.embed(["First text", "Second text"])
        assert res.dimensions == 3
        assert len(res.embeddings) == 2

    def test_embed_single_helper_returns_1d_list(self, client, requests_mock):
        payload = {
            "model": "nomic-embed-text",
            "embeddings": [[0.5, 0.6, 0.7]],
        }
        requests_mock.post("http://mock-ollama:11434/api/embed", json=payload)

        vec = client.embed_single("Test sentence")
        assert vec == [0.5, 0.6, 0.7]

    def test_embed_empty_input_raises_value_error(self, client):
        with pytest.raises(ValueError, match="Embedding input text cannot be empty"):
            client.embed("   ")

        with pytest.raises(ValueError, match="Embedding input list cannot be empty"):
            client.embed([])

    def test_embed_model_not_found_404(self, client, requests_mock):
        requests_mock.post(
            "http://mock-ollama:11434/api/embed",
            status_code=404,
            text="model not found",
        )
        with pytest.raises(OllamaModelNotFoundError):
            client.embed("Test text", model="missing-embed")

    def test_embed_invalid_dimensions_raises_api_error(self, client, requests_mock):
        # Empty vector dimension
        requests_mock.post(
            "http://mock-ollama:11434/api/embed",
            json={"embeddings": [[]]},
        )
        with pytest.raises(OllamaAPIError, match="zero dimensions"):
            client.embed("Test text")

    def test_embed_inconsistent_dimensions_raises_api_error(self, client, requests_mock):
        requests_mock.post(
            "http://mock-ollama:11434/api/embed",
            json={"embeddings": [[0.1, 0.2], [0.3]]},
        )
        with pytest.raises(OllamaAPIError, match="Inconsistent embedding dimensions"):
            client.embed(["Text 1", "Text 2"])

    def test_embed_non_numeric_vector_raises_api_error(self, client, requests_mock):
        requests_mock.post(
            "http://mock-ollama:11434/api/embed",
            json={"embeddings": [["not", "numbers"]]},
        )
        with pytest.raises(OllamaAPIError, match="Non-numeric values found"):
            client.embed("Test text")
