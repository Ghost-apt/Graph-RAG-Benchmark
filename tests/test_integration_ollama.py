"""Live integration tests for Ollama server and configured models.

These tests connect directly to the real Ollama host and models.
They are tagged with the 'integration' marker so they can be run or skipped:
    python -m pytest -m "not integration"  # Run unit tests only
    python -m pytest -m "integration"      # Run live integration tests only
"""

import pytest

from graphrag_bench.config import get_settings
from graphrag_bench.ollama_client import OllamaClient


@pytest.mark.integration
class TestOllamaLiveIntegration:
    """Integration test suite executing real HTTP calls against live Ollama."""

    @pytest.fixture(scope="class")
    def live_client(self):
        settings = get_settings()
        client = OllamaClient(settings=settings)
        if not client.is_server_reachable():
            pytest.skip(f"Live Ollama server is not reachable at {client.base_url}")
        return client

    def test_live_server_reachable(self, live_client):
        assert live_client.is_server_reachable() is True

    def test_live_models_present(self, live_client):
        models = live_client.list_models()
        assert len(models) > 0

        # Check configured LLM model
        settings = get_settings()
        assert live_client.is_model_installed(settings.ollama_llm_model) is True

        # Check configured embedding model
        assert live_client.is_model_installed(settings.ollama_embed_model) is True

    def test_live_llm_generation(self, live_client):
        settings = get_settings()
        res = live_client.generate(
            prompt="Reply with the single word: OK",
            model=settings.ollama_llm_model,
        )
        assert res.response is not None
        assert len(res.response.strip()) > 0
        assert res.done is True

    def test_live_embedding_generation(self, live_client):
        settings = get_settings()
        emb_res = live_client.embed(
            texts="Knowledge Graph benchmarks evaluation.",
            model=settings.ollama_embed_model,
        )
        assert emb_res.dimensions == 768
        assert len(emb_res.embeddings) == 1
        assert len(emb_res.embeddings[0]) == 768

    def test_live_batch_embedding_generation(self, live_client):
        settings = get_settings()
        emb_res = live_client.embed(
            texts=["Vector RAG baseline", "Graph RAG benchmark"],
            model=settings.ollama_embed_model,
        )
        assert emb_res.dimensions == 768
        assert len(emb_res.embeddings) == 2
        assert len(emb_res.embeddings[0]) == 768
        assert len(emb_res.embeddings[1]) == 768
