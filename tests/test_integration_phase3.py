"""Live integration tests for Phase 3 (Embeddings, ChromaDB, Indexing Pipeline, Semantic Search).

These tests connect directly to the real Ollama host and nomic-embed-text model.
They are tagged with the 'integration' marker:
    python -m pytest -m "integration"
"""

from pathlib import Path

import pytest

from graphrag_bench.config import get_settings
from graphrag_bench.document_repository import DocumentRepository
from graphrag_bench.embedding_service import EmbeddingService
from graphrag_bench.indexing_pipeline import IndexingPipeline
from graphrag_bench.ollama_client import OllamaClient
from graphrag_bench.semantic_search import SemanticSearch
from graphrag_bench.vector_store import VectorStore


@pytest.mark.integration
class TestPhase3LiveIntegration:
    """Integration suite verifying real nomic-embed-text vectors and ChromaDB indexing."""

    @pytest.fixture(scope="class")
    def live_client(self):
        settings = get_settings()
        client = OllamaClient(settings=settings)
        if not client.is_server_reachable():
            pytest.skip(f"Live Ollama server is not reachable at {client.base_url}")
        if not client.is_model_installed(settings.ollama_embed_model):
            pytest.skip(f"Embedding model '{settings.ollama_embed_model}' is not installed")
        return client

    def test_live_embedding_service(self, live_client: OllamaClient) -> None:
        service = EmbeddingService(client=live_client)
        vector = service.embed_text(
            "Graph RAG integrates structured knowledge graphs with vector retrieval."
        )
        assert len(vector) == 768
        assert all(isinstance(v, float) for v in vector)

        batch_res = service.embed_batch(
            [
                "Vector databases index embeddings for nearest neighbor search.",
                "Knowledge graphs capture semantic relationships between entities.",
            ]
        )
        assert batch_res.dimensions == 768
        assert len(batch_res.vectors) == 2
        assert batch_res.duration_ms > 0

    def test_live_indexing_and_semantic_retrieval(
        self, live_client: OllamaClient, tmp_path: Path
    ) -> None:
        # 1. Use real DocumentRepository with Phase 2 canonical data
        repo = DocumentRepository()
        docs = repo.list_documents()
        if not docs:
            pytest.skip("No Phase 2 documents found in data/processed/")

        # 2. Use isolated ChromaDB directory in tmp_path
        chroma_dir = tmp_path / "chroma_live_test"
        store = VectorStore(persist_directory=chroma_dir, collection_name="live_test_collection")
        service = EmbeddingService(client=live_client)

        pipeline = IndexingPipeline(
            repository=repo,
            embedding_service=service,
            vector_store=store,
        )

        # 3. Index canonical documents
        stats = pipeline.index_all(force_reindex=True)
        assert stats.is_success
        assert stats.newly_indexed_chunks == len(docs)
        assert stats.dimensions == 768
        assert store.count() == len(docs)

        # 4. Perform live semantic search
        search_engine = SemanticSearch(
            vector_store=store,
            embedding_service=service,
        )

        response = search_engine.search("What is the Graph RAG benchmarking framework?", top_k=2)
        assert response.total_found > 0
        assert len(response.items) > 0
        top = response.items[0]
        assert top.similarity_score > 0.0
        assert len(top.text) > 0
        assert top.metadata.get("source_filename") is not None
