"""Unit tests for IndexingPipeline."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from graphrag_bench.document_repository import DocumentRepository
from graphrag_bench.embedding_service import EmbeddingResult, EmbeddingService
from graphrag_bench.indexing_pipeline import IndexingPipeline, IndexingStats
from graphrag_bench.schemas import DocumentChunk, DocumentMetadata
from graphrag_bench.vector_store import VectorStore


@pytest.fixture
def sample_processed_docs() -> list[DocumentMetadata]:
    """Sample processed document metadata with chunks."""
    doc1 = DocumentMetadata(
        document_id="doc_alpha",
        filename="alpha.txt",
        file_type="txt",
        file_size_bytes=100,
        sha256="a" * 64,
        chunk_count=2,
        word_count=8,
        char_count=41,
        chunk_size=400,
        chunk_overlap=60,
    )

    doc2 = DocumentMetadata(
        document_id="doc_beta",
        filename="beta.pdf",
        file_type="pdf",
        file_size_bytes=200,
        sha256="b" * 64,
        chunk_count=1,
        word_count=3,
        char_count=15,
        chunk_size=400,
        chunk_overlap=60,
    )

    return [doc1, doc2]


@pytest.fixture
def sample_chunks() -> dict[str, list[DocumentChunk]]:
    return {
        "doc_alpha": [
            DocumentChunk(
                chunk_id="chunk_alpha_0",
                document_id="doc_alpha",
                chunk_index=0,
                text="Chunk zero text alpha",
                word_count=4,
                char_count=21,
                start_char=0,
                end_char=21,
                source_filename="alpha.txt",
                file_type="txt",
            ),
            DocumentChunk(
                chunk_id="chunk_alpha_1",
                document_id="doc_alpha",
                chunk_index=1,
                text="Chunk one text alpha",
                word_count=4,
                char_count=20,
                start_char=21,
                end_char=41,
                source_filename="alpha.txt",
                file_type="txt",
            ),
        ],
        "doc_beta": [
            DocumentChunk(
                chunk_id="chunk_beta_0",
                document_id="doc_beta",
                chunk_index=0,
                text="Beta chunk text",
                word_count=3,
                char_count=15,
                start_char=0,
                end_char=15,
                source_filename="beta.pdf",
                file_type="pdf",
                page_number=1,
            )
        ],
    }


@pytest.fixture
def mock_repo(
    sample_processed_docs: list[DocumentMetadata],
    sample_chunks: dict[str, list[DocumentChunk]],
) -> MagicMock:
    repo = MagicMock(spec=DocumentRepository)
    repo.list_documents.return_value = sample_processed_docs

    def get_chunks_side_effect(doc_id: str):
        return sample_chunks.get(doc_id, [])

    repo.get_chunks.side_effect = get_chunks_side_effect
    return repo


@pytest.fixture
def mock_embedding_service() -> MagicMock:
    service = MagicMock(spec=EmbeddingService)
    service.default_model = "nomic-embed-text"
    service.default_batch_size = 2

    def embed_batch_side_effect(texts: list[str], model: str = "nomic-embed-text"):
        # Produce a dummy 4D embedding for each text
        vectors = [[0.1 * i, 0.2, 0.3, 0.4] for i, _ in enumerate(texts)]
        return EmbeddingResult(
            vectors=vectors,
            model=model,
            dimensions=4,
            duration_ms=12.5,
            input_count=len(texts),
        )

    service.embed_batch.side_effect = embed_batch_side_effect
    return service


@pytest.fixture
def isolated_vector_store(tmp_path: Path) -> VectorStore:
    return VectorStore(
        persist_directory=tmp_path / "chroma_test",
        collection_name="pipeline_test_col",
    )


class TestIndexingPipeline:
    """Test suite for IndexingPipeline execution, caching, and progress."""

    def test_pipeline_initialization(
        self,
        mock_repo: MagicMock,
        mock_embedding_service: MagicMock,
        isolated_vector_store: VectorStore,
    ) -> None:
        pipeline = IndexingPipeline(
            repository=mock_repo,
            embedding_service=mock_embedding_service,
            vector_store=isolated_vector_store,
        )
        status = pipeline.get_indexing_status()
        assert status["total_documents"] == 2
        assert status["total_canonical_chunks"] == 3
        assert status["indexed_vector_count"] == 0
        assert status["pending_chunk_count"] == 3

    def test_index_all_new_chunks(
        self,
        mock_repo: MagicMock,
        mock_embedding_service: MagicMock,
        isolated_vector_store: VectorStore,
    ) -> None:
        pipeline = IndexingPipeline(
            repository=mock_repo,
            embedding_service=mock_embedding_service,
            vector_store=isolated_vector_store,
        )

        stats = pipeline.index_all()
        assert isinstance(stats, IndexingStats)
        assert stats.is_success
        assert stats.total_chunks_considered == 3
        assert stats.newly_indexed_chunks == 3
        assert stats.already_indexed_chunks == 0
        assert stats.dimensions == 4
        assert isolated_vector_store.count() == 3

        # Status after indexing
        status = pipeline.get_indexing_status()
        assert status["indexed_vector_count"] == 3
        assert status["pending_chunk_count"] == 0

    def test_incremental_indexing_skips_already_indexed(
        self,
        mock_repo: MagicMock,
        mock_embedding_service: MagicMock,
        isolated_vector_store: VectorStore,
    ) -> None:
        pipeline = IndexingPipeline(
            repository=mock_repo,
            embedding_service=mock_embedding_service,
            vector_store=isolated_vector_store,
        )

        # First run indexes all 3
        pipeline.index_all()
        mock_embedding_service.embed_batch.reset_mock()

        # Second run should skip everything
        second_stats = pipeline.index_all()
        assert second_stats.total_chunks_considered == 3
        assert second_stats.newly_indexed_chunks == 0
        assert second_stats.already_indexed_chunks == 3
        mock_embedding_service.embed_batch.assert_not_called()

    def test_force_reindex_resets_and_reindexes(
        self,
        mock_repo: MagicMock,
        mock_embedding_service: MagicMock,
        isolated_vector_store: VectorStore,
    ) -> None:
        pipeline = IndexingPipeline(
            repository=mock_repo,
            embedding_service=mock_embedding_service,
            vector_store=isolated_vector_store,
        )

        pipeline.index_all()
        assert isolated_vector_store.count() == 3

        # Force reindex
        stats = pipeline.index_all(force_reindex=True)
        assert stats.newly_indexed_chunks == 3
        assert isolated_vector_store.count() == 3

    def test_index_single_document(
        self,
        mock_repo: MagicMock,
        mock_embedding_service: MagicMock,
        isolated_vector_store: VectorStore,
    ) -> None:
        pipeline = IndexingPipeline(
            repository=mock_repo,
            embedding_service=mock_embedding_service,
            vector_store=isolated_vector_store,
        )

        stats = pipeline.index_document("doc_beta")
        assert stats.total_chunks_considered == 1
        assert stats.newly_indexed_chunks == 1
        assert isolated_vector_store.count() == 1
        assert isolated_vector_store.get_by_id("chunk_beta_0") is not None

    def test_progress_callback_invoked(
        self,
        mock_repo: MagicMock,
        mock_embedding_service: MagicMock,
        isolated_vector_store: VectorStore,
    ) -> None:
        pipeline = IndexingPipeline(
            repository=mock_repo,
            embedding_service=mock_embedding_service,
            vector_store=isolated_vector_store,
        )

        callback_calls = []

        def callback(progress: float, message: str) -> None:
            callback_calls.append((progress, message))

        stats = pipeline.index_all(progress_callback=callback)
        assert stats.is_success
        assert len(callback_calls) >= 2  # At least batch updates and completion
        assert callback_calls[-1][0] == 1.0
        assert "completed" in callback_calls[-1][1].lower()

    def test_empty_repository(
        self,
        mock_embedding_service: MagicMock,
        isolated_vector_store: VectorStore,
    ) -> None:
        empty_repo = MagicMock(spec=DocumentRepository)
        empty_repo.list_documents.return_value = []
        pipeline = IndexingPipeline(
            repository=empty_repo,
            embedding_service=mock_embedding_service,
            vector_store=isolated_vector_store,
        )

        stats = pipeline.index_all()
        assert stats.total_chunks_considered == 0
        assert stats.newly_indexed_chunks == 0

    def test_batch_failure_recorded_gracefully(
        self,
        mock_repo: MagicMock,
        mock_embedding_service: MagicMock,
        isolated_vector_store: VectorStore,
    ) -> None:
        mock_embedding_service.embed_batch.side_effect = RuntimeError("Embedding model offline")
        pipeline = IndexingPipeline(
            repository=mock_repo,
            embedding_service=mock_embedding_service,
            vector_store=isolated_vector_store,
        )

        stats = pipeline.index_all()
        assert not stats.is_success
        assert stats.failed_chunks == 3
        assert len(stats.errors) > 0
        assert "Embedding model offline" in stats.errors[0]
