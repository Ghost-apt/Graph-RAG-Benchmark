"""Unit tests for GraphBuilder construction pipeline."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from graphrag_bench.document_repository import DocumentRepository
from graphrag_bench.entity_resolver import EntityResolver
from graphrag_bench.graph_builder import GraphBuilder
from graphrag_bench.graph_extraction_service import GraphExtractionService
from graphrag_bench.graph_schemas import (
    ChunkGraphExtraction,
    ExtractedEntity,
    ExtractedRelationship,
)
from graphrag_bench.graph_store import GraphStore
from graphrag_bench.schemas import DocumentChunk, DocumentMetadata


@pytest.fixture
def mock_repo() -> MagicMock:
    repo = MagicMock(spec=DocumentRepository)

    doc1 = DocumentMetadata(
        document_id="doc_1",
        filename="rag_intro.txt",
        file_type="txt",
        file_size_bytes=500,
        sha256="a" * 64,
        chunk_count=1,
    )
    doc2 = DocumentMetadata(
        document_id="doc_2",
        filename="graph_intro.md",
        file_type="md",
        file_size_bytes=600,
        sha256="b" * 64,
        chunk_count=1,
    )

    chunk1 = DocumentChunk(
        chunk_id="chunk_1",
        document_id="doc_1",
        chunk_index=0,
        text="RAG combines dense semantic vector retrieval with generative language models.",
        word_count=10,
        char_count=77,
        start_char=0,
        end_char=77,
    )
    chunk2 = DocumentChunk(
        chunk_id="chunk_2",
        document_id="doc_2",
        chunk_index=0,
        text="NetworkX facilitates graph construction and multi-hop entity traversal.",
        word_count=9,
        char_count=72,
        start_char=0,
        end_char=72,
    )

    repo.list_documents.return_value = [doc1, doc2]
    repo.get_document.side_effect = lambda did: (
        doc1 if did == "doc_1" else (doc2 if did == "doc_2" else None)
    )
    repo.get_chunks.side_effect = lambda did: (
        [chunk1] if did == "doc_1" else ([chunk2] if did == "doc_2" else [])
    )
    return repo


@pytest.fixture
def mock_extraction_service() -> MagicMock:
    service = MagicMock(spec=GraphExtractionService)
    service.model_name = "qwen3:8b"

    def side_effect_extract(*args, chunk=None, chunk_text=None, chunk_id=None, **kwargs):
        if chunk is not None:
            chunk_text = chunk.text
            chunk_id = chunk.chunk_id
        elif args and chunk_text is None:
            chunk_text = args[0]
            if len(args) > 1 and chunk_id is None:
                chunk_id = args[1]
        chunk_text = chunk_text or ""
        if "RAG" in chunk_text:
            return ChunkGraphExtraction(
                entities=[
                    ExtractedEntity(
                        mention_text="RAG",
                        canonical_name="RAG",
                        entity_type="CONCEPT",
                        description="Retrieval-Augmented Generation",
                    ),
                    ExtractedEntity(
                        mention_text="generative language models",
                        canonical_name="Generative Language Models",
                        entity_type="TECHNOLOGY",
                    ),
                ],
                relationships=[
                    ExtractedRelationship(
                        subject="RAG",
                        predicate="USES",
                        object="Generative Language Models",
                        evidence_quote="RAG combines dense semantic vector retrieval with generative language models.",
                        confidence=0.95,
                    )
                ],
                source_chunk_id=chunk_id,
            )
        else:
            return ChunkGraphExtraction(
                entities=[
                    ExtractedEntity(
                        mention_text="NetworkX",
                        canonical_name="NetworkX",
                        entity_type="FRAMEWORK",
                        description="Graph construction library",
                    ),
                    ExtractedEntity(
                        mention_text="multi-hop entity traversal",
                        canonical_name="Multi-Hop Traversal",
                        entity_type="METHOD",
                    ),
                ],
                relationships=[
                    ExtractedRelationship(
                        subject="NetworkX",
                        predicate="FACILITATES",
                        object="Multi-Hop Traversal",
                        evidence_quote="NetworkX facilitates graph construction and multi-hop entity traversal.",
                        confidence=0.92,
                    )
                ],
                source_chunk_id=chunk_id,
            )

    service.extract_from_chunk.side_effect = side_effect_extract
    return service


def test_build_all_and_incremental(mock_repo, mock_extraction_service, tmp_path: Path):
    graph_dir = tmp_path / "kg"
    builder = GraphBuilder(
        doc_repo=mock_repo,
        extraction_service=mock_extraction_service,
        resolver=EntityResolver(),
        store=GraphStore(),
        graph_dir=graph_dir,
    )

    # First build
    stats = builder.build_all()
    assert stats["chunks_inspected"] == 2
    assert stats["chunks_successful"] == 2
    assert stats["total_entities"] == 4
    assert stats["total_relationships"] == 2
    assert builder.graph_path.exists()
    assert builder.graphml_path.exists()
    assert builder.manifest_path.exists()
    assert builder.extractions_path.exists()

    # Second build without changes should inspect 0 new chunks
    incremental_stats = builder.build_all()
    assert incremental_stats["chunks_inspected"] == 0
    assert incremental_stats["chunks_successful"] == 0


def test_force_rebuild(mock_repo, mock_extraction_service, tmp_path: Path):
    graph_dir = tmp_path / "kg"
    builder = GraphBuilder(
        doc_repo=mock_repo,
        extraction_service=mock_extraction_service,
        graph_dir=graph_dir,
    )

    builder.build_all()
    assert builder.store.node_count == 4

    # Force rebuild
    stats = builder.build_all(force_rebuild=True)
    assert stats["chunks_inspected"] == 2
    assert stats["chunks_successful"] == 2
    assert builder.store.node_count == 4


def test_remove_document(mock_repo, mock_extraction_service, tmp_path: Path):
    graph_dir = tmp_path / "kg"
    builder = GraphBuilder(
        doc_repo=mock_repo,
        extraction_service=mock_extraction_service,
        graph_dir=graph_dir,
    )

    builder.build_all()
    assert builder.store.node_count == 4
    assert builder.store.edge_count == 2

    # Remove doc_1
    edges_removed, nodes_removed = builder.remove_document("doc_1")
    assert edges_removed == 1
    assert nodes_removed == 2  # Both RAG and Generative Language Models were only in doc_1
    assert builder.store.node_count == 2
    assert builder.store.edge_count == 1


def test_get_status(mock_repo, mock_extraction_service, tmp_path: Path):
    graph_dir = tmp_path / "kg"
    builder = GraphBuilder(
        doc_repo=mock_repo,
        extraction_service=mock_extraction_service,
        graph_dir=graph_dir,
    )

    status = builder.get_status()
    assert status["total_documents"] == 2
    assert status["total_repository_chunks"] == 2
    assert status["processed_chunks"] == 0
    assert status["unprocessed_chunks"] == 2

    builder.build_all()
    updated_status = builder.get_status()
    assert updated_status["processed_chunks"] == 2
    assert updated_status["unprocessed_chunks"] == 0
    assert updated_status["node_count"] == 4
    assert updated_status["edge_count"] == 2
