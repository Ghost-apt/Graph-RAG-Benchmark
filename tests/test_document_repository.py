"""Unit tests for DocumentRepository."""

import csv
import io
import json
from pathlib import Path

import pytest

from graphrag_bench.document_repository import DocumentRepository
from graphrag_bench.schemas import DocumentChunk, DocumentMetadata


@pytest.fixture
def temp_repo(tmp_path: Path) -> DocumentRepository:
    """Create a temporary repository inside a pytest temporary directory."""
    return DocumentRepository(base_dir=tmp_path)


@pytest.fixture
def sample_metadata() -> DocumentMetadata:
    return DocumentMetadata(
        doc_id="doc_1234567890abcdef",
        filename="benchmark_report.txt",
        file_format="txt",
        file_size_bytes=1024,
        sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        page_count=1,
        total_words=250,
        total_chunks=2,
    )


@pytest.fixture
def sample_chunks(sample_metadata: DocumentMetadata) -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id=f"{sample_metadata.doc_id}_c0000_11112222",
            doc_id=sample_metadata.doc_id,
            chunk_index=0,
            text="First chunk content for benchmark testing.",
            word_count=6,
            start_char=0,
            end_char=42,
            page_number=1,
        ),
        DocumentChunk(
            chunk_id=f"{sample_metadata.doc_id}_c0001_33334444",
            doc_id=sample_metadata.doc_id,
            chunk_index=1,
            text="Second chunk content continuing the evaluation.",
            word_count=6,
            start_char=43,
            end_char=90,
            page_number=1,
        ),
    ]


class TestDocumentRepository:
    """Tests for local persistence, querying, manifest management, and export."""

    def test_repository_initialization(self, temp_repo: DocumentRepository, tmp_path: Path) -> None:
        assert (tmp_path / "data" / "uploads").is_dir()
        assert (tmp_path / "data" / "processed").is_dir()
        assert (tmp_path / "data" / "processed" / "manifest.json").is_file()

    def test_save_and_retrieve_document(
        self,
        temp_repo: DocumentRepository,
        sample_metadata: DocumentMetadata,
        sample_chunks: list[DocumentChunk],
    ) -> None:
        raw_content = b"Sample text file content"
        success = temp_repo.save_document(
            metadata=sample_metadata,
            chunks=sample_chunks,
            raw_content=raw_content,
        )
        assert success is True

        # Retrieve document
        retrieved_meta = temp_repo.get_document(sample_metadata.doc_id)
        assert retrieved_meta is not None
        assert retrieved_meta.doc_id == sample_metadata.doc_id
        assert retrieved_meta.filename == "benchmark_report.txt"

        # Retrieve chunks
        retrieved_chunks = temp_repo.get_chunks(sample_metadata.doc_id)
        assert len(retrieved_chunks) == 2
        assert retrieved_chunks[0].chunk_id == sample_chunks[0].chunk_id
        assert retrieved_chunks[1].text == sample_chunks[1].text

    def test_duplicate_detection(
        self,
        temp_repo: DocumentRepository,
        sample_metadata: DocumentMetadata,
        sample_chunks: list[DocumentChunk],
    ) -> None:
        temp_repo.save_document(
            metadata=sample_metadata,
            chunks=sample_chunks,
            raw_content=b"Sample content",
        )

        # Query duplicate by SHA-256
        is_dup, dup_doc = temp_repo.is_duplicate(sample_metadata.sha256)
        assert is_dup is True
        assert dup_doc is not None
        assert dup_doc.doc_id == sample_metadata.doc_id

        # Query non-duplicate
        is_dup_fake, fake_doc = temp_repo.is_duplicate("ffffffffffffffffffffffffffffffff")
        assert is_dup_fake is False
        assert fake_doc is None

    def test_list_documents(
        self,
        temp_repo: DocumentRepository,
        sample_metadata: DocumentMetadata,
        sample_chunks: list[DocumentChunk],
    ) -> None:
        assert len(temp_repo.list_documents()) == 0

        temp_repo.save_document(
            metadata=sample_metadata,
            chunks=sample_chunks,
            raw_content=b"Content 1",
        )
        docs = temp_repo.list_documents()
        assert len(docs) == 1
        assert docs[0].doc_id == sample_metadata.doc_id

    def test_delete_document(
        self,
        temp_repo: DocumentRepository,
        sample_metadata: DocumentMetadata,
        sample_chunks: list[DocumentChunk],
    ) -> None:
        temp_repo.save_document(
            metadata=sample_metadata,
            chunks=sample_chunks,
            raw_content=b"Content to delete",
        )
        assert len(temp_repo.list_documents()) == 1

        # Delete
        deleted = temp_repo.delete_document(sample_metadata.doc_id)
        assert deleted is True

        # Verify removal
        assert temp_repo.get_document(sample_metadata.doc_id) is None
        assert len(temp_repo.get_chunks(sample_metadata.doc_id)) == 0
        assert len(temp_repo.list_documents()) == 0

    def test_corrupt_manifest_auto_rebuild(
        self,
        temp_repo: DocumentRepository,
        sample_metadata: DocumentMetadata,
        sample_chunks: list[DocumentChunk],
        tmp_path: Path,
    ) -> None:
        temp_repo.save_document(
            metadata=sample_metadata,
            chunks=sample_chunks,
            raw_content=b"Content for rebuild",
        )

        # Corrupt the manifest file
        manifest_file = tmp_path / "data" / "processed" / "manifest.json"
        manifest_file.write_text("{corrupted json string", encoding="utf-8")

        # New repository instance should recover manifest from documents.json
        recovered_repo = DocumentRepository(base_dir=tmp_path)
        docs = recovered_repo.list_documents()
        assert len(docs) == 1
        assert docs[0].doc_id == sample_metadata.doc_id

    def test_export_chunks_json(
        self,
        temp_repo: DocumentRepository,
        sample_metadata: DocumentMetadata,
        sample_chunks: list[DocumentChunk],
    ) -> None:
        temp_repo.save_document(
            metadata=sample_metadata,
            chunks=sample_chunks,
            raw_content=b"Sample",
        )

        exported_json = temp_repo.export_chunks(doc_id=sample_metadata.doc_id, format="json")
        data = json.loads(exported_json)
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["chunk_id"] == sample_chunks[0].chunk_id

    def test_export_chunks_jsonl(
        self,
        temp_repo: DocumentRepository,
        sample_metadata: DocumentMetadata,
        sample_chunks: list[DocumentChunk],
    ) -> None:
        temp_repo.save_document(
            metadata=sample_metadata,
            chunks=sample_chunks,
            raw_content=b"Sample",
        )

        exported_jsonl = temp_repo.export_chunks(doc_id=sample_metadata.doc_id, format="jsonl")
        lines = [line for line in exported_jsonl.strip().split("\n") if line]
        assert len(lines) == 2
        chunk0 = json.loads(lines[0])
        assert chunk0["chunk_id"] == sample_chunks[0].chunk_id

    def test_export_chunks_csv(
        self,
        temp_repo: DocumentRepository,
        sample_metadata: DocumentMetadata,
        sample_chunks: list[DocumentChunk],
    ) -> None:
        temp_repo.save_document(
            metadata=sample_metadata,
            chunks=sample_chunks,
            raw_content=b"Sample",
        )

        exported_csv = temp_repo.export_chunks(doc_id=sample_metadata.doc_id, format="csv")
        reader = csv.DictReader(io.StringIO(exported_csv))
        rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["chunk_id"] == sample_chunks[0].chunk_id
        assert rows[0]["word_count"] == "6"
