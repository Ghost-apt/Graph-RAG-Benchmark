"""Unit tests for ChromaDB VectorStore."""

from pathlib import Path

import pytest

from graphrag_bench.vector_store import (
    SearchResultItem,
    VectorRecord,
    VectorStore,
)


@pytest.fixture
def temp_store(tmp_path: Path) -> VectorStore:
    """Provide an isolated VectorStore instance in a temporary directory."""
    chroma_dir = tmp_path / "chroma_test"
    return VectorStore(persist_directory=chroma_dir, collection_name="test_collection")


@pytest.fixture
def sample_records() -> list[VectorRecord]:
    """Sample vector records with known 4D vectors for testing."""
    return [
        VectorRecord(
            chunk_id="chunk_doc1_0",
            document_id="doc1",
            text="Vector databases index embeddings for nearest neighbor search.",
            embedding=[1.0, 0.0, 0.0, 0.0],
            metadata={
                "source_filename": "doc1.txt",
                "file_type": "txt",
                "page_number": 1,
                "word_count": 8,
            },
        ),
        VectorRecord(
            chunk_id="chunk_doc1_1",
            document_id="doc1",
            text="Knowledge graphs capture semantic relationships between entities.",
            embedding=[0.0, 1.0, 0.0, 0.0],
            metadata={
                "source_filename": "doc1.txt",
                "file_type": "txt",
                "page_number": 2,
                "word_count": 7,
            },
        ),
        VectorRecord(
            chunk_id="chunk_doc2_0",
            document_id="doc2",
            text="Hybrid retrieval merges dense semantic search and sparse lexical keyword search.",
            embedding=[0.7071, 0.7071, 0.0, 0.0],
            metadata={
                "source_filename": "doc2.pdf",
                "file_type": "pdf",
                "page_number": None,
                "word_count": 10,
            },
        ),
    ]


class TestVectorStore:
    """Test suite for VectorStore CRUD, similarity search, and persistence."""

    def test_initialization(self, temp_store: VectorStore) -> None:
        assert temp_store.count() == 0
        assert temp_store.collection_name == "test_collection"
        assert Path(temp_store.persist_directory).exists()

    def test_add_records_and_count(
        self, temp_store: VectorStore, sample_records: list[VectorRecord]
    ) -> None:
        count_added = temp_store.add_records(sample_records)
        assert count_added == 3
        assert temp_store.count() == 3

    def test_add_empty_records_returns_zero(self, temp_store: VectorStore) -> None:
        assert temp_store.add_records([]) == 0
        assert temp_store.count() == 0

    def test_get_by_id(self, temp_store: VectorStore, sample_records: list[VectorRecord]) -> None:
        temp_store.add_records(sample_records)

        rec = temp_store.get_by_id("chunk_doc1_0")
        assert rec is not None
        assert rec.chunk_id == "chunk_doc1_0"
        assert rec.document_id == "doc1"
        assert "Vector databases" in rec.text
        assert len(rec.embedding) == 4

        # Non-existent
        assert temp_store.get_by_id("non_existent_id") is None

    def test_upsert_records(
        self, temp_store: VectorStore, sample_records: list[VectorRecord]
    ) -> None:
        temp_store.add_records(sample_records[:2])
        assert temp_store.count() == 2

        # Update chunk_doc1_0 and insert chunk_doc2_0
        updated = VectorRecord(
            chunk_id="chunk_doc1_0",
            document_id="doc1",
            text="Updated text for chunk 0",
            embedding=[0.9, 0.1, 0.0, 0.0],
            metadata={"updated": True},
        )
        res = temp_store.upsert_records([updated, sample_records[2]])
        assert res == 2
        assert temp_store.count() == 3

        retrieved = temp_store.get_by_id("chunk_doc1_0")
        assert retrieved is not None
        assert retrieved.text == "Updated text for chunk 0"

    def test_delete_by_ids(
        self, temp_store: VectorStore, sample_records: list[VectorRecord]
    ) -> None:
        temp_store.add_records(sample_records)
        assert temp_store.count() == 3

        deleted = temp_store.delete_by_ids(["chunk_doc1_0", "chunk_doc2_0"])
        assert deleted == 2
        assert temp_store.count() == 1
        assert temp_store.get_by_id("chunk_doc1_0") is None
        assert temp_store.get_by_id("chunk_doc1_1") is not None

    def test_delete_by_document_id(
        self, temp_store: VectorStore, sample_records: list[VectorRecord]
    ) -> None:
        temp_store.add_records(sample_records)
        assert temp_store.count() == 3

        # Delete doc1 (has 2 chunks)
        deleted = temp_store.delete_by_document_id("doc1")
        assert deleted == 2
        assert temp_store.count() == 1

        # Delete non-existent
        deleted_none = temp_store.delete_by_document_id("doc999")
        assert deleted_none == 0
        assert temp_store.count() == 1

    def test_similarity_search_ranking(
        self, temp_store: VectorStore, sample_records: list[VectorRecord]
    ) -> None:
        temp_store.add_records(sample_records)

        # Query exactly aligned with chunk_doc1_0 [1.0, 0.0, 0.0, 0.0]
        results = temp_store.similarity_search([1.0, 0.0, 0.0, 0.0], top_k=3)
        assert len(results) == 3
        top = results[0]
        assert top.chunk_id == "chunk_doc1_0"
        assert top.rank == 1
        assert pytest.approx(top.distance, abs=1e-4) == 0.0
        assert pytest.approx(top.cosine_similarity, abs=1e-4) == 1.0
        assert pytest.approx(top.normalized_score, abs=1e-4) == 1.0
        assert pytest.approx(top.similarity_score, abs=1e-4) == 1.0

        # doc2_0 was [0.7071, 0.7071, 0.0, 0.0], closer to [1,0,0,0] than doc1_1 [0,1,0,0]
        assert results[1].chunk_id == "chunk_doc2_0"
        assert pytest.approx(results[1].cosine_similarity, abs=1e-2) == 0.7071
        assert pytest.approx(results[1].normalized_score, abs=1e-2) == 0.8536

        assert results[2].chunk_id == "chunk_doc1_1"
        assert pytest.approx(results[2].cosine_similarity, abs=1e-4) == 0.0
        assert pytest.approx(results[2].normalized_score, abs=1e-4) == 0.5

    def test_similarity_search_empty_store(self, temp_store: VectorStore) -> None:
        results = temp_store.similarity_search([1.0, 0.0, 0.0, 0.0], top_k=5)
        assert results == []

    def test_similarity_search_with_where_filter(
        self, temp_store: VectorStore, sample_records: list[VectorRecord]
    ) -> None:
        temp_store.add_records(sample_records)

        # Filter by file_type: pdf (only chunk_doc2_0)
        results = temp_store.similarity_search(
            [1.0, 0.0, 0.0, 0.0],
            top_k=5,
            where_filter={"file_type": "pdf"},
        )
        assert len(results) == 1
        assert results[0].chunk_id == "chunk_doc2_0"
        assert results[0].metadata["file_type"] == "pdf"

    def test_get_all_indexed_chunk_ids(
        self, temp_store: VectorStore, sample_records: list[VectorRecord]
    ) -> None:
        temp_store.add_records(sample_records)
        chunk_ids = temp_store.get_all_indexed_chunk_ids()
        assert set(chunk_ids) == {"chunk_doc1_0", "chunk_doc1_1", "chunk_doc2_0"}

    def test_reset_collection(
        self, temp_store: VectorStore, sample_records: list[VectorRecord]
    ) -> None:
        temp_store.add_records(sample_records)
        assert temp_store.count() == 3

        temp_store.reset_collection()
        assert temp_store.count() == 0

        # Verify we can add records again into reset collection
        temp_store.add_records([sample_records[0]])
        assert temp_store.count() == 1

    def test_sanitize_metadata_removes_none_and_formats(self, temp_store: VectorStore) -> None:
        raw_meta = {
            "page_number": None,
            "int_val": 42,
            "float_val": 3.14,
            "str_val": "text",
            "bool_val": True,
            "list_val": ["a", "b"],
        }
        clean = temp_store._sanitize_metadata(raw_meta, "doc_x", "chk_y")
        assert "page_number" not in clean
        assert clean["int_val"] == 42
        assert clean["float_val"] == 3.14
        assert clean["str_val"] == "text"
        assert clean["bool_val"] is True
        assert clean["list_val"] == "['a', 'b']"
        assert clean["document_id"] == "doc_x"
        assert clean["chunk_id"] == "chk_y"

    def test_cosine_similarity_mathematical_precision(self, temp_store: VectorStore) -> None:
        """Verify mathematical precision of cosine distance, raw cosine similarity, and normalized score."""
        records = [
            VectorRecord(
                chunk_id="chk_identical",
                document_id="doc_math",
                text="Identical vector direction",
                embedding=[1.0, 0.0, 0.0, 0.0],
                metadata={"case": "identical"},
            ),
            VectorRecord(
                chunk_id="chk_45deg",
                document_id="doc_math",
                text="45 degree vector angle",
                embedding=[0.70710678, 0.70710678, 0.0, 0.0],
                metadata={"case": "45deg"},
            ),
            VectorRecord(
                chunk_id="chk_orthogonal",
                document_id="doc_math",
                text="Orthogonal 90 degree vector",
                embedding=[0.0, 1.0, 0.0, 0.0],
                metadata={"case": "orthogonal"},
            ),
            VectorRecord(
                chunk_id="chk_opposite",
                document_id="doc_math",
                text="Opposite 180 degree vector",
                embedding=[-1.0, 0.0, 0.0, 0.0],
                metadata={"case": "opposite"},
            ),
        ]
        temp_store.add_records(records)
        assert temp_store.count() == 4

        # Query vector is along x-axis [1, 0, 0, 0]
        results = temp_store.similarity_search([1.0, 0.0, 0.0, 0.0], top_k=4)
        assert len(results) == 4

        # Rank 1: Identical direction (dot product = 1.0)
        # Cosine distance = 1 - 1.0 = 0.0
        # Raw cosine similarity = 1 - 0.0 = 1.0
        # Normalized score = 1 - 0.0/2 = 1.0
        r1 = results[0]
        assert r1.chunk_id == "chk_identical"
        assert pytest.approx(r1.distance, abs=1e-4) == 0.0
        assert pytest.approx(r1.cosine_similarity, abs=1e-4) == 1.0
        assert pytest.approx(r1.normalized_score, abs=1e-4) == 1.0

        # Rank 2: 45 degree angle (dot product = 1/sqrt(2) ≈ 0.7071)
        # Cosine distance ≈ 0.2929
        # Raw cosine similarity ≈ 0.7071
        # Normalized score = 1 - 0.2929/2 ≈ 0.8536
        r2 = results[1]
        assert r2.chunk_id == "chk_45deg"
        assert pytest.approx(r2.distance, abs=1e-3) == 0.2929
        assert pytest.approx(r2.cosine_similarity, abs=1e-3) == 0.7071
        assert pytest.approx(r2.normalized_score, abs=1e-3) == 0.8536

        # Rank 3: Orthogonal (dot product = 0.0)
        # Cosine distance = 1 - 0.0 = 1.0
        # Raw cosine similarity = 1 - 1.0 = 0.0
        # Normalized score = 1 - 1.0/2 = 0.5
        r3 = results[2]
        assert r3.chunk_id == "chk_orthogonal"
        assert pytest.approx(r3.distance, abs=1e-3) == 1.0
        assert pytest.approx(r3.cosine_similarity, abs=1e-3) == 0.0
        assert pytest.approx(r3.normalized_score, abs=1e-3) == 0.5

        # Rank 4: Opposite direction (dot product = -1.0)
        # Cosine distance = 1 - (-1.0) = 2.0
        # Raw cosine similarity = 1 - 2.0 = -1.0
        # Normalized score = 1 - 2.0/2 = 0.0
        r4 = results[3]
        assert r4.chunk_id == "chk_opposite"
        assert pytest.approx(r4.distance, abs=1e-3) == 2.0
        assert pytest.approx(r4.cosine_similarity, abs=1e-3) == -1.0
        assert pytest.approx(r4.normalized_score, abs=1e-3) == 0.0

    def test_search_result_item_invalid_and_missing_distance_handling(self) -> None:
        """Verify SearchResultItem gracefully reconciles missing or non-finite distances."""
        # Test default initialization with missing fields
        item1 = SearchResultItem(
            rank=1,
            chunk_id="c1",
            document_id="d1",
            text="sample text",
            distance=0.2,
        )
        assert item1.distance == 0.2
        assert item1.cosine_similarity == 0.8
        assert item1.normalized_score == 0.9
        assert item1.similarity_score == 0.9

        # Test with legacy similarity_score
        item2 = SearchResultItem(
            rank=2,
            chunk_id="c2",
            document_id="d2",
            text="sample text",
            distance=0.4,
            similarity_score=0.8,
        )
        assert item2.normalized_score == 0.8
        assert item2.cosine_similarity == 0.6
