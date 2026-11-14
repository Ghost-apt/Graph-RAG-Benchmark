"""Unit tests for Knowledge Graph schemas."""

import pytest
from pydantic import ValidationError

from graphrag_bench.graph_schemas import (
    ChunkGraphExtraction,
    ExtractedEntity,
    ExtractedRelationship,
    GraphEntity,
    GraphRelationship,
    normalize_entity_type,
    normalize_predicate,
)


class TestGraphSchemas:
    """Test suite for graph extraction and graph storage schemas."""

    def test_normalize_predicate(self) -> None:
        assert normalize_predicate("evaluates") == "EVALUATES"
        assert normalize_predicate("is a") == "IS_A"
        assert normalize_predicate("integrates-with") == "INTEGRATES_WITH"
        assert normalize_predicate("USES / CALLS") == "USES_CALLS"
        assert normalize_predicate("   ") == "RELATED_TO"

    def test_normalize_entity_type(self) -> None:
        assert normalize_entity_type("CONCEPT") == "CONCEPT"
        assert normalize_entity_type("technology") == "TECHNOLOGY"
        assert normalize_entity_type("ALGORITHM") == "METHOD"
        assert normalize_entity_type("DATABASE") == "SOFTWARE"
        assert normalize_entity_type("university") == "ORGANIZATION"
        assert normalize_entity_type("UnrecognizedCustomType") == "OTHER"

    def test_valid_extracted_entity(self) -> None:
        entity = ExtractedEntity(
            mention_text="  Graph RAG  ",
            canonical_name="Graph RAG",
            entity_type="architecture",  # maps to CONCEPT
            description="Knowledge graph augmented retrieval paradigm",
            aliases=["Graph-based RAG", "G-RAG"],
        )
        assert entity.mention_text == "Graph RAG"
        assert entity.canonical_name == "Graph RAG"
        assert entity.entity_type == "CONCEPT"
        assert entity.aliases == ["Graph-based RAG", "G-RAG"]

    def test_extracted_entity_empty_name_raises(self) -> None:
        with pytest.raises(ValidationError):
            ExtractedEntity(
                mention_text="",
                canonical_name="Valid Name",
            )
        with pytest.raises(ValidationError):
            ExtractedEntity(
                mention_text="Valid Mention",
                canonical_name="   ",
            )

    def test_extracted_entity_aliases_cleaning(self) -> None:
        entity = ExtractedEntity(
            mention_text="ChromaDB",
            canonical_name="ChromaDB",
            entity_type="SOFTWARE",
            aliases=["Chroma", "Chroma", "   ", "Chroma DB"],
        )
        assert entity.aliases == ["Chroma", "Chroma DB"]

    def test_valid_extracted_relationship(self) -> None:
        rel = ExtractedRelationship(
            subject="Hybrid RAG",
            predicate="synthesizes",
            object="Vector Retrieval",
            evidence_quote="Hybrid RAG synthesizes dense vector search with graph traversal.",
            confidence=0.95,
        )
        assert rel.subject == "Hybrid RAG"
        assert rel.predicate == "SYNTHESIZES"
        assert rel.object == "Vector Retrieval"
        assert rel.confidence == 0.95

    def test_extracted_relationship_empty_fields_raise(self) -> None:
        with pytest.raises(ValidationError):
            ExtractedRelationship(
                subject="",
                predicate="USES",
                object="Target",
                evidence_quote="Valid quote.",
            )
        with pytest.raises(ValidationError):
            ExtractedRelationship(
                subject="Source",
                predicate="USES",
                object="",
                evidence_quote="Valid quote.",
            )
        with pytest.raises(ValidationError):
            ExtractedRelationship(
                subject="Source",
                predicate="USES",
                object="Target",
                evidence_quote="   ",
            )

    def test_extracted_relationship_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError):
            ExtractedRelationship(
                subject="Source",
                predicate="USES",
                object="Target",
                evidence_quote="Quote",
                confidence=1.5,
            )
        with pytest.raises(ValidationError):
            ExtractedRelationship(
                subject="Source",
                predicate="USES",
                object="Target",
                evidence_quote="Quote",
                confidence=-0.1,
            )

    def test_chunk_graph_extraction_container(self) -> None:
        extraction = ChunkGraphExtraction(
            entities=[
                ExtractedEntity(
                    mention_text="Qwen", canonical_name="Qwen", entity_type="TECHNOLOGY"
                )
            ],
            relationships=[
                ExtractedRelationship(
                    subject="Qwen",
                    predicate="EXTRACTS",
                    object="Triples",
                    evidence_quote="Qwen extracts triples from chunks.",
                )
            ],
            source_chunk_id="doc1_c0000",
        )
        assert len(extraction.entities) == 1
        assert len(extraction.relationships) == 1
        assert extraction.source_chunk_id == "doc1_c0000"
        data = extraction.model_dump()
        assert "entities" in data
        assert "relationships" in data

    def test_graph_entity_validation(self) -> None:
        node = GraphEntity(
            entity_id="ent_tech_12345",
            canonical_name="NetworkX",
            entity_type="FRAMEWORK",
            description="Python graph manipulation library",
            aliases=["nx"],
            mention_count=3,
            source_document_ids=["doc_1", "doc_2"],
            source_chunk_ids=["chunk_1", "chunk_2"],
        )
        assert node.entity_id == "ent_tech_12345"
        assert node.mention_count == 3
        with pytest.raises(ValidationError):
            GraphEntity(
                entity_id="",
                canonical_name="Invalid",
                entity_type="FRAMEWORK",
            )

    def test_graph_relationship_validation(self) -> None:
        edge = GraphRelationship(
            relationship_id="rel_abc123",
            subject_entity_id="ent_tech_1",
            predicate="IMPLEMENTS",
            object_entity_id="ent_tech_2",
            evidence_quote="System implements Graph RAG algorithms.",
            confidence=0.9,
            source_document_id="doc_1",
            source_chunk_id="chunk_1",
            source_filename="test.txt",
            extraction_model="qwen3:8b",
        )
        assert edge.relationship_id == "rel_abc123"
        assert edge.predicate == "IMPLEMENTS"
        with pytest.raises(ValidationError):
            GraphRelationship(
                relationship_id="",
                subject_entity_id="ent_1",
                predicate="USES",
                object_entity_id="ent_2",
                evidence_quote="Quote",
                source_document_id="doc_1",
                source_chunk_id="chunk_1",
            )
