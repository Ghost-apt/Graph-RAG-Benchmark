"""Unit tests for GraphStore wrapping NetworkX MultiDiGraph."""

from pathlib import Path

import networkx as nx
import pytest

from graphrag_bench.graph_schemas import GraphEntity, GraphRelationship
from graphrag_bench.graph_store import GraphStore


@pytest.fixture
def sample_entities() -> list[GraphEntity]:
    return [
        GraphEntity(
            entity_id="ent_concept_rag",
            canonical_name="RAG",
            entity_type="CONCEPT",
            description="Retrieval-Augmented Generation",
            aliases=["Retrieval Augmented Generation"],
            mention_count=5,
            source_document_ids=["doc_1", "doc_2"],
            source_chunk_ids=["chunk_1", "chunk_2"],
        ),
        GraphEntity(
            entity_id="ent_software_chroma",
            canonical_name="ChromaDB",
            entity_type="SOFTWARE",
            description="Vector database for AI embeddings",
            aliases=["Chroma"],
            mention_count=3,
            source_document_ids=["doc_1"],
            source_chunk_ids=["chunk_1"],
        ),
        GraphEntity(
            entity_id="ent_framework_networkx",
            canonical_name="NetworkX",
            entity_type="FRAMEWORK",
            description="Python library for complex networks",
            aliases=["nx"],
            mention_count=2,
            source_document_ids=["doc_2"],
            source_chunk_ids=["chunk_2"],
        ),
    ]


@pytest.fixture
def sample_relationships() -> list[GraphRelationship]:
    return [
        GraphRelationship(
            relationship_id="rel_rag_uses_chroma",
            subject_entity_id="ent_concept_rag",
            predicate="USES",
            object_entity_id="ent_software_chroma",
            evidence_quote="RAG systems use ChromaDB to retrieve relevant semantic passages.",
            confidence=0.95,
            source_document_id="doc_1",
            source_chunk_id="chunk_1",
            source_filename="rag_overview.txt",
            extraction_model="qwen3:8b",
        ),
        GraphRelationship(
            relationship_id="rel_rag_integrates_networkx",
            subject_entity_id="ent_concept_rag",
            predicate="INTEGRATES_WITH",
            object_entity_id="ent_framework_networkx",
            evidence_quote="Graph RAG integrates with NetworkX for entity traversal.",
            confidence=0.90,
            source_document_id="doc_2",
            source_chunk_id="chunk_2",
            source_filename="graph_rag.md",
            extraction_model="qwen3:8b",
        ),
    ]


def test_add_and_get_entities_and_relationships(sample_entities, sample_relationships):
    store = GraphStore()
    for ent in sample_entities:
        store.add_entity(ent)

    assert store.node_count == 3
    assert store.get_entity("ent_concept_rag").canonical_name == "RAG"

    for rel in sample_relationships:
        store.add_relationship(rel)

    assert store.edge_count == 2
    rel = store.get_relationship("rel_rag_uses_chroma")
    assert rel is not None
    assert rel.predicate == "USES"
    assert rel.evidence_quote.startswith("RAG systems use ChromaDB")


def test_get_neighbors(sample_entities, sample_relationships):
    store = GraphStore()
    for ent in sample_entities:
        store.add_entity(ent)
    for rel in sample_relationships:
        store.add_relationship(rel)

    # RAG has 2 outgoing edges
    out_neighbors = store.get_neighbors("ent_concept_rag", direction="out")
    assert len(out_neighbors) == 2
    target_names = {n["neighbor"].canonical_name for n in out_neighbors}
    assert target_names == {"ChromaDB", "NetworkX"}

    # ChromaDB has 1 incoming edge
    in_neighbors = store.get_neighbors("ent_software_chroma", direction="in")
    assert len(in_neighbors) == 1
    assert in_neighbors[0]["neighbor"].canonical_name == "RAG"
    assert in_neighbors[0]["direction"] == "incoming"


def test_get_subgraph(sample_entities, sample_relationships):
    store = GraphStore()
    for ent in sample_entities:
        store.add_entity(ent)
    for rel in sample_relationships:
        store.add_relationship(rel)

    # Subgraph for ChromaDB (1 hop) includes ChromaDB and RAG
    sub = store.get_subgraph(["ent_software_chroma"], max_hops=1)
    assert "ent_software_chroma" in sub.nodes
    assert "ent_concept_rag" in sub.nodes
    assert "ent_framework_networkx" not in sub.nodes


def test_graph_statistics(sample_entities, sample_relationships):
    store = GraphStore()
    for ent in sample_entities:
        store.add_entity(ent)
    for rel in sample_relationships:
        store.add_relationship(rel)

    stats = store.get_statistics()
    assert stats["node_count"] == 3
    assert stats["edge_count"] == 2
    assert stats["density"] > 0
    assert stats["weakly_connected_components"] == 1
    assert stats["is_weakly_connected"] is True
    assert stats["entity_type_distribution"]["CONCEPT"] == 1
    assert stats["entity_type_distribution"]["SOFTWARE"] == 1
    assert stats["predicate_distribution"]["USES"] == 1
    assert len(stats["top_entities_by_degree"]) == 3
    assert stats["top_entities_by_degree"][0]["canonical_name"] == "RAG"


def test_remove_document_and_orphan_cleanup(sample_entities, sample_relationships):
    store = GraphStore()
    for ent in sample_entities:
        store.add_entity(ent)
    for rel in sample_relationships:
        store.add_relationship(rel)

    # doc_1 supports ChromaDB (only) and RAG (doc_1 + doc_2)
    # Removing doc_1 should remove rel_rag_uses_chroma and orphan ChromaDB
    edges_removed, nodes_removed = store.remove_document("doc_1")
    assert edges_removed == 1
    assert nodes_removed == 1
    assert store.get_entity("ent_software_chroma") is None
    # RAG should still exist because doc_2 also supports it
    rag = store.get_entity("ent_concept_rag")
    assert rag is not None
    assert rag.source_document_ids == ["doc_2"]


def test_save_and_load_json_roundtrip(sample_entities, sample_relationships, tmp_path: Path):
    store = GraphStore()
    for ent in sample_entities:
        store.add_entity(ent)
    for rel in sample_relationships:
        store.add_relationship(rel)

    json_path = tmp_path / "test_graph.json"
    store.save_json(json_path)
    assert json_path.exists()

    new_store = GraphStore()
    new_store.load_json(json_path)

    assert new_store.node_count == store.node_count
    assert new_store.edge_count == store.edge_count
    loaded_rag = new_store.get_entity("ent_concept_rag")
    assert loaded_rag is not None
    assert loaded_rag.canonical_name == "RAG"
    assert loaded_rag.aliases == ["Retrieval Augmented Generation"]
    loaded_rel = new_store.get_relationship("rel_rag_uses_chroma")
    assert loaded_rel is not None
    assert loaded_rel.predicate == "USES"


def test_export_graphml_sanitized(sample_entities, sample_relationships, tmp_path: Path):
    store = GraphStore()
    for ent in sample_entities:
        store.add_entity(ent)
    for rel in sample_relationships:
        store.add_relationship(rel)

    graphml_path = tmp_path / "test_graph.graphml"
    store.export_graphml(graphml_path)
    assert graphml_path.exists()

    # Verify that standard NetworkX read_graphml can read it back without errors
    read_back = nx.read_graphml(graphml_path)
    assert read_back.number_of_nodes() == 3
    assert read_back.number_of_edges() == 2
    node_rag = read_back.nodes["ent_concept_rag"]
    assert node_rag["canonical_name"] == "RAG"
    # Aliases should have been serialized as JSON string in GraphML
    assert "Retrieval Augmented Generation" in node_rag["aliases"]


def test_reset(sample_entities, sample_relationships):
    store = GraphStore()
    for ent in sample_entities:
        store.add_entity(ent)
    for rel in sample_relationships:
        store.add_relationship(rel)

    assert store.node_count == 3
    assert store.edge_count == 2
    store.reset()
    assert store.node_count == 0
    assert store.edge_count == 0
    assert len(store.get_entities()) == 0
    assert len(store.get_relationships()) == 0
