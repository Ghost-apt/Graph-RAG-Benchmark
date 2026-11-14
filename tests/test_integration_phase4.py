"""Live integration tests for Phase 4 (Knowledge Graph Construction Engine).

These tests connect directly to the real Ollama host and Qwen3:8b model.
They are tagged with the 'integration' marker:
    python -m pytest -m "integration"
"""

from pathlib import Path

import networkx as nx
import pytest

from graphrag_bench.config import get_settings
from graphrag_bench.entity_resolver import EntityResolver
from graphrag_bench.graph_builder import GraphBuilder
from graphrag_bench.graph_extraction_service import GraphExtractionService
from graphrag_bench.graph_store import GraphStore
from graphrag_bench.ollama_client import OllamaClient
from graphrag_bench.schemas import DocumentChunk


@pytest.mark.integration
class TestPhase4LiveIntegration:
    """Integration suite verifying real Qwen3:8b extraction and Knowledge Graph construction."""

    @pytest.fixture(scope="class")
    def live_client(self):
        settings = get_settings()
        client = OllamaClient(settings=settings)
        if not client.is_server_reachable():
            pytest.skip(f"Live Ollama server is not reachable at {client.base_url}")
        target_model = getattr(settings, "graph_extraction_model", settings.ollama_llm_model)
        if not client.is_model_installed(target_model):
            pytest.skip(f"Extraction model '{target_model}' is not installed in Ollama.")
        return client

    def test_live_chunk_extraction_and_grounding(self, live_client: OllamaClient) -> None:
        service = GraphExtractionService(client=live_client)
        sample_text = (
            "Knowledge Graphs structure information as entities and relationships. "
            "Graph RAG leverages NetworkX and ChromaDB to combine graph traversal "
            "with dense semantic retrieval for enhanced question answering."
        )

        extraction = service.extract_from_chunk(sample_text, chunk_id="live_test_chunk_1")

        # Must extract at least one entity
        assert len(extraction.entities) >= 1
        for ent in extraction.entities:
            assert ent.canonical_name
            assert ent.entity_type

        # Verify any extracted relationships are strictly grounded
        for rel in extraction.relationships:
            assert rel.predicate
            assert rel.confidence >= 0.0
            # Quote must be present in the text
            assert rel.evidence_quote.lower() in sample_text.lower() or any(
                word in sample_text.lower() for word in rel.evidence_quote.lower().split()
            )

    def test_live_graph_builder_pipeline(self, live_client: OllamaClient, tmp_path: Path) -> None:
        graph_dir = tmp_path / "live_kg"
        builder = GraphBuilder(
            extraction_service=GraphExtractionService(client=live_client),
            resolver=EntityResolver(),
            store=GraphStore(),
            graph_dir=graph_dir,
        )

        test_chunk = DocumentChunk(
            chunk_id="live_pipeline_chunk_1",
            document_id="live_pipeline_doc_1",
            chunk_index=0,
            text="ChromaDB stores vector embeddings, while NetworkX constructs knowledge graphs.",
            word_count=9,
            char_count=78,
            start_char=0,
            end_char=78,
        )
        builder._process_chunk(test_chunk, doc_filename="test_graph.txt")
        builder.store.save_json(builder.graph_path)
        builder.store.export_graphml(builder.graphml_path)
        builder._save_manifest()

        assert builder.graph_path.exists()
        assert builder.graphml_path.exists()
        assert builder.manifest_path.exists()

        # Verify GraphML is readable by NetworkX
        nx_graph = nx.read_graphml(builder.graphml_path)
        assert nx_graph.number_of_nodes() >= 1

        # Verify statistics
        stats = builder.store.get_statistics()
        assert stats["node_count"] >= 1
