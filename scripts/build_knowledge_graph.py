"""Script to construct the Knowledge Graph from canonical document repository chunks."""

import sys
import time
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from graphrag_bench.document_repository import DocumentRepository
from graphrag_bench.entity_resolver import EntityResolver
from graphrag_bench.graph_builder import GraphBuilder
from graphrag_bench.graph_extraction_service import GraphExtractionService
from graphrag_bench.graph_store import GraphStore
from graphrag_bench.logger import setup_logger

setup_logger()


def main():
    print("=" * 65)
    print("      GRAPH RAG BENCHMARK - KNOWLEDGE GRAPH BUILDER")
    print("=" * 65)

    repo = DocumentRepository()
    docs = repo.list_documents()
    print(f"Total Canonical Documents: {len(docs)}")

    resolver = EntityResolver()
    store = GraphStore()
    extraction_service = GraphExtractionService()

    builder = GraphBuilder(
        doc_repo=repo,
        extraction_service=extraction_service,
        resolver=resolver,
        store=store,
    )

    def on_progress(current, total, message):
        print(f"[{current}/{total}] {message}")

    start_t = time.perf_counter()
    results = builder.build_all(force_rebuild=True, progress_callback=on_progress)
    elapsed = time.perf_counter() - start_t

    print("-" * 65)
    print(f"Construction completed in {elapsed:.2f} seconds")
    print(f"Total Chunks Inspected   : {results['chunks_inspected']}")
    print(f"Total Chunks Succeeded   : {results['chunks_successful']}")
    print(f"Total Chunks Failed      : {results['chunks_failed']}")
    print(f"Total Graph Entities     : {results['total_entities']}")
    print(f"Total Graph Relationships: {results['total_relationships']}")
    print(f"Saved Graph JSON Path    : {results['graph_json_path']}")
    print(f"Saved GraphML Path       : {results['graphml_path']}")
    print("=" * 65)


if __name__ == "__main__":
    main()
