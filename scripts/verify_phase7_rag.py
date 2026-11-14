"""End-to-End Benchmark & Data Integrity Verification Script for Phase 7 Grounded RAG.

Executes 6 domain benchmark queries (5 answerable + 1 unanswerable) against
canonical data, verifies citations and grounding, records latency and tokens,
and verifies cryptographic checksums of all underlying database and graph files.
"""

import hashlib
import sys
import time
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from graphrag_bench.answer_schemas import AnswerStatus
from graphrag_bench.config import get_settings
from graphrag_bench.document_repository import DocumentRepository
from graphrag_bench.graph_store import GraphStore
from graphrag_bench.ollama_client import OllamaClient
from graphrag_bench.rag_pipeline import RAGPipeline
from graphrag_bench.vector_store import VectorStore

BENCHMARK_QUERIES = [
    {
        "id": "Q1",
        "category": "Broad Architecture",
        "query": "What is Graph RAG and what are its core architectural components?",
        "expected_status": AnswerStatus.SUCCESS,
        "must_have_citations": True,
    },
    {
        "id": "Q2",
        "category": "Vector & Embeddings",
        "query": "Which vector embedding model is evaluated for Vector RAG in the benchmarking guide?",
        "expected_status": AnswerStatus.SUCCESS,
        "must_have_citations": True,
    },
    {
        "id": "Q3",
        "category": "Multi-Hop Reasoning",
        "query": "How does multi-hop graph traversal enhance document retrieval compared to plain vector search?",
        "expected_status": AnswerStatus.SUCCESS,
        "must_have_citations": True,
    },
    {
        "id": "Q4",
        "category": "Reciprocal Rank Fusion",
        "query": "What is Reciprocal Rank Fusion (RRF) in the context of Hybrid RAG?",
        "expected_status": AnswerStatus.SUCCESS,
        "must_have_citations": True,
    },
    {
        "id": "Q5",
        "category": "Benchmarking & Evaluation",
        "query": "What metrics and evaluation methodologies are used to benchmark Graph RAG systems?",
        "expected_status": AnswerStatus.SUCCESS,
        "must_have_citations": True,
    },
    {
        "id": "Q6",
        "category": "Unanswerable / Out-of-Domain Guardrail",
        "query": "What was the average snowfall in Zurich during January 1985?",
        "expected_status": AnswerStatus.INSUFFICIENT_EVIDENCE,
        "must_have_citations": False,
    },
]

BASELINE_HASHES = {
    "data/processed/chunks.jsonl": "F236374B00197003B1F9933CA8BEEF576CEDDE41BD67E5DF12718AB15FAB16F4",
    "data/processed/documents.json": "CFE029D3AC803D9D7AAFC697037C5296F491B5CE7F658C541A393317A4D54EAB",
    "data/processed/manifest.json": "69BD862C213332BEEB0E6FA0A047B9790AD3400240FDB8371B2F77F57ECDAF39",
    "data/knowledge_graph/knowledge_graph.json": "2E0C8783848B69C730825C20AB6C0D96D674C4195C809E5F4AA71CA89B0AF6DD",
    "data/knowledge_graph/knowledge_graph.graphml": "4BA25CAA4C8259218FEBB874B28EE278352FF7E1A6FF7CA455D41AB5A9F198ED",
    "data/knowledge_graph/manifest.json": "66ABAD41A937157D9D711082AFD39A9CB0A3D77B7B274FCAA0DCD3AD5CAABE8D",
    "data/knowledge_graph/extractions.jsonl": "7769F803E3EFF7AC37565DDF9EE6BF2C2AC4EB6449A4EDB0DC52CE69E42E44C8",
}


def compute_sha256(filepath: Path) -> str:
    """Compute uppercase SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()


def main() -> int:
    """Execute Phase 7 live grounded RAG verification."""
    print("=" * 80)
    print("       GRAPH RAG BENCHMARK — PHASE 7 GROUNDED RAG VERIFICATION")
    print("=" * 80)

    cfg = get_settings()
    client = OllamaClient()

    # 1. Environment & Service Connectivity Check
    print("\n[STEP 1] Validating Ollama Service & Models...")
    if not client.is_server_reachable():
        print("[-] ERROR: Ollama server is not reachable at", cfg.ollama_base_url)
        return 1
    print(f"[+] Ollama server reachable at {cfg.ollama_base_url}")

    models = client.list_models()
    print(f"[+] Installed Ollama models: {', '.join(models)}")
    if not client.is_model_installed(cfg.rag_generation_model):
        print(f"[-] ERROR: Required generation model '{cfg.rag_generation_model}' not found.")
        return 1
    print(f"[+] LLM Generation model confirmed: {cfg.rag_generation_model}")

    # 2. Canonical Datasets Check
    print("\n[STEP 2] Verifying Canonical Storage & Indexes...")
    repo = DocumentRepository()
    all_chunks = repo._get_all_chunks()
    print(f"[+] Document Repository: {len(all_chunks)} canonical chunks found.")

    vstore = VectorStore()
    vcount = vstore.count()
    print(
        f"[+] Vector Database (Chroma): {vcount} vectors in collection '{vstore.collection_name}'."
    )

    gstore = GraphStore()
    gstore.load_json(cfg.graph_path)
    print(f"[+] Knowledge Graph: {gstore.node_count} nodes, {gstore.edge_count} edges loaded.")

    if len(all_chunks) != 3 or vcount != 3 or gstore.node_count != 22:
        print(
            "[-] WARNING: Canonical counts differ from expected baseline (3 chunks, 3 vectors, 22 nodes)."
        )

    # 3. Execute Benchmark Queries
    print("\n[STEP 3] Executing 6 Canonical Benchmark Queries with RAGPipeline...")
    pipeline = RAGPipeline(ollama_client=client)

    all_passed = True
    query_results = []

    for bq in BENCHMARK_QUERIES:
        qid = bq["id"]
        category = bq["category"]
        qtext = bq["query"]
        expected_status = bq["expected_status"]
        must_cite = bq["must_have_citations"]

        print(f"\n{'-' * 80}")
        print(f'Query {qid} [{category}]: "{qtext}"')
        print(f"{'-' * 80}")

        t0 = time.perf_counter()
        ans = pipeline.answer_question(
            request=qtext,
            retrieval_mode="hybrid",
            top_k=5,
            temperature=0.1,
            require_citations=True,
            enable_faithfulness_check=True,
        )
        duration = time.perf_counter() - t0

        print(f"Status              : {ans.status.value.upper()}")
        print(f"Retrieval Latency   : {ans.latency_retrieval_ms:.1f} ms")
        print(f"Generation Latency  : {ans.latency_generation_ms:.1f} ms")
        print(
            f"Total Latency       : {ans.latency_total_ms:.1f} ms (wall clock: {duration * 1000:.1f} ms)"
        )
        print(f"Evidence Items Used : {len(ans.evidence_items)}")
        print(
            f"Citations Extracted : {len(ans.citations)} (Valid: {sum(1 for c in ans.citations if c.is_valid)})"
        )
        prec_str = (
            f"{ans.citation_reference_precision * 100:.1f}%"
            if ans.citation_reference_precision is not None
            else "N/A (No citations)"
        )
        faith_str = (
            f"{ans.faithfulness_score * 100:.1f}%"
            if ans.faithfulness_score is not None
            else "N/A (No claims evaluated)"
        )
        invoked_str = "Yes" if ans.generation_invoked else "No (Short-Circuited)"
        print(f"Generation Invoked  : {invoked_str}")
        print(
            f"Model Invocations   : Total={ans.total_model_invocations} (Extraction={ans.query_extraction_model_calls}, Generation={ans.answer_generation_model_calls})"
        )
        print(f"Citation Ref Prec   : {prec_str}")
        print(f"Faithfulness Score  : {faith_str}")
        if ans.latency_breakdown:
            print(f"Latency Breakdown   : {ans.latency_breakdown}")
        print(f"\nGenerated Answer:\n{ans.answer}\n")

        if ans.citations:
            print("Citations Table:")
            for cit in ans.citations:
                valid_str = "VALID" if cit.is_valid else "INVALID/HALLUCINATED"
                print(
                    f"  [{cit.evidence_id}] {cit.document_title} | Chunk: {cit.chunk_id} | Status: {valid_str}"
                )

        if ans.claims:
            print(f"Claims Evaluated ({len(ans.claims)}):")
            for cl in ans.claims:
                print(
                    f'  [{cl.claim_id}] {cl.verdict} | Claim: "{cl.claim_text[:60]}..." | Score: {cl.grounding_score:.2f}'
                )

        # Check expectations
        status_ok = ans.status == expected_status
        citations_ok = True
        if must_cite:
            citations_ok = len(ans.citations) > 0 and any(c.is_valid for c in ans.citations)
        elif expected_status == AnswerStatus.INSUFFICIENT_EVIDENCE:
            # Should have no fabricated citations
            citations_ok = len([c for c in ans.citations if not c.is_valid]) == 0

        passed = status_ok and citations_ok
        if not passed:
            all_passed = False
            print(
                f"[-] TEST FAILED: expected status={expected_status}, got={ans.status}; citations_ok={citations_ok}"
            )
        else:
            print(f"[+] TEST PASSED for {qid}")

        query_results.append(
            {
                "id": qid,
                "category": category,
                "status": ans.status.value,
                "latency_ms": ans.latency_total_ms,
                "citations_count": len(ans.citations),
                "precision": ans.citation_precision,
                "faithfulness": ans.faithfulness_score,
                "passed": passed,
            }
        )

    # 4. Canonical Hash Immutability Verification
    print(f"\n{'=' * 80}")
    print("[STEP 4] Verifying Post-Execution Canonical Data Immutability...")
    print(f"{'=' * 80}")

    hash_match = True
    for rel_path, expected_hash in BASELINE_HASHES.items():
        p = Path(rel_path)
        if not p.exists():
            print(f"[-] MISSING FILE: {rel_path}")
            hash_match = False
            continue
        current_hash = compute_sha256(p)
        if current_hash != expected_hash:
            print(f"[-] HASH MISMATCH for {rel_path}!")
            print(f"    Expected: {expected_hash}")
            print(f"    Actual  : {current_hash}")
            hash_match = False
        else:
            print(f"[+] IMMUTABLE: {rel_path} matches baseline {current_hash[:16]}...")

    # Summary table
    print("\n" + "=" * 80)
    print("PHASE 7 BENCHMARK SUMMARY TABLE")
    print("=" * 80)
    print(f"{'ID':<4} | {'Category':<32} | {'Status':<22} | {'Latency':<10} | {'Passed'}")
    print("-" * 80)
    for qr in query_results:
        print(
            f"{qr['id']:<4} | {qr['category']:<32} | {qr['status']:<22} | {qr['latency_ms']:<8.1f}ms | {'YES' if qr['passed'] else 'NO'}"
        )
    print("-" * 80)

    if all_passed and hash_match:
        print("\n>>> ALL PHASE 7 BENCHMARK QUERIES PASSED WITH ZERO DATA MUTATIONS! <<<\n")
        return 0
    else:
        print("\n>>> PHASE 7 VERIFICATION ENCOUNTERED DISCREPANCIES. <<<\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
