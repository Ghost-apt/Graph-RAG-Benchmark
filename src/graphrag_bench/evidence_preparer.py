"""Evidence Preparation & Context Packaging Engine for Grounded Question Answering (Phase 7).

Transforms retrieved search candidates (Hybrid, Vector, or Graph) into structured,
deduplicated, and budget-constrained EvidenceItem objects formatted for LLM generation prompts.
"""

from typing import Any

from graphrag_bench.answer_schemas import EvidenceItem
from graphrag_bench.hybrid_schemas import FusionCandidate, HybridSearchResponse
from graphrag_bench.logger import get_logger

logger = get_logger(__name__)


def estimate_tokens(text: str) -> int:
    """Conservatively estimate token count from text using word and character heuristics.

    Heuristic:
    - 1 word is approximately 1.35 tokens in typical BPE/subword tokenizers.
    - 1 token is approximately 3.2 characters in technical text.
    - We take the maximum of both metrics and add ceiling padding to prevent
      underestimating context length.

    Args:
        text: Input string.

    Returns:
        Conservative estimated token count (always >= 0).
    """
    if not text or not text.strip():
        return 0
    text_clean = text.strip()
    words = len(text_clean.split())
    chars = len(text_clean)
    by_chars = int(chars / 3.2 + 0.999)
    by_words = int(words * 1.35 + 0.999)
    return max(by_chars, by_words)


class EvidencePreparer:
    """Prepares, deduplicates, and formats retrieved evidence for grounded LLM prompts."""

    def __init__(self, default_max_context_tokens: int = 3000) -> None:
        """Initialize the EvidencePreparer.

        Args:
            default_max_context_tokens: Default upper bound for total prompt evidence context.
        """
        self.default_max_context_tokens = default_max_context_tokens

    def prepare_from_hybrid(
        self,
        hybrid_response: HybridSearchResponse,
        max_tokens: int | None = None,
        top_k: int = 5,
    ) -> list[EvidenceItem]:
        """Convert a Phase 6 HybridSearchResponse into packaged EvidenceItems.

        Args:
            hybrid_response: Response from HybridRetriever.
            max_tokens: Token budget limit. Defaults to default_max_context_tokens.
            top_k: Maximum number of candidate items to include.

        Returns:
            List of structured EvidenceItem objects assigned citation IDs (S1, S2, etc.).
        """
        budget = max_tokens if max_tokens is not None else self.default_max_context_tokens
        candidates = hybrid_response.results[:top_k]
        return self._package_fusion_candidates(candidates, budget)

    def prepare_from_fusion_candidates(
        self,
        candidates: list[FusionCandidate],
        max_tokens: int | None = None,
        top_k: int = 5,
    ) -> list[EvidenceItem]:
        """Package a list of FusionCandidate objects directly.

        Args:
            candidates: List of FusionCandidate objects.
            max_tokens: Token budget limit.
            top_k: Top K items to package.

        Returns:
            List of structured EvidenceItem objects.
        """
        budget = max_tokens if max_tokens is not None else self.default_max_context_tokens
        return self._package_fusion_candidates(candidates[:top_k], budget)

    def prepare_from_vector(
        self,
        search_items: list[Any],
        max_tokens: int | None = None,
        top_k: int = 5,
    ) -> list[EvidenceItem]:
        """Convert Phase 3 SearchResultItem list into packaged EvidenceItems.

        Args:
            search_items: List of SearchResultItem objects or dicts.
            max_tokens: Token budget limit.
            top_k: Top K items to package.

        Returns:
            List of structured EvidenceItem objects.
        """
        budget = max_tokens if max_tokens is not None else self.default_max_context_tokens
        evidence_items: list[EvidenceItem] = []
        seen_chunks: set[str] = set()
        accumulated_tokens = 0

        for item in search_items[:top_k]:
            chunk_id = getattr(item, "chunk_id", None) or (
                item.get("chunk_id") if isinstance(item, dict) else ""
            )
            if not chunk_id or chunk_id in seen_chunks:
                continue

            doc_id = getattr(item, "document_id", None) or (
                item.get("document_id") if isinstance(item, dict) else "unknown_doc"
            )
            content = getattr(item, "text", None) or (
                item.get("text") if isinstance(item, dict) else ""
            )
            score = getattr(item, "normalized_score", None) or getattr(
                item, "similarity_score", 0.0
            )
            metadata = (
                getattr(item, "metadata", {})
                if hasattr(item, "metadata")
                else (item.get("metadata", {}) if isinstance(item, dict) else {})
            )
            doc_title = (
                metadata.get("filename")
                or metadata.get("source_filename")
                or metadata.get("document_title")
                or doc_id
            )

            item_tokens = estimate_tokens(content)
            if accumulated_tokens + item_tokens > budget:
                # Truncate if reasonable budget remaining
                remaining = budget - accumulated_tokens
                if remaining >= 50:
                    words = content.split()
                    target_words = int(remaining / 1.35)
                    if target_words > 0 and len(words) > target_words:
                        content = (
                            " ".join(words[:target_words]) + " ... [truncated due to context limit]"
                        )
                        item_tokens = estimate_tokens(content)
                    else:
                        break
                else:
                    break

            evidence_id = f"S{len(evidence_items) + 1}"
            seen_chunks.add(chunk_id)
            accumulated_tokens += item_tokens

            evidence_items.append(
                EvidenceItem(
                    evidence_id=evidence_id,
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    document_title=str(doc_title),
                    content=content,
                    token_count=item_tokens,
                    source_modality="vector",
                    retrieval_score=float(score),
                    metadata=metadata,
                )
            )

        return evidence_items

    def prepare_from_graph(
        self,
        graph_results: list[Any],
        max_tokens: int | None = None,
        top_k: int = 5,
        chunk_resolver: Any = None,
    ) -> list[EvidenceItem]:
        """Convert Phase 5 GraphRetrievalResult list into packaged EvidenceItems.

        Args:
            graph_results: List of GraphRetrievalResult objects.
            max_tokens: Token budget limit.
            top_k: Top K items to package.
            chunk_resolver: Optional callable (chunk_id -> str) to resolve verbatim chunk text.

        Returns:
            List of structured EvidenceItem objects.
        """
        budget = max_tokens if max_tokens is not None else self.default_max_context_tokens
        evidence_items: list[EvidenceItem] = []
        seen_chunks: set[str] = set()
        accumulated_tokens = 0

        for res in graph_results[:top_k]:
            chunk_ids = getattr(res, "source_chunk_ids", []) or []
            chunk_id = chunk_ids[0] if chunk_ids else f"graph_{res.result_id}"
            if chunk_id in seen_chunks:
                continue

            doc_ids = getattr(res, "source_document_ids", []) or []
            doc_id = doc_ids[0] if doc_ids else "graph_extracted"
            filenames = getattr(res, "source_filenames", []) or []
            doc_title = filenames[0] if filenames else "Knowledge Graph"

            # Resolve content
            content = ""
            if chunk_resolver and callable(chunk_resolver):
                try:
                    content = chunk_resolver(chunk_id) or ""
                except Exception as exc:
                    logger.debug("Failed resolving chunk text for %s: %s", chunk_id, exc)

            if not content:
                quotes = getattr(res, "evidence_quotes", []) or []
                if quotes:
                    content = "\n".join(f"- {q}" for q in quotes)
                else:
                    path = getattr(res, "path", None)
                    content = str(path) if path else "Knowledge Graph relation traversal"

            score = getattr(res, "score", 0.0)
            item_tokens = estimate_tokens(content)

            if accumulated_tokens + item_tokens > budget:
                remaining = budget - accumulated_tokens
                if remaining >= 50:
                    words = content.split()
                    target_words = int(remaining / 1.35)
                    if target_words > 0 and len(words) > target_words:
                        content = (
                            " ".join(words[:target_words]) + " ... [truncated due to context limit]"
                        )
                        item_tokens = estimate_tokens(content)
                    else:
                        break
                else:
                    break

            evidence_id = f"S{len(evidence_items) + 1}"
            seen_chunks.add(chunk_id)
            accumulated_tokens += item_tokens

            evidence_items.append(
                EvidenceItem(
                    evidence_id=evidence_id,
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    document_title=str(doc_title),
                    content=content,
                    token_count=item_tokens,
                    source_modality="graph",
                    retrieval_score=float(score),
                    metadata={"result_id": getattr(res, "result_id", "")},
                )
            )

        return evidence_items

    def _package_fusion_candidates(
        self,
        candidates: list[FusionCandidate],
        budget: int,
    ) -> list[EvidenceItem]:
        """Internal helper to package FusionCandidate objects with token budgeting."""
        evidence_items: list[EvidenceItem] = []
        seen_chunks: set[str] = set()
        accumulated_tokens = 0

        for cand in candidates:
            chunk_id = cand.chunk_id
            if chunk_id in seen_chunks:
                continue

            content = cand.chunk_text.strip()
            item_tokens = estimate_tokens(content)

            if accumulated_tokens + item_tokens > budget:
                remaining = budget - accumulated_tokens
                if remaining >= 50:
                    words = content.split()
                    target_words = int(remaining / 1.35)
                    if target_words > 0 and len(words) > target_words:
                        content = (
                            " ".join(words[:target_words]) + " ... [truncated due to context limit]"
                        )
                        item_tokens = estimate_tokens(content)
                    else:
                        break
                else:
                    break

            evidence_id = f"S{len(evidence_items) + 1}"
            seen_chunks.add(chunk_id)
            accumulated_tokens += item_tokens

            modality_str = "hybrid"
            source_enum_val = str(cand.retrieval_sources).upper()
            if "VECTOR" in source_enum_val and "GRAPH" not in source_enum_val:
                modality_str = "vector"
            elif "GRAPH" in source_enum_val and "VECTOR" not in source_enum_val:
                modality_str = "graph"

            evidence_items.append(
                EvidenceItem(
                    evidence_id=evidence_id,
                    chunk_id=cand.chunk_id,
                    document_id=cand.document_id,
                    document_title=cand.source_filename,
                    content=content,
                    token_count=item_tokens,
                    source_modality=modality_str,  # type: ignore[arg-type]
                    retrieval_score=float(cand.rrf_score),
                    metadata={
                        "fused_rank": cand.fused_rank,
                        "vector_rank": cand.vector_rank,
                        "graph_rank": cand.graph_rank,
                        "evidence_quotes": cand.evidence_quotes,
                        **cand.source_metadata,
                    },
                )
            )

        return evidence_items

    @staticmethod
    def format_evidence_context(evidence_items: list[EvidenceItem]) -> str:
        """Format a list of EvidenceItems into a structured context block for LLM prompts.

        Surrounded by clear delimiters to prevent prompt injection and clearly
        demarcate citation markers [S1], [S2], etc. Delimiter strings inside untrusted
        chunk contents are safely escaped.

        Args:
            evidence_items: List of prepared EvidenceItem objects.

        Returns:
            Formatted string representation of evidence context.
        """
        if not evidence_items:
            return "<<<EVIDENCE_BLOCK_START>>>\nNo evidence available.\n<<<EVIDENCE_BLOCK_END>>>"

        blocks: list[str] = ["<<<EVIDENCE_BLOCK_START>>>"]
        for item in evidence_items:
            header = (
                f"[{item.evidence_id}] Source Document: {item.document_title} "
                f"(Chunk ID: {item.chunk_id} | Modality: {item.source_modality} | "
                f"Score: {item.retrieval_score:.4f})"
            )
            # Defend against delimiter prompt injection: escape delimiters inside untrusted content
            safe_content = (
                item.content.strip()
                .replace("<<<EVIDENCE_BLOCK_END>>>", "[ESCAPED_DELIMITER_BLOCK_END]")
                .replace("<<<EVIDENCE_BLOCK_START>>>", "[ESCAPED_DELIMITER_BLOCK_START]")
            )
            blocks.append(f"{header}\nContent:\n{safe_content}\n")

        blocks.append("<<<EVIDENCE_BLOCK_END>>>")
        return "\n".join(blocks)
