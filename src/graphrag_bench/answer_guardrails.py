"""Answer Guardrails & Factuality Safety Module for Grounded QA (Phase 7).

Implements pre-generation context checks, refusal phrase detection,
delimiters sanitization, and fallback policies for ungrounded generation.

=============================================================================
MATHEMATICAL SPECIFICATION OF RETRIEVAL RELEVANCE SCALES
=============================================================================

Different retrieval modalities and ranking algorithms operate on disparate
mathematical scales that MUST be explicitly distinguished and never conflated:

1. ChromaDB Cosine Distance (d):
   - Definition: d = 1.0 - (u . v) / (||u|| * ||v||)
   - Range: d in [0.0, 2.0]
   - Perfect semantic alignment yields d = 0.0; orthogonal yields d = 1.0; opposite yields d = 2.0.

2. Raw Cosine Similarity (s_raw):
   - Definition: s_raw = 1.0 - d
   - Range: s_raw in [-1.0, 1.0]
   - Negative values represent opposing semantic directions.

3. Normalized Cosine Score (s_norm):
   - Definition: s_norm = 1.0 - (d / 2.0)
   - Range: s_norm in [0.0, 1.0]
   - Strict bounded unit-interval metric utilized for vector relevance thresholding.
   - Default vector threshold: s_norm >= 0.50 (equivalent to d <= 1.0 or s_raw >= 0.0).

4. Graph Path Score (s_graph):
   - Definition: s_graph = prod(c_e for e in path) * (gamma ^ hop_count)
   - Range: s_graph in [0.0, 1.0]
   - Represents joint relationship extraction confidence penalized by exponential hop decay gamma in (0, 1].
   - Default graph threshold: s_graph >= 0.40.

5. Reciprocal Rank Fusion Score (s_rrf):
   - Definition: s_rrf = sum(w_m / (k + r_m) for m in modalities)
   - Range: s_rrf in [0.0, (w_vec + w_graph) / (k + 1)]
   - With standard benchmark settings (k=60, w_vec=1.0, w_graph=1.0):
     Max possible RRF score (rank 1 in both modalities) is 2 / (60 + 1) = 0.03279.
   - CAUTION: RRF scores reflect order-based rank reciprocal sums, NOT cosine similarities.
     Comparing an RRF score directly to a 0.50 threshold is mathematically invalid.
     Hybrid relevance must evaluate modality-specific normalized scores (s_norm or s_graph)
     or decoupled rank indicators.
=============================================================================
"""

from typing import NamedTuple

from graphrag_bench.answer_schemas import AnswerCitation, AnswerStatus, EvidenceItem
from graphrag_bench.logger import get_logger

logger = get_logger(__name__)

STANDARD_INSUFFICIENT_EVIDENCE_MESSAGE = (
    "Based on the provided evidence, there is insufficient information to answer this question."
)

INSUFFICIENT_EVIDENCE_PHRASES: tuple[str, ...] = (
    "insufficient information",
    "insufficient evidence",
    "not enough information",
    "cannot be answered based on the provided",
    "does not mention",
    "does not contain information",
    "no information provided",
    "evidence does not provide",
    "not mentioned in the provided",
    "cannot determine from the provided",
)


class PreCheckResult(NamedTuple):
    """Result of pre-generation guardrail evaluation."""

    can_proceed: bool
    status: AnswerStatus
    early_answer: str | None
    warnings: list[str]


class PostCheckResult(NamedTuple):
    """Result of post-generation guardrail evaluation."""

    status: AnswerStatus
    sanitized_answer: str
    warnings: list[str]


class AnswerGuardrails:
    """Pre- and post-generation safety checks for grounded question answering."""

    def __init__(
        self,
        min_evidence_items: int = 1,
        weak_evidence_threshold: float = 0.001,
        min_relevance_threshold: float | None = None,
        min_vector_threshold: float = 0.50,
        min_graph_threshold: float = 0.40,
    ) -> None:
        """Initialize guardrails with calibrated modality thresholds.

        Args:
            min_evidence_items: Minimum number of evidence items required to proceed.
            weak_evidence_threshold: Score threshold below which a warning is logged.
            min_relevance_threshold: Optional relevance threshold override below which generation is short-circuited.
            min_vector_threshold: Bounded normalized vector score threshold (s_norm in [0, 1]). Defaults to 0.50.
            min_graph_threshold: Bounded graph path score threshold (s_graph in [0, 1]). Defaults to 0.40.
        """
        self.min_evidence_items = min_evidence_items
        self.weak_evidence_threshold = weak_evidence_threshold
        self.min_relevance_threshold = min_relevance_threshold
        self.min_vector_threshold = min_vector_threshold
        self.min_graph_threshold = min_graph_threshold

    def evaluate_pre_generation(
        self,
        query: str,
        evidence_items: list[EvidenceItem],
        min_relevance_threshold: float | None = None,
    ) -> PreCheckResult:
        """Evaluate pre-generation constraints on query and retrieved context.

        Decouples modality-specific relevance metrics:
        - Vector evidence is evaluated against normalized cosine score s_norm in [0, 1].
        - Graph evidence is evaluated against graph path score s_graph in [0, 1].
        - In hybrid mode, evidence is considered relevant if EITHER modality satisfies
          its respective threshold (preventing high graph relevance from being discarded
          due to low vector relevance, or vice versa).
        - Raw RRF scores (~0.015 - 0.033) are never conflated with cosine thresholds.

        Args:
            query: User query string.
            evidence_items: List of retrieved evidence items.
            min_relevance_threshold: Optional relevance threshold override to short-circuit generation.

        Returns:
            PreCheckResult indicating whether LLM generation should be executed.
        """
        warnings: list[str] = []

        if not query or not query.strip():
            return PreCheckResult(
                can_proceed=False,
                status=AnswerStatus.FAILED,
                early_answer="Query cannot be empty.",
                warnings=["Empty query provided."],
            )

        if len(evidence_items) < self.min_evidence_items:
            logger.info(
                "Insufficient evidence retrieved (%d items); short-circuiting generation.",
                len(evidence_items),
            )
            return PreCheckResult(
                can_proceed=False,
                status=AnswerStatus.INSUFFICIENT_EVIDENCE,
                early_answer=STANDARD_INSUFFICIENT_EVIDENCE_MESSAGE,
                warnings=["Zero or insufficient evidence items retrieved."],
            )

        effective_threshold = (
            min_relevance_threshold
            if min_relevance_threshold is not None
            else self.min_relevance_threshold
        )

        # Calibrate modality-specific checks across evidence items
        v_thresh = (
            effective_threshold if effective_threshold is not None else self.min_vector_threshold
        )
        g_thresh = (
            min(effective_threshold, self.min_graph_threshold)
            if effective_threshold is not None
            else self.min_graph_threshold
        )

        has_relevant_evidence = False
        relevance_scores: list[float] = []

        for item in evidence_items:
            norm_val = item.metadata.get("normalized_score")
            graph_val = item.metadata.get("graph_score")

            v_score = float(norm_val) if norm_val is not None else None
            g_score = float(graph_val) if graph_val is not None else None

            item_scores: list[float] = []
            if v_score is not None:
                item_scores.append(v_score)
                if v_score >= v_thresh:
                    has_relevant_evidence = True
            if g_score is not None:
                item_scores.append(g_score)
                if g_score >= g_thresh:
                    has_relevant_evidence = True

            if item_scores:
                relevance_scores.append(max(item_scores))
            else:
                # Fallback if metadata lacks explicit modality scores
                ret_sc = float(item.retrieval_score)
                relevance_scores.append(ret_sc)
                if effective_threshold is not None:
                    if ret_sc >= effective_threshold:
                        has_relevant_evidence = True
                elif ret_sc >= self.weak_evidence_threshold:
                    has_relevant_evidence = True

        top_relevance = max(relevance_scores) if relevance_scores else 0.0

        if effective_threshold is not None and not has_relevant_evidence:
            msg = (
                f"Retrieved evidence top relevance ({top_relevance:.4f}) is below required "
                f"relevance threshold (vector={v_thresh:.2f}, graph={g_thresh:.2f}); short-circuiting generation."
            )
            logger.info(msg)
            return PreCheckResult(
                can_proceed=False,
                status=AnswerStatus.INSUFFICIENT_EVIDENCE,
                early_answer=STANDARD_INSUFFICIENT_EVIDENCE_MESSAGE,
                warnings=[msg],
            )

        if top_relevance < self.weak_evidence_threshold:
            msg = f"Retrieved evidence has very low relevance score ({top_relevance:.6f})."
            warnings.append(msg)
            logger.warning(msg)

        return PreCheckResult(
            can_proceed=True,
            status=AnswerStatus.SUCCESS,
            early_answer=None,
            warnings=warnings,
        )

    def evaluate_post_generation(
        self,
        answer: str,
        citations: list[AnswerCitation],
        require_citations: bool = True,
    ) -> PostCheckResult:
        """Evaluate post-generation output, sanitize leaks, and classify status.

        Args:
            answer: Raw or cleaned answer generated by LLM.
            citations: Extracted citation objects.
            require_citations: Whether citations were required.

        Returns:
            PostCheckResult with final operational status and sanitized answer.
        """
        warnings: list[str] = []
        sanitized = answer.strip()

        # Sanitize internal delimiter leaks if LLM regurgitated them
        leak_delimiters = [
            "<<<EVIDENCE_BLOCK_START>>>",
            "<<<EVIDENCE_BLOCK_END>>>",
            "<<<EVIDENCE_START>>>",
            "<<<EVIDENCE_END>>>",
        ]
        for delim in leak_delimiters:
            if delim in sanitized:
                sanitized = sanitized.replace(delim, "").strip()
                warnings.append(f"Prompt delimiter '{delim}' leaked into answer and was sanitized.")

        # Check for insufficient evidence phrases
        lower_ans = sanitized.lower()
        valid_citations = [c for c in citations if c.is_valid]
        is_refusal_intent = (
            len(sanitized) < 150
            or lower_ans.startswith(
                (
                    "based on the provided evidence, there is insufficient",
                    "there is insufficient information",
                )
            )
            or not valid_citations
        )
        if is_refusal_intent and any(
            phrase in lower_ans for phrase in INSUFFICIENT_EVIDENCE_PHRASES
        ):
            logger.info("Model output indicates insufficient evidence to answer query.")
            return PostCheckResult(
                status=AnswerStatus.INSUFFICIENT_EVIDENCE,
                sanitized_answer=sanitized,
                warnings=warnings,
            )

        # Check citation presence
        valid_citations = [c for c in citations if c.is_valid]
        if require_citations and not valid_citations:
            msg = "Answer contains no valid citations attributing facts to retrieved evidence."
            warnings.append(msg)
            logger.warning(msg)

        return PostCheckResult(
            status=AnswerStatus.SUCCESS,
            sanitized_answer=sanitized,
            warnings=warnings,
        )
