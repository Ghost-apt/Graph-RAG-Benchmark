"""Claim Extraction & Grounding Faithfulness Checker for Grounded QA (Phase 7).

Decomposes generated answers into atomic factual claims, checks them against
the retrieved source evidence, and computes an overall grounding faithfulness score.
"""

import json
import re
from typing import Literal

from graphrag_bench.answer_schemas import AnswerClaim, EvidenceItem
from graphrag_bench.logger import get_logger
from graphrag_bench.ollama_client import OllamaClient
from graphrag_bench.prompt_builder import PromptBuilder

logger = get_logger(__name__)


class FaithfulnessChecker:
    """Verifies that statements in a generated answer are grounded in retrieved evidence."""

    def __init__(
        self,
        ollama_client: OllamaClient | None = None,
        prompt_builder: PromptBuilder | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize the FaithfulnessChecker.

        Args:
            ollama_client: Injected OllamaClient or None.
            prompt_builder: Injected PromptBuilder or None.
            model: Model identifier for evaluation (defaults to ollama_client default).
        """
        self.ollama_client = ollama_client or OllamaClient()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.model = model

    def split_into_heuristic_claims(self, text: str) -> list[str]:
        """Split text into sentence-level candidate claims using punctuation rules.

        Args:
            text: Answer text.

        Returns:
            List of non-trivial declarative sentence strings.
        """
        # Remove citation markers temporarily to split cleanly
        clean = re.sub(r"\[S\d+(?:[\s,]+S\d+)*\]", "", text)
        sentences = re.split(r"(?<=[.!?])\s+", clean.strip())
        claims: list[str] = []
        for s in sentences:
            s_clean = s.strip()
            # Ignore short greetings, disclaimers, or empty lines
            if len(s_clean.split()) >= 4 and not s_clean.lower().startswith(
                ("based on", "according to", "in conclusion", "hello", "hi")
            ):
                claims.append(s_clean)
        return claims

    def evaluate_heuristic_grounding(
        self,
        claim: str,
        evidence_items: list[EvidenceItem],
    ) -> tuple[Literal["SUPPORTED", "UNSUPPORTED", "UNCERTAIN"], str, float]:
        """Perform heuristic lexical/entity overlap grounding evaluation.

        Args:
            claim: Single claim string.
            evidence_items: Available evidence items.

        Returns:
            Tuple of (verdict, reasoning, grounding_score).
        """
        if not evidence_items:
            return "UNSUPPORTED", "No evidence context available to support claim.", 0.0

        claim_words = {w.lower().strip(".,;:?!'\"()[]{}") for w in claim.split() if len(w) > 3}
        if not claim_words:
            return "UNCERTAIN", "Claim too short or contains only stop words.", 0.5

        best_overlap_ratio = 0.0
        best_source_id = ""

        for item in evidence_items:
            ev_words = {
                w.lower().strip(".,;:?!'\"()[]{}") for w in item.content.split() if len(w) > 3
            }
            if not ev_words:
                continue
            common = claim_words.intersection(ev_words)
            ratio = len(common) / len(claim_words)
            if ratio > best_overlap_ratio:
                best_overlap_ratio = ratio
                best_source_id = item.evidence_id

        if best_overlap_ratio >= 0.50:
            verdict: Literal["SUPPORTED", "UNSUPPORTED", "UNCERTAIN"] = "SUPPORTED"
            reasoning = (
                f"Lexical overlap of {best_overlap_ratio:.1%} with source [{best_source_id}]."
            )
        elif best_overlap_ratio >= 0.25:
            verdict = "UNCERTAIN"
            reasoning = f"Partial lexical overlap of {best_overlap_ratio:.1%} with source [{best_source_id}]."
        else:
            verdict = "UNSUPPORTED"
            reasoning = f"Low lexical overlap ({best_overlap_ratio:.1%}) with retrieved evidence."

        return verdict, reasoning, best_overlap_ratio

    def check_faithfulness(
        self,
        answer: str,
        evidence_items: list[EvidenceItem],
        use_llm: bool = True,
    ) -> tuple[list[AnswerClaim], float | None]:
        """Verify the factual claims in an answer against evidence items.

        NOTE ON SEMANTICS:
        - Heuristic verification uses token and n-gram lexical overlap between the extracted
          claim sentence and retrieved evidence content. This serves as a proxy for presence
          of factual terms, but CANNOT establish formal semantic entailment or verify
          counterfactual logical implications.
        - When zero claims are extracted (e.g. empty answer, refusal disclaimer, or conversational
          filler), the faithfulness score returns None (represented as N/A), indicating that
          grounding is not applicable rather than mathematically 0.0 or 1.0.

        Args:
            answer: Generated answer text.
            evidence_items: Evidence items provided to generation prompt.
            use_llm: Whether to attempt LLM-based claim verification before falling back to heuristics.

        Returns:
            Tuple of (list_of_claims, overall_faithfulness_score or None).
        """
        if not answer.strip() or not evidence_items:
            return [], None

        # Try LLM extraction if enabled
        raw_claims: list[str] = []
        if use_llm:
            try:
                raw_claims = self._extract_claims_llm(answer)
            except Exception as exc:
                logger.debug("LLM claim extraction failed, using heuristic: %s", exc)

        if not raw_claims:
            raw_claims = self.split_into_heuristic_claims(answer)

        if not raw_claims:
            return [], None

        evaluated_claims: list[AnswerClaim] = []
        supported_count = 0

        for idx, claim_text in enumerate(raw_claims, start=1):
            claim_id = f"C{idx}"

            # Extract any cited sources present in the claim if answer had citations
            cited = re.findall(r"\[(S\d+)\]", answer)
            cited_sources = sorted(set(cited))

            verdict: Literal["SUPPORTED", "UNSUPPORTED", "UNCERTAIN"]
            reasoning: str
            score: float

            if use_llm:
                try:
                    verdict, reasoning, score = self._verify_claim_llm(claim_text, evidence_items)
                except Exception as exc:
                    logger.debug("LLM verification failed for claim '%s': %s", claim_text, exc)
                    verdict, reasoning, score = self.evaluate_heuristic_grounding(
                        claim_text, evidence_items
                    )
            else:
                verdict, reasoning, score = self.evaluate_heuristic_grounding(
                    claim_text, evidence_items
                )

            if verdict == "SUPPORTED":
                supported_count += 1

            evaluated_claims.append(
                AnswerClaim(
                    claim_id=claim_id,
                    claim_text=claim_text,
                    cited_sources=cited_sources,
                    verdict=verdict,
                    reasoning=reasoning,
                    grounding_score=score,
                )
            )

        total = len(evaluated_claims)
        faithfulness: float | None = (supported_count / total) if total > 0 else None

        return evaluated_claims, faithfulness

    def _extract_claims_llm(self, answer: str) -> list[str]:
        """Extract claims via OllamaClient structured JSON call."""
        sys_prompt, user_prompt = self.prompt_builder.build_claim_extraction_prompt(answer)
        resp = self.ollama_client.generate(
            prompt=user_prompt,
            system=sys_prompt,
            model=self.model,
            format="json",
            timeout=30,
        )
        data = json.loads(resp.response)
        if isinstance(data, list):
            return [str(c).strip() for c in data if str(c).strip()]
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    return [str(c).strip() for c in v if str(c).strip()]
        return []

    def _verify_claim_llm(
        self,
        claim: str,
        evidence_items: list[EvidenceItem],
    ) -> tuple[Literal["SUPPORTED", "UNSUPPORTED", "UNCERTAIN"], str, float]:
        """Verify claim via OllamaClient JSON call."""
        sys_prompt, user_prompt = self.prompt_builder.build_claim_verification_prompt(
            claim, evidence_items
        )
        resp = self.ollama_client.generate(
            prompt=user_prompt,
            system=sys_prompt,
            model=self.model,
            format="json",
            timeout=30,
        )
        data = json.loads(resp.response)
        raw_verdict = str(data.get("verdict", "UNCERTAIN")).upper().strip()
        verdict: Literal["SUPPORTED", "UNSUPPORTED", "UNCERTAIN"] = "UNCERTAIN"
        if raw_verdict in {"SUPPORTED", "UNSUPPORTED", "UNCERTAIN"}:
            verdict = raw_verdict  # type: ignore[assignment]

        reasoning = str(data.get("reasoning", ""))
        score = float(data.get("grounding_score", 1.0 if verdict == "SUPPORTED" else 0.0))
        return verdict, reasoning, max(0.0, min(1.0, score))
