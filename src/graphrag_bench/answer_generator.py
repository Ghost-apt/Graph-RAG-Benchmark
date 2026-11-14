"""Answer Generation Engine for Grounded Question Answering (Phase 7).

Coordinates LLM generation via OllamaClient, enforcing citation prompts,
extracting/isolating reasoning tokens, and recording inference metrics.
"""

import re
import time
from collections.abc import Iterator
from typing import Any

from graphrag_bench.answer_schemas import EvidenceItem
from graphrag_bench.config import get_settings
from graphrag_bench.evidence_preparer import estimate_tokens
from graphrag_bench.logger import get_logger
from graphrag_bench.ollama_client import (
    OllamaClient,
    OllamaError,
)
from graphrag_bench.prompt_builder import PromptBuilder

logger = get_logger(__name__)


def strip_think_tags(text: str) -> tuple[str, str | None]:
    """Separate reasoning tokens <think>...</think> from the final answer text.

    Args:
        text: Raw output string from generation model.

    Returns:
        Tuple of (clean_answer, thinking_content_or_None).
    """
    pattern = r"<think>(.*?)</think>"
    match = re.search(pattern, text, flags=re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        clean = re.sub(pattern, "", text, flags=re.DOTALL).strip()
        return clean, thinking
    return text.strip(), None


class AnswerGenerator:
    """Generates grounded answers from evidence context using local LLMs."""

    def __init__(
        self,
        ollama_client: OllamaClient | None = None,
        prompt_builder: PromptBuilder | None = None,
        model: str | None = None,
        default_temperature: float | None = None,
        default_timeout: int | None = None,
    ) -> None:
        """Initialize AnswerGenerator with dependencies and configuration.

        Args:
            ollama_client: Injected OllamaClient or default instance.
            prompt_builder: Injected PromptBuilder or default instance.
            model: Target Ollama LLM model name.
            default_temperature: Default sampling temperature.
            default_timeout: Default request timeout in seconds.
        """
        cfg = get_settings()
        self.ollama_client = ollama_client or OllamaClient()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.model = model or cfg.rag_generation_model
        self.default_temperature = (
            default_temperature if default_temperature is not None else cfg.rag_temperature
        )
        self.default_timeout = (
            default_timeout if default_timeout is not None else cfg.rag_generation_timeout
        )

    def generate(
        self,
        query: str,
        evidence_items: list[EvidenceItem],
        temperature: float | None = None,
        max_tokens: int | None = None,
        require_citations: bool = True,
        timeout: float | None = None,
        think: bool | None = False,
    ) -> tuple[str, str | None, dict[str, Any]]:
        """Generate a grounded answer for a user query given evidence context.

        Args:
            query: User natural language question.
            evidence_items: List of prepared EvidenceItem context chunks.
            temperature: Inference temperature.
            max_tokens: Maximum response tokens (num_predict).
            require_citations: Whether to mandate citation markers.
            timeout: Generation timeout in seconds.
            think: Whether to allow thinking tokens for reasoning models (default False).

        Returns:
            Tuple of (clean_answer_text, thinking_text_or_None, stats_dict).

        Raises:
            OllamaError: If the underlying Ollama call fails.
        """
        system_prompt, user_prompt = self.prompt_builder.build_qa_prompt(
            query=query,
            evidence_items=evidence_items,
            require_citations=require_citations,
        )

        temp = temperature if temperature is not None else self.default_temperature
        req_timeout = timeout if timeout is not None else self.default_timeout

        options: dict[str, Any] = {
            "temperature": temp,
        }
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        prompt_est_tokens = estimate_tokens(system_prompt) + estimate_tokens(user_prompt)

        t_start = time.perf_counter()
        logger.info(
            "Generating answer with model '%s' (temp=%.2f, timeout=%ds, evidence_count=%d)",
            self.model,
            temp,
            req_timeout,
            len(evidence_items),
        )

        try:
            resp = self.ollama_client.generate(
                prompt=user_prompt,
                system=system_prompt,
                model=self.model,
                options=options,
                think=think,
                timeout=req_timeout,
            )
            raw_text = resp.response
        except OllamaError as exc:
            logger.error("Answer generation failed via Ollama: %s", exc)
            raise

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        clean_answer, thinking = strip_think_tags(raw_text)
        answer_est_tokens = estimate_tokens(clean_answer)

        stats: dict[str, Any] = {
            "model": self.model,
            "latency_ms": latency_ms,
            "prompt_tokens_estimated": prompt_est_tokens,
            "answer_tokens_estimated": answer_est_tokens,
            "eval_count": resp.eval_count,
            "prompt_eval_count": resp.prompt_eval_count,
            "total_duration_ns": resp.total_duration_ns,
        }

        return clean_answer, thinking, stats

    def generate_stream(
        self,
        query: str,
        evidence_items: list[EvidenceItem],
        temperature: float | None = None,
        max_tokens: int | None = None,
        require_citations: bool = True,
        timeout: float | None = None,
        think: bool | None = False,
    ) -> Iterator[str]:
        """Stream generated response tokens.

        Args:
            query: User natural language question.
            evidence_items: List of prepared EvidenceItem context chunks.
            temperature: Inference temperature.
            max_tokens: Maximum response tokens (num_predict).
            require_citations: Whether to mandate citation markers.
            timeout: Generation timeout in seconds.
            think: Whether to allow thinking tokens.

        Yields:
            Incremental string tokens from LLM.
        """
        system_prompt, user_prompt = self.prompt_builder.build_qa_prompt(
            query=query,
            evidence_items=evidence_items,
            require_citations=require_citations,
        )

        temp = temperature if temperature is not None else self.default_temperature
        req_timeout = timeout if timeout is not None else self.default_timeout

        options: dict[str, Any] = {
            "temperature": temp,
        }
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        yield from self.ollama_client.generate_stream(
            prompt=user_prompt,
            system=system_prompt,
            model=self.model,
            options=options,
            think=think,
            timeout=req_timeout,
        )
