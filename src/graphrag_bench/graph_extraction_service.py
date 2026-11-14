"""Structured LLM extraction service for Knowledge Graph Construction.

Utilizes local Qwen3:8b via OllamaClient to extract grounded entities and relationships from
document chunks with:
- Strict evidence quote validation against original chunk text
- Entity mention verification in chunk context
- Pydantic schema validation and JSON fence cleaning
- Rejection of hallucinated entities and unsupported triples
- Granular latency profiling and structured error reporting
"""

import json
import re
import time
import unicodedata
from typing import Any

from graphrag_bench.config import Settings, get_settings
from graphrag_bench.graph_schemas import (
    ChunkGraphExtraction,
    ExtractedEntity,
    ExtractedRelationship,
)
from graphrag_bench.logger import get_logger
from graphrag_bench.ollama_client import (
    OllamaAPIError,
    OllamaClient,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
)
from graphrag_bench.schemas import DocumentChunk

logger = get_logger("graph_extraction_service")


class ExtractionError(Exception):
    """Base exception for graph extraction failures."""


EXTRACTION_SYSTEM_PROMPT = """You are a precise, factual information extraction engine for knowledge graph construction.
Your task is to extract entities and directional relationships ONLY from the supplied source text.

CRITICAL RULES:
1. Extract entities and relationships ONLY from the supplied text. Do NOT introduce external knowledge.
2. Do NOT invent entities or relationships. Do NOT infer relations not explicitly stated.
3. Every relationship MUST include an exact verbatim quotation from the source text as 'evidence_quote'.
4. Allowed entity types: PERSON, ORGANIZATION, CONCEPT, TECHNOLOGY, PROGRAMMING_LANGUAGE, FRAMEWORK, SOFTWARE, DATASET, METHOD, LOCATION, OTHER.
5. If no entities or relationships are present in the text, return empty arrays.
6. Output MUST be valid JSON strictly matching the requested format.
"""

EXTRACTION_USER_PROMPT_TEMPLATE = """Source Text to analyze:
\"\"\"
{chunk_text}
\"\"\"

Extract all key entities and factual relationships asserted in the text above.

Return a JSON object with this EXACT structure:
{{
  "entities": [
    {{
      "mention_text": "<exact mention in text>",
      "canonical_name": "<standardized entity name>",
      "entity_type": "<one of PERSON, ORGANIZATION, CONCEPT, TECHNOLOGY, PROGRAMMING_LANGUAGE, FRAMEWORK, SOFTWARE, DATASET, METHOD, LOCATION, OTHER>",
      "description": "<short description strictly grounded in the text>",
      "aliases": ["<alias1>", "<alias2>"]
    }}
  ],
  "relationships": [
    {{
      "subject": "<mention or canonical name of source entity>",
      "predicate": "<standardized relationship phrase e.g. EVALUATES, USES, INTEGRATES_WITH, DEVELOPED_BY, PART_OF>",
      "object": "<mention or canonical name of target entity>",
      "evidence_quote": "<exact verbatim quote from source text supporting this relationship>",
      "confidence": 1.0
    }}
  ]
}}
"""


class GraphExtractionService:
    """Service for extracting grounded knowledge graph elements from DocumentChunks."""

    def __init__(
        self,
        client: OllamaClient | None = None,
        ollama_client: OllamaClient | None = None,
        settings: Settings | None = None,
        model: str | None = None,
        model_name: str | None = None,
        min_confidence: float = 0.50,
        timeout: int | None = None,
    ) -> None:
        """Initialize extraction service.

        Args:
            client: Optional pre-configured OllamaClient.
            ollama_client: Alias for client.
            settings: Optional application settings.
            model: Extraction model name override (defaults to settings.ollama_llm_model).
            model_name: Alias for model.
            min_confidence: Minimum confidence threshold for relationships (0.0 to 1.0).
            timeout: LLM extraction timeout in seconds (defaults to settings.graph_extraction_timeout or 600s).
        """
        self.settings = settings or get_settings()
        self.client = client or ollama_client or OllamaClient(settings=self.settings)
        chosen_model = (
            model
            or model_name
            or getattr(self.settings, "graph_extraction_model", self.settings.ollama_llm_model)
        )
        self.model = chosen_model
        self.model_name = chosen_model
        self.min_confidence = min_confidence
        self.timeout = (
            timeout
            if timeout is not None
            else getattr(self.settings, "graph_extraction_timeout", 600)
        )

    def extract_from_chunk(
        self,
        chunk: DocumentChunk | str | None = None,
        chunk_id: str | None = None,
        chunk_text: str | None = None,
        temperature: float = 0.0,
    ) -> ChunkGraphExtraction:
        """Extract validated entities and grounded relationships from a DocumentChunk or text.

        Args:
            chunk: Phase 2 DocumentChunk instance or raw text string.
            chunk_id: Optional chunk identifier when passing raw text string.
            chunk_text: Optional explicit chunk text override.
            temperature: LLM sampling temperature (default 0.0 for deterministic extraction).

        Returns:
            ChunkGraphExtraction with validated entities and relationships.

        Raises:
            ExtractionError: If model fails or output cannot be parsed.
        """
        if isinstance(chunk, str):
            resolved_text = (chunk_text or chunk).strip()
            resolved_chunk_id = chunk_id or ""
            word_count = len(resolved_text.split())
        elif isinstance(chunk, DocumentChunk):
            resolved_text = (chunk_text or chunk.text).strip()
            resolved_chunk_id = chunk_id or chunk.chunk_id
            word_count = chunk.word_count
        else:
            resolved_text = (chunk_text or "").strip()
            resolved_chunk_id = chunk_id or ""
            word_count = len(resolved_text.split())

        if not resolved_text:
            return ChunkGraphExtraction(source_chunk_id=resolved_chunk_id)

        prompt = EXTRACTION_USER_PROMPT_TEMPLATE.format(chunk_text=resolved_text)
        options: dict[str, Any] = {"temperature": temperature}

        start_time = time.perf_counter()
        logger.debug(
            "Extracting knowledge graph triples from chunk '%s' (len: %d words, model: %s)",
            resolved_chunk_id,
            word_count,
            self.model,
        )

        try:
            resp = self.client.generate(
                prompt=prompt,
                model=self.model,
                system=EXTRACTION_SYSTEM_PROMPT,
                options=options,
                format="json",
                think=False,
                timeout=self.timeout,
            )
        except (
            OllamaConnectionError,
            OllamaTimeoutError,
            OllamaModelNotFoundError,
            OllamaAPIError,
        ) as exc:
            logger.error("Ollama extraction failure for chunk '%s': %s", resolved_chunk_id, exc)
            raise ExtractionError(
                f"LLM extraction error for chunk '{resolved_chunk_id}': {exc}"
            ) from exc
        except Exception as exc:
            logger.error("Unexpected generation error on chunk '%s': %s", resolved_chunk_id, exc)
            raise ExtractionError(f"Unexpected extraction failure: {exc}") from exc

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        if hasattr(resp, "response"):
            raw_text = resp.response.strip()
        elif isinstance(resp, dict) and "response" in resp:
            raw_text = str(resp["response"]).strip()
        else:
            raw_text = str(resp).strip()

        # Parse JSON from response
        parsed_data = self._parse_json_payload(raw_text, resolved_chunk_id)

        # Validate with Pydantic
        raw_entities = parsed_data.get("entities", [])
        raw_relationships = parsed_data.get("relationships", [])

        # Step 7: Filter and validate against source chunk ground truth
        valid_entities, valid_relationships = self._validate_and_ground_extractions(
            raw_entities=raw_entities,
            raw_relationships=raw_relationships,
            chunk_text=resolved_text,
            chunk_id=resolved_chunk_id,
        )

        return ChunkGraphExtraction(
            entities=valid_entities,
            relationships=valid_relationships,
            source_chunk_id=resolved_chunk_id,
            raw_response=raw_text,
            extraction_duration_ms=round(duration_ms, 2),
        )

    def _parse_json_payload(self, raw_text: str, chunk_id: str) -> dict[str, Any]:
        """Extract and parse JSON object from LLM response text."""
        # Strip markdown code fences if model enclosed JSON in ```json ... ```
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except Exception:  # noqa: S110
            pass

        # Fallback: attempt regex extraction of outermost JSON object {...}
        match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
                if isinstance(parsed, dict):
                    return parsed
            except Exception:  # noqa: S110
                pass

        logger.warning(
            "Could not parse valid JSON from LLM extraction for chunk '%s'. Returning empty extractions. Raw snippet: %s",
            chunk_id,
            raw_text[:120],
        )
        return {"entities": [], "relationships": []}

    def _validate_and_ground_extractions(
        self,
        raw_entities: list[Any],
        raw_relationships: list[Any],
        chunk_text: str,
        chunk_id: str,
    ) -> tuple[list[ExtractedEntity], list[ExtractedRelationship]]:
        """Validate that all mentions and evidence quotes are grounded in chunk text."""
        normalized_chunk_text = unicodedata.normalize("NFKC", chunk_text)
        chunk_lower = normalized_chunk_text.lower()
        chunk_words_collapsed = re.sub(r"\s+", " ", chunk_lower).strip()

        # 1. Validate Entities
        valid_entities: list[ExtractedEntity] = []
        entity_name_set: set[str] = set()

        for raw_e in raw_entities:
            try:
                entity = ExtractedEntity.model_validate(raw_e)
                # Verify mention text is present in the chunk
                mention_norm = re.sub(r"\s+", " ", entity.mention_text.lower()).strip()
                canonical_norm = re.sub(r"\s+", " ", entity.canonical_name.lower()).strip()

                if mention_norm in chunk_words_collapsed or canonical_norm in chunk_words_collapsed:
                    valid_entities.append(entity)
                    entity_name_set.add(entity.canonical_name.lower())
                    entity_name_set.add(entity.mention_text.lower())
                    for a in entity.aliases:
                        entity_name_set.add(a.lower())
                else:
                    logger.debug(
                        "Rejected ungrounded entity '%s' in chunk '%s'",
                        entity.canonical_name,
                        chunk_id,
                    )
            except Exception as exc:
                logger.debug("Skipping invalid entity model in chunk '%s': %s", chunk_id, exc)

        # 2. Validate Relationships
        valid_relationships: list[ExtractedRelationship] = []

        for raw_r in raw_relationships:
            try:
                rel = ExtractedRelationship.model_validate(raw_r)

                # Check minimum confidence threshold
                if rel.confidence < self.min_confidence:
                    logger.debug(
                        "Rejected relationship (%s -> %s) below confidence threshold (%.2f < %.2f)",
                        rel.subject,
                        rel.object,
                        rel.confidence,
                        self.min_confidence,
                    )
                    continue

                # Verify evidence quote is a substring of the source chunk
                quote_norm = unicodedata.normalize("NFKC", rel.evidence_quote)
                quote_collapsed = re.sub(r"\s+", " ", quote_norm.lower()).strip()
                chunk_stripped = re.sub(r"[*_#`]", "", chunk_words_collapsed)
                quote_stripped = re.sub(r"[*_#`]", "", quote_collapsed)

                is_quote_grounded = (
                    quote_collapsed in chunk_words_collapsed or quote_stripped in chunk_stripped
                )

                if not quote_collapsed or not is_quote_grounded:
                    logger.warning(
                        "Rejected hallucinated quote in chunk '%s': '%s'",
                        chunk_id,
                        rel.evidence_quote[:80],
                    )
                    continue

                # Verify subject and object mentions are supported by the chunk or entities
                sub_norm = re.sub(r"\s+", " ", rel.subject.lower()).strip()
                obj_norm = re.sub(r"\s+", " ", rel.object.lower()).strip()

                sub_in_chunk = (
                    sub_norm in chunk_words_collapsed
                    or sub_norm in chunk_stripped
                    or sub_norm in entity_name_set
                    or sub_norm in quote_collapsed
                    or sub_norm in quote_stripped
                )
                obj_in_chunk = (
                    obj_norm in chunk_words_collapsed
                    or obj_norm in chunk_stripped
                    or obj_norm in entity_name_set
                    or obj_norm in quote_collapsed
                    or obj_norm in quote_stripped
                )

                if not sub_in_chunk or not obj_in_chunk:
                    logger.warning(
                        "Rejected relationship with ungrounded arguments in chunk '%s': (%s - %s -> %s)",
                        chunk_id,
                        rel.subject,
                        rel.predicate,
                        rel.object,
                    )
                    continue

                valid_relationships.append(rel)

            except Exception as exc:
                logger.debug("Skipping invalid relationship model in chunk '%s': %s", chunk_id, exc)

        return valid_entities, valid_relationships
