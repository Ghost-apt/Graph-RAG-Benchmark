"""
Query Entity Extraction Module.
Extracts named entities, technologies, and concepts from user queries.
Prioritizes fast, deterministic graph-index substring matching before falling back to LLM inference.
"""

import json
import re
import time
import unicodedata

from graphrag_bench.config import Settings, get_settings
from graphrag_bench.graph_retrieval_schemas import (
    ExtractionMethod,
    QueryEntity,
    QueryEntityExtractionResult,
)
from graphrag_bench.graph_store import GraphStore
from graphrag_bench.logger import get_logger
from graphrag_bench.ollama_client import OllamaClient, OllamaError

logger = get_logger("query_entity_extractor")

QUERY_EXTRACTION_SYSTEM_PROMPT = """You are a technical entity extraction engine.
Extract the key technical concepts, systems, models, algorithms, or named entities from the user's search query.

CRITICAL RULES:
1. Extract ONLY entities that are explicitly mentioned or referenced in the query.
2. Do NOT invent concepts not implied by the query.
3. Return output strictly in valid JSON format:
{
  "entities": [
    {
      "mention_text": "<exact words from query>",
      "canonical_name": "<standard name>",
      "entity_type": "<CONCEPT, TECHNOLOGY, SOFTWARE, FRAMEWORK, METHOD, OTHER>"
    }
  ]
}
"""


def normalize_term(text: str) -> str:
    """Normalize string using NFKC, lowercase folding, and whitespace collapsing."""
    if not text:
        return ""
    folded = unicodedata.normalize("NFKC", text).lower().strip()
    return re.sub(r"\s+", " ", folded)


class QueryEntityExtractor:
    """Extracts entities from user queries using graph index shortcuts and bounded LLM fallback."""

    def __init__(
        self,
        store: GraphStore | None = None,
        client: OllamaClient | None = None,
        settings: Settings | None = None,
        model: str | None = None,
        timeout: int = 30,
    ) -> None:
        """Initialize query entity extractor.

        Args:
            store: Optional GraphStore instance containing known graph entities for fast matching.
            client: Optional OllamaClient instance.
            settings: Optional application settings.
            model: Optional model override (defaults to settings.ollama_llm_model).
            timeout: Maximum LLM call timeout in seconds for query extraction.
        """
        self.settings = settings or get_settings()
        self.store = store
        self.client = client
        self.model = model or getattr(self.settings, "ollama_llm_model", "qwen3:8b")
        self.timeout = timeout
        self.last_extraction_result: QueryEntityExtractionResult | None = None

    def extract_query_entities(
        self,
        query: str,
        force_llm: bool = False,
    ) -> QueryEntityExtractionResult:
        """Extract entities from the user query.

        Args:
            query: User's natural language question.
            force_llm: If True, bypasses exact dictionary matching and calls LLM.

        Returns:
            QueryEntityExtractionResult containing extracted entities and latency metadata.
        """
        t0 = time.perf_counter()
        clean_query = query.strip() if query else ""
        if not clean_query:
            res = QueryEntityExtractionResult(
                query=query,
                entities=[],
                duration_ms=0.0,
                llm_invoked=False,
            )
            self.last_extraction_result = res
            return res

        # 1. Deterministic Fast-Path: Match against known entities in GraphStore
        if self.store and not force_llm:
            deterministic_matches = self._match_deterministic(clean_query)
            if deterministic_matches:
                dt_ms = (time.perf_counter() - t0) * 1000.0
                logger.debug(
                    "Deterministic match found %d entities for query '%s' in %.2fms",
                    len(deterministic_matches),
                    clean_query,
                    dt_ms,
                )
                res = QueryEntityExtractionResult(
                    query=clean_query,
                    entities=deterministic_matches,
                    duration_ms=round(dt_ms, 2),
                    llm_invoked=False,
                )
                self.last_extraction_result = res
                return res

        # 2. LLM Fallback: Invoke Qwen for complex or paraphrased queries
        entities, llm_called = self._extract_with_llm(clean_query)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        res = QueryEntityExtractionResult(
            query=clean_query,
            entities=entities,
            duration_ms=round(dt_ms, 2),
            llm_invoked=llm_called,
        )
        self.last_extraction_result = res
        return res

    def _match_deterministic(self, query: str) -> list[QueryEntity]:
        """Scan query text against canonical names and aliases in the knowledge graph."""
        if not self.store:
            return []

        all_entities = self.store.get_entities()
        query_norm = normalize_term(query)
        matches: list[QueryEntity] = []
        seen_canonicals: set[str] = set()

        # Build list of (surface_term, canonical_name, entity_type, is_alias)
        # Sort by length descending to match longest phrases first (e.g. 'Vector RAG' before 'RAG')
        candidates: list[tuple[str, str, str, bool]] = []
        for ent in all_entities:
            c_name = ent.canonical_name
            candidates.append((c_name, c_name, ent.entity_type, False))
            for alias in ent.aliases:
                if alias.strip():
                    candidates.append((alias.strip(), c_name, ent.entity_type, True))

        candidates.sort(key=lambda x: len(x[0]), reverse=True)

        for surface_term, canonical_name, entity_type, is_alias in candidates:
            if canonical_name in seen_canonicals:
                continue

            term_norm = normalize_term(surface_term)
            if not term_norm or len(term_norm) < 2:
                continue

            # Word boundary regex search to prevent false partial substring matches
            # e.g. 'rag' should match 'what is rag?' but not 'courage'
            pattern = r"(?<!\w)" + re.escape(term_norm) + r"(?!\w)"
            match = re.search(pattern, query_norm)
            if match:
                # Find exact casing in original query
                start_idx = match.start()
                end_idx = match.end()
                mention_text = query[start_idx:end_idx] if end_idx <= len(query) else surface_term

                matches.append(
                    QueryEntity(
                        mention_text=mention_text,
                        canonical_name=canonical_name,
                        entity_type=entity_type,
                        normalized_term=term_norm,
                        extraction_method=(
                            ExtractionMethod.ALIAS_MATCH
                            if is_alias
                            else ExtractionMethod.EXACT_MATCH
                        ),
                        confidence=1.0,
                    )
                )
                seen_canonicals.add(canonical_name)

        return matches

    def _extract_with_llm(self, query: str) -> tuple[list[QueryEntity], bool]:
        """Call Ollama LLM to extract entities from query text with fallback error handling."""
        if not self.client:
            self.client = OllamaClient(settings=self.settings)

        prompt = f'Search Query: "{query}"\n\nExtract all key entities and technical concepts mentioned in the query above in JSON.'
        try:
            resp = self.client.generate(
                prompt=prompt,
                model=self.model,
                system=QUERY_EXTRACTION_SYSTEM_PROMPT,
                format="json",
                think=False,
                timeout=self.timeout,
                options={"temperature": 0.0, "num_predict": 250},
            )
            raw_text = resp.response.strip()
            return self._parse_llm_response(raw_text, query), True
        except (OllamaError, ValueError, Exception) as exc:
            logger.warning(
                "Query entity extraction LLM call failed for '%s': %s. Returning rule-based fallback.",
                query,
                exc,
            )
            # Rule-based fallback: treat non-stopword query tokens as candidate entities
            return self._heuristic_fallback(query), True

    def _parse_llm_response(self, raw_text: str, query: str) -> list[QueryEntity]:
        """Parse structured entities from LLM JSON response."""
        entities: list[QueryEntity] = []
        try:
            # Strip markdown code blocks if present
            cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned).strip()
            data = json.loads(cleaned)
            raw_entities = data.get("entities", []) if isinstance(data, dict) else []

            query_norm = normalize_term(query)
            for item in raw_entities:
                if not isinstance(item, dict):
                    continue
                mention = str(item.get("mention_text", "")).strip()
                canonical = str(item.get("canonical_name", mention)).strip()
                etype = str(item.get("entity_type", "OTHER")).strip().upper()

                if not mention and not canonical:
                    continue

                display_mention = mention or canonical
                display_canonical = canonical or display_mention
                norm_term = normalize_term(display_mention)

                # Ensure entity has some ground in query
                if norm_term in query_norm or normalize_term(display_canonical) in query_norm:
                    entities.append(
                        QueryEntity(
                            mention_text=display_mention,
                            canonical_name=display_canonical,
                            entity_type=etype or "OTHER",
                            normalized_term=norm_term,
                            extraction_method=ExtractionMethod.LLM_EXTRACTION,
                            confidence=0.85,
                        )
                    )
        except Exception as exc:
            logger.warning(
                "Failed to parse query extraction JSON: %s. Raw: %s", exc, raw_text[:100]
            )
            return self._heuristic_fallback(query)

        return entities or self._heuristic_fallback(query)

    def _heuristic_fallback(self, query: str) -> list[QueryEntity]:
        """Simple non-LLM token extraction fallback for offline or failed LLM states."""
        stop_words = {
            "what",
            "is",
            "how",
            "does",
            "why",
            "which",
            "who",
            "when",
            "where",
            "can",
            "the",
            "a",
            "an",
            "and",
            "or",
            "in",
            "on",
            "with",
            "for",
            "to",
            "of",
            "between",
            "tell",
            "me",
            "about",
            "compare",
            "explain",
            "difference",
        }
        words = re.findall(r"\b[A-Za-z0-9_-]+\b", query)
        content_words = [w for w in words if w.lower() not in stop_words and len(w) > 2]
        if not content_words:
            return []

        # Return single multi-word or first salient content term
        joined = " ".join(content_words[:3])
        return [
            QueryEntity(
                mention_text=joined,
                canonical_name=joined.title(),
                entity_type="OTHER",
                normalized_term=normalize_term(joined),
                extraction_method=ExtractionMethod.NORMALIZED_MATCH,
                confidence=0.5,
            )
        ]
