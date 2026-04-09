import logging

from chromadb import Collection

from nlp import extract_keywords
from providers import LLMProvider
from tools import Tool, ToolResult

log = logging.getLogger(__name__)


CONFIDENCE_THRESHOLD = 1.0
HIGH_CONFIDENCE_REMINDER = "\n\n---\nBase your answer on the information above. You may reason about or build upon this content, but do not invent facts not found here."
LOW_CONFIDENCE_REMINDER = "\n\n---\nThese results may not be relevant to the question. If none are relevant, say you don't have that information rather than guessing."
MAX_CONTEXT_CHARS = 6000


class SearchHandbookTool(Tool):
    name = "search_handbook"
    description = (
        "Search the UIC student handbook. You MUST provide at least 3 different query variations "
        "for better coverage. Include synonyms, different phrasings, and both broad and specific terms."
    )
    parameters = {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "description": "At least 3 search query variations (required). Include: the original phrasing, a synonym/rephrase, and a broader or more specific version.",
            },
            "n_results": {
                "type": "integer",
                "description": "Number of results to retrieve per query (default 3)",
                "default": 3,
            },
        },
        "required": ["queries"],
    }

    def __init__(self, provider: LLMProvider, collection: Collection):
        self._provider = provider
        self._collection = collection

    def _search(self, queries: list[str], n_results: int, where_filter: dict | None = None) -> list[tuple[str, float]]:
        """Search for each query, return deduplicated (doc, distance) pairs preserving order."""
        seen: set[str] = set()
        docs: list[tuple[str, float]] = []
        for q in queries:
            embedding = self._provider.embed(q, "search_query")
            try:
                params = {
                    "query_embeddings": [embedding],
                    "n_results": n_results,
                    "include": ["documents", "distances"],
                }
                if where_filter:
                    params["where"] = where_filter
                results = self._collection.query(**params)
                result_docs = results.get("documents", [[]])[0]
                result_distances = results.get("distances", [[]])[0]
                for doc, dist in zip(result_docs, result_distances):
                    if doc not in seen:
                        seen.add(doc)
                        docs.append((doc, dist))
            except Exception:
                continue
        return docs

    def execute(self, **kwargs) -> ToolResult:
        queries = kwargs.get("queries", [])
        n_results = kwargs.get("n_results", 3)

        # Support single string for model flexibility
        if isinstance(queries, str):
            queries = [queries]

        if not queries:
            return ToolResult(content="Error: queries is required", success=False)

        log.info(f"Search queries: {queries} (n_results={n_results})")

        # Primary: pure vector search
        docs = self._search(queries, n_results)

        log.info(f"Vector search returned {len(docs)} docs")
        for i, (doc, dist) in enumerate(docs):
            log.debug(f"  [{i}] dist={dist:.3f} ({len(doc)} chars) {doc[:80]}...")

        # Last resort: keyword filter (may surface tagged docs that vector search missed)
        if not docs:
            all_keywords: list[str] = []
            for q in queries:
                all_keywords.extend(extract_keywords(q, use_fallback=True, include_verb=True))
            all_keywords = list(set(all_keywords))

            log.info(f"Keyword fallback with: {all_keywords}")
            if all_keywords:
                or_conditions = [{f"tag_{kw}": {"$eq": True}} for kw in all_keywords]
                where_filter = {"$or": or_conditions} if len(or_conditions) > 1 else or_conditions[0]
                docs = self._search(queries, n_results, where_filter)
                log.info(f"Keyword fallback returned {len(docs)} docs")

        if not docs:
            return ToolResult(
                content=(
                    f"No results found for queries: {queries}. "
                    "Try rephrasing with different keywords or a broader search term."
                ),
                success=False,
            )

        # Compute average distance to determine confidence
        avg_distance = sum(dist for _, dist in docs) / len(docs)
        high_confidence = avg_distance < CONFIDENCE_THRESHOLD
        confidence_label = "high confidence" if high_confidence else "low confidence"
        reminder = HIGH_CONFIDENCE_REMINDER if high_confidence else LOW_CONFIDENCE_REMINDER

        log.info(f"Avg distance={avg_distance:.3f}, confidence={confidence_label}")

        # Cap total context size
        parts: list[str] = []
        total = 0
        for doc, _ in docs:
            if total + len(doc) > MAX_CONTEXT_CHARS:
                break
            parts.append(doc)
            total += len(doc)

        context = "\n\n".join(parts)
        result_count = f"({len(parts)} of {len(docs)} results shown, {confidence_label})"

        log.info(f"Returning {len(parts)} of {len(docs)} docs ({total} chars)")

        return ToolResult(
            content=f"{result_count}\n\n{context}{reminder}",
            success=True,
        )
