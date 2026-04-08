import logging

from chromadb import Collection

from nlp import extract_keywords
from providers import LLMProvider
from tools import Tool, ToolResult

log = logging.getLogger(__name__)


GROUNDING_REMINDER = "\n\n---\nAnswer using ONLY the information above. If the answer is not found above, say so."
MAX_CONTEXT_CHARS = 6000


class SearchHandbookTool(Tool):
    name = "search_handbook"
    description = (
        "Search the UIC student handbook. Provide multiple query variations "
        "for better coverage (e.g. synonyms, different phrasings)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": "3-5 search query variations to find information in the handbook. Use different phrasings, synonyms, and specificity levels for better results.",
            },
            "n_results": {
                "type": "integer",
                "description": "Number of results to retrieve per query (default 5)",
                "default": 5,
            },
        },
        "required": ["queries"],
    }

    def __init__(self, provider: LLMProvider, collection: Collection):
        self._provider = provider
        self._collection = collection

    def _search(self, queries: list[str], n_results: int, where_filter: dict | None = None) -> list[str]:
        """Search for each query, return deduplicated docs preserving order."""
        seen: set[str] = set()
        docs: list[str] = []
        for q in queries:
            embedding = self._provider.embed(q, "search_query")
            try:
                params = {
                    "query_embeddings": [embedding],
                    "n_results": n_results,
                    "include": ["documents"],
                }
                if where_filter:
                    params["where"] = where_filter
                results = self._collection.query(**params)
                for doc in results.get("documents", [[]])[0]:
                    if doc not in seen:
                        seen.add(doc)
                        docs.append(doc)
            except Exception:
                continue
        return docs

    def execute(self, **kwargs) -> ToolResult:
        queries = kwargs.get("queries", [])
        n_results = kwargs.get("n_results", 5)

        # Support single string for model flexibility
        if isinstance(queries, str):
            queries = [queries]

        if not queries:
            return ToolResult(content="Error: queries is required", success=False)

        log.info(f"Search queries: {queries} (n_results={n_results})")

        # Primary: pure vector search
        docs = self._search(queries, n_results)

        log.info(f"Vector search returned {len(docs)} docs")
        for i, doc in enumerate(docs):
            log.debug(f"  [{i}] ({len(doc)} chars) {doc[:80]}...")

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

        # Cap context size to avoid overflowing the model's context window
        parts: list[str] = []
        total = 0
        for doc in docs:
            if total + len(doc) > MAX_CONTEXT_CHARS:
                break
            parts.append(doc)
            total += len(doc)

        context = "\n\n".join(parts)
        result_count = f"({len(parts)} of {len(docs)} results shown)"

        log.info(f"Returning {len(parts)} of {len(docs)} docs ({total} chars)")

        return ToolResult(
            content=f"{result_count}\n\n{context}{GROUNDING_REMINDER}",
            success=True,
        )
