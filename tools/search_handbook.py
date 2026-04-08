from chromadb import Collection

from nlp import extract_keywords
from providers import LLMProvider
from tools import Tool, ToolResult


GROUNDING_REMINDER = "\n\n---\nAnswer using ONLY the information above. If the answer is not found above, say so."
MAX_CONTEXT_CHARS = 3000


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
                "description": "2-3 search query variations to find information in the handbook. Use different phrasings for better results.",
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

    def _search(self, queries: list[str], n_results: int, where_filter: dict | None) -> list[str]:
        """Search for each query, return deduplicated docs preserving order."""
        seen: set[str] = set()
        docs: list[str] = []
        for q in queries:
            embedding = self._provider.embed(q, "search_query")
            try:
                results = self._collection.query(
                    query_embeddings=[embedding],
                    n_results=n_results,
                    where=where_filter,
                    include=["documents"],
                )
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

        # Collect keywords from all queries for metadata filtering
        all_keywords: list[str] = []
        for q in queries:
            all_keywords.extend(extract_keywords(q, use_fallback=True, include_verb=True))
        all_keywords = list(set(all_keywords))

        where_filter = None
        if all_keywords:
            or_conditions = [{f"tag_{kw}": {"$eq": True}} for kw in all_keywords]
            where_filter = {"$or": or_conditions} if len(or_conditions) > 1 else or_conditions[0]

        docs = self._search(queries, n_results, where_filter)
        used_fallback = False

        # Fall back to pure vector search if keyword filter returned nothing
        if not docs and where_filter:
            docs = self._search(queries, n_results, None)
            used_fallback = True

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

        if used_fallback:
            result_count += " [broad search — keyword filter matched nothing]"

        return ToolResult(
            content=f"{result_count}\n\n{context}{GROUNDING_REMINDER}",
            success=True,
        )
