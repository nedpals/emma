from chromadb import Collection

from nlp import extract_keywords
from providers import LLMProvider
from tools import Tool, ToolResult


GROUNDING_REMINDER = "\n\n---\nAnswer using ONLY the information above. If the answer is not found above, say so."
MAX_CONTEXT_CHARS = 3000


class SearchHandbookTool(Tool):
    name = "search_handbook"
    description = "Search the UIC student handbook for information relevant to a query. Returns matching passages from the handbook. Use a specific, targeted query for best results."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query describing what information to find in the handbook",
            },
            "n_results": {
                "type": "integer",
                "description": "Number of results to retrieve per search (default 5)",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    def __init__(self, provider: LLMProvider, collection: Collection):
        self._provider = provider
        self._collection = collection

    def _query(self, embedding: list[float], n_results: int, where_filter: dict | None) -> list[str]:
        try:
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=n_results,
                where=where_filter,
                include=["documents"],
            )
            return results.get("documents", [[]])[0]
        except Exception:
            return []

    def execute(self, **kwargs) -> ToolResult:
        query = kwargs.get("query", "")
        n_results = kwargs.get("n_results", 5)

        if not query:
            return ToolResult(content="Error: query is required", success=False)

        query_embedding = self._provider.embed(query, "search_query")

        # Try with keyword filter first for precision
        input_keywords = extract_keywords(query, use_fallback=True, include_verb=True)

        where_filter = None
        if input_keywords:
            or_conditions = [{f"tag_{kw}": {"$eq": True}} for kw in input_keywords]
            where_filter = {"$or": or_conditions} if len(or_conditions) > 1 else or_conditions[0]

        docs = self._query(query_embedding, n_results, where_filter)
        used_fallback = False

        # Fall back to pure vector search if keyword filter returned nothing
        if not docs and where_filter:
            docs = self._query(query_embedding, n_results, None)
            used_fallback = True

        if not docs:
            return ToolResult(
                content=(
                    f"No results found for query '{query}'. "
                    "Try rephrasing with different keywords or a broader search term."
                ),
                success=False,
            )

        # Cap context size to avoid overflowing the model's context window
        parts = []
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
