from chromadb import Collection

from nlp import extract_keywords
from providers import LLMProvider
from tools import Tool, ToolResult


GROUNDING_REMINDER = "\n\n---\nAnswer using ONLY the information above. If the answer is not found above, say so."


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
                "description": "Number of results to retrieve per search (default 10)",
                "default": 10,
            },
        },
        "required": ["query"],
    }

    def __init__(self, provider: LLMProvider, collection: Collection):
        self._provider = provider
        self._collection = collection

    def execute(self, **kwargs) -> ToolResult:
        query = kwargs.get("query", "")
        n_results = kwargs.get("n_results", 10)

        if not query:
            return ToolResult(content="Error: query is required", success=False)

        # 1. Embed the query
        query_embedding = self._provider.embed(query, "search_query")

        # 2. Extract keywords for metadata filtering
        input_keywords = extract_keywords(query, use_fallback=True, include_verb=True)

        where_filter = None
        if input_keywords:
            or_conditions = [{f"tag_{kw}": {"$eq": True}} for kw in input_keywords]
            where_filter = {"$or": or_conditions} if len(or_conditions) > 1 else or_conditions[0]

        # 3. Query vector store
        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter,
                include=["documents"],
            )
            docs = results.get("documents", [[]])[0]
        except Exception:
            docs = []

        if not docs:
            return ToolResult(
                content="No relevant information found in the handbook." + GROUNDING_REMINDER,
                success=True,
            )

        context = "\n\n".join(docs)
        return ToolResult(content=context + GROUNDING_REMINDER, success=True)
