import numpy as np
from chromadb import Collection

from nlp import extract_keywords
from providers import LLMProvider
from tools import Tool, ToolResult


GROUNDING_REMINDER = "\n\n---\nAnswer using ONLY the information above. If the answer is not found above, say so."


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    a = np.array(vec1)
    b = np.array(vec2)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class SearchHandbookTool(Tool):
    name = "search_handbook"
    description = "Search the UIC student handbook for information relevant to a query. Returns matching passages from the handbook."
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

        # 1. Generate alternative queries
        alt_prompt = [
            {
                "role": "user",
                "content": (
                    "You are an expert query generator and translator. Rephrase the following "
                    "question in 2-3 different ways to improve search results in a vector database "
                    "containing a school handbook. If the question is not in English, provide the "
                    "English translation first. Respond ONLY with the alternative queries, each on "
                    f"a new line.\n\nQuestion: '{query}'"
                ),
            }
        ]
        alt_queries_str = self._provider.generate(alt_prompt, temperature=0.2)
        alt_queries = [q.strip() for q in alt_queries_str.split("\n") if q.strip()]

        # 2. Get embeddings for all queries
        all_queries = [query] + alt_queries
        all_embeddings = [self._provider.embed(q, "search_query") for q in all_queries]

        # 3. Extract keywords for metadata filtering
        input_keywords = extract_keywords(query, use_fallback=True, include_verb=True)
        for alt_q in alt_queries:
            input_keywords.extend(extract_keywords(alt_q, use_fallback=True, include_verb=True))
        input_keywords = list(set(input_keywords))

        # 4. Build metadata filter
        where_filter = None
        if input_keywords:
            or_conditions = [{f"tag_{kw}": {"$eq": True}} for kw in input_keywords]
            where_filter = {"$or": or_conditions} if len(or_conditions) > 1 else or_conditions[0]

        # 5. Query vector store for each embedding
        all_docs: dict[str, list[float]] = {}
        for emb in all_embeddings:
            try:
                results = self._collection.query(
                    query_embeddings=[emb],
                    n_results=n_results,
                    where=where_filter,
                    include=["documents"],
                )
                docs = results.get("documents", [[]])[0]
                for doc in docs:
                    if doc not in all_docs:
                        all_docs[doc] = emb
            except Exception:
                continue

        if not all_docs:
            return ToolResult(
                content="No relevant information found in the handbook." + GROUNDING_REMINDER,
                success=True,
            )

        # 6. Re-rank by cosine similarity to original query
        query_embedding = all_embeddings[0]
        scored = [
            (doc, _cosine_similarity(query_embedding, emb))
            for doc, emb in all_docs.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        context = "\n\n".join(doc for doc, _ in scored)
        return ToolResult(content=context + GROUNDING_REMINDER, success=True)
