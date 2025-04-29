import numpy as np

import meta

from chromadb import Collection

from llm import generate_response, get_embedding
from models import Message, UserMessage, SystemMessage
from nlp import extract_keywords
    
def cosine_similarity(vec1, vec2):
    """Calculates cosine similarity between two vectors."""
    # Ensure vectors are numpy arrays
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    # Handle potential zero vectors
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

# this prompt will be used to generate a search query to look up in the document
def load_search_prompt(input: str, chat_history: list[Message] = []) -> list[Message]:
    """
    Load the prompt messages for generating a search query.
    
    Args:
        input: The user's current question
        chat_history: Previous conversation messages
    
    Returns:
        List of messages to generate an effective search query
    """
    return [
        SystemMessage("You are a search query optimizer. Your task is to extract key terms and concepts from the conversation to create effective search queries. Focus on specific terms, names, and important concepts."),
        *chat_history,
        UserMessage(f"For the question: '{input}', extract 2-4 key terms or phrases that would be most relevant for searching the school handbook. Focus on specific nouns, policies, or procedures mentioned. Respond with just the search terms, no explanation needed.")
    ]

# this will be the base prompt for the chatbot to ask the user a question both for history aware and non-history aware
# Define general tone and structure, incorporate specifics from meta
base_system_prompt_template = f"""
{meta.full_description}

**Your Tone:** Be helpful, informative, and maintain a slightly formal, respectful tone appropriate for a university setting. Be concise and clear in your answers.

**General Instructions:**
- Answer questions based *only* on the provided context below.
- Do not use introductory phrases like "Based on the context..." or "The context states...". Answer directly.
- If the answer is not found in the context, state clearly: "I don't have information on that specific topic based on the handbook." Do not invent answers or provide information outside the handbook.
- {meta.additional_prompt}

Context:
---
{{context}}
---
""".strip()

def load_alternative_query_prompt(input: str) -> list[Message]:
    """
    Generates a prompt to ask the LLM for alternative phrasings of the user's query.
    """
    return [
        SystemMessage("You are an expert query generator. Your goal is to rephrase the user's question in 2-3 different ways to improve search results in a vector database containing a school handbook. Focus on using synonyms, different sentence structures, or breaking down the question if complex. Respond ONLY with the alternative queries, each on a new line. Do not include the original query."),
        UserMessage(f"Generate alternative search queries for: '{input}'")
    ]

# this will be the initial prompt for the chatbot to ask the user a question
def load_prompt(input: str, context: str, chat_history: list[Message] | None = None) -> list[Message]:
    """
    Load the prompt messages for the chatbot's response generation.
    Returns a list of dictionaries for the LLM.
    """
    # Format the template ONLY with the actual context now
    formatted_system_prompt = base_system_prompt_template.format(context=context)

    # If non-history aware, structure prompt differently
    if chat_history is None:
        return [
            SystemMessage(formatted_system_prompt),
            UserMessage(f"Question: {input}"),
        ]
    return [
        SystemMessage(formatted_system_prompt),
        *chat_history,
        UserMessage(input),
    ]

class RetrievalChain:
    def __init__(self, vector_collection: Collection, history_aware: bool = True):
        self.vector_collection = vector_collection
        self.history_aware = history_aware

    def invoke(self, inputs: dict, config = {}) -> dict:
        """
        Invoke the retrieval chain with the provided inputs.
        """
        chat_history = inputs.get("chat_history", [])
        input_text = inputs.get("input", "")
        n_results = inputs.get("n_results", 10)

        if not self.history_aware:
            # If history is not aware, clear the chat history
            chat_history = None

        # 1. Generate alternative queries if the input is too complex
        alternative_query_prompt = load_alternative_query_prompt(input_text)
        alternative_queries_str = generate_response(alternative_query_prompt, temperature=0.2)
        alternative_queries = [q.strip() for q in alternative_queries_str.split('\n') if q.strip()]
        print(f"Generated alternative queries: {alternative_queries}")

        # 1.2. Get embedding for the search query
        query_embedding = get_embedding(input_text, 'search_query')
        all_queries = [input_text] + alternative_queries
        all_query_embeddings = [query_embedding] + [get_embedding(q, 'search_query') for q in alternative_queries]

        # 2. Extract keywords from the USER INPUT for filtering
        # Using spaCy for potentially better keyword extraction from the query
        input_keywords = extract_keywords(input_text, use_fallback=True, include_verb=True)

        print(f"Extracted keywords from input for filtering: {input_keywords}")

        # 3. Build a dynamic 'where' filter using the '$in' operator
        where_filter = None
        if input_keywords:
            # Create an $or condition: match if *any* tag corresponding to an input keyword exists and is True.
            # We check for equality ($eq) with True for each tag field.
            or_conditions = [{f"tag_{keyword}": {"$eq": True}} for keyword in input_keywords]
            where_filter = {"$or": or_conditions} if len(or_conditions) > 1 else or_conditions[0]

            print(f"Applying dynamic metadata filter: {where_filter}")
        else:
            print("No keywords extracted from input for filtering.")

        # 4. Retrieve context from the vector store
        all_retrieved_docs_with_embeddings = {} # Use dict for easy deduplication based on content

        for i, query_embedding in enumerate(all_query_embeddings):
            current_query = all_queries[i]
            print(f"Querying vector store with: '{current_query}'")
            try:
                results = self.vector_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results, # Retrieve n_results for EACH query
                    where=where_filter, # Filter disabled for now
                    include=['documents']
                )
                docs = results.get('documents', [[]])[0]
                print(f"  -> Retrieved {len(docs)} docs for this query.")
                for doc, emb in zip(docs, all_query_embeddings):
                    if doc not in all_retrieved_docs_with_embeddings:
                        all_retrieved_docs_with_embeddings[doc] = emb
            except Exception as e:
                print(f"Error querying for '{current_query}': {e}")

        unique_docs = list(all_retrieved_docs_with_embeddings.keys())
        print(f"Retrieved {len(unique_docs)} unique documents total before re-ranking.")

        if unique_docs:
            # Calculate similarity scores against the ORIGINAL query embedding
            scores = [
                cosine_similarity(query_embedding, all_retrieved_docs_with_embeddings[doc])
                for doc in unique_docs
            ]

            # Sort documents based on scores (descending)
            scored_docs = sorted(zip(unique_docs, scores), key=lambda item: item[1], reverse=True)

            # Extract sorted documents
            final_context_docs = [doc for doc, score in scored_docs]
            print(f"Re-ranked documents. Top score: {scores[scored_docs.index(scored_docs[0])]:.4f}" if scored_docs else "N/A")
        else:
            final_context_docs = []

        print(f"Retrieved {len(final_context_docs)} unique documents total.")
        print(f"Final Context documents: {final_context_docs}")
        context = "\n\n".join(final_context_docs) if final_context_docs else "No relevant context found."

        # 3. Generate final response
        final_prompt_messages = load_prompt(input_text, context, chat_history)
        return {
            "answer": generate_response(final_prompt_messages, temperature=0.5)
        }

