import time
import os

from typing import Iterable, Tuple

from requests.exceptions import ConnectionError

from llm import get_embedding
from extractor import extract_content
from vector_store import load_vector_store
from nlp import extract_keywords, nlp

def embed_documents(docs: Iterable[Tuple[int, str, str]], max_docs_per_request: int = 2, extra_tags: list[str] = None):
    vector_store = load_vector_store()
    
    for i, doc in enumerate(docs):
        while True:
            try:
                print(f"Embedding split {i+1}")
                
                page_number, context_meta, content = doc
                content_keywords = extract_keywords(content)

                # Additional: extract keywords from context and
                context_keywords = extract_keywords(context_meta)
                content_keywords = list(set(content_keywords + context_keywords))

                # Add extra tags if provided
                if extra_tags:
                    content_keywords.extend(extra_tags)

                content_keywords = list(set(content_keywords))  # Remove duplicates

                metadata = {
                    "page_number": page_number,
                    "context": context_meta,
                    # Add keyword tags
                    **{f"tag_{keyword}": True for keyword in content_keywords}
                }
                print(f"  - Extracted keywords: {content_keywords[:10]}...")

                embedding = get_embedding(content, 'search_document')
                vector_store.add(
                    embeddings=[embedding],
                    documents=[content],
                    metadatas=[metadata],
                    ids=[f"doc_{i}"]
                )

                if i % max_docs_per_request == 0:
                    time.sleep(1)

                break
            except (KeyError, ConnectionError):
                print("Error occurred, retrying in 5 seconds")
                time.sleep(5)
                continue

if __name__ == "__main__":
    if nlp is None:
        print("Exiting due to missing spaCy model.")
    else:
        max_docs_per_request = int(os.environ.get("MAX_EMBED_COUNT", "2"))
        print("Extracting content...")
        docs_iterable = extract_content() # Assuming this returns an iterable/generator

        print("Starting embedding process...")
        embed_documents(docs_iterable, max_docs_per_request, extra_tags=["uic", "school", "university", "university of immaculate conception"])
        print("Embedding process finished.")
