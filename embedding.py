import time
import os

from typing import Iterable, Tuple

from requests.exceptions import ConnectionError

from llm import provider
from extractor import extract_content
from vector_store import load_vector_store
from nlp import extract_keywords, nlp

MAX_CHUNK_CHARS = 1200


def split_chunk(text: str) -> list[str]:
    """Split text into pieces at paragraph boundaries, each under MAX_CHUNK_CHARS."""
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        # If adding this paragraph would exceed the limit, flush current group
        if current and current_len + para_len + 2 > MAX_CHUNK_CHARS:  # +2 for \n\n
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0

        current.append(para)
        current_len += para_len + 2  # +2 for the \n\n separator

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def embed_documents(docs: Iterable[Tuple[int, str, str]], max_docs_per_request: int = 2, extra_tags: list[str] = None):
    vector_store = load_vector_store()
    doc_id = 0

    for i, doc in enumerate(docs):
        page_number, context_meta, content = doc

        # Prepend context to content for better embeddings
        enriched_content = f"{context_meta}\n\n{content}" if context_meta else content

        # Split oversized chunks
        chunks = split_chunk(enriched_content)

        for chunk in chunks:
            while True:
                try:
                    print(f"Embedding doc {doc_id + 1} (from segment {i + 1}, {len(chunk)} chars)")

                    content_keywords = extract_keywords(chunk)
                    context_keywords = extract_keywords(context_meta)
                    all_keywords = list(set(content_keywords + context_keywords))

                    if extra_tags:
                        all_keywords.extend(extra_tags)
                    all_keywords = list(set(all_keywords))

                    metadata = {
                        "page_number": page_number,
                        "context": context_meta,
                        **{f"tag_{keyword}": True for keyword in all_keywords}
                    }

                    embedding = provider.embed(chunk, 'search_document')
                    vector_store.add(
                        embeddings=[embedding],
                        documents=[chunk],
                        metadatas=[metadata],
                        ids=[f"doc_{doc_id}"]
                    )

                    doc_id += 1

                    if doc_id % max_docs_per_request == 0:
                        time.sleep(1)

                    break
                except (KeyError, ConnectionError):
                    print("Error occurred, retrying in 5 seconds")
                    time.sleep(5)
                    continue

    print(f"Embedded {doc_id} documents from {i + 1} segments.")


if __name__ == "__main__":
    if nlp is None:
        print("Exiting due to missing spaCy model.")
    else:
        max_docs_per_request = int(os.environ.get("MAX_EMBED_COUNT", "2"))
        print("Extracting content...")
        docs_iterable = extract_content()

        print("Starting embedding process...")
        embed_documents(docs_iterable, max_docs_per_request, extra_tags=["uic", "school", "university", "university of immaculate conception"])
        print("Embedding process finished.")
