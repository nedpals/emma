from typing import Literal, List, Iterable
from llm import embeddings
from extractor import extract_content_from_env
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from requests.exceptions import ConnectionError

import time
import os

local = os.environ.get("LOCAL", "1") == "1"

def load_vector_store(source: Literal["local", "supabase"] = "", documents: List[Document] = []) -> VectorStore:
    if len(source) == 0:
        source = "local" if local else "supabase"

    if source == "local":
        from langchain_community.vectorstores.chroma import Chroma

        if documents and len(documents) > 0:
            return Chroma.from_documents(
                documents,
                persist_directory="./embeddings_db",
                embedding=embeddings)

        return Chroma(
            persist_directory="./embeddings_db",
            embedding_function=embeddings)
    else:
        import supabase
        from langchain_community.vectorstores.supabase import SupabaseVectorStore

        supabaseUrl = os.environ["SUPABASE_URL"]
        supabaseKey = os.environ["SUPABASE_SERVICE_KEY"]
        supabaseClient = supabase.Client(supabaseUrl, supabaseKey)

        if documents and len(documents) > 0:
            return SupabaseVectorStore.from_documents(
                client=supabaseClient,
                table_name="documents",
                query_name="match_documents",
                embedding=embeddings,
                documents=documents)

        return SupabaseVectorStore(
            client=supabaseClient,
            table_name="documents",
            query_name="match_documents",
            embedding=embeddings)

def from_splitter(doc: Iterable[Document]):
    text_splitter = RecursiveCharacterTextSplitter()
    return text_splitter.split_documents(doc)

def initiate_embed(docs: Iterable[Document]):
    store_type = "local"
    # store_type = "local" if local else "supabase"

    for i, doc in enumerate(docs):
        while True:
            try:
                print(f"Embedding split {i+1} of {len(docs)}")

                # mistral has a limit of 16384 tokens per request so we need to split the document into chunks
                # and then embed each chunk individually and then combine the chunks into a single embedding
                # this will be a very long process so we should save the embeddings to disk and then load them later
                load_vector_store(store_type, [doc])

                if i % 2 == 0:
                    # sleep for 1 second every other split to
                    # avoid hitting the mistral rate limit
                    time.sleep(1)

                break
            except KeyError:
                print("Key error, retrying in 5 seconds")
                time.sleep(5)
                continue
            except ConnectionError:
                print("Connection error, retrying in 5 seconds")
                time.sleep(5)
                continue

if __name__ == "__main__":
    initiate_embed(
        extract_content_from_env() if os.environ.get("EXTRACTOR", "") == "llmsherpa" else from_splitter(extract_content_from_env())
    )
