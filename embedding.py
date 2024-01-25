from llm import embeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time

def load_embeddings():
    return Chroma(persist_directory="./embeddings_db", embedding_function=embeddings)

def initiate_embed():
    loader = PyPDFLoader("Handbook 2018.pdf")
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter()
    all_splits = text_splitter.split_documents(pages)

    # mistral has a limit of 16384 tokens per request so we need to split the document into chunks
    # and then embed each chunk individually and then combine the chunks into a single embedding
    # this will be a very long process so we should save the embeddings to disk and then load them later

    i = -1

    for split in all_splits:
        i += 1
        print(f"Embedding split {i+1} of {len(all_splits)}")
        Chroma.from_documents(documents=[split], embedding=embeddings, persist_directory="./embeddings_db")
        if i % 2 == 0:
            time.sleep(1) # sleep for 1 second every other split to avoid hitting the mistral rate limit

if __name__ == "__main__":
    initiate_embed()
