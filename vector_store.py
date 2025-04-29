from chromadb import PersistentClient

def load_vector_store():
    """
    Load the vector store from the local database.
    """
    client = PersistentClient(path="./embeddings_db")
    vector_store = client.get_or_create_collection("documents")
    return vector_store