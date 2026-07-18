import chromadb
from chromadb.utils import embedding_functions

_client: chromadb.ClientAPI | None = None


def get_client(persist_dir: str) -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=persist_dir)
    return _client


def _embedding_fn():
    # ONNX all-MiniLM-L6-v2, runs on CPU, cached under $HOME on first use.
    # No Ollama / second LLM required for embeddings.
    return embedding_functions.DefaultEmbeddingFunction()


def get_reference_collection(persist_dir: str):
    client = get_client(persist_dir)
    return client.get_or_create_collection(
        "prompt_reference", embedding_function=_embedding_fn()
    )


def get_history_collection(persist_dir: str):
    client = get_client(persist_dir)
    return client.get_or_create_collection(
        "prompt_history", embedding_function=_embedding_fn()
    )
