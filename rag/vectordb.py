# rag/vectordb.py
"""
ChromaDB vector store management.
Handles collection creation, persistence, and client access.
"""

import os
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from rag.embeddings import get_embedding_function

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_PERSIST_DIR  = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME      = "python_pandas_docs"


def get_chroma_client(persist_dir: str = DEFAULT_PERSIST_DIR) -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client."""
    os.makedirs(persist_dir, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir)


def get_vectorstore(persist_dir: str = DEFAULT_PERSIST_DIR) -> Chroma:
    """
    Return a LangChain Chroma vectorstore backed by the local persist directory.
    Works whether or not documents have been ingested yet.
    """
    embedding_fn = get_embedding_function()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        persist_directory=persist_dir,
    )


def collection_exists(persist_dir: str = DEFAULT_PERSIST_DIR) -> bool:
    """Return True if the ChromaDB collection has been populated."""
    if not os.path.exists(persist_dir):
        return False
    try:
        client = get_chroma_client(persist_dir)
        col = client.get_collection(COLLECTION_NAME)
        return col.count() > 0
    except Exception:
        # Collection does not exist yet — that is expected on first run
        return False
